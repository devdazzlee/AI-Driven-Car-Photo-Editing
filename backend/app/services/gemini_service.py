"""
Gemini API service for car image processing.

Uses Gemini API (gemini-3.1-flash-image-preview) for all image processing.
Single API call: remove reflections, clean floor, maintain car color, keep walls/floor intact.
"""

import base64
import io
import logging
import time
import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

GEMINI_MODEL = "gemini-3.1-flash-image-preview"
REQUEST_TIMEOUT_MS = 360_000  # 6 minutes
MAX_INPUT_SIZE = 1024  # Match Gemini's 1K output size
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10

# --- Prompts ---

ENHANCE_PROMPT = (
    "Professional automotive photo retouching. Perform the following edits perfectly:\n\n"

    "1. STUDIO BACKGROUND: Replace all background walls with a smooth, clean, pure white studio background. Remove all background clutter, doors, and lights.\n"
    "2. FLOOR: Clean the floor of dirt and tire tracks, but maintain its exact original texture, pattern, and perspective.\n"
    "3. REMOVE UGLY GLARE: Find every bright white studio light reflection, brilliant specular highlight, and white glare square on the car's body (hood, roof, sides) and glass. Eliminate them completely.\n"
    "4. RESTORE PAINT: Wherever you removed the white glare, seamlessly paint in the true native color of the car (e.g., if the car is black, fill those spots with deep black paint). Blend it perfectly into the surrounding panels.\n"
    "5. GLOSSY FINISH: The car must remain deeply glossy, reflective, and beautifully polished. Maintain its exact original contour and shape.\n\n"

    "Return ONLY the edited image."
)

BACKGROUND_REMOVAL_PROMPT = (
    "Remove the background from this car photo. "
    "Replace the background with a clean solid white background. "
    "Keep the car exactly as it is - preserve all details, colors, and reflections. "
    "Return only the edited image, no text."
)

BACKGROUND_REMOVAL_TRANSPARENT_PROMPT = (
    "Remove the background from this car photo. "
    "Make the background fully transparent. "
    "Keep the car exactly as it is - preserve all details, colors, and reflections. "
    "Return only the edited image with transparent background, no text."
)

# Gemini 3.1 Flash Image only accepts these aspect ratios
_ALLOWED_ASPECT_RATIOS = (
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4",
    "8:1", "9:16", "16:9", "21:9",
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resize_for_api(pil_img: Image.Image, max_side: int = MAX_INPUT_SIZE) -> Image.Image:
    """Resize image so longest side <= max_side."""
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    if w >= h:
        new_w, new_h = max_side, int(h * max_side / w)
    else:
        new_w, new_h = int(w * max_side / h), max_side
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _aspect_ratio_str(w: int, h: int) -> str:
    """Map image dimensions to nearest allowed Gemini aspect ratio."""
    if h == 0:
        return "1:1"
    actual = w / h
    best_ratio = "1:1"
    best_diff = float("inf")
    for ratio in _ALLOWED_ASPECT_RATIOS:
        num, den = map(int, ratio.split(":"))
        target = num / den
        diff = abs(actual - target)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
    return best_ratio


def _get_client():
    """Lazy-load Gemini client."""
    from google import genai
    api_key = GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required. Set it in backend/.env")
    return genai.Client(api_key=api_key)


def _extract_image_from_response(response) -> Image.Image:
    """Extract PIL Image from Gemini API response."""
    result_bytes = None
    parts = getattr(response, "parts", None) or (
        response.candidates[0].content.parts if response.candidates else []
    )
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            result_bytes = data if isinstance(data, bytes) else base64.b64decode(data)
            break
    if result_bytes is None:
        raise RuntimeError("Gemini did not return an image")
    return Image.open(io.BytesIO(result_bytes)).convert("RGB")


def _pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    """Convert PIL Image to JPEG bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _call_gemini_with_retry(client, prompt: str, img_bytes: bytes, aspect: str, label: str):
    """Call Gemini API with retry logic for server errors and timeouts."""
    from google.genai import types

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Gemini API attempt %d/%d for %s", attempt, MAX_RETRIES, label)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.85,
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=aspect),
                    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
                ),
            )
            return response
        except Exception as e:
            error_str = str(e)
            is_retryable = any(code in error_str for code in (
                "503", "500", "502", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                "timed out", "ReadTimeout", "TimeoutError",
            ))
            if is_retryable and attempt < MAX_RETRIES:
                logger.warning(
                    "AI server busy (attempt %d/%d) for %s: %s — retrying in %ds...",
                    attempt, MAX_RETRIES, label, error_str, RETRY_DELAY_SECONDS,
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            logger.error("Gemini API failed after %d attempts for %s: %s", attempt, label, error_str)
            raise RuntimeError(
                "Processing failed due to high server demand. Please try again in a few minutes."
            ) from e


# ---------------------------------------------------------------------------
# Post-processing checks
# ---------------------------------------------------------------------------

def _find_car_center_x(gray: np.ndarray) -> float:
    """Find horizontal center-of-mass of the darkest region (car body)."""
    threshold = np.median(gray) * 0.85
    dark_mask = (gray < threshold).astype(np.float64)
    col_weights = dark_mask.sum(axis=0)
    total = col_weights.sum()
    if total == 0:
        return 0.5
    x_coords = np.arange(gray.shape[1], dtype=np.float64)
    return (col_weights * x_coords).sum() / total / gray.shape[1]


def _is_flipped(original: Image.Image, result: Image.Image) -> bool:
    """Detect horizontal flip using car mass position and column correlation."""
    size = (128, 128)
    orig_gray = np.array(original.resize(size).convert("L"), dtype=np.float64)
    res_gray = np.array(result.resize(size).convert("L"), dtype=np.float64)

    # Method 1: Car mass center-of-gravity
    orig_cx = _find_car_center_x(orig_gray)
    res_cx = _find_car_center_x(res_gray)
    mass_flipped = False
    cx_diff = abs(orig_cx - res_cx)
    if cx_diff > 0.1:
        mirrored_cx = 1.0 - res_cx
        if abs(orig_cx - mirrored_cx) < abs(orig_cx - res_cx):
            mass_flipped = True

    # Method 2: Column profile correlation
    orig_profile = orig_gray.mean(axis=0)
    res_profile = res_gray.mean(axis=0)
    res_flipped = res_profile[::-1]
    normal_corr = np.corrcoef(orig_profile, res_profile)[0, 1]
    flipped_corr = np.corrcoef(orig_profile, res_flipped)[0, 1]
    corr_flipped = flipped_corr > normal_corr and abs(flipped_corr - normal_corr) > 0.02

    is_flip = mass_flipped or corr_flipped
    if is_flip:
        logger.warning("Flip detected (mass=%s, corr=%s) — correcting", mass_flipped, corr_flipped)
    return is_flip


def _apply_brightness(pil_img: Image.Image, boost: float) -> Image.Image:
    """Apply brightness boost to image. boost=1.0 means no change, 1.5 means 50% brighter."""
    if boost <= 1.0:
        return pil_img
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(pil_img)
    result = enhancer.enhance(boost)
    logger.info("Applied brightness boost: %.2f", boost)
    return result


def _get_car_mask_rembg(pil_img: Image.Image) -> np.ndarray:
    """
    Return a boolean car mask using rembg.

    Threshold at > 128 keeps only high-confidence car pixels.
    The strict threshold is intentional: _restore_car_color builds an extended
    application mask internally, so median shifts are always computed on clean
    car pixels only.
    """
    try:
        from rembg import remove
        mask_pil = remove(pil_img, only_mask=True)
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")
        return np.array(mask_pil) > 128
    except Exception as e:
        logger.error("rembg mask generation failed: %s", e)
        return np.ones((pil_img.height, pil_img.width), dtype=bool)


def _restore_car_color(
    original: Image.Image,
    gemini_result: Image.Image,
    mask_np: np.ndarray,
) -> Image.Image:
    """
    Restore the original car color to Gemini's output using LAB median shifts.

    Gemini handles background removal / reflection cleanup fully. This function
    corrects the color drift and boundary mis-renders Gemini introduces.

    Only called for standard background-removal mode. Enhance-preserve mode relies
    entirely on Gemini's output — applying a LAB shift there would re-brighten
    areas Gemini intentionally darkened (reflection removal), undoing Gemini's work.

    Problems addressed:
      (A) Colored car boundary bleed (lower bumper / spoiler against dark floor):
          rembg assigns low confidence → pixels excluded from strict mask → Gemini
          background-white bleeds into car edge. Fixed by dilation + AB chroma guard.

      (B) Neutral car (black/grey) grey halo on body panels:
          rembg assigns low confidence to bright-reflection areas near white backdrop
          → excluded from strict mask. The global shift is too small to fix these.
          Fixed by dilation + Gemini-brightness guard (grey zone L < 195 = car surface;
          pure white L ≥ 195 = true background) + bidirectional deviation correction.

      (C) Large local mis-renders in standard mode (standard only):
          After global shift, pixels where |result_L − orig_L| > 40 are Gemini
          mistakes (car surface wrongly brightened or darkened). Blended 85% back to
          original to fix grey halos and white boundary bleeds simultaneously.

    Median vs mean:
      Reflection pixels (bright white, AB ≈ 128) are outliers inside the car mask.
      Mean is pulled toward neutral; median ignores outliers and always tracks the
      dominant paint color, making the correction universal across all car colors.
    """
    orig_np   = np.array(original).astype(np.uint8)
    gemini_np = np.array(gemini_result).astype(np.uint8)

    orig_lab   = cv2.cvtColor(orig_np,   cv2.COLOR_RGB2Lab).astype(np.float32)
    gemini_lab = cv2.cvtColor(gemini_np, cv2.COLOR_RGB2Lab).astype(np.float32)

    result_lab = gemini_lab.copy()  # start from Gemini output; background stays intact

    if not mask_np.any():
        return gemini_result

    # --- Step 1: compute median shifts using the strict car mask ---
    shifts: dict[int, float] = {}
    for ch, name in [(0, "L"), (1, "A"), (2, "B")]:
        orig_median = float(np.median(orig_lab[mask_np, ch]))
        gem_median  = float(np.median(gemini_lab[mask_np, ch]))
        shifts[ch]  = orig_median - gem_median
        logger.info("LAB %s median-shift: %+.1f  (orig=%.1f  gem=%.1f)",
                    name, shifts[ch], orig_median, gem_median)

    # --- Step 2: build extended application mask ---
    mask_uint8 = mask_np.astype(np.uint8) * 255

    # (a) Closing fills concave voids within the car silhouette
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed_mask  = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel) > 128

    car_median_a = float(np.median(orig_lab[mask_np, 1]))
    car_median_b = float(np.median(orig_lab[mask_np, 2]))
    car_chroma   = np.sqrt((car_median_a - 128.0) ** 2 + (car_median_b - 128.0) ** 2)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask  = cv2.dilate(mask_uint8, dilate_kernel, iterations=2) > 128
    boundary_zone = dilated_mask & ~closed_mask

    if car_chroma > 12:
        # (b-colored) Colored car: AB chroma guard prevents including neutral floor/background.
        ab_tolerance = max(22.0, car_chroma * 0.9)
        ab_dist = np.sqrt(
            (orig_lab[:, :, 1] - car_median_a) ** 2 +
            (orig_lab[:, :, 2] - car_median_b) ** 2
        )
        extended_mask = closed_mask | (boundary_zone & (ab_dist < ab_tolerance))
        logger.info("Colored car (chroma=%.1f): dilation + AB guard (tol=%.1f)",
                    car_chroma, ab_tolerance)
    else:
        # (b-neutral) Neutral car (black/grey/white/silver): chroma cannot distinguish
        # car from background. Instead guard on Gemini-output brightness:
        #   grey zone (L < 195) = car surface Gemini mis-rendered → include
        #   pure white (L ≥ 195) = true background Gemini correctly placed → exclude
        gem_not_bg = gemini_lab[:, :, 0] < 195
        extended_mask = closed_mask | (boundary_zone & gem_not_bg)
        logger.info("Neutral car (chroma=%.1f): dilation + Gemini-brightness guard", car_chroma)

    # --- Step 3: apply global median shifts to extended mask ---
    for ch in [0, 1, 2]:
        result_lab[extended_mask, ch] = np.clip(
            gemini_lab[extended_mask, ch] + shifts[ch], 0, 255
        )

    # --- Step 4: pixel-level deviation correction ---
    # The global shift handles overall color drift but cannot fix large local
    # Gemini mis-renders (grey halos on dark cars, whitened boundary panels).
    # In standard mode Gemini must NOT change car appearance — any pixel
    # deviating > 40 LAB-L from the original is a Gemini mistake: blend 85% back.
    l_orig   = orig_lab[:, :, 0]
    l_result = result_lab[:, :, 0]
    deviation = l_result - l_orig  # +ve = Gemini over-brightened, -ve = Gemini darkened
    large_dev = extended_mask & (np.abs(deviation) > 40)
    if large_dev.any():
        for ch in [0, 1, 2]:
            result_lab[large_dev, ch] = (
                orig_lab[large_dev, ch] * 0.85 +
                result_lab[large_dev, ch] * 0.15
            )
        logger.info(
            "Deviation correction: %d pixels (over-bright=%d, over-dark=%d)",
            int(large_dev.sum()),
            int((extended_mask & (deviation > 40)).sum()),
            int((extended_mask & (deviation < -40)).sum()),
        )

    result_rgb = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
    logger.info("Car color restored via LAB median-shifts with extended mask")
    return Image.fromarray(result_rgb)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_car_image(
    image_data: bytes,
    filename: str,
    mode: str = "enhance-preserve",
    output_format: str = "png",
    background: str = "white",
    lighting_boost: float = 1.0,
) -> bytes:
    """
    Process car image.

    enhance-preserve: single Gemini call to clean background, remove reflections, keep floor/walls
    standard: full image to Gemini for background removal
    """
    from app.services.image_utils import load_image

    pil_img = load_image(image_data, filename)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    orig_w, orig_h = pil_img.size

    # Resize large images for API
    pil_img = _resize_for_api(pil_img)

    if mode == "enhance-preserve":
        prompt = ENHANCE_PROMPT
    elif background == "transparent":
        prompt = BACKGROUND_REMOVAL_TRANSPARENT_PROMPT
    else:
        prompt = BACKGROUND_REMOVAL_PROMPT

    w, h = pil_img.size
    aspect = _aspect_ratio_str(w, h)
    img_bytes = _pil_to_jpeg_bytes(pil_img)
    client = _get_client()

    response = _call_gemini_with_retry(client, prompt, img_bytes, aspect, filename)
    result_pil = _extract_image_from_response(response)

    # Resize result to match input dimensions
    if result_pil.size != pil_img.size:
        result_pil = result_pil.resize(pil_img.size, Image.Resampling.LANCZOS)

    # Post-processing color restoration — standard background-removal mode only.
    # In enhance-preserve mode Gemini owns the car appearance completely (removes
    # reflections, corrects color). Applying a LAB shift on top would re-brighten
    # areas Gemini intentionally darkened, undoing the reflection removal.
    # For standard mode (background swap to white) Gemini must not change the car,
    # so any color drift or boundary bleed is corrected here.
    if mode != "enhance-preserve" and background != "transparent":
        mask_np = _get_car_mask_rembg(pil_img)
        result_pil = _restore_car_color(pil_img, result_pil, mask_np)
        if result_pil.size != pil_img.size:
            result_pil = result_pil.resize(pil_img.size, Image.Resampling.LANCZOS)

    # Post-processing: flip detection
    if _is_flipped(pil_img, result_pil):
        result_pil = ImageOps.mirror(result_pil)

    # Apply brightness boost if requested
    result_pil = _apply_brightness(result_pil, lighting_boost)

    # Convert to requested format
    out_buf = io.BytesIO()
    fmt = output_format.lower()
    if fmt == "png":
        result_pil.save(out_buf, format="PNG")
    elif fmt in ("jpg", "jpeg"):
        result_pil.save(out_buf, format="JPEG", quality=95)
    elif fmt == "webp":
        result_pil.save(out_buf, format="WEBP", quality=95)
    else:
        result_pil.save(out_buf, format="PNG")

    return out_buf.getvalue()
