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

GEMINI_MODEL = "gemini-3-pro-image-preview"
REQUEST_TIMEOUT_MS = 360_000  # 6 minutes
MAX_INPUT_SIZE = 1024  # Match Gemini's 1K output size
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10


# --- Prompts ---

ENHANCE_PROMPT = (
    "Edit this car dealership photo with these exact instructions:\n\n"

    "1. COMPOSITION AND FRAMING (MOST IMPORTANT RULE — THIS IS CRITICAL):\n"
    "   - Do NOT change the camera angle, perspective, zoom level, or framing in any way.\n"
    "   - The car must appear at the exact same size, position and angle as the original.\n"
    "   - Do not zoom in or out. Do not reframe or reposition the car.\n"
    "   - Do NOT flip or mirror the image horizontally or vertically.\n"
    "   - Return the image at exactly the same dimensions as the input.\n"
    "   - Never cut off bumpers, mirrors, roof, hood, trunk or any other part of the car.\n"
    "   - Every part visible in the original must remain visible in the output.\n"
    "   - Do NOT crop, cut off or remove any part of the car. The complete car must be fully visible "
    "in the final image exactly as in the original. If the car extends to the edge of the original "
    "image, it must extend to the same edge in the output. Cutting off any part of the car is "
    "unacceptable.\n\n"

    "2. BACKGROUND — WALLS AND CEILING:\n"
    "   Remove ALL objects from walls and ceiling without exception. This includes garage doors, "
    "door frames, hinges, handles, studio lights, light fixtures, ceiling rigs, lighting rails, "
    "cables, wires, trolleys, carts, and any other equipment. "
    "Make walls and ceiling completely flat, solid white — a single uniform white with no variation. "
    "The entire background must be one flat white color from top to bottom and edge to edge. "
    "Do NOT add any gradients, curved shadows, arcs, dark transitions, or tonal variations anywhere "
    "in the background. Do NOT render any room shape, ceiling curve, or architectural feature. "
    "Do NOT add any vignetting or darkening near the top or edges of the image. "
    "The background must look like a pure flat white photography backdrop — perfectly uniform. "
    "The only exception is the wall-floor corner which must remain visible and natural.\n\n"

    "3. BACKGROUND — FLOOR (ABSOLUTE RULE — DO NOT VIOLATE):\n"
    "   THE FLOOR COLOR, BRIGHTNESS, AND TONE MUST BE 100% IDENTICAL TO THE ORIGINAL IMAGE.\n"
    "   This is non-negotiable. The floor in your output must look like the same physical floor "
    "as in the original photo — same color, same shade, same brightness level.\n"
    "   What you MAY do to the floor (ONLY these things):\n"
    "     - Remove puddles, standing water, wet spots, and moisture reflections\n"
    "     - Remove visible tire marks, oil stains, or dirt smudges\n"
    "   What you MUST NEVER do to the floor:\n"
    "     - NEVER change the floor color (not even slightly)\n"
    "     - NEVER lighten or brighten the floor\n"
    "     - NEVER darken the floor\n"
    "     - NEVER make the floor look uniform, studio-style, or flat\n"
    "     - NEVER replace the floor material or texture\n"
    "     - NEVER add gradients or vignette to the floor\n"
    "   The floor's natural variation in tone (from perspective, distance, shadows) is CORRECT "
    "and must be preserved exactly as in the original. Do not flatten or homogenize it.\n"
    "   EXACT COLOR RULE: The floor RGB color will be provided below. Your output floor "
    "MUST match it. If the floor is dark grey (e.g. RGB 80,80,80) — keep it dark grey. "
    "If the floor is light grey — keep it light grey. If the floor is concrete-colored — "
    "keep it concrete-colored. Any floor color change from the original is unacceptable.\n\n"

    "4. REFLECTIONS:\n"
    "   Remove all white light reflections from the hood, roof, doors and fenders. "
    "Remove all glare and bright spots from ALL glass surfaces: front windshield, rear window, "
    "side windows, and quarter windows. Every glass surface must look dark, deep and clear "
    "with no bright reflections, no white glare patches and no light spots of any kind. "
    "The rear window in particular must be dark and reflection-free.\n"
    "   IMPORTANT — DO NOT TOUCH CHROME OR METALLIC TRIM: "
    "Chrome moldings, chrome strips, chrome accents, metallic trim pieces, and any decorative metallic elements on the car body are NOT reflections. "
    "Do NOT remove, dull, darken, or alter any chrome or metallic trim. "
    "These are design features of the car and must remain exactly as they appear in the original — shiny, reflective and bright. "
    "Only remove unwanted studio light reflections from painted body panels. "
    "Never touch any chrome or metallic decorative elements.\n\n"

    "5. CAR COLOR:\n"
    "   Keep the exact original car color including its vibrancy, saturation and brightness. "
    "Do not flatten or dull the paint finish. Do not darken or lighten the paint. "
    "Do not change the hue or saturation.\n\n"

    "6. TIRES:\n"
    "   Make tires deep black and clean. Remove all dust and discoloration from rubber surface only.\n\n"

    "7. WHEELS AND RIMS:\n"
    "   Do NOT alter, distort, fill or change wheel covers, hubcaps, rims or spokes in any way. "
    "Do not fill wheels with black. Do not remove rim details or spokes. "
    "Do not add or duplicate wheel covers that are not in the original. "
    "Keep all wheel details exactly as in the original. Any wheel distortion is unacceptable. "
    "If you cannot clean tires without distorting wheels, leave wheels exactly as they are.\n\n"

    "8. COLOR ACCURACY (THIS IS CRITICAL):\n"
    "   Do not add any rainbow effects, color shifts, prismatic colors or any color distortion "
    "to any part of the image. All colors must remain true to the original. "
    "If you cannot process a specific area without causing color distortion, leave that area "
    "exactly as it is in the original.\n\n"

    "9. VIBRANCY AND SATURATION:\n"
    "   The car paint must look vibrant, rich and natural exactly like the original. "
    "Do not reduce the saturation or vibrancy of the car paint. Do not make the car "
    "look flat, dull or matte. The car should look shiny and vibrant exactly as it did "
    "in the original photo.\n\n"

    "10. CAR PARTS, COMPONENTS AND BODY PANELS (THIS IS ABSOLUTELY CRITICAL):\n"
    "   - Do NOT change the color of ANY car part or component. "
    "Headlights, tail lights, headlight surrounds, grille, grille surround, bumpers, body panels, "
    "hood, roof, doors, door handles, side mirrors, pillar trim, step bars, running boards, "
    "and every other component must remain EXACTLY the same color as in the original photo. "
    "No color changes to any car part are acceptable under any circumstances.\n"
    "   - Do NOT add, create, invent or hallucinate any car parts, badges, decorative elements, "
    "chrome surrounds, lighting elements, or any other components that are NOT visible in the "
    "original image. You must never add anything new to the car — only clean the background.\n"
    "   - Do NOT repaint, recolor, alter or modify any part of the car body, trim, or components. "
    "The car must look physically identical to the original — same parts, same colors, same finish. "
    "If a part is black in the original it must stay black. "
    "If a part is grey in the original it must stay grey. "
    "If a part is chrome in the original it must stay chrome. "
    "ONLY the background (walls and floor) and unwanted light reflections on painted panels should change. "
    "Every single car component must be pixel-identical in color and shape to the original.\n\n"

    "Return only the edited image with no text or watermarks."
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
                    temperature=0.0,
                    top_p=1.0,
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=aspect, image_size="1K"),
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


def _validate_composition(original: Image.Image, result: Image.Image) -> bool:
    """
    Check that Gemini kept the car in the same position and scale.

    Uses normalized 2D cross-correlation at 64×64 thumbnail scale.
    A value above 0.60 means the two images are structurally aligned.

    Why this matters for the diff composite:
      diff = gemini - original is upscaled 6× to full resolution.
      If Gemini shifted the car 5px at 1024px, that becomes a 30px shift at 6000px.
      The diff then contains bright/dark halos around every car edge.
      Applied to the full-res original those halos create a ghosted double-car.
      This check detects the misalignment before compositing so we can retry.
    """
    size = (64, 64)
    orig_arr = np.array(original.resize(size).convert("L"), dtype=np.float64)
    res_arr  = np.array(result.resize(size).convert("L"), dtype=np.float64)

    orig_norm = orig_arr - orig_arr.mean()
    res_norm  = res_arr  - res_arr.mean()
    orig_std  = orig_norm.std()
    res_std   = res_norm.std()

    if orig_std < 1e-6 or res_std < 1e-6:
        return True  # uniform image, can't measure — pass through

    correlation = float((orig_norm * res_norm).mean() / (orig_std * res_std))
    logger.info("Composition correlation: %.3f", correlation)
    return correlation > 0.60


def _check_color_accuracy(original: Image.Image, result: Image.Image, client,
                          prompt: str, img_bytes: bytes, aspect: str, label: str,
                          mask_np: np.ndarray | None = None) -> Image.Image:
    """
    If car color drifted > 5%, retry Gemini once and keep the better result.

    Uses car mask pixels only — whole-image average includes background which Gemini
    changes intentionally (walls, floor), causing false positives that skip needed retries
    and miss real car color changes (white → silver) that are hidden in the average.
    """
    orig_np = np.array(original)
    res_np  = np.array(result)

    if mask_np is not None and mask_np.any():
        # Resize mask to match the 1024px images
        mh, mw = mask_np.shape
        ih, iw = orig_np.shape[:2]
        if (mh, mw) != (ih, iw):
            mask_resized = np.array(
                Image.fromarray(mask_np.astype(np.uint8) * 255, mode='L').resize((iw, ih))
            ) > 128
        else:
            mask_resized = mask_np
        orig_car = orig_np[mask_resized].astype(np.float32)
        res_car  = res_np[mask_resized].astype(np.float32)
        if orig_car.size == 0:
            orig_car = orig_np.reshape(-1, 3).astype(np.float32)
            res_car  = res_np.reshape(-1, 3).astype(np.float32)
    else:
        orig_car = orig_np.reshape(-1, 3).astype(np.float32)
        res_car  = res_np.reshape(-1, 3).astype(np.float32)

    orig_avg   = orig_car.mean(axis=0)
    res_avg    = res_car.mean(axis=0)
    color_diff = np.abs(orig_avg - res_avg).mean() / 255.0

    logger.info(
        "Car color check: orig RGB(%.0f,%.0f,%.0f) vs gemini RGB(%.0f,%.0f,%.0f) drift=%.1f%%",
        orig_avg[0], orig_avg[1], orig_avg[2],
        res_avg[0],  res_avg[1],  res_avg[2],
        color_diff * 100,
    )

    if color_diff <= 0.05:  # 5% threshold — catches white→silver (≈25% drift)
        return result

    logger.warning("Car color drift %.1f%% exceeds 5%% for %s — retrying once", color_diff * 100, label)
    try:
        retry_response = _call_gemini_with_retry(client, prompt, img_bytes, aspect, f"{label}[retry]")
        retry_pil      = _extract_image_from_response(retry_response)
        retry_np       = np.array(retry_pil)
        if mask_np is not None and mask_np.any():
            rh, rw = retry_np.shape[:2]
            if (mh, mw) != (rh, rw):
                rmask = np.array(
                    Image.fromarray(mask_np.astype(np.uint8) * 255, mode='L').resize((rw, rh))
                ) > 128
            else:
                rmask = mask_resized
            retry_car  = retry_np[rmask].astype(np.float32)
            retry_avg  = retry_car.mean(axis=0) if retry_car.size else retry_np.reshape(-1,3).astype(np.float32).mean(axis=0)
        else:
            retry_avg  = retry_np.reshape(-1, 3).astype(np.float32).mean(axis=0)
        retry_diff = np.abs(orig_avg - retry_avg).mean() / 255.0
        if retry_diff < color_diff:
            logger.info("Retry better car color (%.1f%% vs %.1f%%) for %s", retry_diff * 100, color_diff * 100, label)
            return retry_pil
        logger.info("Original result better car color (%.1f%% vs %.1f%%) for %s", color_diff * 100, retry_diff * 100, label)
    except Exception:
        logger.warning("Color retry failed for %s, using first result", label)
    return result


def _get_car_mask_rembg(pil_img: Image.Image) -> np.ndarray:
    """Get boolean mask of the car using rembg."""
    try:
        from rembg import remove
        mask_pil = remove(pil_img, only_mask=True)
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')
        return np.array(mask_pil) > 128
    except Exception as e:
        logger.error("Failed to generate RMBG mask: %s", e)
        return np.ones((pil_img.height, pil_img.width), dtype=bool)


def _get_average_color(pil_img: Image.Image, mask_np: np.ndarray) -> np.ndarray:
    """Get average RGB color of the masked area."""
    img_np = np.array(pil_img)
    if not mask_np.any():
        return np.array([0.0, 0.0, 0.0])
    return img_np[mask_np].mean(axis=0)


def _restore_car_color(original: Image.Image, gemini_result: Image.Image, mask_np: np.ndarray) -> Image.Image:
    """
    Restore original car color to Gemini's fully-processed output using LAB mean shifts.

    Gemini handles 100% of the work: reflection removal, background cleaning, everything.
    This function ONLY corrects the color drift Gemini introduces via pure mean shifts —
    no std normalization, no scaling — preserving Gemini's structure (reflections removed).

    Why mean-shift only (no std normalization):
      std normalization multiplies every pixel by orig_std/gem_std.
      For black/neutral cars this ratio is unstable (both stds are near zero)
      and amplifies noise into a visible color cast (e.g. black → brownish).
      A plain mean shift moves the whole color distribution by a fixed offset —
      stable for every car color including black, white, and grey.

    LAB channels — car area only:
      L  (brightness) : shift mean → fixes dullness, keeps Gemini's local structure
                         (reflection spots stay dark = reflections stay removed).
      A  (red↔green)  : shift mean → corrects hue drift (e.g. red→brown).
      B  (yellow↔blue): shift mean → corrects hue drift (e.g. blue cast).
      Background       : untouched — Gemini's clean background kept as-is.
    """
    orig_np   = np.array(original).astype(np.uint8)
    gemini_np = np.array(gemini_result).astype(np.uint8)

    orig_lab   = cv2.cvtColor(orig_np,   cv2.COLOR_RGB2Lab).astype(np.float32)
    gemini_lab = cv2.cvtColor(gemini_np, cv2.COLOR_RGB2Lab).astype(np.float32)

    result_lab = gemini_lab.copy()   # start with full Gemini output (background intact)

    if not mask_np.any():
        return gemini_result

    # Median is used instead of mean because reflection pixels (bright white, A≈128, B≈128)
    # are outliers in the car mask. Mean gets pulled toward neutral by those outliers,
    # causing wrong shifts for colored cars (red→brown, blue→grey, etc.).
    # Median ignores outliers — it always reflects the dominant car paint color,
    # making this correction universal across all car colors.
    for ch, name in [(0, "L"), (1, "A"), (2, "B")]:
        orig_median = np.median(orig_lab[mask_np, ch])
        gem_median  = np.median(gemini_lab[mask_np, ch])
        shift       = orig_median - gem_median
        result_lab[mask_np, ch] = np.clip(gemini_lab[mask_np, ch] + shift, 0, 255)
        logger.info("LAB %s median-shift: %+.1f  (orig=%.1f  gem=%.1f)", name, shift, orig_median, gem_median)

    result_rgb = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
    logger.info("Car color restored via LAB median-shifts — works for all car colors")
    return Image.fromarray(result_rgb)


def _sample_floor_color(pil_img: Image.Image, car_mask_np: np.ndarray) -> tuple[int, int, int]:
    """
    Sample the average floor color from the original image.
    Floor = background pixels in the bottom 40% of the frame.
    Returns (R, G, B) as integers.
    """
    h, w = car_mask_np.shape
    background_mask = ~car_mask_np
    floor_row_start = int(h * 0.60)
    floor_row_mask = np.zeros((h, w), dtype=bool)
    floor_row_mask[floor_row_start:, :] = True
    floor_mask = background_mask & floor_row_mask
    if not floor_mask.any():
        return (200, 200, 200)
    img_np = np.array(pil_img)
    avg = img_np[floor_mask].mean(axis=0)
    return (int(avg[0]), int(avg[1]), int(avg[2]))


def _restore_floor_from_original(
    original: Image.Image,
    processed: Image.Image,
    car_mask_np: np.ndarray,
) -> Image.Image:
    """
    Correct Gemini's floor color recoloring while keeping Gemini's cleaned floor texture.

    Root cause history:
      - Corner sampling FAILED: studio photos have white backdrop curtains in corners.
        Original corner = white curtain (RGB≈230), Gemini corner = white wall (RGB≈235).
        Drift = 5 → "negligible" → no correction. But actual floor centre changed by 90 units.
      - Pixel replacement FAILED: restores original dirt/wet spots.
      - Affine (mean+std) FAILED: orig_std >> proc_std on wet concrete → scale blows up.

    Correct approach — background mask + centre-strip floor sampling:
      1. Use rembg car mask to identify background (non-car) pixels.
      2. Sample floor from the CENTRE strip of background pixels in the bottom 40% of frame.
         Centre strip = middle 60% of width, excluding 20% margins on each side.
         This avoids backdrop curtains (which hang at edges) and reliably hits actual floor.
      3. If the centre-strip sample is bright (avg L > 200) AND the original image corners
         are dark (avg L < 150), the centre strip caught backdrop — fall back to full
         background-mask floor pixels (better than nothing).
      4. Apply per-channel mean-shift to all background floor pixels.
         Gemini recolors floor uniformly so global mean-shift is exact.
    """
    if original.size != processed.size:
        original = original.resize(processed.size, Image.Resampling.LANCZOS)

    h, w = car_mask_np.shape
    orig_np  = np.array(original).astype(np.float32)
    proc_np  = np.array(processed).astype(np.float32)

    background_mask = ~car_mask_np

    # --- Define floor zone: background pixels in bottom 40% of frame ---
    floor_row_start = int(h * 0.60)
    floor_row_mask  = np.zeros((h, w), dtype=bool)
    floor_row_mask[floor_row_start:, :] = True
    full_floor_mask = background_mask & floor_row_mask

    # --- Centre strip: exclude 20% margins on each side (backdrop zone) ---
    margin = int(w * 0.20)
    centre_col_mask = np.zeros((h, w), dtype=bool)
    centre_col_mask[:, margin: w - margin] = True
    centre_floor_mask = full_floor_mask & centre_col_mask

    # Choose sampling region: prefer centre strip, fall back to full floor mask
    if centre_floor_mask.sum() >= 200:
        sample_mask = centre_floor_mask
        sample_label = "centre-strip"
    elif full_floor_mask.sum() >= 200:
        sample_mask = full_floor_mask
        sample_label = "full-floor-bg"
    else:
        logger.info("Floor color: too few background floor pixels, skipping")
        return processed

    orig_sample = orig_np[sample_mask]   # (N, 3)
    proc_sample = proc_np[sample_mask]

    orig_floor_mean = orig_sample.mean(axis=0)
    proc_floor_mean = proc_sample.mean(axis=0)
    drift           = orig_floor_mean - proc_floor_mean
    max_drift       = np.abs(drift).max()

    logger.info(
        "Floor color [%s]: orig RGB(%.0f,%.0f,%.0f) → gemini RGB(%.0f,%.0f,%.0f) "
        "drift R%+.1f G%+.1f B%+.1f",
        sample_label,
        orig_floor_mean[0], orig_floor_mean[1], orig_floor_mean[2],
        proc_floor_mean[0], proc_floor_mean[1], proc_floor_mean[2],
        drift[0], drift[1], drift[2],
    )

    if max_drift <= 5:
        logger.info("Floor color: drift %.1f — negligible, no correction", max_drift)
        return processed

    # Apply mean-shift to ALL background floor pixels (not just sample zone)
    result_np = proc_np.copy()
    result_np[full_floor_mask] = np.clip(proc_np[full_floor_mask] + drift, 0, 255)

    return Image.fromarray(result_np.astype(np.uint8))


def _apply_brightness(pil_img: Image.Image, boost: float) -> Image.Image:
    """Apply brightness boost to image. boost=1.0 means no change, 1.5 means 50% brighter."""
    if boost <= 1.0:
        return pil_img
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(pil_img)
    result = enhancer.enhance(boost)
    logger.info("Applied brightness boost: %.2f", boost)
    return result


def _calculate_lab_shifts(
    original: Image.Image, gemini_result: Image.Image, mask_np: np.ndarray
) -> tuple:
    """
    Calculate LAB median shifts needed to correct Gemini's color drift.
    Returns (shift_L, shift_A, shift_B).

    Resolution-independent: calculated at 1024px but valid at any resolution because
    the shift is a scalar offset applied uniformly — not pixel-position dependent.

    Why we filter mask pixels by brightness percentile:
      The rembg car mask captures ALL car pixels: body paint, open trunk interior,
      window glass, tire rubber. When a trunk is open the dark interior (L≈5-30) is
      included. Its low L drags the median down and produces a large spurious negative
      L shift that DARKENS the entire car. Example: orig L=29 (trunk-dominated),
      gem L=60 → shift=-31 → white car body shifts from L=230 to L=199 (very dark).

      Fix: use only pixels in the 30th–80th percentile of L in the ORIGINAL car area.
        - Bottom 30% excluded: trunk interior, deep shadows, tire rubber, dark windows
        - Top 20% excluded: specular reflections, white highlights
        - Middle 50% = car body paint — the dominant, representative color

      This is robust across all car colors:
        - White car: selects body (L=180-230), skips trunk (L<50) and reflections (L>240)
        - Black car: selects body (L=20-50), skips deep shadow (L<20) and specular (L>50)
        - Any car: percentile-based so it adapts to the actual distribution
    """
    if not mask_np.any():
        return (0.0, 0.0, 0.0)
    orig_np  = np.array(original).astype(np.uint8)
    gem_np   = np.array(gemini_result).astype(np.uint8)
    orig_lab = cv2.cvtColor(orig_np, cv2.COLOR_RGB2Lab).astype(np.float32)
    gem_lab  = cv2.cvtColor(gem_np,  cv2.COLOR_RGB2Lab).astype(np.float32)

    # Build brightness-filtered mask using the original image's L distribution
    # within the car mask. Use 30th–80th percentile range → car body paint only.
    l_vals = orig_lab[mask_np, 0]
    l_lo   = float(np.percentile(l_vals, 30))
    l_hi   = float(np.percentile(l_vals, 80))
    paint_mask = mask_np & (orig_lab[:, :, 0] >= l_lo) & (orig_lab[:, :, 0] <= l_hi)

    # Fallback: if percentile filter removes too many pixels, use full mask
    if paint_mask.sum() < 50:
        paint_mask = mask_np
        logger.info("LAB shift: paint mask too sparse, falling back to full car mask")

    logger.info(
        "LAB shift sampling: %d paint pixels (L %.0f–%.0f) from %d total car pixels",
        paint_mask.sum(), l_lo, l_hi, mask_np.sum(),
    )

    shifts = []
    for ch, name in [(0, "L"), (1, "A"), (2, "B")]:
        orig_med = float(np.median(orig_lab[paint_mask, ch]))
        gem_med  = float(np.median(gem_lab[paint_mask, ch]))
        shift    = orig_med - gem_med
        shifts.append(shift)
        logger.info("LAB %s shift: %+.1f  (orig=%.1f  gem=%.1f)", name, shift, orig_med, gem_med)
    return tuple(shifts)


def _apply_lab_shifts(
    target: Image.Image, mask_np: np.ndarray, shifts: tuple
) -> Image.Image:
    """Apply pre-calculated LAB shifts to the car area of any image at any resolution."""
    if not mask_np.any() or all(abs(s) < 0.5 for s in shifts):
        return target
    t_np  = np.array(target).astype(np.uint8)
    t_lab = cv2.cvtColor(t_np, cv2.COLOR_RGB2Lab).astype(np.float32)
    for ch, shift in enumerate(shifts):
        t_lab[mask_np, ch] = np.clip(t_lab[mask_np, ch] + shift, 0, 255)
    logger.info("Applied LAB shifts %s at %s", tuple(round(s, 1) for s in shifts), target.size)
    return Image.fromarray(cv2.cvtColor(t_lab.astype(np.uint8), cv2.COLOR_Lab2RGB))


def _scale_mask(mask_np: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize a boolean mask to (target_w, target_h) with LANCZOS then threshold."""
    mask_img = Image.fromarray((mask_np.astype(np.uint8) * 255), mode='L')
    return np.array(mask_img.resize((target_w, target_h), Image.Resampling.LANCZOS)) > 128




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

    # Resize to 1024px for Gemini API
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

    # Sample original car color and HSV data using rembg mask
    mask_np = _get_car_mask_rembg(pil_img)
    original_car_color = _get_average_color(pil_img, mask_np)
    r, g, b = int(original_car_color[0]), int(original_car_color[1]), int(original_car_color[2])

    # Sample HSV saturation and brightness from car area
    orig_np = np.array(pil_img)
    orig_hsv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    if mask_np.any():
        s_val = int(orig_hsv[mask_np, 1].mean())
        v_val = int(orig_hsv[mask_np, 2].mean())
    else:
        s_val, v_val = 128, 200

    # Sample original floor color — inject into prompt so Gemini knows the exact floor color
    fr, fg, fb = _sample_floor_color(pil_img, mask_np)

    color_instruction = (
        f"\n\nCAR COLOR IS CRITICAL: The car color is RGB ({r}, {g}, {b}). "
        f"Saturation level is {s_val}. Brightness level is {v_val}. "
        "These are the EXACT values you must maintain. The output car must have these exact same values. "
        "Do not change these under any circumstances."
        f"\n\nFLOOR COLOR MEASUREMENT (DO NOT IGNORE): "
        f"I have measured the original floor color from this exact image. "
        f"The floor is RGB ({fr}, {fg}, {fb}). "
        f"Brightness level: {int((fr + fg + fb) / 3)}. "
        f"Your output floor MUST match this EXACTLY. "
        f"If your output floor brightness differs by more than 10 units from {int((fr + fg + fb) / 3)}, "
        f"you have failed this instruction. "
        f"The floor color MUST remain RGB ({fr}, {fg}, {fb}) — same exact shade, same tone. "
        f"Do NOT produce a brighter floor. Do NOT produce a darker floor. "
        f"Only remove puddles or wet spots while keeping the exact same floor color."
    )

    if mode == "enhance-preserve":
        prompt_with_color = prompt + "\n\n11. " + color_instruction.lstrip()
    else:
        prompt_with_color = prompt + color_instruction

    response = _call_gemini_with_retry(client, prompt_with_color, img_bytes, aspect, filename)
    result_pil = _extract_image_from_response(response)

    # Resize result to match input dimensions
    if result_pil.size != pil_img.size:
        result_pil = result_pil.resize(pil_img.size, Image.Resampling.LANCZOS)

    # Post-processing: color accuracy check (retry if color drifted badly)
    result_pil = _check_color_accuracy(
        pil_img, result_pil, client, prompt_with_color, img_bytes, aspect, filename,
        mask_np=mask_np,
    )
    if result_pil.size != pil_img.size:
        result_pil = result_pil.resize(pil_img.size, Image.Resampling.LANCZOS)

    # Flip detection — Gemini occasionally mirrors the car; correct if detected.
    if _is_flipped(pil_img, result_pil):
        result_pil = ImageOps.mirror(result_pil)

    # Calculate LAB colour shifts at 1024px resolution.
    # These scalar offsets are resolution-independent and will be applied at full res.
    lab_shifts = _calculate_lab_shifts(pil_img, result_pil, mask_np)

    # Upscale Gemini's output directly to full resolution.
    # Gemini handles reflection removal, background cleaning, and floor correction natively.
    # We do not mix original pixels — that caused seam artifacts (patches) at reflection
    # boundaries where original and Gemini pixels met with different values.
    if result_pil.size != (orig_w, orig_h):
        result_pil = result_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    result_pil = _apply_lab_shifts(result_pil, _scale_mask(mask_np, orig_w, orig_h), lab_shifts)
    logger.info("Using Gemini output directly (upscaled to %dx%d) — no pixel mixing", orig_w, orig_h)

    # Floor color normalization — correct Gemini's floor recoloring to match original.
    # Reload original at full res for sampling (pil_img is downscaled to 1024px).
    orig_for_floor = load_image(image_data, filename)
    if orig_for_floor.mode != "RGB":
        orig_for_floor = orig_for_floor.convert("RGB")
    if orig_for_floor.size != result_pil.size:
        orig_for_floor = orig_for_floor.resize(result_pil.size, Image.Resampling.LANCZOS)
    floor_mask_fullres = _scale_mask(mask_np, result_pil.width, result_pil.height)
    result_pil = _restore_floor_from_original(orig_for_floor, result_pil, floor_mask_fullres)
    del orig_for_floor

    # Apply brightness boost if requested
    result_pil = _apply_brightness(result_pil, lighting_boost)

    # Convert to requested format at maximum quality.
    # NEF is Nikon's proprietary binary RAW format — cannot be written by any Python library.
    # TIFF is the correct lossless professional equivalent for processed RAW exports.
    out_buf = io.BytesIO()
    fmt = output_format.lower()
    if fmt == "png":
        result_pil.save(out_buf, format="PNG", compress_level=0)
    elif fmt in ("jpg", "jpeg"):
        result_pil.save(out_buf, format="JPEG", quality=100, subsampling=0)
    elif fmt == "webp":
        result_pil.save(out_buf, format="WEBP", quality=100, lossless=False)
    elif fmt in ("tif", "tiff", "nef"):
        # TIFF with LZW compression: lossless, smaller than uncompressed, widely supported
        # in Photoshop, Lightroom, and all professional editing software
        result_pil.save(out_buf, format="TIFF", compression="tiff_lzw")
    else:
        result_pil.save(out_buf, format="PNG", compress_level=0)

    return out_buf.getvalue()
