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

    "2. BACKGROUND — WALLS:\n"
    "   Remove ALL objects from walls without exception. This includes garage doors, door frames, "
    "hinges, handles, studio lights, light fixtures, cables, wires, trolleys, carts, and any other "
    "equipment. Make walls completely smooth and white. Nothing should be visible on the walls "
    "except the wall corner which must remain visible and natural.\n\n"

    "3. BACKGROUND — FLOOR:\n"
    "   Clean the floor of all dirt, dust, marks, tire tracks and uneven patches. "
    "Keep the exact same floor color and texture — do not replace with a different material or color. "
    "Just make it look professionally cleaned.\n\n"

    "4. REFLECTIONS:\n"
    "   Remove all white light reflections from the hood, roof, doors and fenders. "
    "Remove all glare from windows and windshield. Windows should look dark and clear with no glare.\n\n"

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
                    temperature=0.2,
                    top_p=0.85,
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


def _check_color_accuracy(original: Image.Image, result: Image.Image, client,
                          prompt: str, img_bytes: bytes, aspect: str, label: str) -> Image.Image:
    """If color drifted >10%, retry Gemini call once and keep the better result."""
    orig_avg = np.array(original.resize((64, 64))).mean(axis=(0, 1))
    res_avg = np.array(result.resize((64, 64))).mean(axis=(0, 1))
    color_diff = np.abs(orig_avg - res_avg).mean() / 255.0

    if color_diff <= 0.10:
        return result

    logger.warning("Color drift %.1f%% exceeds 10%% for %s — retrying once", color_diff * 100, label)
    try:
        retry_response = _call_gemini_with_retry(client, prompt, img_bytes, aspect, f"{label}[retry]")
        retry_pil = _extract_image_from_response(retry_response)
        retry_avg = np.array(retry_pil.resize((64, 64))).mean(axis=(0, 1))
        retry_diff = np.abs(orig_avg - retry_avg).mean() / 255.0
        if retry_diff < color_diff:
            logger.info("Retry better color (%.1f%% vs %.1f%%) for %s", retry_diff * 100, color_diff * 100, label)
            return retry_pil
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


def _apply_color_correction(original: Image.Image, processed: Image.Image, mask_np: np.ndarray) -> Image.Image:
    """Correct color shift using channel-wise ratios on car area."""
    original_color = _get_average_color(original, mask_np)
    processed_color = _get_average_color(processed, mask_np)
    
    if np.all(original_color == 0) or np.all(processed_color == 0):
        return processed

    shift = np.abs(original_color - processed_color) / (original_color + 1e-6)
    
    if np.any(shift > 0.10):
        logger.info("Color shift >10%% detected. Original RGB: %s, Processed RGB: %s. Applying correction.", 
                    original_color.round(1), processed_color.round(1))
        
        correction_factor = original_color / (processed_color + 1e-6)
        
        processed_np = np.array(processed).astype(np.float32)
        corrected_np = processed_np.copy()
        
        corrected_pixels = processed_np[mask_np] * correction_factor
        corrected_np[mask_np] = np.clip(corrected_pixels, 0, 255)
        
        return Image.fromarray(corrected_np.astype(np.uint8))
    
    return processed


def _apply_saturation_correction(original: Image.Image, processed: Image.Image, mask_np: np.ndarray) -> Image.Image:
    """Correct saturation and brightness shifts on car area."""
    orig_np = np.array(original)
    proc_np = np.array(processed)
    
    orig_hsv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    proc_hsv = cv2.cvtColor(proc_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    if not mask_np.any():
        return processed
        
    s_orig_mean = orig_hsv[mask_np, 1].mean()
    s_proc_mean = proc_hsv[mask_np, 1].mean()
    
    v_orig_mean = orig_hsv[mask_np, 2].mean()
    v_proc_mean = proc_hsv[mask_np, 2].mean()
    
    if s_orig_mean == 0 or s_proc_mean == 0 or v_orig_mean == 0 or v_proc_mean == 0:
        return processed
        
    s_drop = (s_orig_mean - s_proc_mean) / s_orig_mean
    v_drop = (v_orig_mean - v_proc_mean) / v_orig_mean
    
    s_factor = 1.0
    v_factor = 1.0
    
    apply_correction = False
    
    if s_drop > 0.10:
        s_factor = s_orig_mean / s_proc_mean
        logger.info("Saturation drop >10%% detected. Original S: %.1f, Processed S: %.1f. Applying factor %.2f.", s_orig_mean, s_proc_mean, s_factor)
        apply_correction = True
        
    if v_drop > 0.10:
        v_factor = v_orig_mean / v_proc_mean
        logger.info("Brightness (V) drop >10%% detected. Original V: %.1f, Processed V: %.1f. Applying factor %.2f.", v_orig_mean, v_proc_mean, v_factor)
        apply_correction = True
        
    if not apply_correction:
        return processed
        
    corrected_hsv = proc_hsv.copy()
    
    if s_factor != 1.0:
        corrected_hsv[mask_np, 1] = np.clip(corrected_hsv[mask_np, 1] * s_factor, 0, 255)
    if v_factor != 1.0:
        corrected_hsv[mask_np, 2] = np.clip(corrected_hsv[mask_np, 2] * v_factor, 0, 255)
        
    corrected_rgb = cv2.cvtColor(corrected_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    final_np = proc_np.copy()
    final_np[mask_np] = corrected_rgb[mask_np]
    
    return Image.fromarray(final_np)


def _apply_brightness(pil_img: Image.Image, boost: float) -> Image.Image:
    """Apply brightness boost to image. boost=1.0 means no change, 1.5 means 50% brighter."""
    if boost <= 1.0:
        return pil_img
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(pil_img)
    result = enhancer.enhance(boost)
    logger.info("Applied brightness boost: %.2f", boost)
    return result


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

    # NEW: Sample original car color using RMBG mask
    mask_np = _get_car_mask_rembg(pil_img)
    original_car_color = _get_average_color(pil_img, mask_np)
    r, g, b = int(original_car_color[0]), int(original_car_color[1]), int(original_car_color[2])

    if mode == "enhance-preserve":
        prompt_with_color = prompt + (
            f"\n\n10. CAR COLOR IS CRITICAL: The car in this image is exactly color RGB approximately ({r}, {g}, {b}). "
            "You must preserve this exact color. Do not lighten, darken or shift the hue under any circumstances. "
            "The output car color must match the input car color exactly."
        )
    else:
        prompt_with_color = prompt

    response = _call_gemini_with_retry(client, prompt_with_color, img_bytes, aspect, filename)
    result_pil = _extract_image_from_response(response)

    # Resize result to match input dimensions
    if result_pil.size != pil_img.size:
        result_pil = result_pil.resize(pil_img.size, Image.Resampling.LANCZOS)

    # Post-processing: color accuracy check
    result_pil = _check_color_accuracy(
        pil_img, result_pil, client, prompt_with_color, img_bytes, aspect, filename,
    )
    if result_pil.size != pil_img.size:
        result_pil = result_pil.resize(pil_img.size, Image.Resampling.LANCZOS)

    # Post-processing: exact color correction mapping using RMBG mask
    result_pil = _apply_color_correction(pil_img, result_pil, mask_np)

    # Post-processing: exact saturation/vibrancy mapping using RMBG mask
    result_pil = _apply_saturation_correction(pil_img, result_pil, mask_np)

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
