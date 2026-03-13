"""
General-purpose Replicate FLUX Fill Pro inpainting service.
Used by enhance_preserve_service.py for all AI-powered image editing.
"""

import os
import io
import time
import base64

import cv2
import httpx
import numpy as np
import requests
from PIL import Image

from app.config import REPLICATE_API_TOKEN

# FLUX Fill Pro — best inpainting model available
FLUX_FILL_MODEL = "black-forest-labs/flux-fill-pro"


def _numpy_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert numpy BGR to PIL RGB Image."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _numpy_to_data_uri(img_bgr: np.ndarray) -> str:
    """Convert numpy BGR image to base64 data URI for Replicate API."""
    pil_img = _numpy_to_pil(img_bgr)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _mask_to_data_uri(mask: np.ndarray) -> str:
    """Convert numpy grayscale mask to base64 data URI. White (255) = inpaint area."""
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    pil_mask = Image.fromarray(binary_mask, mode="L")
    buffer = io.BytesIO()
    pil_mask.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _download_result_image(output) -> np.ndarray:
    """Download the result image from Replicate output URL."""
    if isinstance(output, str):
        url = output
    elif hasattr(output, "url"):
        url = output.url
    elif isinstance(output, list) and len(output) > 0:
        url = str(output[0])
    elif hasattr(output, "__iter__"):
        for item in output:
            if isinstance(item, str):
                url = item
                break
            elif hasattr(item, "url"):
                url = item.url
                break
            else:
                url = str(item)
                break
        else:
            raise ValueError(f"Could not extract URL from output: {output}")
    else:
        url = str(output)

    print(f"[FLUX] Downloading result from: {url[:80]}...")

    response = requests.get(url, timeout=180)
    response.raise_for_status()

    pil_img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def flux_fill_pro_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    guidance: float = 30.0,
    steps: int = 50,
    max_dimension: int = 1536,
    timeout_seconds: float = 600.0,
) -> np.ndarray:
    """
    Use FLUX Fill Pro to inpaint masked areas of an image.

    Args:
        image:         BGR numpy array (the full original image)
        mask:          Grayscale numpy array (255 = area to inpaint, 0 = keep original)
        prompt:        Text describing what the inpainted area should look like
        guidance:      How strictly to follow the prompt (1.5-100)
        steps:         Number of denoising steps (15-50)
        max_dimension: Maximum width or height sent to API

    Returns:
        BGR numpy array with masked areas replaced by FLUX-generated content
    """
    token = REPLICATE_API_TOKEN or os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        print("[FLUX] ERROR: REPLICATE_API_TOKEN not set in .env file!")
        print("[FLUX] Falling back to cv2.inpaint (low quality)")
        return cv2.inpaint(image, mask, 10, cv2.INPAINT_NS)

    os.environ["REPLICATE_API_TOKEN"] = token

    original_h, original_w = image.shape[:2]
    mask_px = int(np.count_nonzero(mask > 127))

    if mask_px < 100:
        print(f"[FLUX] Mask too small ({mask_px} px), nothing to inpaint")
        return image

    print(f"[FLUX] FLUX Fill Pro — mask={mask_px} px, prompt='{prompt[:50]}...'")

    scale = 1.0
    api_w, api_h = original_w, original_h

    if max(original_h, original_w) > max_dimension:
        scale = max_dimension / max(original_h, original_w)
        api_w = int(original_w * scale)
        api_h = int(original_h * scale)

    api_w = max(32, (api_w // 32) * 32)
    api_h = max(32, (api_h // 32) * 32)

    image_resized = cv2.resize(image, (api_w, api_h), interpolation=cv2.INTER_LANCZOS4)
    mask_resized = cv2.resize(mask, (api_w, api_h), interpolation=cv2.INTER_NEAREST)
    _, mask_resized = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

    image_uri = _numpy_to_data_uri(image_resized)
    mask_uri = _mask_to_data_uri(mask_resized)

    try:
        import replicate
    except ImportError:
        print("[FLUX] replicate package not installed. Run: pip install replicate")
        return cv2.inpaint(image, mask, 10, cv2.INPAINT_NS)

    try:
        start_time = time.time()

        print(f"[FLUX] Calling Replicate API... (image: {api_w}x{api_h}, steps: {steps}, timeout: {timeout_seconds}s)")

        client = replicate.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=60.0, write=120.0)
        )
        output = client.run(
            FLUX_FILL_MODEL,
            input={
                "image": image_uri,
                "mask": mask_uri,
                "prompt": prompt,
                "guidance": guidance,
                "steps": steps,
                "output_format": "png",
            },
        )

        elapsed = time.time() - start_time
        print(f"[FLUX] API response received in {elapsed:.1f}s")

        result_resized = _download_result_image(output)

        result_full = cv2.resize(
            result_resized, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4
        )

        feather_mask = cv2.GaussianBlur(mask, (21, 21), 7)
        feather_float = feather_mask.astype(np.float32) / 255.0

        output_image = image.copy()
        for c in range(3):
            output_image[:, :, c] = (
                result_full[:, :, c] * feather_float
                + image[:, :, c] * (1.0 - feather_float)
            ).astype(np.uint8)

        print(f"[FLUX] SUCCESS — inpainted {mask_px} px in {elapsed:.1f}s")
        return output_image

    except Exception as e:
        print(f"[FLUX] API ERROR: {e}")
        import traceback

        traceback.print_exc()

        print("[FLUX] Falling back to cv2.inpaint")
        return cv2.inpaint(image, mask, 10, cv2.INPAINT_NS)


def flux_fill_pro_inpaint_region(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    padding: int = 60,
    **kwargs,
) -> np.ndarray:
    """
    Inpaint a small region efficiently by cropping first.
    """
    h, w = image.shape[:2]

    mask_coords = np.where(mask > 127)
    if len(mask_coords[0]) == 0:
        return image

    y1 = int(mask_coords[0].min())
    y2 = int(mask_coords[0].max())
    x1 = int(mask_coords[1].min())
    x2 = int(mask_coords[1].max())

    y1_pad = max(0, y1 - padding)
    y2_pad = min(h, y2 + padding)
    x1_pad = max(0, x1 - padding)
    x2_pad = min(w, x2 + padding)

    crop_image = image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
    crop_mask = mask[y1_pad:y2_pad, x1_pad:x2_pad].copy()

    crop_result = flux_fill_pro_inpaint(crop_image, crop_mask, prompt, **kwargs)

    result = image.copy()
    result[y1_pad:y2_pad, x1_pad:x2_pad] = crop_result

    return result
