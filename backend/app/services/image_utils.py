"""Image loading utilities including RAW (NEF) support."""

import io
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {".nef", ".nrw", ".arw", ".cr2", ".dng"}


def load_image(data: bytes, filename: str) -> Image.Image:
    """
    Load image from bytes. Handles JPEG, PNG, WebP, and RAW (NEF) formats.
    Returns RGB PIL Image.
    """
    ext = Path(filename).suffix.lower()

    if ext in RAW_EXTENSIONS:
        return _load_raw(data, filename)

    return Image.open(io.BytesIO(data)).convert("RGB")


def _extract_embedded_jpeg(data: bytes) -> Optional[bytes]:
    """Extract the largest embedded JPEG preview from a RAW file.

    RAW files (NEF, ARW, CR2, etc.) contain embedded JPEG previews.
    We find all JPEG segments (SOI/EOI markers) and return the largest one,
    which is typically the full-resolution preview.
    """
    jpeg_start = b'\xff\xd8'
    jpeg_end = b'\xff\xd9'
    largest = None
    largest_size = 0

    pos = 0
    while pos < len(data) - 1:
        start = data.find(jpeg_start, pos)
        if start == -1:
            break
        end = data.find(jpeg_end, start + 2)
        if end == -1:
            break
        end += 2  # include the EOI marker
        segment = data[start:end]
        if len(segment) > largest_size:
            largest = segment
            largest_size = len(segment)
        pos = end

    return largest


def _load_raw(data: bytes, filename: str) -> Image.Image:
    """Decode RAW (NEF/ARW/CR2/DNG) to RGB PIL Image.

    Uses the embedded JPEG preview as the primary source. Every RAW file contains
    a full-resolution JPEG processed by the camera's own pipeline (Nikon Picture
    Control, white balance, tone curve) — this is exactly what the photographer
    saw and the correct brightness/color representation.

    rawpy's mathematical decode produces a flat, ~40% darker result because it
    does not apply the camera manufacturer's proprietary tone curves. Using the
    embedded JPEG avoids this brightness mismatch entirely.

    Falls back to rawpy only if no embedded JPEG is found (rare edge case).
    """
    # Primary: embedded JPEG — camera-processed, correct brightness and color
    jpeg_data = _extract_embedded_jpeg(data)
    if jpeg_data and len(jpeg_data) > 10000:
        logger.info("Loaded embedded JPEG preview (%d bytes) from %s", len(jpeg_data), filename)
        return Image.open(io.BytesIO(jpeg_data)).convert("RGB")

    # Fallback: rawpy decode if no embedded JPEG found
    try:
        import rawpy
        import numpy as np
        with rawpy.imread(io.BytesIO(data)) as raw:
            rgb = raw.postprocess(use_camera_wb=True)
        logger.info("Loaded via rawpy decode for %s", filename)
        return Image.fromarray(rgb)
    except (ImportError, OSError) as e:
        logger.warning("rawpy unavailable (%s) for %s", e, filename)

    raise ValueError(
        f"Cannot load RAW file {filename}: no embedded JPEG found and rawpy is unavailable."
    )
