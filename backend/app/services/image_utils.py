"""Image loading utilities including RAW (NEF) support."""

import io
import logging
from pathlib import Path

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


def _extract_embedded_jpeg(data: bytes) -> bytes | None:
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

    Tries rawpy first for full-quality decode. If rawpy is unavailable or blocked
    (e.g. by Windows Application Control), falls back to extracting the embedded
    JPEG preview that all RAW files contain.
    """
    # Try rawpy first for best quality
    try:
        import rawpy
        with rawpy.imread(io.BytesIO(data)) as raw:
            rgb = raw.postprocess(use_camera_wb=True)
        import numpy as np
        return Image.fromarray(rgb)
    except (ImportError, OSError) as e:
        logger.warning("rawpy unavailable (%s), falling back to embedded JPEG extraction for %s", e, filename)

    # Fallback: extract embedded JPEG preview from RAW file
    jpeg_data = _extract_embedded_jpeg(data)
    if jpeg_data and len(jpeg_data) > 10000:  # sanity check: must be a real image
        logger.info("Extracted embedded JPEG preview (%d bytes) from %s", len(jpeg_data), filename)
        return Image.open(io.BytesIO(jpeg_data)).convert("RGB")

    raise ValueError(
        f"Cannot load RAW file {filename}: rawpy is blocked by Windows Application Control "
        "and no embedded JPEG preview was found. Try converting the file to JPEG first."
    )
