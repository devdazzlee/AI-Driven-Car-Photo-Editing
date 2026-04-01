#!/usr/bin/env python3
"""
Automated car photo editing pipeline using Gemini API (gemini-3.1-flash-image-preview).

Uses the Batch API for cost optimization (~50% discount).

Library: google-genai (recommended; google-generativeai is deprecated as of Nov 2025).
Requirements: pip install google-genai python-dotenv Pillow
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Add backend to path for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-image-preview"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_INLINE_BATCH_SIZE = 5  # Inline requests limited to ~20MB total
TARGET_RESOLUTION = 1024

CAR_EDIT_PROMPT = """Edit this car photo for a professional automotive listing. Apply ALL of the following changes:

1. REFLECTION REMOVAL: Completely remove the bright light reflections from the upper body of the car where studio lights hit the paint. Eliminate all specular highlights and glare.

2. ENVIRONMENT STANDARDIZATION: Replace the background with a clean, seamless white studio wall. Remove all visible light fixtures, garage doors, wires, and any other environmental clutter.

3. FLOOR CLEANUP: Clean the floor tiles of all dirt, dust, and marks. Maintain a consistent dark tiled texture—the floor should look professionally cleaned.

4. WHEEL/TIRE DETAIL: Make the tires look deep black and remove all road dust and grime from the rubber. Tires should appear clean and well-maintained.

5. NATURAL COLOR PRESERVATION: The car is grey. Maintain the natural metallic grey color exactly—do not make it too dark or too washed out. Preserve the authentic paint finish.

6. GLASS CLARITY: Remove reflections from the windshield and side windows so they look tinted and clear. Windows should appear clean with no glare or reflections.

Output the edited image only. Maintain the original aspect ratio and composition."""


def _get_client():
    """Create Gemini client."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required. Set it in backend/.env")
    return genai.Client(api_key=api_key)


def _resize_to_target(pil_img: Image.Image, max_side: int = TARGET_RESOLUTION) -> Image.Image:
    """Resize image so longest side is max_side, preserving aspect ratio."""
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    if w >= h:
        new_w, new_h = max_side, int(h * max_side / w)
    else:
        new_w, new_h = int(w * max_side / h), max_side
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


# Gemini 3.1 Flash Image only accepts these aspect ratios
_ALLOWED_ASPECT_RATIOS = (
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4",
    "8:1", "9:16", "16:9", "21:9",
)


def _get_aspect_ratio_str(w: int, h: int) -> str:
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


def _load_and_prepare_image(path: Path) -> tuple[bytes, str, int, int]:
    """Load image, convert to RGB, resize, return (png_bytes, mime, w, h)."""
    pil = Image.open(path).convert("RGB")
    pil = _resize_to_target(pil)
    buf = __import__("io").BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue(), "image/png", pil.width, pil.height


def process_car_batch(
    input_folder: str | Path,
    output_folder: str | Path,
    use_batch_api: bool = True,
    poll_interval: int = 30,
) -> list[dict]:
    """
    Process car images in batch using Gemini API.

    Args:
        input_folder: Path to folder containing car images (.jpg, .jpeg, .png, .webp)
        output_folder: Path to save processed images
        use_batch_api: If True, use Batch API (50% cost discount, async, ~24h target).
                       If False, use real-time API (immediate, standard pricing).
        poll_interval: Seconds between status polls when using Batch API

    Returns:
        List of result dicts: [{"filename": str, "success": bool, "output_path": str?, "error": str?}, ...]

    Cost: Batch API targets ~$0.02/image (50% of $0.039 standard). Real-time ~$0.039/image.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [
        f
        for f in sorted(input_path.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_files:
        logger.warning("No supported images found in %s", input_path)
        return []

    logger.info("Found %d images in %s", len(image_files), input_path)

    client = _get_client()
    from google.genai import types

    if use_batch_api and len(image_files) > 1:
        return _run_batch_api(client, image_files, output_path, poll_interval)
    return _run_realtime_api(client, image_files, output_path, types)


def _run_realtime_api(client, image_files: list[Path], output_path: Path, types) -> list[dict]:
    """Process images one-by-one via real-time generateContent."""
    results = []
    for i, img_path in enumerate(image_files):
        try:
            img_bytes, mime, w, h = _load_and_prepare_image(img_path)
            aspect = _get_aspect_ratio_str(w, h)

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    CAR_EDIT_PROMPT,
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=aspect),
                ),
            )

            result_pil = None
            for part in getattr(response, "parts", []) or (
                response.candidates[0].content.parts if response.candidates else []
            ):
                if getattr(part, "inline_data", None) or getattr(part, "inlineData", None):
                    result_pil = part.as_image()
                    break
            if result_pil is None:
                raise RuntimeError("No image in response")

            out_name = f"{img_path.stem}_processed.png"
            out_file = output_path / out_name
            result_pil.convert("RGB").save(out_file, format="PNG")
            logger.info("[%d/%d] %s -> %s", i + 1, len(image_files), img_path.name, out_name)
            results.append({"filename": img_path.name, "success": True, "output_path": str(out_file)})
        except Exception as e:
            logger.exception("Failed %s", img_path.name)
            results.append({"filename": img_path.name, "success": False, "error": str(e)})
    return results


def _run_batch_api(
    client,
    image_files: list[Path],
    output_path: Path,
    poll_interval: int,
) -> list[dict]:
    """Process images via Batch API (file-based JSONL)."""
    import base64 as b64

    from google.genai import types

    # 1. Upload each image to File API
    uploaded = []
    for img_path in image_files:
        img_bytes, mime, w, h = _load_and_prepare_image(img_path)
        aspect = _get_aspect_ratio_str(w, h)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(img_bytes)
            tmp_path = tf.name
        try:
            f = client.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(
                    display_name=f"car_{img_path.stem}",
                    mime_type=mime,
                ),
            )
            uploaded.append((img_path, f.name, aspect))
        finally:
            os.unlink(tmp_path)

    # 2. Build JSONL with file_data references
    jsonl_path = output_path / "_batch_requests.jsonl"
    with open(jsonl_path, "w") as f:
        for i, (img_path, file_uri, aspect) in enumerate(uploaded):
            req = {
                "key": f"request-{i}",
                "request": {
                    "contents": [
                        {
                            "parts": [
                                {"text": CAR_EDIT_PROMPT},
                                {"file_data": {"file_uri": file_uri, "mime_type": "image/png"}},
                            ]
                        }
                    ],
                    "generation_config": {
                        "response_modalities": ["TEXT", "IMAGE"],
                        "image_config": {"aspect_ratio": aspect},
                    },
                },
            }
            f.write(json.dumps(req) + "\n")

    # 3. Upload JSONL and create batch job
    uploaded_jsonl = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(display_name="car-batch-requests", mime_type="jsonl"),
    )
    jsonl_path.unlink(missing_ok=True)

    batch_job = client.batches.create(
        model=GEMINI_MODEL,
        src=uploaded_jsonl.name,
        config={"display_name": "car-photo-batch"},
    )
    logger.info("Created batch job: %s", batch_job.name)

    # 4. Poll until complete
    completed = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    while True:
        job = client.batches.get(name=batch_job.name)
        state = getattr(job.state, "name", str(job.state))
        if state in completed:
            break
        logger.info("Job state: %s. Waiting %ds...", state, poll_interval)
        time.sleep(poll_interval)

    if state != "JOB_STATE_SUCCEEDED":
        logger.error("Batch job failed: %s", getattr(job, "error", state))
        return [{"filename": p.name, "success": False, "error": f"Batch job {state}"} for p in image_files]

    # 5. Download results and save images
    result_file_name = getattr(job.dest, "file_name", None) or getattr(job.dest, "fileName", None)
    content_bytes = client.files.download(file=result_file_name)
    content = content_bytes.decode("utf-8")

    results = []
    for line in content.splitlines():
        if not line:
            continue
        parsed = json.loads(line)
        key = parsed.get("key", "")
        idx = int(key.replace("request-", "")) if key.startswith("request-") else -1
        img_path = image_files[idx] if 0 <= idx < len(image_files) else None

        if "response" in parsed and parsed["response"]:
            try:
                parts = parsed["response"]["candidates"][0]["content"]["parts"]
                for part in parts:
                    if part.get("inlineData"):
                        data = b64.b64decode(part["inlineData"]["data"])
                        out_name = f"{img_path.stem}_processed.png" if img_path else f"{key}_processed.png"
                        out_file = output_path / out_name
                        Image.open(__import__("io").BytesIO(data)).convert("RGB").save(out_file, format="PNG")
                        results.append({"filename": img_path.name, "success": True, "output_path": str(out_file)})
                        logger.info("Saved %s", out_name)
                        break
            except Exception as e:
                results.append({"filename": img_path.name if img_path else key, "success": False, "error": str(e)})
        elif "error" in parsed:
            results.append({"filename": img_path.name if img_path else key, "success": False, "error": parsed["error"]})

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch process car photos with Gemini API")
    parser.add_argument("input_folder", help="Folder containing car images")
    parser.add_argument("output_folder", help="Folder to save processed images")
    parser.add_argument("--realtime", action="store_true", help="Use real-time API instead of Batch (faster, higher cost)")
    parser.add_argument("--poll", type=int, default=30, help="Batch status poll interval (seconds)")
    args = parser.parse_args()

    results = process_car_batch(
        args.input_folder,
        args.output_folder,
        use_batch_api=not args.realtime,
        poll_interval=args.poll,
    )

    success = sum(1 for r in results if r.get("success"))
    logger.info("Done: %d/%d succeeded", success, len(results))
    for r in results:
        if not r.get("success"):
            logger.warning("  Failed %s: %s", r["filename"], r.get("error", "unknown"))


if __name__ == "__main__":
    main()
