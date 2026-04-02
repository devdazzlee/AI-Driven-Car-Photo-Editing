"""Image processing orchestrator with error handling and logging."""

import asyncio
import json
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import LOGS_DIR, OUTPUT_DIR, RETENTION_HOURS
from app.services.gemini_service import process_car_image, process_car_image_async, MAX_CONCURRENT_GEMINI
from app.services.image_utils import load_image, RAW_EXTENSIONS

logger = logging.getLogger(__name__)

# Single executor thread for running async batch loops in background
_executor = ThreadPoolExecutor(max_workers=1)


class ProcessingLog:
    """Tracks processing status for a job or single image."""

    def __init__(self, job_id: str, total: int = 1):
        self.job_id = job_id
        self.total = total
        self.completed = 0
        self.failed: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.started_at = datetime.utcnow().isoformat()
        self.finished_at: str | None = None
        self.status: str = "pending"  # pending | processing | completed | failed

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "results": self.results,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
        }


# In-memory job store (for M1; use Redis/DB in production)
_jobs: dict[str, ProcessingLog] = {}


def _save_raw_preview(image_data: bytes, filename: str, job_id: str) -> str | None:
    """For RAW files (NEF etc), save a browser-viewable PNG preview of the original. Returns preview filename or None."""
    ext = Path(filename).suffix.lower()
    if ext not in RAW_EXTENSIONS:
        return None
    preview_filename = f"{Path(filename).stem}_original.png"
    preview_path = OUTPUT_DIR / job_id / preview_filename
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    img = load_image(image_data, filename)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(preview_path, format="PNG")
    return preview_filename


def _process_single(
    image_data: bytes,
    filename: str,
    job_id: str,
    opts: dict,
) -> dict[str, Any]:
    """Internal: process one image and update job log."""
    log = _jobs.get(job_id)
    fmt = opts.get("output_format", "png").lower()
    bg = opts.get("background", "white").lower()
    mode = opts.get("processing_mode", "standard").lower()
    try:
        lb = float(opts.get("lighting_boost", "1.0"))
    except (TypeError, ValueError):
        lb = 1.0

    logger.info("Processing %s with mode=%s, format=%s", filename, mode, fmt)

    if mode == "keep-floor-walls":
        # Preserve original: floor, walls, corner - no background removal
        ext = "png" if fmt == "png" else "jpg" if fmt in ("jpeg", "jpg") else "webp"
        output_filename = f"{Path(filename).stem}_processed.{ext}"
        output_path = OUTPUT_DIR / job_id / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = load_image(image_data, filename)  # Handles NEF, PNG, JPEG, WebP
            if ext in ("jpg", "jpeg") and img.mode != "RGB":
                img = img.convert("RGB")
            save_fmt = "JPEG" if ext in ("jpg", "jpeg") else ext.upper()
            img.save(output_path, format=save_fmt, quality=95)
            result = {"original_filename": filename, "processed_filename": output_filename, "success": True}
            preview = _save_raw_preview(image_data, filename, job_id)
            if preview:
                result["original_preview"] = preview
            if log:
                log.completed += 1
                log.results.append(result)
            return result
        except Exception as e:
            logger.exception("Preserve mode failed for %s", filename)
            failed_entry = {"filename": filename, "error": str(e), "success": False}
            if log:
                log.failed.append(failed_entry)
            return failed_entry

    if mode == "enhance-preserve":
        # Remove sky/ceiling, enhance car, adjust lighting - keep floor & walls
        # Lazy import: skimage inpaint can fail on Windows (DLL blocked by App Control)
        from app.services.enhance_preserve_service import enhance_preserve_service

        ext = "png" if fmt == "png" else "jpg" if fmt in ("jpeg", "jpg") else "webp"
        output_filename = f"{Path(filename).stem}_processed.{ext}"
        output_path = OUTPUT_DIR / job_id / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result_bytes = enhance_preserve_service.process(
                image_data,
                filename=filename,
                output_format=ext,
                lighting_boost=lb,
            )
            output_path.write_bytes(result_bytes)
            result = {"original_filename": filename, "processed_filename": output_filename, "success": True}
            preview = _save_raw_preview(image_data, filename, job_id)
            if preview:
                result["original_preview"] = preview
            if log:
                log.completed += 1
                log.results.append(result)
            return result
        except Exception as e:
            logger.exception("Enhance-preserve failed for %s", filename)
            failed_entry = {"filename": filename, "error": str(e), "success": False}
            if log:
                log.failed.append(failed_entry)
            return failed_entry

    # Standard: background removal via Gemini
    if bg == "transparent":
        ext = "png"  # Transparency only supported in PNG
    else:
        ext = "png" if fmt == "png" else "jpg" if fmt in ("jpeg", "jpg") else "webp"
    output_filename = f"{Path(filename).stem}_processed.{ext}"
    output_path = OUTPUT_DIR / job_id / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result_bytes = process_car_image(
            image_data,
            filename=filename,
            mode="standard",
            output_format=ext,
            background=bg,
            lighting_boost=lb,
        )
        output_path.write_bytes(result_bytes)
        result = {"original_filename": filename, "processed_filename": output_filename, "success": True}
        preview = _save_raw_preview(image_data, filename, job_id)
        if preview:
            result["original_preview"] = preview
        if log:
            log.completed += 1
            log.results.append(result)
        return result
    except Exception as e:
        logger.exception("Processing failed for %s", filename)
        failed_entry = {
            "filename": filename,
            "error": str(e),
            "success": False,
        }
        if log:
            log.failed.append(failed_entry)
        return failed_entry


def _cleanup_old_files() -> None:
    """Delete output folders and log files older than RETENTION_HOURS."""
    cutoff = datetime.now(timezone.utc).timestamp() - RETENTION_HOURS * 3600
    deleted = 0
    for job_dir in OUTPUT_DIR.iterdir():
        if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
            shutil.rmtree(job_dir, ignore_errors=True)
            deleted += 1
    for log_file in LOGS_DIR.glob("*.json"):
        if log_file.stat().st_mtime < cutoff:
            log_file.unlink(missing_ok=True)
            deleted += 1
    if deleted:
        logger.info("Cleanup: removed %d old output dirs/logs (retention=%dh)", deleted, RETENTION_HOURS)


async def _process_single_async(
    image_data: bytes,
    filename: str,
    job_id: str,
    opts: dict,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Async version of _process_single — Gemini call is awaited, not blocked."""
    log = _jobs.get(job_id)
    fmt = opts.get("output_format", "png").lower()
    bg = opts.get("background", "white").lower()
    mode = opts.get("processing_mode", "standard").lower()
    try:
        lb = float(opts.get("lighting_boost", "1.0"))
    except (TypeError, ValueError):
        lb = 1.0

    logger.info("Processing %s async with mode=%s, format=%s", filename, mode, fmt)

    if mode == "keep-floor-walls":
        # keep-floor-walls has no Gemini call — run sync path directly
        return _process_single(image_data, filename, job_id, opts)

    ext = "png" if fmt == "png" else "jpg" if fmt in ("jpeg", "jpg") else "webp"
    output_filename = f"{Path(filename).stem}_processed.{ext}"
    output_path = OUTPUT_DIR / job_id / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result_bytes = await process_car_image_async(
            image_data,
            filename=filename,
            mode=mode if mode == "enhance-preserve" else "standard",
            output_format=ext,
            background=bg,
            lighting_boost=lb,
            semaphore=semaphore,
        )
        output_path.write_bytes(result_bytes)
        result = {"original_filename": filename, "processed_filename": output_filename, "success": True}
        preview = _save_raw_preview(image_data, filename, job_id)
        if preview:
            result["original_preview"] = preview
        if log:
            log.completed += 1
            log.results.append(result)
        return result
    except Exception as e:
        logger.exception("Async processing failed for %s", filename)
        failed_entry = {"filename": filename, "error": str(e), "success": False}
        if log:
            log.failed.append(failed_entry)
        return failed_entry


async def _run_batch_async(job_id: str, images: list[tuple[bytes, str]], opts: dict) -> None:
    """Async batch runner — all Gemini calls fire simultaneously, not sequentially."""
    log = _jobs.get(job_id)
    if not log:
        return
    log.status = "processing"

    # One shared semaphore limits concurrent Gemini calls across all images
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)

    tasks = [
        _process_single_async(image_data, filename, job_id, opts, semaphore)
        for image_data, filename in images
    ]
    await asyncio.gather(*tasks)

    log.status = "completed"
    log.finished_at = datetime.utcnow().isoformat()
    log_path = LOGS_DIR / f"{job_id}.json"
    log_path.write_text(json.dumps(log.to_dict(), indent=2))
    _cleanup_old_files()


def _run_batch(job_id: str, images: list[tuple[bytes, str]], opts: dict) -> None:
    """Background thread entry point — runs the async batch in its own event loop."""
    asyncio.run(_run_batch_async(job_id, images, opts))


def start_batch(images: list[tuple[bytes, str]], opts: dict | None = None) -> str:
    """
    Start batch processing in background. Returns job_id immediately.
    Frontend polls /api/status/{job_id} for progress.
    All Gemini calls run concurrently via asyncio.
    """
    job_id = str(uuid.uuid4())
    log = ProcessingLog(job_id, total=len(images))
    _jobs[job_id] = log
    _executor.submit(_run_batch, job_id, images, opts or {})
    return job_id


async def _process_sync_async(images: list[tuple[bytes, str]], opts: dict, job_id: str) -> None:
    """Async core for process_sync — concurrent Gemini calls even for small batches."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)
    tasks = [
        _process_single_async(image_data, filename, job_id, opts, semaphore)
        for image_data, filename in images
    ]
    await asyncio.gather(*tasks)


async def process_sync(images: list[tuple[bytes, str]], opts: dict | None = None) -> dict[str, Any]:
    """
    Process images concurrently (1-3 images).
    All Gemini calls fire simultaneously via asyncio.
    Called from FastAPI async endpoint — no asyncio.run() needed.
    """
    opts = opts or {}
    job_id = str(uuid.uuid4())
    log = ProcessingLog(job_id, total=len(images))
    _jobs[job_id] = log
    log.status = "processing"
    await _process_sync_async(images, opts, job_id)
    log.status = "completed"
    log.finished_at = datetime.utcnow().isoformat()
    log_path = LOGS_DIR / f"{job_id}.json"
    log_path.write_text(json.dumps(log.to_dict(), indent=2))
    _cleanup_old_files()
    return {
        "job_id": job_id,
        "total": log.total,
        "completed": log.completed,
        "failed_count": len(log.failed),
        "results": log.results,
        "failed": log.failed,
        "status": log.status,
    }


def get_job_status(job_id: str) -> dict | None:
    """Get processing status for a job."""
    log = _jobs.get(job_id)
    if not log:
        # Try loading from log file (e.g. after server restart)
        log_path = LOGS_DIR / f"{job_id}.json"
        if log_path.exists():
            data = json.loads(log_path.read_text())
            return data
        return None
    return log.to_dict()


def get_processed_file_path(job_id: str, filename: str) -> Path | None:
    """Resolve path to a processed file for download."""
    path = OUTPUT_DIR / job_id / filename
    return path if path.exists() else None
