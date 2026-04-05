"""Image processing orchestrator with error handling and logging."""

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
from app.services.gemini_service import process_car_image
from app.services.image_utils import load_image, RAW_EXTENSIONS

logger = logging.getLogger(__name__)

# Single worker for sequential processing (avoids memory spikes on 4GB VPS)
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
    img.save(preview_path, format="PNG", compress_level=0)
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
        ext = ("jpg" if fmt in ("jpg", "jpeg") else
               "tif" if fmt in ("tif", "tiff", "nef") else
               "webp" if fmt == "webp" else "png")
        output_filename = f"{Path(filename).stem}_processed.{ext}"
        output_path = OUTPUT_DIR / job_id / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = load_image(image_data, filename)  # Handles NEF, PNG, JPEG, WebP
            if ext == "jpg" and img.mode != "RGB":
                img = img.convert("RGB")
            if ext == "jpg":
                img.save(output_path, format="JPEG", quality=100, subsampling=0)
            elif ext == "tif":
                img.save(output_path, format="TIFF", compression="tiff_lzw")
            elif ext == "webp":
                img.save(output_path, format="WEBP", quality=100)
            else:
                img.save(output_path, format="PNG", compress_level=0)
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

        ext = ("jpg" if fmt in ("jpg", "jpeg") else
               "tif" if fmt in ("tif", "tiff", "nef") else
               "webp" if fmt == "webp" else "png")
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
        ext = ("jpg" if fmt in ("jpg", "jpeg") else
               "tif" if fmt in ("tif", "tiff", "nef") else
               "webp" if fmt == "webp" else "png")
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


def _run_batch(job_id: str, images: list[tuple[bytes, str]], opts: dict) -> None:
    """Background worker: process all images for a job."""
    log = _jobs.get(job_id)
    if not log:
        return
    log.status = "processing"
    for image_data, filename in images:
        _process_single(image_data, filename, job_id, opts)
    log.status = "completed"
    log.finished_at = datetime.utcnow().isoformat()
    log_path = LOGS_DIR / f"{job_id}.json"
    log_path.write_text(json.dumps(log.to_dict(), indent=2))
    _cleanup_old_files()


def start_batch(images: list[tuple[bytes, str]], opts: dict | None = None) -> str:
    """
    Start batch processing in background. Returns job_id immediately.
    Frontend polls /api/status/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())
    log = ProcessingLog(job_id, total=len(images))
    _jobs[job_id] = log
    _executor.submit(_run_batch, job_id, images, opts or {})
    return job_id


def process_sync(images: list[tuple[bytes, str]], opts: dict | None = None) -> dict[str, Any]:
    """
    Process images synchronously (single or small batch).
    Use for 1-3 images when instant response is preferred.
    """
    opts = opts or {}
    job_id = str(uuid.uuid4())
    log = ProcessingLog(job_id, total=len(images))
    _jobs[job_id] = log
    log.status = "processing"
    for image_data, filename in images:
        _process_single(image_data, filename, job_id, opts)
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
