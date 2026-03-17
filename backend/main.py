"""
Car Image AI - Backend
Gemini API (gemini-3.1-flash-image-preview) for car photo enhancement.
"""

from pathlib import Path
import shutil
import time

from dotenv import load_dotenv

# Load .env from backend directory before any other imports
load_dotenv(Path(__file__).resolve().parent / ".env")

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import LOGS_DIR, OUTPUT_DIR, RETENTION_HOURS
from app.routers.process import router as process_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _cleanup_old_files():
    """Delete logs and outputs older than RETENTION_HOURS to prevent disk bloat."""
    if RETENTION_HOURS <= 0:
        return
    cutoff = time.time() - (RETENTION_HOURS * 3600)
    removed_logs = 0
    removed_outputs = 0
    for log_path in LOGS_DIR.glob("*.json"):
        try:
            if log_path.stat().st_mtime < cutoff:
                log_path.unlink()
                removed_logs += 1
        except OSError:
            pass
    for job_dir in OUTPUT_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        try:
            files = [f for f in job_dir.rglob("*") if f.is_file()]
            mtime = max(f.stat().st_mtime for f in files) if files else 0
            if mtime < cutoff:
                shutil.rmtree(job_dir, ignore_errors=True)
                removed_outputs += 1
        except OSError:
            pass
    if removed_logs or removed_outputs:
        logger.info("Cleanup: removed %d old logs, %d output dirs (older than %dh)", removed_logs, removed_outputs, RETENTION_HOURS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_files()
    logger.info("Car Image AI backend ready (Gemini API)")
    yield
    logger.info("Shutdown complete")


app = FastAPI(
    title="Car Image AI",
    description="AI-powered car photo editing - Gemini API (gemini-3.1-flash-image-preview)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ai-driven-car-photo-editing-e4cg.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(process_router)


@app.get("/")
def root():
    return {"message": "Car Image AI API", "docs": "/docs"}
