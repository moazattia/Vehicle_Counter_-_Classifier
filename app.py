"""FastAPI application for Vehicle Counter & Classifier."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

import config
import pipeline
from schemas import (
    HealthResponse,
    ProcessRequest,
    ResultsResponse,
    StatusResponse,
    UploadResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Job state store (upload info only) ────────────────

@dataclass
class JobState:
    input_path: str = ""
    filename: str = ""


jobs: dict[str, JobState] = {}


def _get_job(job_id: str) -> JobState:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return jobs[job_id]


def _status_file(job_id: str) -> Path:
    return config.OUTPUT_DIR / job_id / "status.json"


def _read_status(job_id: str) -> dict:
    """Read status JSON from disk. Returns default if not yet created."""
    sf = _status_file(job_id)
    if sf.exists():
        try:
            return json.loads(sf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"status": "processing", "progress_percent": 0.0, "counts": {}}
    return {"status": "uploaded", "progress_percent": 0.0, "counts": {}}


WORKER_SCRIPT = str(config.BASE_DIR / "worker.py")


# ── Startup / shutdown ───────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Loading YOLO model …")
    model = pipeline.load_model()
    logger.info(f"Model loaded on device: {model.device}")
    yield


app = FastAPI(
    title="Vehicle Counter & Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    model = pipeline.load_model()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=str(model.device),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{ext}'. Allowed: {config.ALLOWED_EXTENSIONS}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f} MB). Max: {config.MAX_FILE_SIZE_MB} MB")

    job_id = uuid.uuid4().hex[:12]
    save_path = config.DATA_DIR / f"{job_id}{ext}"
    save_path.write_bytes(contents)

    jobs[job_id] = JobState(input_path=str(save_path), filename=file.filename or "unknown")

    logger.info(f"Uploaded {file.filename} → {save_path} ({size_mb:.1f} MB)")
    return UploadResponse(job_id=job_id, filename=file.filename or "unknown")


@app.post("/process/{job_id}", response_model=StatusResponse)
async def process_video(job_id: str, params: ProcessRequest | None = None):
    job = _get_job(job_id)

    # check if already processing/done
    current = _read_status(job_id)
    if current["status"] in ("processing", "done"):
        raise HTTPException(status_code=400, detail=f"Job is already {current['status']}")

    line_pos = params.line_position if params else config.LINE_POSITION
    confidence = params.confidence if params else config.CONFIDENCE_THRESHOLD

    output_dir = config.OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # write initial processing status
    pipeline._write_status(output_dir / "status.json", {
        "status": "processing",
        "progress_percent": 0.0,
        "counts": {},
    })

    # launch worker as a completely separate subprocess
    proc = subprocess.Popen(
        [sys.executable, WORKER_SCRIPT,
         job.input_path, str(output_dir), job_id,
         str(line_pos), str(confidence)],
    )

    logger.info(f"Started processing job {job_id} (PID {proc.pid})")
    return StatusResponse(job_id=job_id, status="processing")


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    _get_job(job_id)  # ensure job exists
    status = _read_status(job_id)
    return StatusResponse(
        job_id=job_id,
        status=status.get("status", "uploaded"),
        progress_percent=status.get("progress_percent", 0.0),
        counts=status.get("counts", {}),
    )


@app.get("/results/{job_id}", response_model=ResultsResponse)
async def get_results(job_id: str):
    _get_job(job_id)
    status = _read_status(job_id)
    if status.get("status") != "done":
        raise HTTPException(status_code=400, detail=f"Job not done yet (status: {status.get('status')})")
    return ResultsResponse(
        job_id=job_id,
        total_vehicles=status.get("total", 0),
        counts=status.get("counts", {}),
        video_duration_seconds=round(status.get("duration", 0.0), 2),
        processed_at=status.get("processed_at", ""),
    )


@app.get("/download/video/{job_id}")
async def download_video(job_id: str):
    _get_job(job_id)
    status = _read_status(job_id)
    if status.get("status") != "done":
        raise HTTPException(status_code=400, detail="Processing not complete")
    path = Path(status["video_out"])
    if not path.exists():
        raise HTTPException(status_code=500, detail="Output video not found")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}_annotated.mp4")


@app.get("/download/report/{job_id}")
async def download_report(job_id: str):
    _get_job(job_id)
    status = _read_status(job_id)
    if status.get("status") != "done":
        raise HTTPException(status_code=400, detail="Processing not complete")
    path = Path(status["report_out"])
    if not path.exists():
        raise HTTPException(status_code=500, detail="Report not found")
    return FileResponse(path, media_type="text/markdown", filename=f"{job_id}_report.md")


@app.get("/download/chart/{job_id}")
async def download_chart(job_id: str):
    _get_job(job_id)
    status = _read_status(job_id)
    if status.get("status") != "done":
        raise HTTPException(status_code=400, detail="Processing not complete")
    path = Path(status["chart_out"])
    if not path.exists():
        raise HTTPException(status_code=500, detail="Chart not found")
    return FileResponse(path, media_type="image/png", filename=f"{job_id}_chart.png")


# ── Run directly ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.API_HOST, port=config.API_PORT, reload=True)
