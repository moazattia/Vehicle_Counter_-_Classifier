from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str = "uploaded"


class ProcessRequest(BaseModel):
    line_position: float = Field(0.5, ge=0.0, le=1.0, description="Counting line as fraction of frame height")
    confidence: float = Field(0.3, ge=0.05, le=1.0, description="Detection confidence threshold")


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress_percent: float = 0.0
    counts: dict[str, int] = {}


class ResultsResponse(BaseModel):
    job_id: str
    total_vehicles: int
    counts: dict[str, int]
    video_duration_seconds: float
    processed_at: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    device: str
