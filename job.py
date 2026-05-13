import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"      # sitting in the queue, not picked up yet
    PROCESSING = "processing"  # whisper is currently chewing on it
    DONE = "done"            # transcription complete
    FAILED = "failed"        # something went wrong, check error field


class TranscriptionJob(BaseModel):
    """internal representation of a job -- stored in memory (swap for redis in prod)"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    status: JobStatus = JobStatus.PENDING
    language: Optional[str] = None   # None = whisper auto-detects, pass "en" to force english etc
    model_size: str = "base"         # tiny/base/small/medium/large -- larger = slower but better
    result: Optional[str] = None     # the actual transcript text
    error: Optional[str] = None      # populated if status == FAILED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None  # how long the audio was


class SubmitResponse(BaseModel):
    """what we return when a job is accepted"""
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """what we return when polling for status"""
    job_id: str
    filename: str
    status: JobStatus
    language: Optional[str]
    model_size: str
    result: Optional[str]
    error: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]


class QueueStatsResponse(BaseModel):
    """quick overview of whats going on in the queue -- handy for debugging"""
    total_jobs: int
    pending: int
    processing: int
    done: int
    failed: int
