import logging
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from models.job import (
    JobStatus,
    JobStatusResponse,
    QueueStatsResponse,
    SubmitResponse,
    TranscriptionJob,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# file size limit -- 100mb should cover most audio files
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024

# only accept audio formats whisper can actually handle
ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mpeg"}

ALLOWED_MODEL_SIZES = {"tiny", "base", "small", "medium", "large"}


def _get_worker(request: Request):
    """helper to grab worker from app state -- avoids importing global"""
    worker = getattr(request.app.state, "worker", None)
    if worker is None:
        # this shouldnt happen but better to fail loudly than silently
        raise HTTPException(status_code=503, detail="transcription worker not available")
    return worker


@router.post("/transcribe", response_model=SubmitResponse, status_code=202)
async def submit_transcription(
    request: Request,
    file: Annotated[UploadFile, File(description="audio file to transcribe")],
    language: Annotated[Optional[str], Form()] = None,
    model_size: Annotated[str, Form()] = "base",
):
    """
    upload an audio file and get back a job_id.
    poll GET /transcribe/{job_id} to check status and retrieve the transcript.

    - **file**: mp3, wav, flac, m4a, ogg, webm supported
    - **language**: optional ISO 639-1 code (e.g. 'en', 'de', 'pl'). leave blank for auto-detect
    - **model_size**: tiny | base | small | medium | large. bigger = slower but more accurate
    """
    worker = _get_worker(request)

    # validate model size early so we dont waste time reading the file
    if model_size not in ALLOWED_MODEL_SIZES:
        raise HTTPException(
            status_code=422,
            detail=f"invalid model_size '{model_size}'. choose from: {sorted(ALLOWED_MODEL_SIZES)}"
        )

    # check file extension -- content-type can be spoofed so we check both
    filename = file.filename or "upload"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type '{ext}'. allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # read file into memory -- for very large files you'd stream to disk instead
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        mb = len(file_bytes) / 1024 / 1024
        raise HTTPException(
            status_code=413,
            detail=f"file too large ({mb:.1f}mb). max allowed is {MAX_FILE_SIZE_BYTES // 1024 // 1024}mb"
        )

    # build the job and drop it in the queue
    job = TranscriptionJob(
        filename=filename,
        language=language,
        model_size=model_size,
    )

    job_id = await worker.enqueue(job, file_bytes, ext)
    logger.info(f"accepted upload '{filename}' as job {job_id}")

    return SubmitResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"job accepted. poll GET /api/v1/transcribe/{job_id} for results"
    )


@router.post("/transcribe/batch", status_code=202)
async def submit_batch(
    request: Request,
    files: Annotated[list[UploadFile], File(description="multiple audio files")],
    language: Annotated[Optional[str], Form()] = None,
    model_size: Annotated[str, Form()] = "base",
):
    """
    submit multiple files at once -- all get queued and processed concurrently up to the worker limit.
    returns a list of job_ids in the same order as the uploaded files.
    """
    worker = _get_worker(request)

    if not files:
        raise HTTPException(status_code=400, detail="no files provided")

    if len(files) > 20:
        # dont let someone DDOS the queue with 500 files in one request
        raise HTTPException(status_code=400, detail="max 20 files per batch request")

    if model_size not in ALLOWED_MODEL_SIZES:
        raise HTTPException(status_code=422, detail=f"invalid model_size '{model_size}'")

    results = []
    for upload in files:
        filename = upload.filename or "upload"
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext not in ALLOWED_EXTENSIONS:
            # skip bad files but report them -- dont fail the whole batch
            results.append({"filename": filename, "error": f"unsupported type '{ext}', skipped"})
            continue

        file_bytes = await upload.read()
        if len(file_bytes) == 0 or len(file_bytes) > MAX_FILE_SIZE_BYTES:
            results.append({"filename": filename, "error": "file empty or too large, skipped"})
            continue

        job = TranscriptionJob(filename=filename, language=language, model_size=model_size)
        job_id = await worker.enqueue(job, file_bytes, ext)
        results.append({"filename": filename, "job_id": job_id, "status": JobStatus.PENDING})

    return {"batch_size": len(files), "jobs": results}


@router.get("/transcribe/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request):
    """
    poll this endpoint to check if transcription is done.
    when status == 'done', the 'result' field contains the transcript text.
    when status == 'failed', the 'error' field explains what went wrong.
    """
    worker = _get_worker(request)
    job = worker.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")

    return JobStatusResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status,
        language=job.language,
        model_size=job.model_size,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
        duration_seconds=job.duration_seconds,
    )


@router.get("/transcribe", response_model=list[JobStatusResponse])
async def list_jobs(request: Request, status: Optional[JobStatus] = None):
    """
    list all jobs, optionally filtered by status.
    e.g. GET /api/v1/transcribe?status=pending  to see whats waiting in queue
    """
    worker = _get_worker(request)
    jobs = worker.get_all_jobs()

    if status:
        jobs = [j for j in jobs if j.status == status]

    # sort newest first -- makes more sense when checking recent uploads
    jobs.sort(key=lambda j: j.created_at, reverse=True)

    return [
        JobStatusResponse(
            job_id=j.job_id,
            filename=j.filename,
            status=j.status,
            language=j.language,
            model_size=j.model_size,
            result=j.result,
            error=j.error,
            created_at=j.created_at,
            completed_at=j.completed_at,
            duration_seconds=j.duration_seconds,
        )
        for j in jobs
    ]


@router.get("/queue/stats", response_model=QueueStatsResponse)
async def queue_stats(request: Request):
    """quick snapshot of queue health -- useful for monitoring dashboards"""
    worker = _get_worker(request)
    jobs = worker.get_all_jobs()

    return QueueStatsResponse(
        total_jobs=len(jobs),
        pending=sum(1 for j in jobs if j.status == JobStatus.PENDING),
        processing=sum(1 for j in jobs if j.status == JobStatus.PROCESSING),
        done=sum(1 for j in jobs if j.status == JobStatus.DONE),
        failed=sum(1 for j in jobs if j.status == JobStatus.FAILED),
    )


@router.delete("/transcribe/{job_id}", status_code=204)
async def delete_job(job_id: str, request: Request):
    """
    remove a completed or failed job from memory.
    you cant cancel a job thats currently processing -- whisper doesnt support mid-run cancellation.
    """
    worker = _get_worker(request)
    job = worker.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")

    if job.status == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="cant delete a job thats currently processing"
        )

    del worker.jobs[job_id]
    logger.info(f"deleted job {job_id}")
    return JSONResponse(status_code=204, content=None)
