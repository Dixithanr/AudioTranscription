import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import transcription
from services.worker import TranscriptionWorker

# set up a basic logger so we can see whats hapening in console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# global worker instance -- bit of a singleton but works fine for single process
worker: TranscriptionWorker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """start background worker on startup, clean shutdown on exit"""
    global worker

    logger.info("starting transcription worker...")
    worker = TranscriptionWorker(max_concurrent=3)  # dont set this too high, whisper is heavy

    # kick off the worker loop as a background task
    worker_task = asyncio.create_task(worker.run())

    # expose worker to routers via app state
    app.state.worker = worker

    yield  # app runs here

    # gracefull shutdown
    logger.info("shutting down worker, waiting for in-progress jobs...")
    worker.stop()
    await worker_task
    logger.info("worker stopped cleanly")


app = FastAPI(
    title="Speech-to-Text Transcription API",
    description="Async transcription using OpenAI Whisper with concurrent upload handling",
    version="1.0.0",
    lifespan=lifespan
)

# allow all origins for dev -- tighten this up before going to prod!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcription.router, prefix="/api/v1", tags=["transcription"])


@app.get("/health")
async def health_check():
    """simple liveness probe -- useful for k8s or just checking if the thing is up"""
    return {"status": "ok", "worker_running": worker is not None and worker.running}
