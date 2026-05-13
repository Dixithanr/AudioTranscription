import asyncio
import logging
import os
import tempfile
import time
from datetime import datetime
from typing import Dict

from models.job import JobStatus, TranscriptionJob

logger = logging.getLogger(__name__)


class TranscriptionWorker:
    """
    background worker that pulls jobs from a queue and runs whisper on them.

    uses asyncio.Semaphore to cap concurrency -- whisper loads full model into
    memory per process so running too many at once will OOM your machine real fast.
    we run whisper in a thread pool so it doesnt block the event loop (its sync code).
    """

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # in-memory job store -- replace with postgres or redis for anything serious
        self.jobs: Dict[str, TranscriptionJob] = {}

        # queue holds (job_id, file_bytes, file_extension) tuples
        self.queue: asyncio.Queue = asyncio.Queue()

        self.running = False
        self._whisper_model = None  # lazy load -- dont want to wait at startup

    def stop(self):
        """signal worker to stop after finishing current batch"""
        self.running = False

    async def run(self):
        """main worker loop -- keep consuming from queue until stopped"""
        self.running = True
        logger.info(f"worker started, max_concurrent={self.max_concurrent}")

        while self.running:
            try:
                # timeout so we can check self.running periodically
                job_id, file_bytes, file_ext = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
                # spin up a task per job so multiple can run at same time
                asyncio.create_task(self._process_job(job_id, file_bytes, file_ext))
            except asyncio.TimeoutError:
                continue  # no jobs right now, loop back and check running flag
            except Exception as e:
                logger.error(f"unexpected error in worker loop: {e}")

        logger.info("worker loop exited")

    async def enqueue(self, job: TranscriptionJob, file_bytes: bytes, file_ext: str) -> str:
        """
        add a job to the store and put it on the queue.
        returns job_id immediately -- caller polls for results.
        """
        self.jobs[job.job_id] = job
        await self.queue.put((job.job_id, file_bytes, file_ext))
        logger.info(f"enqueued job {job.job_id} ({job.filename}), queue size ~{self.queue.qsize()}")
        return job.job_id

    async def _process_job(self, job_id: str, file_bytes: bytes, file_ext: str):
        """
        actually run whisper on the file.
        semaphore ensures we dont exceed max_concurrent running at same time.
        """
        async with self.semaphore:
            job = self.jobs.get(job_id)
            if not job:
                logger.warning(f"job {job_id} not found in store, skipping")
                return

            job.status = JobStatus.PROCESSING
            logger.info(f"processing job {job_id}, model={job.model_size}")

            # write bytes to a temp file -- whisper needs a file path not bytes
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=file_ext, delete=False
                ) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                # run whisper in thread pool so we dont block async event loop
                # whisper.transcribe() is pure sync/blocking code
                result = await asyncio.get_event_loop().run_in_executor(
                    None,  # uses default threadpool
                    self._run_whisper,
                    tmp_path,
                    job.model_size,
                    job.language
                )

                job.result = result["text"].strip()
                job.language = result.get("language")  # whisper tells us what it detected
                job.duration_seconds = result.get("duration")
                job.status = JobStatus.DONE
                job.completed_at = datetime.utcnow()
                logger.info(f"job {job_id} done, detected lang={job.language}")

            except Exception as e:
                # dont let one bad file kill the worker
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.utcnow()
                logger.error(f"job {job_id} failed: {e}")

            finally:
                # always clean up the temp file, even if we crash
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def _run_whisper(self, file_path: str, model_size: str, language: str | None) -> dict:
        """
        synchronous whisper call -- runs in a thread.
        lazy-loads the model on first call then reuses it.
        NOTE: if you change model_size between requests this wont reload -- add
        a model cache dict keyed by size if you need multiple sizes simultaneously.
        """
        import whisper  # import here so startup doesnt fail if whisper isnt installed yet

        if self._whisper_model is None or self._whisper_model.dims.n_mels != self._model_size_check(model_size):
            logger.info(f"loading whisper model '{model_size}'...")
            t0 = time.time()
            self._whisper_model = whisper.load_model(model_size)
            logger.info(f"model loaded in {time.time()-t0:.1f}s")

        transcribe_kwargs = {"fp16": False}  # fp16=False avoids warnings on CPU
        if language:
            transcribe_kwargs["language"] = language

        result = self._whisper_model.transcribe(file_path, **transcribe_kwargs)
        return result

    def _model_size_check(self, size: str) -> int:
        """
        hacky way to detect if loaded model matches requested size.
        whisper uses n_mels as a proxy -- 80 for base/small/medium, 128 for large.
        not perfect but good enough for single-size deployments.
        """
        size_map = {"tiny": 80, "base": 80, "small": 80, "medium": 80, "large": 128}
        return size_map.get(size, 80)

    def get_job(self, job_id: str) -> TranscriptionJob | None:
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> list[TranscriptionJob]:
        return list(self.jobs.values())
