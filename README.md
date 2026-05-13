# Speech-to-Text Transcription API

Async transcription API built with FastAPI + OpenAI Whisper. Handles concurrent uploads via an asyncio job queue with a configurable concurrency cap.

## Setup

```bash
pip install -r requirements.txt
```

> Whisper also needs ffmpeg on your PATH:
> - Ubuntu: `sudo apt install ffmpeg`
> - Mac: `brew install ffmpeg`
> - Windows: download from https://ffmpeg.org/download.html

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs at: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/transcribe` | Upload single audio file, get job_id |
| POST | `/api/v1/transcribe/batch` | Upload up to 20 files at once |
| GET | `/api/v1/transcribe/{job_id}` | Poll for status / get transcript |
| GET | `/api/v1/transcribe` | List all jobs (filter by ?status=pending) |
| DELETE | `/api/v1/transcribe/{job_id}` | Remove a completed job |
| GET | `/api/v1/queue/stats` | Queue health snapshot |
| GET | `/health` | Liveness probe |

## Example usage

### Single file upload

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@interview.mp3" \
  -F "model_size=base" \
  -F "language=en"
```

Response:
```json
{
  "job_id": "3f2a1b4c-...",
  "status": "pending",
  "message": "job accepted. poll GET /api/v1/transcribe/3f2a1b4c-... for results"
}
```

### Poll for result

```bash
curl http://localhost:8000/api/v1/transcribe/3f2a1b4c-...
```

Response when done:
```json
{
  "job_id": "3f2a1b4c-...",
  "status": "done",
  "result": "Hello, this is the transcribed text...",
  "language": "en",
  "duration_seconds": 47.3
}
```

### Batch upload

```bash
curl -X POST http://localhost:8000/api/v1/transcribe/batch \
  -F "files=@file1.mp3" \
  -F "files=@file2.wav" \
  -F "model_size=small"
```

## Concurrency model

```
Upload requests (concurrent, no limit)
        │
        ▼
  asyncio.Queue
        │
        ▼
TranscriptionWorker  ←── asyncio.Semaphore(max_concurrent=3)
        │
        ├── Thread 1: whisper.transcribe(file1)
        ├── Thread 2: whisper.transcribe(file2)
        └── Thread 3: whisper.transcribe(file3)
```

- Uploads are accepted immediately and return a `job_id` (HTTP 202)
- The worker pulls from the queue and runs up to `max_concurrent` Whisper jobs in parallel
- Each Whisper call runs in a thread pool via `run_in_executor` so it doesn't block the event loop
- The Whisper model is lazy-loaded on first job and reused for subsequent ones

## Whisper model sizes

| Model | ~VRAM | Speed | Notes |
|-------|-------|-------|-------|
| tiny | 1GB | fastest | ok for clear speech |
| base | 1GB | fast | good default |
| small | 2GB | moderate | noticeably better |
| medium | 5GB | slow | good for noisy audio |
| large | 10GB | slowest | best accuracy |

## Production notes

- Replace the in-memory job store with Redis or Postgres for persistence across restarts
- Add auth (API key or OAuth) before exposing this publicly
- Consider running multiple uvicorn workers behind a load balancer, with a shared Redis queue
- `max_concurrent` in `main.py` should match your CPU/GPU capacity -- 3 is conservative for CPU
