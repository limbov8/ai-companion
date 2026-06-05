# AI Companion

Personal AI companion monolith for voice conversation, note taking, memory, retrieval, prompts, and extensible tools.

## What is included

- FastAPI backend with conversation orchestration.
- Browser voice UI with microphone capture, barge-in controls, and streaming-friendly endpoints.
- Local model service interfaces for Whisper ASR, Qwen embeddings, and Qwen TTS.
- DeepSeek chat client with multi-turn conversation support.
- pgvector/Postgres memory schema and Docker Compose.
- Prompt registry and prompt eval templates.
- Extensible tool and skill registry with web search as the first tool.

## Quick start

1. Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY`.
2. Run everything, including local ASR, embedding, and TTS model preload on the GPU:

```powershell
make run
```

3. Open `http://localhost:8086`.

The `make run` target creates `.env` when missing, starts Postgres with pgvector, installs local model dependencies, and runs the FastAPI/UI server with strict local GPU model preload. Startup fails if Whisper ASR, Qwen embedding, or Qwen TTS cannot load.

If Docker Desktop is not running and you only want to run the local GPU model API/UI:

```powershell
make run-no-db
```

The Docker Postgres service is exposed on host port `5433` by default to avoid collisions with a local Postgres on `5432`. If you already have a `.env`, update `DATABASE_URL` to:

```powershell
postgresql+psycopg://ai_companion:ai_companion@localhost:5433/ai_companion
```

Check GPU usage after the server starts:

```powershell
make gpu-status
```

Or start the server in the background and automatically show `nvidia-smi` after health passes:

```powershell
make run-check
```

Use `make run-check-no-db` for the same model/GPU check while Docker is down.

## Manual start

Start Postgres with pgvector:

```powershell
docker compose up -d db
```

Install Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Run the API:

```powershell
uvicorn server.main:app --host 0.0.0.0 --port 8086
```

The local GPU models are lazy-loaded service adapters. They can be run one at a time on a single GPU instead of keeping ASR, embedding, and TTS resident at all times.
