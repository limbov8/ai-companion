PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8086
PG_PORT ?= 5433

.PHONY: help init-env install install-audio-tools install-qwen-asr install-local-models check-docker db run run-no-db run-check run-check-no-db run-api run-api-local docker-run test gpu-status stop logs clean

help:
	@echo "AI Companion commands"
	@echo "  make run                  Start db and API/UI with all three local GPU models preloaded"
	@echo "  make run-no-db            Start API/UI with local GPU models, skipping Docker/Postgres"
	@echo "  make run-check            Start local model server in background and show nvidia-smi"
	@echo "  make run-check-no-db      Background local model check, skipping Docker/Postgres"
	@echo "  make run-api-local        Run API/UI with strict local GPU model preload"
	@echo "  make run-api              Run API/UI without forcing local model preload"
	@echo "  make db                   Start Postgres with pgvector"
	@echo "  make docker-run           Run db and API with Docker Compose"
	@echo "  make install              Install Python dev dependencies"
	@echo "  make install-audio-tools  Install repo-local SoX for Qwen TTS on Windows"
	@echo "  make install-qwen-asr     Install Qwen ASR without breaking Qwen TTS"
	@echo "  make install-local-models Install GPU/local model dependencies"
	@echo "  make test                 Run tests"
	@echo "  make gpu-status           Show nvidia-smi GPU status"
	@echo "  make stop                 Stop Docker Compose services"
	@echo "  make logs                 Tail Docker Compose logs"

init-env:
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host 'Created .env from .env.example' } else { Write-Host '.env already exists' }"

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-audio-tools:
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_sox.ps1

install-qwen-asr:
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_qwen_asr.ps1 -Python '$(PYTHON)'

install-local-models: install-audio-tools
	$(PYTHON) -m pip install -e ".[dev,local-models]"
	$(MAKE) install-qwen-asr

check-docker:
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_docker.ps1

db: check-docker
	PG_PORT=$(PG_PORT) docker compose up -d db

run: init-env db install-local-models run-api-local

run-no-db: init-env install-local-models run-api-local

run-check: init-env db install-local-models
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_local_and_check.ps1 -Python '$(PYTHON)' -HostName '$(HOST)' -Port $(PORT)

run-check-no-db: init-env install-local-models
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_local_and_check.ps1 -Python '$(PYTHON)' -HostName '$(HOST)' -Port $(PORT)

run-api:
	$(PYTHON) -m uvicorn server.main:app --host $(HOST) --port $(PORT)

run-api-local:
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_api_local.ps1 -Python '$(PYTHON)' -HostName '$(HOST)' -Port $(PORT)

docker-run: init-env check-docker
	PG_PORT=$(PG_PORT) docker compose up --build

test:
	$(PYTHON) -m pytest -q

gpu-status:
	nvidia-smi

stop:
	docker compose down

logs:
	docker compose logs -f

clean:
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force; if (Test-Path .pytest_cache) { Remove-Item .pytest_cache -Recurse -Force }"
