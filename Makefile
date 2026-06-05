PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8080

.PHONY: help init-env install install-local-models db run run-api docker-run test stop logs clean

help:
	@echo "AI Companion commands"
	@echo "  make run                  Create .env, start pgvector db, install deps, run API/UI"
	@echo "  make run-api              Run API/UI only"
	@echo "  make db                   Start Postgres with pgvector"
	@echo "  make docker-run           Run db and API with Docker Compose"
	@echo "  make install              Install Python dev dependencies"
	@echo "  make install-local-models Install GPU/local model dependencies"
	@echo "  make test                 Run tests"
	@echo "  make stop                 Stop Docker Compose services"
	@echo "  make logs                 Tail Docker Compose logs"

init-env:
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host 'Created .env from .env.example' } else { Write-Host '.env already exists' }"

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-local-models:
	$(PYTHON) -m pip install -e ".[dev,local-models]"

db:
	docker compose up -d db

run: init-env db install run-api

run-api:
	$(PYTHON) -m uvicorn server.main:app --host $(HOST) --port $(PORT)

docker-run: init-env
	docker compose up --build

test:
	$(PYTHON) -m pytest -q

stop:
	docker compose down

logs:
	docker compose logs -f

clean:
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force; if (Test-Path .pytest_cache) { Remove-Item .pytest_cache -Recurse -Force }"
