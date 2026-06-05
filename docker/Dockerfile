FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md config.yml ./
COPY server ./server
COPY prompts ./prompts
COPY web ./web

RUN pip install --no-cache-dir -e .

EXPOSE 8080
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
