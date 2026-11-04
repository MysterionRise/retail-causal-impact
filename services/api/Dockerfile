# path: services/api/Dockerfile
# Causal engine API + AG-UI copilot endpoint.
# Build context is the repository root:  docker build -f services/api/Dockerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CAUSAL_ROOT=/app

WORKDIR /app

# Build tooling for the few packages that compile on install (econml/dowhy deps)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential make \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[api]"

COPY config ./config
COPY scripts ./scripts
COPY services ./services

# The engine needs the pipeline artifacts. They are produced at image build
# time with the fast profile so the container starts instantly; mount
# ./data, ./models over them to use a different run.
RUN mkdir -p data/raw data/interim data/processed reports/figures reports/tables models \
    && python -m scripts.run_pipeline --config config/default.yaml --data_source synth --stage all

EXPOSE 8000
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
