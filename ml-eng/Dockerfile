# syntax=docker/dockerfile:1
#
# Production Dockerfile for the OpenTrace RAG API (FastAPI service).
# Intended for GCE (VM + Docker or Cloud Run).
#
# Key properties for 12-factor / GCE:
# - Does NOT bake any .env files or secrets.
# - All configuration (Qdrant, BQ, LLM endpoint, RAG_* vars, caches) comes from
#   environment variables or mounted secret files at runtime.
# - Model caches (fastembed, HF) are expected to be on a persistent volume or
#   downloaded on first start (they are large and versioned separately).
#
# Build (from the data-team root):
#   docker build -f ml-eng/Dockerfile -t opentrace-rag-api:latest .
#
# Run locally (prod-like, with env vars):
#   docker run --rm -p 8080:8080 \
#     -e QDRANT_URL=... \
#     -e QDRANT_API_KEY=... \
#     -e RAG_LLM_BASE_URL=http://host.docker.internal:1234/v1 \
#     -e RAG_GENERATE_TEMPERATURE=0.5 \
#     opentrace-rag-api:latest
#
# For Cloud Run / GCE: set the same env vars (or use Secret Manager) and
# point FASTEMBED_CACHE_PATH / HF_HOME to a mounted volume if you want to
# persist the embedding models across restarts.

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Sensible defaults; override in deployment
    PORT=8080

WORKDIR /app

# Minimal system deps (fastembed/onnx and some tokenizers benefit from these)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps for the RAG service
COPY ml/rag/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Production ASGI server
RUN pip install --no-cache-dir "uvicorn[standard]" gunicorn

# Copy only the Python package tree needed at runtime
# (the 'ml' package lives under PYTHONPATH=/app)
COPY ml/ ./ml/

# Make 'ml.rag...' importable exactly like the local PYTHONPATH=. usage
ENV PYTHONPATH=/app

EXPOSE 8080

# Default production command.
# Use gunicorn + uvicorn workers for real deployments (better multi-core).
# Single uvicorn for the simplest Cloud Run / small GCE setups.
CMD ["sh", "-c", "uvicorn ml.rag.app.api:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${UVICORN_WORKERS:-1} --log-level ${LOG_LEVEL:-info}"]

# Alternative (recommended for production):
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "ml.rag.app.api:app"]