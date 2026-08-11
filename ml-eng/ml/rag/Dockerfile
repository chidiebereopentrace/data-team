# RAG API for Hugging Face Spaces (build from repo root: docker build -f ml/rag/Dockerfile .)
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install repo deps + RAG deps (BQ, LangGraph, FastAPI)
COPY requirements.txt .
COPY ml/rag/requirements.txt ml/rag/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r ml/rag/requirements.txt

# Copy only what RAG needs so the image stays smaller
COPY ml ./ml

# HF Spaces use port 7860; frontend will call this API
ENV PYTHONPATH=/app
# Vector search uses Qdrant (not a local on-disk index). Set at runtime, e.g.:
#   QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NEWS, QDRANT_COLLECTION_RESEARCH_PAPERS,
#   QDRANT_COLLECTION_* for news / academic / policies / public_reports / formation / OTA
#   (see README / vector_retriever.py).

# Optional: pass HF_API_TOKEN at build time (or set at runtime)
ARG HF_API_TOKEN
ENV HF_API_TOKEN=${HF_API_TOKEN}

EXPOSE 7860 8501
CMD ["uvicorn", "ml.rag.api:app", "--host", "0.0.0.0", "--port", "7860"]
