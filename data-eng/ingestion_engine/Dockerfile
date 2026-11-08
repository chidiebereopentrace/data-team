FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY manifests ./manifests
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    INGEST_OUT=/storage \
    INGEST_MANIFESTS=/app/manifests
CMD ["sh", "-c", "rm -rf /storage/landing /storage/prepared /storage/quarantine /storage/raw && ingestion-engine --execute"]
