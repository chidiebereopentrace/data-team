"""Upload export artifacts to GCS (or local fallback for dev/tests)."""
from __future__ import annotations

import logging
import os
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MIME_BY_EXT = {
    ".csv": "text/csv",
    ".png": "image/png",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".html": "text/html",
}


def mime_type_for_filename(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return _MIME_BY_EXT.get(ext, "application/octet-stream")


def _gcs_bucket_name() -> str:
    return os.environ.get("RAG_ARTIFACT_GCS_BUCKET", "").strip()


def _signed_url_ttl_seconds() -> int:
    raw = os.environ.get("RAG_ARTIFACT_SIGNED_URL_TTL_SECONDS", "3600").strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return 3600


def _max_artifact_bytes() -> int:
    raw = os.environ.get("RAG_ARTIFACT_MAX_BYTES", str(10 * 1024 * 1024)).strip()
    try:
        return max(1024, int(raw))
    except ValueError:
        return 10 * 1024 * 1024


def upload_artifact(data: bytes, filename: str) -> dict[str, Any]:
    """
    Store artifact bytes and return metadata including a download URL.

    Production: GCS bucket + V4 signed URL (requires RAG_ARTIFACT_GCS_BUCKET).
    Dev/test: writes under RAG_ARTIFACT_LOCAL_DIR or system temp; url is file:// path.
    """
    if len(data) > _max_artifact_bytes():
        raise ValueError(
            f"Artifact exceeds size limit ({len(data)} > {_max_artifact_bytes()} bytes)"
        )

    mime = mime_type_for_filename(filename)
    artifact_id = f"art_{uuid.uuid4().hex[:12]}"
    bucket = _gcs_bucket_name()

    if bucket:
        return _upload_gcs(data, filename, artifact_id, mime, bucket)

    local_root = os.environ.get("RAG_ARTIFACT_LOCAL_DIR", "").strip()
    base = Path(local_root) if local_root else Path(tempfile.gettempdir()) / "rag_artifacts"
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"{artifact_id}_{filename}"
    dest.write_bytes(data)
    logger.info("artifact stored locally path=%s bytes=%d", dest, len(data))
    return {
        "id": artifact_id,
        "filename": filename,
        "mime_type": mime,
        "url": dest.as_uri(),
        "byte_size": len(data),
        "gcs_uri": None,
    }


def _upload_gcs(
    data: bytes,
    filename: str,
    artifact_id: str,
    mime: str,
    bucket: str,
) -> dict[str, Any]:
    from google.cloud import storage  # lazy import

    prefix = os.environ.get("RAG_ARTIFACT_GCS_PREFIX", "rag-exports").strip().strip("/")
    blob_name = f"{prefix}/{artifact_id}/{filename}"
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(blob_name)
    blob.upload_from_string(data, content_type=mime)
    ttl = _signed_url_ttl_seconds()
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(seconds=ttl),
        method="GET",
    )
    gcs_uri = f"gs://{bucket}/{blob_name}"
    logger.info("artifact uploaded gcs=%s bytes=%d", gcs_uri, len(data))
    return {
        "id": artifact_id,
        "filename": filename,
        "mime_type": mime,
        "url": url,
        "byte_size": len(data),
        "gcs_uri": gcs_uri,
    }


__all__ = ["mime_type_for_filename", "upload_artifact"]
