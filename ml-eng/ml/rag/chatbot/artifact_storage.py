"""Upload export artifacts to S3-compatible storage, GCS, or local fallback."""
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


def _s3_bucket_name() -> str:
    return os.environ.get("AWS_S3_BUCKET_NAME", "").strip()


def _s3_credentials_ready() -> bool:
    return bool(
        _s3_bucket_name()
        and os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
        and os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
        and os.environ.get("AWS_ENDPOINT_URL", "").strip()
    )


def _artifact_prefix() -> str:
    raw = (
        os.environ.get("RAG_ARTIFACT_S3_PREFIX")
        or os.environ.get("RAG_ARTIFACT_GCS_PREFIX")
        or "rag-exports"
    )
    return raw.strip().strip("/")


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

    Priority: S3-compatible (Railway neat-icebox via AWS_*) → GCS → local file://.
    """
    if len(data) > _max_artifact_bytes():
        raise ValueError(
            f"Artifact exceeds size limit ({len(data)} > {_max_artifact_bytes()} bytes)"
        )

    mime = mime_type_for_filename(filename)
    artifact_id = f"art_{uuid.uuid4().hex[:12]}"

    if _s3_credentials_ready():
        return _upload_s3(data, filename, artifact_id, mime, _s3_bucket_name())

    gcs_bucket = _gcs_bucket_name()
    if gcs_bucket:
        return _upload_gcs(data, filename, artifact_id, mime, gcs_bucket)

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
        "s3_uri": None,
        "storage_uri": dest.as_uri(),
    }


def _upload_s3(
    data: bytes,
    filename: str,
    artifact_id: str,
    mime: str,
    bucket: str,
) -> dict[str, Any]:
    import boto3  # pyright: ignore[reportMissingImports]

    endpoint = os.environ.get("AWS_ENDPOINT_URL", "").strip()
    region = (
        os.environ.get("AWS_DEFAULT_REGION", "").strip()
        or os.environ.get("AWS_REGION", "").strip()
        or "auto"
    )
    prefix = _artifact_prefix()
    key = f"{prefix}/{artifact_id}/{filename}"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "").strip(),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip(),
        region_name=region,
    )
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=mime)
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=_signed_url_ttl_seconds(),
    )
    s3_uri = f"s3://{bucket}/{key}"
    logger.info("artifact uploaded s3=%s bytes=%d", s3_uri, len(data))
    return {
        "id": artifact_id,
        "filename": filename,
        "mime_type": mime,
        "url": url,
        "byte_size": len(data),
        "gcs_uri": None,
        "s3_uri": s3_uri,
        "storage_uri": s3_uri,
    }


def _upload_gcs(
    data: bytes,
    filename: str,
    artifact_id: str,
    mime: str,
    bucket: str,
) -> dict[str, Any]:
    from google.cloud import storage  # lazy import

    prefix = _artifact_prefix()
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
        "s3_uri": None,
        "storage_uri": gcs_uri,
    }


def refresh_artifact_url(artifact_id: str, filename: str) -> dict[str, Any]:
    """
    Re-sign a download URL for an existing artifact (same object key as upload).

    Key layout: ``{prefix}/{artifact_id}/{filename}``.
    TTL: ``RAG_ARTIFACT_SIGNED_URL_TTL_SECONDS`` (default 3600, min 60).
    """
    aid = (artifact_id or "").strip()
    fname = (filename or "").strip()
    if not aid or not fname:
        raise ValueError("artifact_id and filename are required")
    if "/" in aid or ".." in aid or "/" in fname or ".." in fname:
        raise ValueError("artifact_id and filename must be path-safe (no slashes)")

    ttl = _signed_url_ttl_seconds()
    prefix = _artifact_prefix()
    key = f"{prefix}/{aid}/{fname}"

    if _s3_credentials_ready():
        import boto3  # pyright: ignore[reportMissingImports]

        bucket = _s3_bucket_name()
        endpoint = os.environ.get("AWS_ENDPOINT_URL", "").strip()
        region = (
            os.environ.get("AWS_DEFAULT_REGION", "").strip()
            or os.environ.get("AWS_REGION", "").strip()
            or "auto"
        )
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip(),
            region_name=region,
        )
        try:
            client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            raise FileNotFoundError(f"Artifact not found: {aid}/{fname}") from exc
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl,
        )
        return {
            "id": aid,
            "filename": fname,
            "url": url,
            "expires_in_seconds": ttl,
            "storage_uri": f"s3://{bucket}/{key}",
        }

    gcs_bucket = _gcs_bucket_name()
    if gcs_bucket:
        from google.cloud import storage  # lazy import

        client = storage.Client()
        blob = client.bucket(gcs_bucket).blob(key)
        if not blob.exists():
            raise FileNotFoundError(f"Artifact not found: {aid}/{fname}")
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=ttl),
            method="GET",
        )
        return {
            "id": aid,
            "filename": fname,
            "url": url,
            "expires_in_seconds": ttl,
            "storage_uri": f"gs://{gcs_bucket}/{key}",
        }

    local_root = os.environ.get("RAG_ARTIFACT_LOCAL_DIR", "").strip()
    base = Path(local_root) if local_root else Path(tempfile.gettempdir()) / "rag_artifacts"
    dest = base / f"{aid}_{fname}"
    if not dest.is_file():
        raise FileNotFoundError(f"Artifact not found: {aid}/{fname}")
    return {
        "id": aid,
        "filename": fname,
        "url": dest.as_uri(),
        "expires_in_seconds": ttl,
        "storage_uri": dest.as_uri(),
    }


__all__ = ["mime_type_for_filename", "upload_artifact", "refresh_artifact_url"]
