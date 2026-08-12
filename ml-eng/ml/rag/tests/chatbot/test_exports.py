"""Tests for export builders and artifact storage."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.artifact_storage import upload_artifact
from ml.rag.chatbot.export_runner import run_exports
from ml.rag.chatbot.exports.chart_builder import build_chart
from ml.rag.chatbot.exports.csv_builder import build_csv
from ml.rag.chatbot.exports.tabular import rows_from_bq_results


_SAMPLE_ROWS = [
    {"year": 2020, "production_tonnes": 1200},
    {"year": 2021, "production_tonnes": 1350},
    {"year": 2022, "production_tonnes": 1400},
]


def test_build_csv() -> None:
    data, name = build_csv(_SAMPLE_ROWS, filename="maize.csv")
    assert name == "maize.csv"
    assert b"year" in data
    assert b"production_tonnes" in data


def test_build_chart_png() -> None:
    data, name = build_chart(_SAMPLE_ROWS, title="Maize production", filename="maize.png")
    assert name == "maize.png"
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_rows_from_bq_results() -> None:
    bq = [
        {
            "source": "bigquery",
            "content": "{'year': 2020, 'value': 10}",
            "metadata": {},
        }
    ]
    rows = rows_from_bq_results(bq)
    assert len(rows) == 1
    assert rows[0]["year"] == 2020


def test_upload_artifact_local(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ARTIFACT_LOCAL_DIR", str(tmp_path))
    monkeypatch.delenv("RAG_ARTIFACT_GCS_BUCKET", raising=False)
    monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    meta = upload_artifact(b"hello,csv", "test.csv")
    assert meta["filename"] == "test.csv"
    assert meta["mime_type"] == "text/csv"
    assert meta["byte_size"] == 9
    assert meta["url"].startswith("file://")


def test_upload_artifact_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from unittest import mock

    monkeypatch.setenv("AWS_S3_BUCKET_NAME", "neat-icebox")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example.railway.app")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "auto")
    monkeypatch.delenv("RAG_ARTIFACT_GCS_BUCKET", raising=False)

    fake_client = mock.Mock()
    fake_client.generate_presigned_url.return_value = (
        "https://storage.example.railway.app/neat-icebox/rag-exports/art_abc/test.csv?X-Amz-Signature=1"
    )
    fake_boto3 = mock.Mock()
    fake_boto3.client.return_value = fake_client

    with mock.patch.dict(sys.modules, {"boto3": fake_boto3}):
        meta = upload_artifact(b"hello,csv", "test.csv")

    fake_boto3.client.assert_called_once()
    kwargs = fake_boto3.client.call_args.kwargs
    assert kwargs["endpoint_url"] == "https://storage.example.railway.app"
    fake_client.put_object.assert_called_once()
    put_kwargs = fake_client.put_object.call_args.kwargs
    assert put_kwargs["Bucket"] == "neat-icebox"
    assert put_kwargs["Body"] == b"hello,csv"
    assert put_kwargs["ContentType"] == "text/csv"
    assert "rag-exports/" in put_kwargs["Key"]
    assert put_kwargs["Key"].endswith("/test.csv")
    fake_client.generate_presigned_url.assert_called_once()
    assert meta["url"].startswith("https://")
    assert meta["s3_uri"] is not None
    assert meta["s3_uri"].startswith("s3://neat-icebox/")
    assert meta["gcs_uri"] is None


def test_run_exports_csv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_ARTIFACT_LOCAL_DIR", str(tmp_path))
    monkeypatch.delenv("RAG_ARTIFACT_GCS_BUCKET", raising=False)
    monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    bq = [{"source": "bigquery", "content": str(_SAMPLE_ROWS[0]), "metadata": {}}]
    for i, row in enumerate(_SAMPLE_ROWS[1:], start=1):
        bq.append({"source": "bigquery", "content": str(row), "metadata": {}})
    state = {"acf_band_label": "Strong", "acf_score": 80, "acf_explanation": "Good evidence"}
    artifacts = run_exports(
        export_kind="csv",
        query="Maize production Kenya",
        answer="Production rose.",
        bq_results=bq,
        citations=[{"id": 1, "kind": "bigquery", "text": "BQ row"}],
        state=state,
        export_enabled=True,
        plan_type="Agribusinesses",
    )
    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "csv"
    assert artifacts[0]["citation_ids"] == [1]


def test_run_exports_blocked_without_route_flag() -> None:
    with pytest.raises(ValueError, match="export not enabled"):
        run_exports(
            export_kind="csv",
            query="q",
            answer="a",
            bq_results=[],
            citations=[],
            state={},
            export_enabled=False,
            plan_type="Agribusinesses",
        )
