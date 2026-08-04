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
    meta = upload_artifact(b"hello,csv", "test.csv")
    assert meta["filename"] == "test.csv"
    assert meta["mime_type"] == "text/csv"
    assert meta["byte_size"] == 9
    assert meta["url"].startswith("file://")


def test_run_exports_csv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_ARTIFACT_LOCAL_DIR", str(tmp_path))
    monkeypatch.delenv("RAG_ARTIFACT_GCS_BUCKET", raising=False)
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
