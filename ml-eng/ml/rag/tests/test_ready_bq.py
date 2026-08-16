"""Readiness / GCP-BQ probe unit tests (no live BigQuery)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ml.rag.app import api as api_mod


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("RAG_LLM_API_KEY", "sk-test")
    monkeypatch.delenv("BQ_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64", raising=False)
    monkeypatch.delenv("RAG_REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    return TestClient(api_mod.app)


def test_ready_skips_bq_when_project_unset(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["bq"]["ok"] is True
    assert body["bq"].get("skipped")


def test_ready_not_ready_when_bq_project_without_credentials(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BQ_PROJECT", "demo-project")
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["bq"]["ok"] is False
    assert "BQ_PROJECT+GCP credentials" in body["missing_config_keys"]


def test_ready_gcp_json_ok_but_bq_client_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sa = tmp_path / "sa.json"
    sa.write_text(json.dumps({"type": "service_account", "project_id": "demo"}), encoding="utf-8")
    monkeypatch.setenv("BQ_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa))

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("bq unreachable")

    monkeypatch.setattr(api_mod, "_bq_readiness", lambda: {
        "project_set": True,
        "project": "demo-project",
        "gcp": {"json_ok": True, "credentials_path_set": True},
        "ok": False,
        "error": "bq unreachable",
    })
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["bq"]["ok"] is False
