"""Tests for resolved LLM model config and owl-alpha removal."""
from __future__ import annotations

import logging
import os

import pytest
from fastapi.testclient import TestClient

from ml.rag.app import api as api_mod
from ml.rag.llm_chat import DEFAULT_LLM_MODEL_ID, llm_model_id
from ml.rag.llm_model_config import (
    log_resolved_llm_models,
    resolved_llm_models,
    warn_router_models,
)
from ml.rag.local_env import apply_lm_studio_defaults


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "RAG_LLM_MODEL_ID",
        "RAG_LLM_MODEL_FREE",
        "RAG_LLM_MODEL_FARMERS",
        "RAG_LLM_MODEL_GOVERNMENT",
        "RAG_LLM_MODEL_NGOS",
        "RAG_LLM_MODEL_AGRIBUSINESSES",
        "RAG_LLM_MODEL_INTEGRATED",
        "RAG_BQ_REASONER_MODEL_ID",
        "RAG_BQ_NL2SQL_MODEL_ID",
        "RAG_SUMMARY_MODEL_ID",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def ready_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("RAG_LLM_API_KEY", "sk-test")
    monkeypatch.delenv("BQ_PROJECT", raising=False)
    return TestClient(api_mod.app)


def test_apply_lm_studio_defaults_does_not_seed_owl_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("RAG_LLM_MODEL_ID", raising=False)
    apply_lm_studio_defaults()
    assert os.environ.get("RAG_LLM_MODEL_ID", "").strip() == ""
    assert llm_model_id() == DEFAULT_LLM_MODEL_ID


def test_resolved_llm_models_defaults_to_qwen() -> None:
    models = resolved_llm_models()
    assert models["chat_default"] == DEFAULT_LLM_MODEL_ID
    assert models["summary"] == DEFAULT_LLM_MODEL_ID
    assert models["bq_nl2sql"] == DEFAULT_LLM_MODEL_ID
    assert models["bq_reasoner"] == DEFAULT_LLM_MODEL_ID
    assert models["code_default"] == DEFAULT_LLM_MODEL_ID
    for plan_id, slug in models["plan_models"].items():
        assert slug == DEFAULT_LLM_MODEL_ID, plan_id


def test_warn_router_models_flags_owl_alpha(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("RAG_LLM_MODEL_ID", "openrouter/owl-alpha")
    caplog.set_level(logging.WARNING)
    flagged = warn_router_models()
    assert any("openrouter/owl-alpha" in item for item in flagged)
    assert any("router models detected" in r.message for r in caplog.records)


def test_log_resolved_llm_models_emits_info(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    models = log_resolved_llm_models()
    assert models["chat_default"] == DEFAULT_LLM_MODEL_ID
    assert any("LLM models:" in r.message for r in caplog.records)


def test_ready_includes_llm_models(ready_client: TestClient) -> None:
    resp = ready_client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert "llm_models" in body
    assert body["llm_models"]["chat_default"] == DEFAULT_LLM_MODEL_ID
