"""Tests for OpenRouter session_id injection and NL2SQL purpose labeling."""
from __future__ import annotations

from unittest import mock

import pytest

from ml.rag.observability import openrouter_run_context


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "HF_API_TOKEN",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "RAG_OPENROUTER_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)


def test_llm_chat_injects_openrouter_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-key")

    captured: dict = {}

    def fake_post(url: str, **kwargs: object) -> mock.Mock:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["headers"] = kwargs.get("headers")
        resp = mock.Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        resp.raise_for_status = mock.Mock()
        return resp

    with mock.patch("ml.rag.llm_chat.requests.post", side_effect=fake_post):
        from ml.rag.llm_chat import llm_chat_complete

        with openrouter_run_context("trace-abc123"):
            out = llm_chat_complete([{"role": "user", "content": "hi"}], purpose="nl2sql")

    assert out == "ok"
    payload = captured.get("json") or {}
    assert payload.get("session_id") == "trace-abc123"
    headers = captured.get("headers") or {}
    assert headers.get("x-session-id") == "trace-abc123"


def test_llm_chat_skips_session_id_for_lm_studio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://127.0.0.1:1234/v1")

    captured: dict = {}

    def fake_post(url: str, **kwargs: object) -> mock.Mock:
        captured["json"] = kwargs.get("json")
        resp = mock.Mock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "x"}}], "usage": {}}
        resp.raise_for_status = mock.Mock()
        return resp

    with mock.patch("ml.rag.llm_chat.requests.post", side_effect=fake_post):
        from ml.rag.llm_chat import llm_chat_complete

        with openrouter_run_context("trace-abc123"):
            llm_chat_complete([{"role": "user", "content": "hi"}])

    payload = captured.get("json") or {}
    assert "session_id" not in payload


def test_llm_chat_openrouter_session_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-key")
    monkeypatch.setenv("RAG_OPENROUTER_SESSION_ID", "off")

    captured: dict = {}

    def fake_post(url: str, **kwargs: object) -> mock.Mock:
        captured["json"] = kwargs.get("json")
        resp = mock.Mock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "x"}}], "usage": {}}
        resp.raise_for_status = mock.Mock()
        return resp

    with mock.patch("ml.rag.llm_chat.requests.post", side_effect=fake_post):
        from ml.rag.llm_chat import llm_chat_complete

        with openrouter_run_context("trace-abc123"):
            llm_chat_complete([{"role": "user", "content": "hi"}])

    payload = captured.get("json") or {}
    assert "session_id" not in payload


def test_call_llama_for_sql_passes_nl2sql_purpose(monkeypatch: pytest.MonkeyPatch) -> None:
    with mock.patch("ml.rag.retrievers.bq_retriever.llm_chat_complete") as mock_llm:
        mock_llm.return_value = "SELECT 1"
        from ml.rag.retrievers.bq_retriever import _call_llama_for_sql

        _call_llama_for_sql([{"role": "user", "content": "q"}])
        mock_llm.assert_called_once()
        assert mock_llm.call_args.kwargs.get("purpose") == "nl2sql"


def test_nl2sql_span_metadata_on_empty_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_BQ_NL2SQL_ENABLED", "0")
    updates: list[dict] = []

    with mock.patch(
        "ml.rag.retrievers.bq_retriever.update_current_span_metadata",
        side_effect=lambda meta: updates.append(meta),
    ):
        with mock.patch("ml.rag.retrievers.bq_retriever.observed_span") as mock_span:
            mock_span.return_value.__enter__ = mock.Mock(return_value=None)
            mock_span.return_value.__exit__ = mock.Mock(return_value=False)
            from ml.rag.retrievers.bq_retriever import BQRetriever

            retriever = BQRetriever(project_id="p", nl2sql_enabled=True)
            with mock.patch.object(retriever, "_nl_to_sql_one", return_value=""):
                out = retriever._nl_to_sql_queries("test question", table_hints=[])

    assert out == []
    assert updates
    assert updates[-1]["mode"] == "per_hint"
    assert updates[-1]["sql_query_count"] == 0
