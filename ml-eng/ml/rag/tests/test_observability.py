"""Unit tests for Langfuse observability helpers (no live Langfuse server)."""
from __future__ import annotations

import os

from concurrent.futures import ThreadPoolExecutor

import pytest

from ml.rag.observability import (
    RagTraceHandle,
    build_rag_invoke_config,
    get_openrouter_run_id,
    infer_rag_route,
    is_tracing_enabled,
    openrouter_run_context,
    openrouter_sessions_enabled,
    rag_trace_context,
    run_with_tracing_context,
    safe_llm_trace_input,
    sql_hash,
    summarize_rag_result_for_trace,
)


@pytest.fixture(autouse=True)
def _clear_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_HOST",
        "LANGFUSE_TRACING_RELEASE",
        "RAG_LLM_BASE_URL",
        "RAG_OPENROUTER_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)


def test_is_tracing_disabled_without_keys() -> None:
    assert is_tracing_enabled() is False


def test_is_tracing_enabled_with_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langfuse")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert is_tracing_enabled() is True


def test_build_rag_invoke_config_no_keys() -> None:
    cfg = build_rag_invoke_config(session_id="sess-1", plan_type="Farmers", category="Government")
    assert cfg.get("metadata", {}).get("langfuse_session_id") == "sess-1"
    assert "plan_type:Farmers" in cfg.get("metadata", {}).get("langfuse_tags", [])
    assert "category:Government" in cfg.get("metadata", {}).get("langfuse_tags", [])
    assert cfg.get("callbacks") in (None, [])


def test_build_rag_invoke_config_includes_release_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_RELEASE", "abc123")
    cfg = build_rag_invoke_config(session_id="sess-1")
    assert "release:abc123" in cfg.get("metadata", {}).get("langfuse_tags", [])


def test_build_rag_invoke_config_merges_base() -> None:
    cfg = build_rag_invoke_config(
        base_config={"metadata": {"custom": "x"}, "callbacks": []},
        session_id="abc",
    )
    assert cfg["metadata"]["custom"] == "x"
    assert cfg["metadata"]["langfuse_session_id"] == "abc"


def test_sql_hash_deterministic() -> None:
    assert sql_hash("SELECT 1") == sql_hash("SELECT 1")
    assert sql_hash("SELECT 1") != sql_hash("SELECT 2")


def test_summarize_rag_result_for_trace() -> None:
    summary = summarize_rag_result_for_trace(
        {
            "vector_news_results": [1, 2],
            "rerank_mode": "cohere",
            "answer": "ok",
        }
    )
    assert summary["vector_news_count"] == 2
    assert summary["rerank_mode"] == "cohere"
    assert summary["route"] == "full_rag"


def test_run_with_tracing_context_runs_fn() -> None:
    assert run_with_tracing_context(lambda x: x + 1, 1)() == 2


def test_openrouter_run_context_sets_and_restores() -> None:
    assert get_openrouter_run_id() is None
    with openrouter_run_context("run-a"):
        assert get_openrouter_run_id() == "run-a"
    assert get_openrouter_run_id() is None


def test_openrouter_sessions_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    assert openrouter_sessions_enabled() is True
    monkeypatch.setenv("RAG_OPENROUTER_SESSION_ID", "off")
    assert openrouter_sessions_enabled() is False
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.delenv("RAG_OPENROUTER_SESSION_ID", raising=False)
    assert openrouter_sessions_enabled() is False


def test_run_with_tracing_context_preserves_contextvar() -> None:
    with openrouter_run_context("thread-run"):
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(run_with_tracing_context(get_openrouter_run_id))
            assert fut.result() == "thread-run"


def test_rag_trace_context_sets_openrouter_without_langfuse() -> None:
    with rag_trace_context(session_id="s1", trace_input={"query": "hi"}) as handle:
        assert handle.span is None
        assert get_openrouter_run_id() is not None
        assert len(get_openrouter_run_id() or "") >= 8
    assert get_openrouter_run_id() is None


def test_infer_rag_route_meta() -> None:
    assert infer_rag_route({"is_meta_query": True}) == "meta"


def test_infer_rag_route_full_rag() -> None:
    assert infer_rag_route({"vector_news_results": [{"content": "x"}]}) == "full_rag"


def test_safe_llm_trace_input_truncates_user_message() -> None:
    long_msg = "x" * 1000
    out = safe_llm_trace_input(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": long_msg}],
        "test-model",
    )
    assert out["model"] == "test-model"
    assert out["message_count"] == 2
    assert len(out["last_user_message"]) == 500


def test_rag_trace_context_noop_without_keys() -> None:
    with rag_trace_context(session_id="s1", trace_input={"query": "hi"}) as handle:
        assert handle.span is None
        assert get_openrouter_run_id() is not None
        handle.update_output({"answer": "hello", "is_meta_query": True})
    assert get_openrouter_run_id() is None


def test_rag_trace_handle_update_output_no_span() -> None:
    handle = RagTraceHandle(span=None)
    handle.update_output({"answer": "ok"})  # should not raise


def test_llm_chat_complete_no_raise_without_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BASE_URL", "")
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    from ml.rag.llm_chat import llm_chat_complete

    assert llm_chat_complete([{"role": "user", "content": "hi"}]) == ""
