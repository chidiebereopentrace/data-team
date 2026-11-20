"""Partial answers when a stage overruns, and the multilingual rerank default."""
from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from unittest import mock

import pytest
from ml.rag.chatbot import graph
from ml.rag.chatbot.generator import SourceRef, _finalize_generation_result, generate
from ml.rag.chatbot.stage_budget import (
    remaining_seconds,
    request_budget,
    stage_budget_seconds,
)
from ml.rag.llm_chat import llm_chat_complete
from ml.rag.rerank_client import DEFAULT_RERANK_MODEL_ID, rerank_model_id


def test_stage_budget_uses_configured_seconds(monkeypatch) -> None:
    monkeypatch.setenv("RAG_STAGE_RETRIEVE_BUDGET_S", "2")
    assert stage_budget_seconds("RAG_STAGE_RETRIEVE_BUDGET_S", default_s=12.0) == 2.0


@pytest.mark.parametrize("configured,expected", [(None, 120), (40, 40)])
def test_pipeline_budget_allows_long_generation_and_honours_override(monkeypatch, configured, expected):
    if configured is None:
        monkeypatch.delenv("RAG_REQUEST_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("RAG_REQUEST_TIMEOUT_S", str(configured))

    def invoke(initial, config):
        assert expected - 1 < remaining_seconds() <= expected
        return {"answer": "done", "citations": []}

    monkeypatch.setattr(graph, "_get_compiled_graph", lambda: SimpleNamespace(invoke=invoke))
    assert graph.run_rag("test")["answer"] == "done"


def test_generate_distinguishes_incomplete_document_search_from_empty_results(monkeypatch) -> None:
    monkeypatch.delenv("RAG_ALLOW_UNGROUNDED", raising=False)
    result = generate(
        "Which reports cover irrigation in Senegal?", [],
        decomposition={"geography": ["Senegal"]}, retrieval_partial=True,
    )
    assert "document search did not complete" in result.answer
    assert "Senegal" in result.answer
    assert result.citations == []


def test_openrouter_rerank_default_is_multilingual(monkeypatch) -> None:
    monkeypatch.delenv("RAG_RERANK_MODEL_ID", raising=False)
    assert DEFAULT_RERANK_MODEL_ID == "cohere/rerank-multilingual-v3.0"
    assert rerank_model_id() == "cohere/rerank-multilingual-v3.0"


def test_generate_does_not_dump_source_text_when_model_times_out() -> None:
    item = {
        "content": "Maize prices in Kenya rose through the 2024 season.",
        "source": "news",
        "_context_kind": "news",
        "metadata": {
            "title": "Kenya maize brief",
            "publisher": "FEWS NET",
            "published_at": "2024-06-01",
        },
    }
    with mock.patch("ml.rag.chatbot.generator._call_llama", return_value=""):
        result = generate(
            "what happened to maize prices in Kenya",
            [item],
            task_mode="chat",
            decomposition={"geography": ["Kenya"]},
        )
    assert "could not complete a reliable answer" in result.answer
    assert "Kenya maize brief" not in result.answer
    assert result.citations == []


def test_warehouse_stall_preserves_completed_document_leg(monkeypatch) -> None:
    release = Event()
    plan = {"fact_tables": {"yield": "agg_production_country_year"}}
    monkeypatch.setattr(graph, "node_bq_reason", lambda _state: {"bq_sql_plan": plan})

    def blocked_warehouse(_state):
        release.wait(5)
        return {"bq_results": []}

    monkeypatch.setattr(graph, "node_bq_retrieve", blocked_warehouse)
    evidence = {"content": "A policy passage", "source": "policies"}
    monkeypatch.setattr(graph, "node_parallel_retrieve", lambda _state: {"vector_results": [evidence]})
    try:
        with request_budget(graph._GENERATION_RESERVE_S + 0.2):
            result = graph.node_retrieve_legs({"query": "a comparison"})
        assert result["vector_results"] == [evidence]
        assert result["bq_sql_plan"] == plan
        assert result["bq_sql_debug"] == [{"status": "bq_timeout"}]
    finally:
        release.set()


def test_document_coverage_requires_a_marker_in_the_localized_answer() -> None:
    topic = "What does the policy say about water conservation?"
    ref = SourceRef(1, {
        "content": "Le plan demande de réduire les pertes d'eau.",
        "source": "policies",
        "metadata": {"document_requirements": [topic]},
    }, "Plan national")
    kwargs = {"inline_citations": True, "require_document_citations": True}
    omitted = _finalize_generation_result("Le plan demande de réduire les pertes d'eau.", [ref], **kwargs)
    cited = _finalize_generation_result("Le plan demande de réduire les pertes d'eau [1].", [ref], **kwargs)
    assert omitted.citations == []
    assert omitted.cited_document_requirements == ()
    assert cited.cited_document_requirements == (topic,)


def test_llm_stops_waiting_at_deadline_when_transport_does_not(monkeypatch) -> None:
    release = Event()
    entered = Event()
    finished = Event()
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://example.test/v1")

    def stalled(*_args, **_kwargs):
        entered.set()
        release.wait(3)
        finished.set()
        return mock.Mock()

    monkeypatch.setattr("ml.rag.llm_chat.requests.post", stalled)
    try:
        with request_budget(0.25):
            assert llm_chat_complete([{"role": "user", "content": "test"}]) == ""
        assert entered.is_set()
        assert not finished.is_set()
    finally:
        release.set()
