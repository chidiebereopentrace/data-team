"""Web-fallback node: the turn becomes a gap only when no row or document survived."""

from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.graph import (
    _route_after_web_fallback,
    node_insufficient_context,
    node_web_fallback,
)
from ml.rag.retrievers.web_retriever import WebFallbackResult


def _run_rate_limited(reranked: list[dict]) -> dict:
    state = {"query": "rice production senegal 2024", "reranked_context": reranked}
    with mock.patch("ml.rag.chatbot.graph.needs_web_fallback", return_value=True):
        with mock.patch(
            "ml.rag.chatbot.graph.retrieve_web_fallback_detailed",
            return_value=WebFallbackResult(items=[], status="rate_limited", reason="RATE_LIMIT: 429"),
        ):
            return node_web_fallback(state)


def test_rate_limited_with_nothing_retrieved_sets_insufficient() -> None:
    out = _run_rate_limited([])
    assert out.get("web_fallback_status") == "rate_limited"
    assert out.get("insufficient_context") is True
    assert "reranked_context" not in out
    assert "web_results" not in out


def test_rate_limited_with_one_document_still_answers() -> None:
    doc = {
        "content": "[Academic] Rice yields in the Senegal River valley.",
        "_context_kind": "academic",
        "metadata": {"authors": "X", "publication_year": "2010"},
    }
    out = _run_rate_limited([doc])
    assert out.get("insufficient_context") is not True


def test_route_after_web_fallback_picks_insufficient_branch() -> None:
    assert _route_after_web_fallback({"insufficient_context": True}) == "insufficient_context"
    assert _route_after_web_fallback({"insufficient_context": False}) == "generate"
    assert _route_after_web_fallback({}) == "generate"


def test_node_insufficient_context_names_the_gap_without_citations() -> None:
    out = node_insufficient_context(
        {
            "query": "rice production in Senegal in 2031",
            "decomposition": {"geography": ["Senegal"], "time_start": "2031-01-01", "time_end": "2031-12-31"},
            "turn_contract": {"measure_id": "production"},
        }
    )
    assert "Senegal" in out["answer"]
    assert "2031" in out["answer"]
    assert out["citations"] == []
    assert out["answer_outcome"] == "gap"
