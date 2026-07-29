"""Tests for the web-fallback graph node guardrails.

These verify the most important guardrail behavior: when supplemental web
search is rate-limited and the reranked internal context is also weak, the
graph must route to the deterministic "insufficient context" branch instead
of silently letting the generator fabricate an answer from stale chunks.
"""

from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.graph import (
    _INSUFFICIENT_CONTEXT_ANSWER,
    _route_after_web_fallback,
    node_insufficient_context,
    node_web_fallback,
)
from ml.rag.retrievers.web_retriever import WebFallbackResult


def _weak_reranked() -> list[dict]:
    """One vaguely-related academic chunk — below the usability threshold."""
    return [
        {
            "content": "[Academic] An old paper that doesn't really answer the question.",
            "_context_kind": "academic",
            "source": "academic",
            "metadata": {"authors": "X", "publication_year": "2010"},
        }
    ]


def test_node_web_fallback_rate_limited_with_weak_internal_sets_insufficient() -> None:
    state = {"query": "rice production senegal 2024", "reranked_context": _weak_reranked()}
    with mock.patch(
        "ml.rag.chatbot.graph.needs_web_fallback",
        return_value=True,
    ):
        with mock.patch(
            "ml.rag.chatbot.graph.retrieve_web_fallback_detailed",
            return_value=WebFallbackResult(items=[], status="rate_limited", reason="RATE_LIMIT: 429"),
        ):
            out = node_web_fallback(state)

    assert out.get("web_fallback_status") == "rate_limited"
    assert out.get("insufficient_context") is True
    # Critical: we did NOT append fake web chunks or extend reranked_context.
    assert "reranked_context" not in out
    assert "web_results" not in out


def test_node_web_fallback_rate_limited_with_strong_internal_does_not_block() -> None:
    """If internal context is already strong, a rate-limited web call should NOT flip
    the turn to insufficient — we can still answer from internal sources."""
    strong = [
        {"content": "[News] x", "_context_kind": "news", "metadata": {"title": "n"}},
        {"content": "[News] y", "_context_kind": "news", "metadata": {"title": "n"}},
        {"content": "[News] z", "_context_kind": "news", "metadata": {"title": "n"}},
        {"content": "[Structured data] {}", "_context_kind": "bigquery", "metadata": {"sql": "SELECT 1"}},
    ]
    state = {"query": "q", "reranked_context": strong}
    with mock.patch(
        "ml.rag.chatbot.graph.needs_web_fallback",
        return_value=True,
    ):
        with mock.patch(
            "ml.rag.chatbot.graph.retrieve_web_fallback_detailed",
            return_value=WebFallbackResult(items=[], status="rate_limited", reason="RATE_LIMIT"),
        ):
            out = node_web_fallback(state)

    assert out.get("web_fallback_status") == "rate_limited"
    assert out.get("insufficient_context") is not True


def test_route_after_web_fallback_picks_insufficient_branch() -> None:
    assert _route_after_web_fallback({"insufficient_context": True}) == "insufficient_context"
    assert _route_after_web_fallback({"insufficient_context": False}) == "generate"
    assert _route_after_web_fallback({}) == "generate"


def test_node_insufficient_context_returns_canned_answer_no_citations() -> None:
    out = node_insufficient_context({"web_fallback_status": "rate_limited", "web_fallback_reason": "x"})
    assert out["answer"] == _INSUFFICIENT_CONTEXT_ANSWER
    assert out["citations"] == []
