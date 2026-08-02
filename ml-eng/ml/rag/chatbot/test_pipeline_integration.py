"""End-to-end RAG pipeline integration tests.

Sprint 1, Week 3 (Jul 2026): exercises the real compiled LangGraph via
``run_rag()`` while mocking only the external boundaries (vector/BQ retrievers,
the LLM call, reranker model, and web fallback) so the test is deterministic and
runs anywhere — no Qdrant, BigQuery, or LLM backend required.

What this locks in (query → response), across the Weeks 1–3 work:
  1. Data-query path returns answer + citations + an ACF signal.
  2. Direct-answer-first: a preamble-opening LLM answer is cleaned end-to-end.
  3. Inline [N] citations resolve to real structured citation objects.
  4. No-data path returns the structured gap message (no fabrication).
  5. Meta/product path short-circuits with ACF HIGH.
  6. session_id threads through cleanly (Week-2 session isolation regression guard).
  7. `usage` accounting key is present on the result.
"""
from __future__ import annotations

from unittest import mock

import pytest

# The compiled graph needs langgraph at import time; skip gracefully if absent
# (e.g. a minimal dev venv). Runs fully in CI / the provisioned environment.
pytest.importorskip("langgraph")

from ml.rag.chatbot import graph as graph_mod
from ml.rag.chatbot.graph import run_rag


# ---------------------------------------------------------------------------
# Canned retrieval fixtures
# ---------------------------------------------------------------------------

def _news_chunk() -> dict:
    return {
        "content": "Senegal raised its rice self-sufficiency target for 2024.",
        "score": 0.91,
        "metadata": {
            "doc_kind": "news_article",
            "title": "Senegal rice policy shift",
            "publisher": "AgriNews",
            "published_at": "2023-06-01",
            "geo_country_primary": "Senegal",
            "url": "https://example.com/senegal-rice",
        },
    }


def _academic_chunk() -> dict:
    return {
        "content": "Field trials show drought-tolerant rice varieties raise yields.",
        "score": 0.87,
        "metadata": {
            "doc_kind": "academic_article",
            "article_title": "Drought-tolerant rice in West Africa",
            "authors": "Diallo, A.",
            "publication_year": "2022",
            "geo_country_primary": "Senegal",
            "doi": "10.1234/dt-rice",
        },
    }


def _ota_chunk() -> dict:
    return {
        "content": "Rice output projected to rise 8% next season.",
        "score": 0.80,
        "metadata": {
            "doc_kind": "ota_insight",
            "metric_text": "Rice output +8%",
            "geo_country_primary": "Senegal",
            "as_of_date": "2024-01-01",
            "ota_record_id": "ota-senegal-rice-1",
        },
    }


class _FakeBQRetriever:
    """Stand-in for BQRetriever that returns no structured rows."""

    def retrieve(self, *args, **kwargs):
        return []


def _install_pipeline_mocks(stack, *, news=None, academic=None, ota=None, llm_answer="ok"):
    """Patch every external boundary of the graph for deterministic runs."""
    stack.enter_context(
        mock.patch.object(graph_mod, "_retrieve_bq_tables", return_value=[])
    )
    stack.enter_context(mock.patch.object(graph_mod, "BQRetriever", _FakeBQRetriever))
    stack.enter_context(
        mock.patch.object(graph_mod, "_retrieve_news", return_value=news or [])
    )
    stack.enter_context(
        mock.patch.object(graph_mod, "_retrieve_academic", return_value=academic or [])
    )
    stack.enter_context(
        mock.patch.object(graph_mod, "_retrieve_ota", return_value=ota or [])
    )
    # Deterministic reranker: identity passthrough truncated to top_k.
    stack.enter_context(
        mock.patch.object(
            graph_mod, "rerank", side_effect=lambda q, items, top_k=20: list(items)[:top_k]
        )
    )
    # Keep web fallback out of the way (internal context decides the path).
    stack.enter_context(
        mock.patch.object(graph_mod, "needs_web_fallback", return_value=False)
    )
    # Mock the LLM call inside the generator.
    stack.enter_context(
        mock.patch("ml.rag.chatbot.generator._call_llama", return_value=llm_answer)
    )


# ---------------------------------------------------------------------------
# 1. Data-query happy path — answer + citations + ACF signal
# ---------------------------------------------------------------------------

def test_pipeline_data_query_returns_answer_citations_and_acf() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            news=[_news_chunk()],
            academic=[_academic_chunk()],
            ota=[_ota_chunk()],
            llm_answer="Senegal raised its rice self-sufficiency target for 2024.[1]",
        )
        result = run_rag("What is Senegal's rice policy for 2024?")

    assert result.get("answer")
    assert "I don't have OpenTrace data" not in result["answer"]
    # ACF Path B signal present on every response.
    assert result.get("acf_band")
    assert isinstance(result.get("acf_score"), (int, float))
    assert result.get("acf_note") or result.get("acf_explanation")
    # Citations resolved from the inline [1].
    assert result.get("citations")
    assert result["citations"][0]["id"] == 1
    # Usage accounting wired.
    assert "usage" in result


# ---------------------------------------------------------------------------
# 2. Direct-answer-first backstop fires end-to-end
# ---------------------------------------------------------------------------

def test_pipeline_strips_preamble_end_to_end() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            news=[_news_chunk()],
            academic=[_academic_chunk()],
            llm_answer="Based on the context, Senegal raised its rice target.[1]",
        )
        result = run_rag("Senegal rice policy?")

    assert not result["answer"].lower().startswith("based on the context")
    assert "Senegal raised its rice target." in result["answer"]


# ---------------------------------------------------------------------------
# 3. No-data path — structured gap message, no fabrication
# ---------------------------------------------------------------------------

def test_pipeline_no_context_returns_structured_gap() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        # All retrievers empty → nothing reaches the generator.
        _install_pipeline_mocks(stack, news=[], academic=[], ota=[], llm_answer="should not appear")
        result = run_rag("What are tulip exports from Antarctica?")

    assert "I don't have OpenTrace data" in result["answer"]
    assert "ACF: no evidence" in result["answer"]
    assert result.get("citations") == []


# ---------------------------------------------------------------------------
# 4. Meta path short-circuits with ACF HIGH
# ---------------------------------------------------------------------------

def test_pipeline_meta_query_short_circuits_high_acf() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(graph_mod, "is_meta_query", return_value=True))
        stack.enter_context(
            mock.patch(
                "ml.rag.chatbot.assistant_identity.generate_meta_answer",
                return_value="I am ADZA, OpenTrace's agricultural intelligence assistant.",
            )
        )
        result = run_rag("Who are you?")

    assert "ADZA" in result["answer"]
    assert result.get("acf_band") == "strong"
    assert result.get("acf_score") == 90


# ---------------------------------------------------------------------------
# 5. session_id threads through cleanly (Week-2 regression guard)
# ---------------------------------------------------------------------------

def test_pipeline_accepts_session_id() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            news=[_news_chunk()],
            academic=[_academic_chunk()],
            llm_answer="Senegal rice policy summary.[1]",
        )
        result = run_rag("Senegal rice policy?", session_id="itest_session_123")

    assert result.get("answer")
    assert result.get("acf_band")
    assert "usage" in result
