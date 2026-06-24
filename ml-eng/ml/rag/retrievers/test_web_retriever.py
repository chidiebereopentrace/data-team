"""Unit tests for supplemental web fallback retrieval."""

from __future__ import annotations

import os
from unittest import mock

from ml.rag.retrievers.web_retriever import (
    _build_wiki_search_query,
    _retrieve_wikipedia,
    needs_web_fallback,
    retrieve_web_fallback,
    route_after_rerank,
)


def _news_chunk() -> dict:
    return {
        "content": "[News] Rice policy in Senegal",
        "_context_kind": "news",
        "metadata": {"title": "Senegal rice"},
    }


def _bq_chunk() -> dict:
    return {
        "content": "[Structured data] {'country': 'Senegal'}",
        "_context_kind": "bigquery",
        "metadata": {"sql": "SELECT 1"},
    }


def _academic_chunk() -> dict:
    return {
        "content": "[Academic] Study on rice",
        "_context_kind": "academic",
        "metadata": {"authors": "Author A", "publication_year": "2020"},
    }


def test_needs_web_fallback_disabled_by_default() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAG_WEB_FALLBACK_ENABLED", None)
    assert not needs_web_fallback([_news_chunk(), _bq_chunk()], enabled=False)


def test_needs_web_fallback_low_chunk_count() -> None:
    assert needs_web_fallback([_academic_chunk()], enabled=True)


def test_needs_web_fallback_no_news_no_bq() -> None:
    chunks = [_academic_chunk(), _academic_chunk(), _academic_chunk()]
    assert needs_web_fallback(chunks, enabled=True)


def test_needs_web_fallback_sufficient_internal_context() -> None:
    chunks = [_news_chunk(), _bq_chunk(), _academic_chunk()]
    assert not needs_web_fallback(chunks, enabled=True)


def test_route_after_rerank() -> None:
    weak = {"reranked_context": [_academic_chunk()]}
    strong = {"reranked_context": [_news_chunk(), _bq_chunk(), _academic_chunk()]}
    with mock.patch(
        "ml.rag.retrievers.web_retriever.needs_web_fallback",
        side_effect=lambda ctx, **_: len(ctx) < 3,
    ):
        assert route_after_rerank(weak) == "web_fallback"
        assert route_after_rerank(strong) == "generate"


def test_build_wiki_search_query_includes_geography() -> None:
    q = _build_wiki_search_query(
        "rice production policies",
        {"geography": ["Senegal"], "entities": ["rice"]},
    )
    assert "Senegal" in q
    assert "rice" in q


def test_retrieve_wikipedia_parses_search_and_summary() -> None:
    search_resp = mock.Mock()
    search_resp.raise_for_status = mock.Mock()
    search_resp.json.return_value = {
        "query": {"search": [{"title": "Agriculture in Senegal"}]},
    }
    summary_resp = mock.Mock()
    summary_resp.status_code = 200
    summary_resp.raise_for_status = mock.Mock()
    summary_resp.json.return_value = {
        "title": "Agriculture in Senegal",
        "extract": "Agriculture is a major sector in Senegal with rice as a staple crop.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Agriculture_in_Senegal"}},
    }

    with mock.patch("ml.rag.retrievers.web_retriever.requests.get", side_effect=[search_resp, summary_resp]):
        items = _retrieve_wikipedia("rice Senegal", top_k=1, timeout_s=5.0)

    assert len(items) == 1
    assert items[0]["_context_kind"] == "web_wikipedia"
    assert "Senegal" in items[0]["content"]
    assert "wikipedia.org" in items[0]["metadata"]["url"]


def test_retrieve_web_fallback_wiki_then_tavily() -> None:
    wiki_items = [
        {
            "content": "Wiki summary about rice.",
            "source": "web_wikipedia",
            "_context_kind": "web_wikipedia",
            "metadata": {"title": "Rice", "url": "https://en.wikipedia.org/wiki/Rice"},
        }
    ]
    with mock.patch.dict(os.environ, {"RAG_WEB_FALLBACK_ENABLED": "1"}):
        with mock.patch(
            "ml.rag.retrievers.web_retriever._retrieve_wikipedia",
            return_value=wiki_items,
        ) as wiki_mock:
            with mock.patch(
                "ml.rag.retrievers.web_retriever._retrieve_tavily",
            ) as tavily_mock:
                out = retrieve_web_fallback("rice in Senegal", {"geography": ["Senegal"]})

    assert len(out) == 1
    wiki_mock.assert_called_once()
    tavily_mock.assert_not_called()


def test_retrieve_web_fallback_tavily_when_wiki_empty() -> None:
    tavily_items = [
        {
            "content": "News snippet about Senegal rice policy.",
            "source": "web_search",
            "_context_kind": "web_search",
            "metadata": {"title": "Senegal rice", "url": "https://example.com/a"},
        }
    ]
    with mock.patch.dict(os.environ, {"RAG_WEB_FALLBACK_ENABLED": "1"}):
        with mock.patch(
            "ml.rag.retrievers.web_retriever._retrieve_wikipedia",
            return_value=[],
        ):
            with mock.patch(
                "ml.rag.retrievers.web_retriever._retrieve_tavily",
                return_value=tavily_items,
            ) as tavily_mock:
                out = retrieve_web_fallback("rice policy Senegal", None)

    assert len(out) == 1
    assert out[0]["_context_kind"] == "web_search"
    tavily_mock.assert_called_once()
