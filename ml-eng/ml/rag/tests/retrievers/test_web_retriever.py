"""Unit tests for supplemental web fallback retrieval."""

from __future__ import annotations

import os
from unittest import mock

from ml.rag.retrievers.web_retriever import (
    WebFallbackResult,
    _build_wiki_search_query,
    _retrieve_tavily,
    _retrieve_wikipedia,
    _shape_wiki_queries,
    _wiki_title_passes,
    needs_web_fallback,
    reset_tavily_quota,
    retrieve_web_fallback,
    retrieve_web_fallback_detailed,
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


def test_shape_wiki_queries_entity_country_primary() -> None:
    shaped = _shape_wiki_queries(
        "How does maize production compare across regions and what policies matter most?",
        {"geography": ["Kenya"], "entities": ["Maize"]},
    )
    assert shaped[0] == "Maize Kenya"
    # Long question is not used as the sole primary query.
    assert "compare across" not in shaped[0].lower()


def test_shape_wiki_africa_default_appends_africa() -> None:
    shaped = _shape_wiki_queries(
        "which country has the best agricultural activity in 2020",
        {"entities": [], "geography": [], "africa_default": True},
    )
    assert any("africa" in s.lower() for s in shaped)


def test_wiki_title_passes_drops_switzerland_on_africa_default() -> None:
    from ml.rag.retrievers.web_retriever import _wiki_title_passes

    assert not _wiki_title_passes(
        "Agriculture in Switzerland",
        countries=[],
        entity_tokens=["agriculture"],
        africa_default=True,
    )
    assert _wiki_title_passes(
        "Agriculture in Africa",
        countries=[],
        entity_tokens=["agriculture"],
        africa_default=True,
    )


def test_build_wiki_search_query_includes_geography() -> None:
    q = _build_wiki_search_query(
        "rice production policies",
        {"geography": ["Senegal"], "entities": ["rice"]},
    )
    assert "Senegal" in q
    assert "rice" in q


def test_wiki_title_passes_drops_conflicting_country() -> None:
    assert not _wiki_title_passes(
        "Agriculture in Nigeria",
        countries=["Kenya"],
        entity_tokens=["maize"],
    )
    assert _wiki_title_passes(
        "Agriculture in Kenya",
        countries=["Kenya"],
        entity_tokens=["maize"],
    )
    # Universal topical title kept when it does not name a conflicting country.
    assert _wiki_title_passes(
        "Maize",
        countries=["Kenya"],
        entity_tokens=["maize"],
    )


def test_retrieve_wikipedia_prefers_opensearch() -> None:
    opensearch_resp = mock.Mock()
    opensearch_resp.raise_for_status = mock.Mock()
    opensearch_resp.json.return_value = [
        "rice Senegal",
        ["Agriculture in Senegal"],
        [""],
        ["https://en.wikipedia.org/wiki/Agriculture_in_Senegal"],
    ]
    summary_resp = mock.Mock()
    summary_resp.status_code = 200
    summary_resp.raise_for_status = mock.Mock()
    summary_resp.json.return_value = {
        "title": "Agriculture in Senegal",
        "extract": "Agriculture is a major sector in Senegal with rice as a staple crop.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Agriculture_in_Senegal"}},
    }
    get = mock.Mock(side_effect=[opensearch_resp, summary_resp])

    with mock.patch("ml.rag.retrievers.web_retriever.requests.get", get):
        items = _retrieve_wikipedia(
            "rice Senegal",
            top_k=1,
            timeout_s=5.0,
            countries=["Senegal"],
            entity_tokens=["rice"],
        )

    assert len(items) == 1
    assert items[0]["_context_kind"] == "web_wikipedia"
    assert "Senegal" in items[0]["content"]
    assert "wikipedia.org" in items[0]["metadata"]["url"]
    first_params = get.call_args_list[0].kwargs.get("params") or get.call_args_list[0][1].get("params")
    assert first_params["action"] == "opensearch"


def test_retrieve_wikipedia_falls_back_to_list_search() -> None:
    empty_open = mock.Mock()
    empty_open.raise_for_status = mock.Mock()
    empty_open.json.return_value = ["q", [], [], []]
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

    with mock.patch(
        "ml.rag.retrievers.web_retriever.requests.get",
        side_effect=[empty_open, search_resp, summary_resp],
    ):
        items = _retrieve_wikipedia("rice Senegal", top_k=1, timeout_s=5.0, countries=["Senegal"])

    assert len(items) == 1


def test_retrieve_wikipedia_section_when_summary_thin() -> None:
    opensearch_resp = mock.Mock()
    opensearch_resp.raise_for_status = mock.Mock()
    opensearch_resp.json.return_value = ["Maize", ["Maize"], [""], ["https://en.wikipedia.org/wiki/Maize"]]
    summary_resp = mock.Mock()
    summary_resp.status_code = 200
    summary_resp.raise_for_status = mock.Mock()
    summary_resp.json.return_value = {
        "title": "Maize",
        "extract": "Short stub.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Maize"}},
    }
    sections_resp = mock.Mock()
    sections_resp.raise_for_status = mock.Mock()
    sections_resp.json.return_value = {
        "parse": {"sections": [{"toclevel": 1, "index": "1", "line": "Description"}]},
    }
    section_text_resp = mock.Mock()
    section_text_resp.raise_for_status = mock.Mock()
    section_text_resp.json.return_value = {
        "parse": {
            "text": {
                "*": "<p>Maize is a cereal grain first domesticated by indigenous peoples in southern Mexico.</p>"
            }
        },
    }

    with mock.patch(
        "ml.rag.retrievers.web_retriever.requests.get",
        side_effect=[opensearch_resp, summary_resp, sections_resp, section_text_resp],
    ):
        items = _retrieve_wikipedia(
            "Maize Kenya",
            top_k=1,
            timeout_s=5.0,
            countries=["Kenya"],
            entity_tokens=["maize"],
        )

    assert len(items) == 1
    assert "cereal grain" in items[0]["content"]
    assert items[0]["metadata"]["wiki_section_used"] is True


def test_retrieve_web_fallback_wiki_then_tavily() -> None:
    wiki_items = [
        {
            "content": "Wiki summary about rice.",
            "source": "web_wikipedia",
            "_context_kind": "web_wikipedia",
            "metadata": {"title": "Rice", "url": "https://en.wikipedia.org/wiki/Rice"},
        }
    ]
    with mock.patch.dict(
        os.environ,
        {"RAG_WEB_FALLBACK_ENABLED": "1", "RAG_WEB_WIKI_TOP_K": "1"},
    ):
        with mock.patch(
            "ml.rag.retrievers.web_retriever._retrieve_wikipedia",
            return_value=wiki_items,
        ) as wiki_mock:
            with mock.patch(
                "ml.rag.retrievers.web_retriever._retrieve_tavily",
            ) as tavily_mock:
                out = retrieve_web_fallback(
                    "rice in Senegal",
                    {"geography": ["Senegal"], "entities": ["rice"]},
                )

    assert len(out) == 1
    wiki_mock.assert_called_once()
    assert wiki_mock.call_args.args[0] == "rice Senegal"
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
                return_value=(tavily_items, "ok", ""),
            ) as tavily_mock:
                out = retrieve_web_fallback(
                    "What rice policy matters in Senegal?",
                    {"geography": ["Senegal"], "entities": ["rice"]},
                )

    assert len(out) == 1
    assert out[0]["_context_kind"] == "web_search"
    tavily_mock.assert_called_once()
    # Tavily uses the primary shaped query (entity + country), not the long question.
    assert tavily_mock.call_args.args[0] == "rice Senegal"


# --- Guardrail tests (rate limit, quota, structured result) ---


def test_retrieve_tavily_rate_limit_no_retry(monkeypatch) -> None:
    """A 429 / quota error must surface as ``rate_limited`` and NOT trigger backoff retry."""
    reset_tavily_quota()
    monkeypatch.setenv("RAG_TAVILY_DAILY_LIMIT", "100")
    # Make backoff non-zero so we'd notice an accidental retry on a rate-limit.
    monkeypatch.setenv("RAG_TAVILY_BACKOFF_S", "0.01")

    rate_limited = mock.Mock(return_value=(None, [], "RATE_LIMIT: 429 Too Many Requests"))
    with mock.patch(
        "ml.web_data_mining.agentic.tavily_tools.is_tavily_configured",
        return_value=True,
    ):
        with mock.patch(
            "ml.web_data_mining.agentic.tavily_tools.tavily_search_news",
            rate_limited,
        ):
            items, status, reason = _retrieve_tavily(
                "rice senegal", top_k=2, time_start=None, time_end=None
            )

    assert items == []
    assert status == "rate_limited"
    assert "RATE_LIMIT" in reason
    # Critical: do NOT retry on a rate-limit signal — that would just burn quota.
    assert rate_limited.call_count == 1


def test_retrieve_tavily_transient_error_retries_once(monkeypatch) -> None:
    """Transient (non-rate-limit) errors should retry exactly once after backoff."""
    reset_tavily_quota()
    monkeypatch.setenv("RAG_TAVILY_DAILY_LIMIT", "100")
    monkeypatch.setenv("RAG_TAVILY_BACKOFF_S", "0.01")

    good_results = [
        {"title": "T", "content": "Some news content about rice in Senegal.", "url": "https://x"}
    ]
    side_effect = [
        (None, [], "connection reset"),  # first call: transient
        ("ok", good_results, None),       # retry: succeeds
    ]
    call = mock.Mock(side_effect=side_effect)
    with mock.patch(
        "ml.web_data_mining.agentic.tavily_tools.is_tavily_configured",
        return_value=True,
    ):
        with mock.patch(
            "ml.web_data_mining.agentic.tavily_tools.tavily_search_news",
            call,
        ):
            items, status, _reason = _retrieve_tavily(
                "rice senegal", top_k=2, time_start=None, time_end=None
            )

    assert call.call_count == 2
    assert status == "ok"
    assert len(items) == 1


def test_retrieve_tavily_local_quota_exhausted_skips_call(monkeypatch) -> None:
    """When the in-process daily counter has hit the limit, do not call Tavily at all."""
    reset_tavily_quota()
    monkeypatch.setenv("RAG_TAVILY_DAILY_LIMIT", "1")

    call = mock.Mock(return_value=("x", [{"title": "T", "content": "x" * 50, "url": "https://x"}], None))
    with mock.patch(
        "ml.web_data_mining.agentic.tavily_tools.is_tavily_configured",
        return_value=True,
    ):
        with mock.patch(
            "ml.web_data_mining.agentic.tavily_tools.tavily_search_news",
            call,
        ):
            # First call consumes the only quota slot
            _items1, status1, _r1 = _retrieve_tavily(
                "q1", top_k=1, time_start=None, time_end=None
            )
            # Second call must be blocked locally
            items2, status2, reason2 = _retrieve_tavily(
                "q2", top_k=1, time_start=None, time_end=None
            )

    assert status1 == "ok"
    assert call.call_count == 1  # only the first attempt actually hit Tavily
    assert items2 == []
    assert status2 == "rate_limited"
    assert "quota" in reason2.lower()


def test_retrieve_web_fallback_detailed_propagates_rate_limit(monkeypatch) -> None:
    """Wiki empty + Tavily rate-limited must surface rate_limited (not ok) so the graph
    can refuse to answer instead of papering over with stale internal docs."""
    reset_tavily_quota()
    monkeypatch.setenv("RAG_WEB_FALLBACK_ENABLED", "1")

    with mock.patch(
        "ml.rag.retrievers.web_retriever._retrieve_wikipedia",
        return_value=[],
    ):
        with mock.patch(
            "ml.rag.retrievers.web_retriever._retrieve_tavily",
            return_value=([], "rate_limited", "RATE_LIMIT: 429"),
        ):
            result = retrieve_web_fallback_detailed("rice policy senegal", None)

    assert isinstance(result, WebFallbackResult)
    assert result.items == []
    assert result.status == "rate_limited"
    assert "RATE_LIMIT" in result.reason
