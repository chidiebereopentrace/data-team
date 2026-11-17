from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

from ml.rag.chatbot import graph


def _decompose(monkeypatch, query, **facets):
    monkeypatch.setattr(graph, "decompose_query", lambda *args, **kwargs: facets.copy())
    monkeypatch.setattr(
        graph, "enrich_query_to_english",
        lambda *args, **kwargs: SimpleNamespace(answer_lang="en", english_query="", needs_enrich=False),
    )
    monkeypatch.setattr(graph, "metacog_enabled", lambda: False)
    return graph.node_decompose({"query": query})


def test_yield_comparison_searches_ota_public_reports_and_news(monkeypatch):
    state = _decompose(
        monkeypatch, "Compare maize yields in Kenya in 2018 versus 2023.",
        geography=["Kenya"], entities=["maize"], job="compare",
    )
    bq = Mock(return_value={"bq_results": [{"content": "yield rows"}]})
    ota = Mock(return_value=[{"content": "OTA yield insight"}])
    reports = Mock(return_value=[{"content": "Public crop survey with yield figures"}])
    news = Mock(return_value=[{"content": "News coverage of the yield release"}])
    other = Mock(return_value=[])
    monkeypatch.setattr(graph, "node_bq_retrieve", bq)
    monkeypatch.setattr(graph, "_retrieve_ota", ota)
    monkeypatch.setattr(graph, "_retrieve_public_reports", reports)
    monkeypatch.setattr(graph, "_retrieve_news", news)
    for name in ("academic_papers", "policies", "formation", "legacy_research"):
        monkeypatch.setattr(graph, f"_retrieve_{name}", other)
    monkeypatch.setattr(graph, "_use_legacy_research_collection", lambda: True)

    result = graph.node_retrieve_legs(state)

    assert state["route_candidate"] == "full_rag"
    bq.assert_called_once()
    ota.assert_called_once()
    reports.assert_called_once()
    news.assert_called_once()
    other.assert_not_called()
    assert result["bq_results"][0]["content"] == "yield rows"
    assert result["vector_ota_results"][0]["content"] == "OTA yield insight"
    assert result["vector_public_reports_results"][0]["content"] == "Public crop survey with yield figures"
    assert result["vector_news_results"][0]["content"] == "News coverage of the yield release"
    assert state["source_selection"]["qdrant"]["reason"] == "indicator_context"


def test_data_only_export_skips_qdrant(monkeypatch):
    state = _decompose(
        monkeypatch, "Give me only the data for maize yield in Kenya in 2023 as CSV.",
        geography=["Kenya"], entities=["maize"], job="fact",
    )
    assert state["source_selection"]["bigquery"]["selected"]
    assert state["source_selection"]["qdrant"]["reason"] == "data_only_request"
    assert not state["source_selection"]["qdrant"]["selected"]


def test_indicator_search_allows_reports_and_news_published_after_the_observation_year(monkeypatch):
    query = "What was the maize yield in Kenya in 2023?"
    state = _decompose(
        monkeypatch, query, geography=["Kenya"], entities=["maize"], job="fact",
        time_start="2023-01-01", time_end="2023-12-31",
    )
    vector = Mock()
    vector.return_value.retrieve.return_value = []
    monkeypatch.setattr(graph, "VectorRetriever", vector)

    for retrieve in (graph._retrieve_public_reports, graph._retrieve_news):
        vector.return_value.retrieve.reset_mock()
        retrieve(state)
        calls = vector.return_value.retrieve.call_args_list
        assert any(query in call.args for call in calls)
        assert all(call.kwargs.get("published_at_to") is None for call in calls)


def test_policy_content_skips_warehouse_even_when_it_mentions_yield(monkeypatch):
    state = _decompose(
        monkeypatch, "What does Kenya's agricultural policy say about maize yields?",
        geography=["Kenya"], entities=["maize"], job="fact",
    )
    bq = Mock()
    monkeypatch.setattr(graph, "node_bq_reason", bq)
    monkeypatch.setattr(
        graph, "node_parallel_retrieve", Mock(return_value={"vector_results": [{"content": "policy excerpt"}]}),
    )
    result = graph.node_retrieve_legs(state)

    assert state["route_candidate"] == "full_rag"
    assert state["task_mode"] == "research"
    bq.assert_not_called()
    assert result["vector_results"][0]["content"] == "policy excerpt"
    assert state["source_selection"]["bigquery"]["reason"] == "document_content_requested"


def test_named_action_plan_routes_to_document_requirements(monkeypatch):
    state = _decompose(
        monkeypatch,
        "What does Kenya’s National Climate Change Action Plan 2023–2027 say about agricultural adaptation?",
        geography=["Kenya"], job="fact",
    )
    assert not state["source_selection"]["bigquery"]["selected"]
    assert state["source_selection"]["qdrant"]["reason"] == "document_content_requested"
    assert state["turn_contract"]["documents"]


def test_livelihood_and_policy_question_starts_both_legs_in_parallel(monkeypatch):
    state = _decompose(
        monkeypatch,
        "Compare agricultural employment share in Kenya in 2018 versus 2023 and explain how policy contributed.",
        geography=["Kenya"], job="compare",
    )
    started = Barrier(2, timeout=3)
    real_bq_reason = graph.node_bq_reason

    def bq_reason(state):
        started.wait()
        return real_bq_reason(state)

    def docs(_state):
        started.wait()
        return {"vector_results": [{"content": "policy evidence"}]}

    monkeypatch.setattr(graph, "node_bq_reason", bq_reason)
    monkeypatch.setattr(graph, "node_bq_retrieve", lambda state: {"bq_results": [{"content": "employment series"}]})
    monkeypatch.setattr(graph, "node_parallel_retrieve", docs)
    result = graph.node_retrieve_legs(state)

    assert state["route_candidate"] == "full_rag"
    assert state["source_selection"]["bigquery"]["requirements"][0]["selected"]
    assert result["bq_sql_plan"]["selected_tables"] == ["fct_employment"]
    assert result["bq_results"] and result["vector_results"]


def test_missing_country_does_not_block_scoped_document_evidence(monkeypatch):
    state = _decompose(
        monkeypatch,
        "What were maize prices, and what did the Agricultural Modernisation Policy recommend?",
        entities=["maize"], job="fact",
        evidence_topics=["Agricultural Modernisation Policy recommendations"],
    )
    assert state["route_candidate"] == "full_rag"
    assert state["source_selection"]["qdrant"]["selected"]
    # A document title/tag alone is not a country filter for a price series.
    assert not state["source_selection"]["bigquery"]["selected"]
    assert state["source_selection"]["bigquery"]["requirements"][0]["reason"] == "geography_required"


def test_unscoped_price_request_asks_for_clarification(monkeypatch):
    state = _decompose(monkeypatch, "What were maize prices?", entities=["maize"], job="fact")
    assert state["route_candidate"] == "clarify"
    assert not state["source_selection"]["bigquery"]["selected"]
    assert not state["source_selection"]["qdrant"]["selected"]


def test_policy_first_request_with_observed_yield_comparison_uses_both_sources(monkeypatch):
    state = _decompose(
        monkeypatch,
        "What does Kenya's agricultural policy say about maize yields? Compare observed yields in 2018 versus 2023.",
        geography=["Kenya"], entities=["maize"], job="compare",
    )
    assert state["source_selection"]["bigquery"]["selected"]
    assert state["source_selection"]["qdrant"]["selected"]
