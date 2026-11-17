"""Coverage retry decisions and evidence propagation through the compiled graph."""
from __future__ import annotations

from typing import Any

import pytest

_SKIPPED_WAREHOUSE_PLAN = {"skip_bq": True, "reason": "fixture"}


@pytest.fixture
def replay_retrieval(monkeypatch: pytest.MonkeyPatch):
    from ml.rag.chatbot import graph
    from ml.rag.chatbot.generator import GenerationResult

    monkeypatch.setenv("RAG_WEB_FALLBACK_ENABLED", "0")
    monkeypatch.setattr(graph, "metacog_enabled", lambda: False)

    def replay(
        first: list[str],
        retry: list[str],
        *,
        decomposition: dict[str, Any] | None = None,
        initial: graph.RAGGraphState | None = None,
        warehouse_first_rows: list[dict[str, Any]] | None = None,
    ):
        vector_calls = []
        warehouse_calls = []
        generated_from = []

        def retrieve(state):
            vector_calls.append(state)
            contents = first if len(vector_calls) == 1 else retry
            chunks = [
                {
                    "content": content,
                    "metadata": {"url": f"https://example.com/source-{i}"},
                    "score": 0.9,
                }
                for i, content in enumerate(contents)
            ]
            return {"vector_news_results": chunks, "vector_results": chunks}

        def generate(query, context, **kwargs):
            generated_from.extend(item["content"] for item in context)
            return GenerationResult(answer="Fixture answer", citations=[])

        def retrieve_warehouse(state):
            warehouse_calls.append(state)
            return {"bq_results": warehouse_first_rows or []}

        monkeypatch.setattr(graph, "node_decompose", lambda state: {
            "decomposition": decomposition or {"geography": ["Kenya"], "entities": ["rice"]},
            "route_candidate": "full_rag",
            "answer_lang": "en",
        })
        monkeypatch.setattr(graph, "node_parallel_retrieve", retrieve)
        plan = _SKIPPED_WAREHOUSE_PLAN
        if warehouse_first_rows:
            plan = {"selected_tables": ["prod_country_year"]}
        monkeypatch.setattr(graph, "node_bq_reason", lambda state: {"bq_sql_plan": plan})
        monkeypatch.setattr(graph, "node_bq_retrieve", retrieve_warehouse)
        monkeypatch.setattr(graph, "rerank", lambda query, documents, **kwargs: documents)
        monkeypatch.setattr(graph, "generate", generate)

        result = graph.build_graph().invoke({
            "query": "What affects rice farming in Kenya?",
            "coverage_retry": 0,
            "profile_scope_retry": 0,
            "rerank_top_k": 1,
            **(initial or {}),
        })
        assert len(warehouse_calls) == 1
        return result, vector_calls, generated_from

    return replay


def test_covered_turn_keeps_merged_and_ranked_evidence(replay_retrieval) -> None:
    result, calls, generated = replay_retrieval(
        ["Kenya rice farms rely on irrigation.", "Kenya rice growers manage water availability."],
        [],
    )
    assert len(calls) == 1
    assert result["coverage_retry"] == 0
    assert len(result["merged_context"]) == 2
    assert len(result["reranked_context"]) == 1
    assert generated == [result["reranked_context"][0]["content"]]
    assert result["bq_sql_plan"] == _SKIPPED_WAREHOUSE_PLAN


def test_unchanged_scope_does_not_trigger_an_identical_search(replay_retrieval) -> None:
    result, calls, generated = replay_retrieval(
        ["Water availability influences farm yields."],
        ["Kenya rice farms rely on irrigation."],
    )
    assert len(calls) == 1
    assert result["coverage_retry"] == 0
    assert result["reranked_context"][0]["content"] == "[News] Water availability influences farm yields."
    assert generated == [result["reranked_context"][0]["content"]]


def test_keeps_completed_warehouse_and_document_evidence(replay_retrieval) -> None:
    result, calls, generated = replay_retrieval(
        ["Water availability influences farm yields."],
        [],
        initial={"rerank_top_k": 2},
        warehouse_first_rows=[{
            "content": "Reported water availability was 30 percent.",
            "source": "bigquery",
            "metadata": {"value_semantics": {"measure_value": 30}},
        }],
    )
    assert len(calls) == 1
    assert result["coverage_retry"] == 0
    assert {item["source"] for item in result["reranked_context"]} == {"news", "bigquery"}
    assert set(generated) == {item["content"] for item in result["reranked_context"]}


@pytest.mark.parametrize("geography_source,time_source", [("profile", "query"), ("query", "ui")])
def test_widened_retry_keeps_explicit_scope(replay_retrieval, geography_source, time_source) -> None:
    result, calls, generated = replay_retrieval(
        [],
        ["Kenya rice farms rely on irrigation."],
        decomposition={
            "geography": ["Kenya"], "entities": ["rice"], "geography_source": geography_source,
            "time_source": time_source, "time_start": "2024-01-01", "time_end": "2024-12-31",
        },
        initial={
            "query": (
                "What affected rice farming in 2024?"
                if geography_source == "profile"
                else "What affects rice farming in Kenya?"
            ),
            "geo_override": "Kenya" if geography_source == "profile" else "",
            "time_start_override": "2024-01-01" if time_source == "ui" else "",
            "time_end_override": "2024-12-31" if time_source == "ui" else "",
            "turn_contract": {"time_spec": {"start": "2024-01-01", "end": "2024-12-31", "hard_filter": True}},
        },
    )
    assert len(calls) == 2
    assert result["profile_scope_retry"] == 1
    assert result["coverage_retry"] == 0
    assert result["decomposition"] == calls[1]["decomposition"]
    assert result["decomposition"]["geography"] == ([] if geography_source == "profile" else ["Kenya"])
    assert result["geo_override"] == ""
    assert result["time_start_override"] == result["time_end_override"] == ""
    assert result["decomposition"]["time_start"] == ("" if time_source == "ui" else "2024-01-01")
    assert result["decomposition"]["time_end"] == ("" if time_source == "ui" else "2024-12-31")
    assert result["turn_contract"]["time_spec"]["hard_filter"] == (time_source != "ui")
    assert generated == ["[News] Kenya rice farms rely on irrigation."]
