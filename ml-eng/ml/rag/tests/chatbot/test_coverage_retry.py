"""Coverage retry decisions and evidence propagation through the compiled graph."""
from __future__ import annotations

from typing import Any

import pytest

from ml.rag.chatbot.coverage_retry import should_coverage_retry, should_retry_bq, slots_missing_from_evidence

_SKIPPED_WAREHOUSE_PLAN = {"skip_bq": True, "reason": "fixture"}


def test_slots_missing_when_empty() -> None:
    state = {
        "decomposition": {"geography": ["Kenya"], "entities": ["maize"]},
        "bq_results": [],
        "reranked_context": [],
    }
    assert slots_missing_from_evidence(state) is True


def test_slots_present_in_bq() -> None:
    state = {
        "decomposition": {"geography": ["Kenya"], "entities": ["maize"]},
        "bq_results": [{"content": "Kenya maize production 2020 was 4Mt"}],
        "coverage_retry": 0,
        "route_candidate": "full_rag",
    }
    assert slots_missing_from_evidence(state) is False
    assert should_coverage_retry(state) is False


def test_retry_only_once() -> None:
    state = {
        "decomposition": {"geography": ["Kenya"]},
        "bq_results": [],
        "reranked_context": [],
        "coverage_retry": 1,
        "route_candidate": "full_rag",
    }
    assert should_coverage_retry(state) is False


def test_discarded_candidates_and_failed_jobs_do_not_cover_facets() -> None:
    state = {
        "decomposition": {"geography": ["Senegal"], "entities": ["millet"]},
        "bq_results": [{"content": "Senegal millet failed", "metadata": {"status": "validation_failed"}}],
        "merged_context": [{"content": "Senegal millet report rejected by reranking"}],
        "reranked_context": [{"content": "Senegal rice prices"}],
    }
    assert slots_missing_from_evidence(state)


def test_identical_warehouse_retry_cannot_repair_invalid_sql() -> None:
    assert not should_retry_bq({
        "bq_sql_plan": {"selected_tables": ["fct_prices"]},
        "bq_sql_debug": [{"status": "validation_failed"}],
    })


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
        warehouse_retry_rows: list[dict[str, Any]] | None = None,
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
            rows = warehouse_first_rows if len(warehouse_calls) == 1 else warehouse_retry_rows
            return {"bq_results": rows or []}

        monkeypatch.setattr(graph, "node_decompose", lambda state: {
            "decomposition": decomposition or {"geography": ["Kenya"], "entities": ["rice"]},
            "route_candidate": "full_rag",
            "answer_lang": "en",
        })
        monkeypatch.setattr(graph, "node_parallel_retrieve", retrieve)
        plan = _SKIPPED_WAREHOUSE_PLAN
        if warehouse_first_rows or warehouse_retry_rows is not None:
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
        assert len(warehouse_calls) == (1 if warehouse_retry_rows is None else 2)
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


def test_successful_retry_reaches_generation(replay_retrieval) -> None:
    result, calls, generated = replay_retrieval(
        ["Water availability influences farm yields."],
        ["Kenya rice farms rely on irrigation."],
    )
    assert len(calls) == 2
    assert result["coverage_retry"] == 1
    assert result["reranked_context"][0]["content"] == "[News] Kenya rice farms rely on irrigation."
    assert generated == [result["reranked_context"][0]["content"]]


def test_empty_retry_keeps_first_pass_evidence(replay_retrieval) -> None:
    result, calls, generated = replay_retrieval(["Water availability influences farm yields."], [])
    assert len(calls) == 2
    assert result["coverage_retry"] == 1
    assert result["merged_context"][0]["content"] == "[News] Water availability influences farm yields."
    assert generated == [result["merged_context"][0]["content"]]


def test_old_warehouse_rows_do_not_make_empty_vector_retry_successful(replay_retrieval) -> None:
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
    assert len(calls) == 2
    assert result["coverage_retry"] == 1
    assert {item["source"] for item in result["reranked_context"]} == {"news", "bigquery"}
    assert set(generated) == {item["content"] for item in result["reranked_context"]}


@pytest.mark.parametrize("source", ["news", "bigquery"])
def test_retry_uses_evidence_when_only_one_leg_returns_it(replay_retrieval, source: str) -> None:
    fact = "Kenya rice production was 150,000 tonnes in 2024."
    warehouse_rows = [{
        "content": fact,
        "source": "bigquery",
        "metadata": {"value_semantics": {"measure_value": 150_000}},
    }] if source == "bigquery" else []
    result, calls, generated = replay_retrieval(
        [], [fact] if source == "news" else [], warehouse_retry_rows=warehouse_rows,
    )
    assert len(calls) == 2
    assert result["coverage_retry"] == 1
    evidence, = result["reranked_context"]
    assert evidence["source"] == source
    assert generated == [evidence["content"]]


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
