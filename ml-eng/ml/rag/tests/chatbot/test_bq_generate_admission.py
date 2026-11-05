"""Tests for BQ execute-flag sourcing on bind-first planned path."""
from __future__ import annotations

from ml.rag.chatbot.bq_execute_state import (
    bq_execute_flags,
    collect_pre_queries,
    ensure_planned_attempt_debug,
    plan_indicates_warehouse_attempt,
)
from ml.rag.chatbot.generator import generate, is_mergeable_bq_evidence
from ml.rag.chatbot.generation_plan import build_generation_plan
from unittest import mock
import os


def test_collect_pre_queries_prefers_post_retrieve_over_empty_plan() -> None:
    plan = {"bq_sql_queries": [], "bind_contracts": {"agg_x": {}}, "plan_source": "class_engine"}
    debug = [{"sql": "SELECT 1", "job_id": "j1", "status": "ok"}]
    qs = collect_pre_queries(plan, state_queries=["SELECT 1"], bq_sql_debug=debug)
    assert qs == ["SELECT 1"]
    assert plan_indicates_warehouse_attempt(plan)


def test_bind_path_empty_job_sets_structured_bq_empty() -> None:
    plan = {
        "bq_sql_queries": [],
        "bind_contracts": {"agg_production_country_year": {"table_id": "agg_production_country_year"}},
        "plan_source": "class_engine",
        "retrieve_mode": "planned",
    }
    debug = [
        {
            "sql": "SELECT production_qty FROM t WHERE country_iso3 = 'KEN'",
            "job_id": "job-1",
            "status": "ok",
        }
    ]
    pre = collect_pre_queries(plan, state_queries=[], bq_sql_debug=debug)
    flags = bq_execute_flags(
        debug,
        pre_queries=pre,
        usable_bq=False,
        warehouse_attempted=True,
    )
    assert flags["structured_bq_empty"] is True
    assert flags["structured_bq_unavailable"] is False


def test_planned_intent_without_a_job_records_compile_error() -> None:
    plan = {
        "plan_source": "class_engine",
        "query_intents": [{"goal": "maize price", "tables": ["agg_prices_country_month"]}],
        "bq_sql_queries": ["SELECT price_avg FROM agg_prices_country_month"],
    }
    debug = ensure_planned_attempt_debug(plan, [], usable_bq=False)
    assert debug[0]["status"] == "compile_error"
    assert "agg_prices_country_month" in debug[0]["sql"]
    assert "no BigQuery job" in debug[0]["prep_error"]
    flags = bq_execute_flags(
        debug,
        pre_queries=collect_pre_queries(plan, bq_sql_debug=debug),
        usable_bq=False,
        warehouse_attempted=True,
    )
    assert flags["structured_bq_never_executed"] is True
    assert flags["structured_bq_unavailable"] is False


def test_empty_job_is_not_replaced_by_compile_error() -> None:
    plan = {"plan_source": "class_engine", "bind_contracts": {"agg_prices_country_month": {}}}
    existing = [{"sql": "SELECT 1", "status": "empty_result", "job_id": "job-9"}]
    assert ensure_planned_attempt_debug(plan, existing, usable_bq=False) == existing


def test_no_plan_no_debug_is_unavailable_not_empty() -> None:
    flags = bq_execute_flags([], pre_queries=[], usable_bq=False, warehouse_attempted=False)
    assert flags["structured_bq_unavailable"] is True
    assert flags["structured_bq_empty"] is False


def test_mergeable_bq_without_value_semantics() -> None:
    item = {
        "content": "[Structured data] production for Kenya 2016 was 100000",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {"country_name": "Kenya", "year": 2016},
    }
    assert is_mergeable_bq_evidence(item)


def test_generate_opens_llm_for_mergeable_bq_without_semantics() -> None:
    decomposition = {
        "geography": ["Kenya"],
        "entities": ["rice"],
        "primary_measures": ["production"],
    }
    item = {
        "content": "[Structured data] production for Kenya 2016",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {"country_name": "Kenya", "year": 2016, "product_name": "Rice"},
    }
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        mock_llm.return_value = "Kenya produced rice in 2016."
        plan = build_generation_plan(
            "what is the production of rice in kenya in 2016",
            task_mode="fact_lookup",
            reranked_context=[item],
            decomposition=decomposition,
        )
        result = generate(
            "what is the production of rice in kenya in 2016",
            [item],
            task_mode="fact_lookup",
            decomposition=decomposition,
            generation_plan=plan,
            usable_bq=True,
        )
    mock_llm.assert_called_once()
    assert "rice" in result.answer.lower()


def test_generate_answers_from_a_report_when_the_warehouse_misses() -> None:
    decomposition = {
        "geography": ["Kenya"],
        "entities": ["rice"],
        "primary_measures": ["production"],
    }
    item = {
        "content": "A ministry report discusses rice production in Kenya.",
        "source": "public_report",
        "_context_kind": "public_report",
        "metadata": {"title": "Kenya rice note"},
    }
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        mock_llm.return_value = "Reports discuss rice production in Kenya, without a warehouse figure."
        result = generate(
            "what is the production of rice in kenya in 2016",
            [item],
            task_mode="fact_lookup",
            decomposition=decomposition,
            structured_bq_validation_failed=True,
            warehouse_was_attempted=True,
        )
    mock_llm.assert_called_once()
    prompt = mock_llm.call_args.args[0][0]["content"]
    assert "Do not invent totals, yields, prices, or a time series." in prompt
    assert "rice" in result.answer.lower()


def test_generate_typed_gap_when_warehouse_empty_flag() -> None:
    decomposition = {
        "geography": ["Kenya"],
        "entities": ["rice"],
        "primary_measures": ["production"],
    }
    plan = build_generation_plan(
        "what is the production of rice in kenya in 2016",
        task_mode="fact_lookup",
        reranked_context=[],
        decomposition=decomposition,
    )
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_ALLOW_UNGROUNDED", None)
            result = generate(
                "what is the production of rice in kenya in 2016",
                [],
                task_mode="fact_lookup",
                decomposition=decomposition,
                generation_plan=plan,
                structured_bq_empty=True,
                warehouse_was_attempted=True,
            )
    mock_llm.assert_not_called()
    assert "no rows" in result.answer.lower() or "warehouse" in result.answer.lower()


def test_generate_typed_gap_when_warehouse_attempted_without_job() -> None:
    decomposition = {
        "geography": ["Kenya"],
        "entities": ["rice"],
        "primary_measures": ["production"],
    }
    plan = build_generation_plan(
        "what is the production of rice in kenya in 2016",
        task_mode="fact_lookup",
        reranked_context=[],
        decomposition=decomposition,
    )
    with mock.patch("ml.rag.chatbot.generator._call_llama") as mock_llm:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_ALLOW_UNGROUNDED", None)
            result = generate(
                "what is the production of rice in kenya in 2016",
                [],
                task_mode="fact_lookup",
                decomposition=decomposition,
                generation_plan=plan,
                warehouse_was_attempted=True,
            )
    mock_llm.assert_not_called()
    low = result.answer.lower()
    assert "no production figures" in low
    assert "kenya" in low
    assert "warehouse" in low
