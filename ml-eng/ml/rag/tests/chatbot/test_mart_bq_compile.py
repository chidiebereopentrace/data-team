"""Tests for resolve_geo_filter_values and schema-driven intents."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.bq_sql_patterns import try_sql_pattern
from ml.rag.chatbot.bq_table_schema_yaml import (
    compile_intent_for_table,
    resolve_geo_filter_values,
)
from ml.rag.chatbot.fact_bq_plan import build_fact_bq_plan
from ml.rag.chatbot.retrieval_contract import build_retrieval_contract
from ml.rag.chatbot.bq_table_schema_yaml import list_mart_table_index

_KNOWN = {r["table_id"] for r in list_mart_table_index()}


@pytest.mark.parametrize(
    "table_id,label,expected",
    [
        ("fct_production", "Nigeria", "NGA"),
        ("agg_production_annual", "Nigeria", "Nigeria"),
        ("agg_production_annual", "KEN", "Kenya"),
        ("fct_production", "NGA", "NGA"),
    ],
)
def test_resolve_geo_filter_values(table_id: str, label: str, expected: str) -> None:
    assert resolve_geo_filter_values(table_id, [label]) == [expected]


def test_compile_intent_single_country_not_custom() -> None:
    intent = compile_intent_for_table(
        "fct_production",
        measure_id="production",
        query="What was maize production in Nigeria in 2022?",
        geo_labels=["Nigeria"],
        year_hint="2022",
        multi_country=False,
        time_start="2022-01-01",
        time_end="2022-12-31",
    )
    assert intent["pattern"] != "custom"
    assert intent["pattern"] == "rank_by_sum"
    assert "country_iso3" in intent["grain"]


def test_retrieval_contract_single_country_production_intent(monkeypatch) -> None:
    monkeypatch.setenv("RAG_SQL_COMPILER", "0")
    q = "What was maize production in Nigeria in 2022?"
    dec = {
        "geography": ["Nigeria"],
        "entities": ["maize", "production"],
        "time_start": "2022-01-01",
        "time_end": "2022-12-31",
    }
    contract = build_retrieval_contract(q, decomposition=dec, known_tables=_KNOWN)
    assert contract.bq_intents
    assert contract.bq_intents[0]["pattern"] != "custom"


def test_try_sql_patterns_nigeria_maize_2022() -> None:
    intent = compile_intent_for_table(
        "fct_production",
        measure_id="production",
        query="What was maize production in Nigeria in 2022?",
        geo_labels=["Nigeria"],
        year_hint="2022",
        multi_country=False,
        time_start="2022-01-01",
        time_end="2022-12-31",
    )
    hit = try_sql_pattern(
        intent,
        project_id="proj",
        dataset="mart_dev",
        query="What was maize production in Nigeria in 2022?",
        time_start="2022-01-01",
        time_end="2022-12-31",
        geo_country="Nigeria",
        selected_tables={"fct_production"},
    )
    assert hit is not None
    assert "stg_" not in hit["sql"]
    assert "NGA" in hit["sql"]
    assert "2022" in hit["sql"]


def test_fact_bq_plan_single_country_pattern() -> None:
    plan = build_fact_bq_plan(
        "Nigeria maize production 2022",
        decomposition={
            "geography": ["Nigeria"],
            "entities": ["maize"],
            "time_start": "2022-01-01",
            "time_end": "2022-12-31",
        },
        known_tables=_KNOWN,
        task_mode="fact_lookup",
    )
    assert plan is not None
    assert plan["query_intents"][0]["pattern"] != "custom"
