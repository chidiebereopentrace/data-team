"""Smoke tests for mart_dev BQ catalog and indicator taxonomy."""
from __future__ import annotations

from ml.rag.chatbot.agri_measure_ontology import MEASURES, resolve_measure
from ml.rag.chatbot.bq_table_schema_yaml import (
    list_mart_table_index,
    pack_mart_table_hints,
)
from ml.rag.chatbot.mart_indicator_classes import (
    all_class_codes,
    class_for_query,
    do_not_mix_tables,
    facts_for_class,
)
from ml.rag.chatbot.retrieval_contract import build_retrieval_contract, choose_agg_vs_fact


def test_indicator_taxonomy_has_fourteen_classes():
    codes = all_class_codes()
    assert len(codes) >= 14
    assert "PROD" in codes
    assert "FS" in codes


def test_class_for_query_production():
    hits = class_for_query("What is maize production in Nigeria?")
    assert "PROD" in hits


def test_facts_for_class_prod():
    facts = facts_for_class("PROD")
    assert "fct_production" in facts
    assert "fct_yield" in facts


def test_do_not_mix_production_yield():
    reason = do_not_mix_tables("fct_production", "fct_yield")
    assert reason


def test_list_mart_table_index_non_empty():
    rows = list_mart_table_index()
    assert rows
    ids = {r["table_id"] for r in rows}
    assert "fct_production" in ids


def test_pack_mart_hints_includes_filtering_guidance():
    hints, _ = pack_mart_table_hints(["fct_production"])
    assert hints
    blob = "\n".join(hints)
    assert "filtering_guidance" in blob.lower() or "Filtering guidance" in blob
    assert "production_grain" in blob.lower()


def test_ontology_uses_mart_tables():
    hit = resolve_measure("maize production in Kenya 2022", {})
    assert hit is not None
    spec = MEASURES[hit.measure.id]
    assert any(t.startswith("fct_") or t.startswith("agg_") for t in spec.candidate_tables)


def test_retrieval_contract_mart_tables(monkeypatch):
    monkeypatch.setenv("RAG_SQL_COMPILER", "0")
    contract = build_retrieval_contract(
        "maize production in Nigeria 2022",
        decomposition={"entities": ["maize"], "geography": ["NGA"], "time_start": "2022-01-01"},
        known_tables={r["table_id"] for r in list_mart_table_index()},
    )
    assert contract.bq_tables
    assert contract.bq_tables[0].startswith(("fct_", "agg_"))


def test_choose_agg_vs_fact_national_production():
    routed = choose_agg_vs_fact(
        "fct_production",
        query="national maize production by country",
        multi_country=True,
        year_hint="2022",
        iso_count=16,
    )
    assert routed == "agg_production_country_year"
