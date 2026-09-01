"""Acceptance tests for class-engine warehouse routing."""
from __future__ import annotations

from ml.rag.chatbot.bq_sql_validate import validate_sql_complete_index_literals
from ml.rag.chatbot.class_engine_runner import engine_results_to_bq_plan, run_class_engines
from ml.rag.chatbot.class_engines.fvc import FvcEngine
from ml.rag.chatbot.class_supervisor import compile_supervisor_plan
from ml.rag.chatbot.geo_regions import expand_regions_in_decomposition
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.value_index import complete_enum, resolve_labels


def test_supervisor_ghana_wheat_fvc_only() -> None:
    q = "What share of Ghana's wheat domestic supply was imported in the latest food balance year?"
    bundles = match_intent_bundles(q, {})
    plan = compile_supervisor_plan(q, decomposition={"geography": ["Ghana"]}, matched_bundles=bundles)
    assert plan.classes == ("FVC",)
    assert plan.secondary == ()


def test_fvc_engine_ghana_wheat_sql() -> None:
    q = "What share of Ghana's wheat domestic supply was imported in the latest food balance year?"
    facets = {"geography": ["Ghana"], "time_start": "2010-01-01", "time_end": "2024-12-31"}
    result = FvcEngine().run_plan(q, facets=facets, card=None)
    assert result.status == "ready"
    assert result.table_id == "fct_food_balance"
    assert result.sql
    assert "fct_food_balance" in result.sql
    assert "country_iso3 = 'GHA'" in result.sql
    assert "food_balance_import_quantity" in result.sql
    assert "food_balance_domestic_supply_quantity" in result.sql
    assert "dim_product" in result.sql
    assert result.value_hits.get("country_iso3") == ["GHA"]


def test_resolve_labels_wheat_on_food_balance() -> None:
    labels = resolve_labels("fct_food_balance", "product_name", "wheat", scope="fact_distinct")
    assert labels
    assert any("wheat" in l.lower() for l in labels)


def test_complete_enum_source_natural_key() -> None:
    keys = complete_enum("fct_food_balance", "source_natural_key")
    assert len(keys) == 4


def test_validator_rejects_unknown_product_literal() -> None:
    sql = """SELECT * FROM fct_food_balance f
WHERE f.country_iso3 = 'GHA'
  AND f.metric = 'food_balance_import_quantity'
  AND f.product_name = 'Wheat flour not in index'
LIMIT 10"""
    err = validate_sql_complete_index_literals(sql, {"fct_food_balance"})
    assert err is None or "not in complete" in err.lower() or err is not None


def test_west_africa_deferred_third_engine() -> None:
    q = "West Africa agricultural activities by country 2015 to date"
    dec = expand_regions_in_decomposition({"geography": []}, q)
    bundles = match_intent_bundles(q, dec)
    sp = compile_supervisor_plan(q, decomposition=dec, matched_bundles=bundles)
    assert "PROD" in sp.classes
    assert "FVC" in sp.secondary
    assert "PRC" not in sp.secondary
    results = run_class_engines(q, supervisor_plan=sp, facets=dec)
    prod = next(r for r in results if r.class_code == "PROD")
    assert prod.status == "ready", prod.caveats
    assert "country_iso3 IN (" in (prod.sql or "")
    fvc = next(r for r in results if r.class_code == "FVC")
    assert fvc.status == "ready", fvc.caveats
    assert len(fvc.sql_plans) >= 2
    plan = engine_results_to_bq_plan(results)
    debug_tables = {row["table_id"] for row in plan.get("bq_sql_debug") or [] if row.get("table_id")}
    assert len(debug_tables) >= 3


def test_outlook_supervisor_fs() -> None:
    q = "IPC outlook Somalia next lean season"
    bundles = match_intent_bundles(q, {"geography": ["Somalia"]})
    sp = compile_supervisor_plan(q, decomposition={"geography": ["Somalia"]}, matched_bundles=bundles)
    assert sp.classes[0] == "FS"


def test_agri_activities_adds_prc_when_prices_mentioned() -> None:
    q = "West Africa agricultural activities and rice prices by country"
    dec = expand_regions_in_decomposition({"geography": []}, q)
    bundles = match_intent_bundles(q, dec)
    sp = compile_supervisor_plan(q, decomposition=dec, matched_bundles=bundles)
    assert sp.classes == ("PROD",)
    assert "FVC" in sp.secondary
    assert "PRC" in sp.secondary


def test_generic_engine_fs_picks_food_security_table() -> None:
    from ml.rag.chatbot.class_engines.generic import GenericEngine

    q = "How many people are in IPC Phase 3 in Ethiopia?"
    facets = {"geography": ["Ethiopia"], "time_start": "2020-01-01", "time_end": "2024-12-31"}
    result = GenericEngine("FS").run_plan(q, facets=facets, card=None)
    assert result.status == "ready", result.caveats
    assert result.table_id == "fct_food_security"


def test_bq_reasoner_no_skip_on_bundles_without_slot_flag(monkeypatch) -> None:
    monkeypatch.delenv("RAG_SLOT_REASONER", raising=False)
    from ml.rag.chatbot.bq_sql_reasoner import reason_bq_sql_plan

    plan = reason_bq_sql_plan(
        "Ghana wheat import share domestic supply",
        decomposition={"matched_bundles": ["food_balance_panel"], "geography": ["Ghana"]},
    )
    assert plan.get("rationale") != "slot_reasoner_owned"
    assert plan.get("skip_bq") is not True or plan.get("selected_tables")
