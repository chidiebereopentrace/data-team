"""Tests for geo ISO3 resolution and West Africa panel SQL."""
from __future__ import annotations

from ml.rag.chatbot.class_engine_runner import engine_results_to_bq_plan, run_class_engines
from ml.rag.chatbot.class_engines.fvc import FvcEngine
from ml.rag.chatbot.class_engines.prod import ProdEngine
from ml.rag.chatbot.class_supervisor import compile_supervisor_plan
from ml.rag.chatbot.geo_iso3 import infer_country_iso3_from_query, resolve_geography_iso3
from ml.rag.chatbot.geo_regions import expand_regions_in_decomposition
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.retrievers.web_retriever import needs_web_fallback, route_after_rerank


def test_resolve_geography_iso3_west_africa_sixteen() -> None:
    q = "West Africa agricultural activities by country 2015 to date"
    dec = expand_regions_in_decomposition({"geography": []}, q)
    iso = resolve_geography_iso3(q, geography=dec.get("geography"), expanded_regions=dec.get("expanded_regions"))
    assert len(iso) == 16
    assert "GHA" in iso
    assert "NGA" in iso


def test_infer_country_not_com_from_compare() -> None:
    assert infer_country_iso3_from_query("compare production across countries") is None
    assert infer_country_iso3_from_query("agricultural commodities report") is None


def test_prod_engine_west_africa_panel_planned() -> None:
    q = "West Africa agricultural activities by country 2015 to date"
    dec = expand_regions_in_decomposition(
        {"geography": [], "time_start": "2015-01-01", "time_end": "2024-12-31"},
        q,
    )
    result = ProdEngine().run_plan(q, facets=dec, card=None)
    assert result.status == "planned", result.caveats
    assert result.sql is None
    assert result.table_id == "agg_production_country_year"
    assert result.bind_contract is not None
    assert len(result.value_hits.get("country_iso3") or []) == 16


def test_fvc_engine_agri_activities_planned() -> None:
    q = "West Africa agricultural activities by country 2015 to date"
    dec = expand_regions_in_decomposition({"geography": [], "time_start": "2015-01-01"}, q)
    result = FvcEngine().run_plan(q, facets=dec, card=None)
    assert result.status == "planned", result.caveats
    assert result.bind_contract is not None
    assert result.table_id in ("fct_food_balance", "fct_trade")


def test_west_africa_multi_table_bq_plan() -> None:
    q = "West Africa agricultural activities by country 2015 to date"
    dec = expand_regions_in_decomposition({"geography": [], "time_start": "2015-01-01"}, q)
    bundles = match_intent_bundles(q, dec)
    sp = compile_supervisor_plan(q, decomposition=dec, matched_bundles=bundles)
    assert sp.classes == ("PROD",)
    assert "FVC" in sp.secondary
    assert "PRC" not in sp.secondary
    results = run_class_engines(q, supervisor_plan=sp, facets=dec)
    plan = engine_results_to_bq_plan(results)
    assert plan.get("bind_contracts")
    assert plan.get("query_intents")
    assert not plan.get("bq_sql_queries")
    debug_tables = {row["table_id"] for row in plan.get("bq_sql_debug") or [] if row.get("table_id")}
    assert "agg_production_country_year" in debug_tables
    assert len(debug_tables) >= 2


def test_needs_web_fallback_skips_analytical() -> None:
    assert not needs_web_fallback([], task_mode="analytical", bq_warehouse_attempted=False)


def test_route_after_rerank_skips_web_on_bq_plan() -> None:
    state = {
        "reranked_context": [],
        "bq_results": [],
        "task_mode": "chat",
        "bq_sql_plan": {
            "engine_results": [{"status": "timeout", "sql": "SELECT 1"}],
            "bq_sql_queries": ["SELECT 1"],
        },
        "bq_sql_queries": ["SELECT 1"],
    }
    assert route_after_rerank(state) == "generate"
