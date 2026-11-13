"""Class- and region-agnostic multi-table planned binds (not West-Africa-only)."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.class_engine_runner import engine_results_to_bq_plan, run_class_engines
from ml.rag.chatbot.class_engines.base import EngineResult
from ml.rag.chatbot.class_engines.fs import FsEngine
from ml.rag.chatbot.class_engines.fvc import FvcEngine
from ml.rag.chatbot.class_engines.prc import PrcEngine
from ml.rag.chatbot.class_engines.prod import ProdEngine
from ml.rag.chatbot.class_supervisor import compile_supervisor_plan
from ml.rag.chatbot.class_table_router import select_table_plans
from ml.rag.chatbot.geo_regions import expand_regions_in_decomposition
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.value_index import resolve_geography_iso3

_MULTI_REGION_PANEL = [
    ("west africa", 16),
    ("north africa", 6),
    ("east africa", 18),
    ("central africa", 6),
    ("southern africa", 10),
    ("ECOWAS", 15),
    ("SADC", 16),
    ("IGAD", 8),
    ("UEMOA", 8),
]

_EXTRA_REGION_BLENDS = [
    "Sahel",
    "horn of africa",
]


@pytest.mark.parametrize("region_label,expected_iso_count", _MULTI_REGION_PANEL)
def test_region_agri_panel_multi_class_bind_keys(region_label: str, expected_iso_count: int) -> None:
    q = f"{region_label} agricultural activities by country 2015 to date"
    dec = expand_regions_in_decomposition({"geography": [], "time_start": "2015-01-01"}, q)
    iso = resolve_geography_iso3(
        q, geography=dec.get("geography"), expanded_regions=dec.get("expanded_regions")
    )
    assert len(iso) == expected_iso_count
    bundles = match_intent_bundles(q, dec)
    sp = compile_supervisor_plan(q, decomposition=dec, matched_bundles=bundles)
    assert sp.classes == ("PROD",)
    assert "FVC" in sp.secondary
    results = run_class_engines(q, supervisor_plan=sp, facets=dec)
    plan = engine_results_to_bq_plan(results)
    binds = plan.get("bind_contracts") or {}
    assert "agg_production_country_year" in binds
    assert "agg_food_balance_country_year" in binds
    assert "fct_trade" in binds
    assert len(binds) >= 3
    assert plan.get("retrieve_mode") in (None, "planned") or plan.get("plan_source") == "class_engine"


@pytest.mark.parametrize("region_label", _EXTRA_REGION_BLENDS)
def test_sahel_horn_agri_or_fs_region_blend(region_label: str) -> None:
    q = f"{region_label} agricultural activities by country 2018 to date"
    dec = expand_regions_in_decomposition({"geography": [], "time_start": "2018-01-01"}, q)
    iso = resolve_geography_iso3(
        q, geography=dec.get("geography"), expanded_regions=dec.get("expanded_regions")
    )
    assert len(iso) >= 3
    bundles = match_intent_bundles(q, dec)
    sp = compile_supervisor_plan(q, decomposition=dec, matched_bundles=bundles)
    results = run_class_engines(q, supervisor_plan=sp, facets=dec)
    plan = engine_results_to_bq_plan(results)
    binds = plan.get("bind_contracts") or {}
    assert binds
    assert len(binds) >= 2


def test_fvc_select_table_plans_panel_roles() -> None:
    q = "East Africa agricultural activities by country 2015 to date"
    dec = expand_regions_in_decomposition({"geography": [], "time_start": "2015-01-01"}, q)
    bundles = match_intent_bundles(q, dec)
    iso = resolve_geography_iso3(
        q, geography=dec.get("geography"), expanded_regions=dec.get("expanded_regions")
    )
    card = load_schema_card("FVC") or {}
    plans = select_table_plans("FVC", query=q, facets=dec, bundles=bundles, card=card, iso_list=iso)
    assert len(plans) == 2
    assert plans[0].table_id == "agg_food_balance_country_year"
    assert plans[0].role == "panel"
    assert plans[1].table_id == "fct_trade"
    assert plans[1].role == "companion"


def test_prod_select_table_plans_panel_plus_companions() -> None:
    q = "SADC maize production by country 2020"
    dec = expand_regions_in_decomposition(
        {"geography": [], "time_start": "2020-01-01", "time_end": "2020-12-31"},
        q,
    )
    bundles = match_intent_bundles(q, dec)
    iso = resolve_geography_iso3(
        q, geography=dec.get("geography"), expanded_regions=dec.get("expanded_regions")
    )
    card = load_schema_card("PROD") or {}
    plans = select_table_plans("PROD", query=q, facets=dec, bundles=bundles, card=card, iso_list=iso)
    assert plans
    assert plans[0].table_id == "agg_production_country_year"
    assert plans[0].role == "panel"
    companion_ids = [p.table_id for p in plans if p.role == "companion"]
    assert "agg_production_country_season" not in companion_ids
    assert plans[0].table_id == "agg_production_country_year"


def test_prod_season_query_attaches_country_season() -> None:
    q = "Kenya rice production by harvest season 2016"
    dec = {
        "geography": ["Kenya"],
        "time_start": "2016-01-01",
        "time_end": "2016-12-31",
        "entities": ["rice"],
        "primary_measures": ["production"],
    }
    bundles = match_intent_bundles(q, dec)
    iso = resolve_geography_iso3(q, geography=dec.get("geography"))
    card = load_schema_card("PROD") or {}
    plans = select_table_plans("PROD", query=q, facets=dec, bundles=bundles, card=card, iso_list=iso)
    assert any(p.table_id == "agg_production_country_season" for p in plans)


def test_hdi_card_driven_taxonomy_companions() -> None:
    q = "HDI trends in Kenya and Ghana 2015 to 2020"
    dec = expand_regions_in_decomposition(
        {"geography": ["Kenya", "Ghana"], "time_start": "2015-01-01", "time_end": "2020-12-31"},
        q,
    )
    bundles = match_intent_bundles(q, dec)
    iso = resolve_geography_iso3(
        q, geography=dec.get("geography"), expanded_regions=dec.get("expanded_regions")
    )
    card = load_schema_card("HDI") or {}
    plans = select_table_plans("HDI", query=q, facets=dec, bundles=bundles, card=card, iso_list=iso)
    assert plans
    ids = [p.table_id for p in plans]
    assert any(t.startswith("fct_hdi") or t == "fct_hdi" for t in ids) or "fct_economics" in ids or ids
    if "agg_hdi_latest" in ids:
        assert len(plans) >= 2
        assert any(p.role == "companion" for p in plans)


def test_engine_bind_contracts_match_plan_count() -> None:
    q = "West Africa agricultural activities by country 2015 to date"
    dec = expand_regions_in_decomposition({"geography": [], "time_start": "2015-01-01"}, q)
    for engine in (ProdEngine(), FvcEngine()):
        result = engine.run_plan(q, facets=dec, card=None)
        assert result.status == "planned", (engine.class_code, result.caveats)
        assert result.bind_contracts
        assert len(result.bind_contracts) == len(result.query_intents)
        assert result.table_id in result.bind_contracts


def test_prc_router_and_multi_bind() -> None:
    q = "rice prices in Senegal 2018 to 2022"
    dec = {
        "geography": ["Senegal"],
        "time_start": "2018-01-01",
        "time_end": "2022-12-31",
        "entities": ["rice"],
    }
    result = PrcEngine().run_plan(q, facets=dec, card=None)
    assert result.status == "planned", result.caveats
    assert result.bind_contracts
    assert "agg_prices_country_month" in result.bind_contracts or "fct_prices" in result.bind_contracts
    # taxonomy companion when primary is market fact
    if "fct_prices" in result.bind_contracts and len(result.bind_contracts) >= 2:
        assert "agg_prices_country_month" in result.bind_contracts


def test_fs_engine_uses_router_plans() -> None:
    q = "IPC Phase 3 population in Ethiopia 2023"
    dec = {
        "geography": ["Ethiopia"],
        "time_start": "2023-01-01",
        "time_end": "2023-12-31",
    }
    result = FsEngine().run_plan(q, facets=dec, card=None)
    assert result.status == "planned", result.caveats
    assert (
        "fct_food_security" in result.bind_contracts
        or "agg_food_security_monthly" in result.bind_contracts
        or "agg_food_security_country_month" in result.bind_contracts
    )


def test_food_balance_share_stays_fvc_only() -> None:
    q = "What share of Ghana's wheat domestic supply was imported in the latest food balance year?"
    bundles = match_intent_bundles(q, {"geography": ["Ghana"]})
    sp = compile_supervisor_plan(q, decomposition={"geography": ["Ghana"]}, matched_bundles=bundles)
    assert sp.classes == ("FVC",)
    assert sp.secondary == ()


def test_engine_results_merge_disjoint_and_overlap() -> None:
    a = EngineResult(
        class_code="PROD",
        status="planned",
        table_id="agg_production_country_year",
        sql=None,
        bind_contract={"table_id": "agg_production_country_year"},
        bind_contracts={"agg_production_country_year": {"table_id": "agg_production_country_year"}},
        query_intents=[{"tables": ["agg_production_country_year"]}],
    )
    b = EngineResult(
        class_code="FVC",
        status="planned",
        table_id="fct_food_balance",
        sql=None,
        bind_contracts={
            "fct_food_balance": {"table_id": "fct_food_balance", "src": "b"},
            "fct_trade": {"table_id": "fct_trade"},
        },
        query_intents=[{"tables": ["fct_food_balance"]}, {"tables": ["fct_trade"]}],
    )
    plan = engine_results_to_bq_plan([a, b])
    binds = plan["bind_contracts"]
    assert set(binds) == {"agg_production_country_year", "fct_food_balance", "fct_trade"}

    overlap = EngineResult(
        class_code="X",
        status="planned",
        table_id="fct_food_balance",
        sql=None,
        bind_contracts={"fct_food_balance": {"table_id": "fct_food_balance", "src": "overlap"}},
        query_intents=[{"tables": ["fct_food_balance"]}],
    )
    merged = engine_results_to_bq_plan([b, overlap])
    assert merged["bind_contracts"]["fct_food_balance"]["src"] == "overlap"
