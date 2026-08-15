"""Regression fixtures for measure ontology + control-plane upgrade."""
from __future__ import annotations

from unittest.mock import patch

from ml.rag.chatbot.agri_measure_ontology import (
    MEASURES,
    effective_tables,
    resolve_measure,
    wants_africa_panel,
)
from ml.rag.chatbot.analytical_intent import is_analytical_query
from ml.rag.chatbot.fact_bq_plan import build_fact_bq_plan
from ml.rag.chatbot.query_decomposer import (
    apply_africa_default_scope,
    wants_africa_default_scope,
    wants_africa_panel_scope,
)
from ml.rag.chatbot.query_enricher import enrich_query_with_memory
from ml.rag.chatbot.task_mode import (
    clarify_answer,
    needs_clarify,
    resolve_task_mode,
)


def test_ontology_has_core_measures() -> None:
    for mid in (
        "production",
        "yield",
        "trade",
        "market_price",
        "food_security_ipc",
        "climate",
        "soil",
        "investor_best_country",
        "data_export_panel",
    ):
        assert mid in MEASURES


def test_resolve_yield_not_production() -> None:
    hit = resolve_measure("What was maize yield in Kenya in 2020?", {"entities": ["maize"], "geography": ["Kenya"]})
    assert hit is not None
    assert hit.measure.id == "yield"
    assert "element='Yield'" in hit.measure.filter_hints


def test_resolve_investor_best_country() -> None:
    q = "Which African country is best for agricultural investment?"
    hit = resolve_measure(q, {"africa_default": True})
    assert hit is not None
    assert hit.measure.id == "investor_best_country"
    assert hit.measure.crop_required is False
    assert hit.measure.country_is_answer is True
    assert is_analytical_query(q, {"africa_default": True})
    assert resolve_task_mode(q, {"africa_default": True, "entities": []}) == "analytical"
    assert not needs_clarify(q, {"africa_default": True, "geography": [], "entities": []})


def test_ipc_no_crop_clarify() -> None:
    q = "What is the IPC food security situation in Somalia right now?"
    dec = {"geography": ["Somalia"], "entities": ["IPC"], "intent": "monitoring"}
    hit = resolve_measure(q, dec)
    assert hit is not None
    assert hit.measure.id == "food_security_ipc"
    assert not needs_clarify(q, dec)
    mode = resolve_task_mode(q, dec)
    assert mode != "clarify"


def test_soil_no_crop_clarify() -> None:
    q = "What is soil organic matter like in Zambia?"
    dec = {"geography": ["Zambia"], "entities": ["soil"], "intent": "descriptive"}
    assert not needs_clarify(q, dec)
    assert resolve_task_mode(q, dec) != "clarify"


def test_retail_price_tables() -> None:
    hit = resolve_measure(
        "Ethiopia retail maize prices",
        {"geography": ["Ethiopia"], "entities": ["maize"]},
    )
    assert hit is not None
    assert hit.measure.id == "market_price"
    tables = effective_tables(hit)
    assert "stg_fews_market_prices" in tables
    assert tables[0] != "stg_faostat_production"


def test_africa_panel_scope() -> None:
    q = "Maize yields for all African countries, past 5 years"
    assert wants_africa_panel(q)
    assert wants_africa_panel_scope(q)
    assert not wants_africa_default_scope(q)
    dec = apply_africa_default_scope({"entities": ["maize"], "geography": []}, q)
    assert dec.get("africa_panel") is True
    assert dec.get("africa_default") is not True
    assert resolve_task_mode(q, dec) == "data_export_only"
    assert not needs_clarify(q, dec)


def test_which_country_ranking_africa_default() -> None:
    q = "Which African country produces the most maize?"
    assert wants_africa_default_scope(q)
    dec = apply_africa_default_scope({"entities": ["maize"], "geography": []}, q)
    assert dec.get("africa_default") is True
    assert resolve_task_mode(q, dec) == "fact_lookup"


def test_fact_plan_yield_element() -> None:
    known = {"stg_faostat_production", "stg_yield_raw_data"}
    plan = build_fact_bq_plan(
        "What was maize yield in Kenya in 2020?",
        decomposition={
            "geography": ["Kenya"],
            "entities": ["maize"],
            "time_end": "2020-12-31",
        },
        known_tables=known,
        task_mode="fact_lookup",
    )
    assert plan is not None
    assert plan["skip_bq"] is False
    blob = str(plan["query_intents"])
    assert "Yield" in blob
    assert "element='Production'" not in blob or "element='Yield'" in blob


def test_fact_plan_trade_not_production() -> None:
    known = {
        "stg_faostat_production",
        "stg_faostat_trade",
        "stg_fews_cross_border_trade",
    }
    plan = build_fact_bq_plan(
        "Uganda coffee export volume 2019",
        decomposition={
            "geography": ["Uganda"],
            "entities": ["coffee"],
            "time_end": "2019-12-31",
        },
        known_tables=known,
        task_mode="fact_lookup",
    )
    assert plan is not None
    selected = plan["selected_tables"]
    assert "stg_faostat_trade" in selected
    assert selected[0] != "stg_faostat_production"


def test_clarify_ux_not_maize_nigeria_only() -> None:
    text = clarify_answer("soil fertility?", decomposition={"geography": [], "entities": []})
    assert "maize production in Nigeria in 2020" not in text.lower()
    assert "Suggested prompts" in text or "soil" in text.lower()


def test_enricher_merges_elliptical_followup() -> None:
    prior = [
        {"role": "user", "content": "What does research say about maize drought tolerance?"},
        {"role": "assistant", "content": "Several studies discuss drought-tolerant maize."},
    ]
    out = enrich_query_with_memory(
        "Country is Niger",
        conversation_summary="",
        recent_turns=prior,
    )
    assert out["enriched"] is True
    assert "maize" in out["enriched_query"].lower()
    assert "Niger" in out["enriched_query"]


def test_reasoner_ontology_fallback_on_empty_llm() -> None:
    from ml.rag.chatbot.bq_sql_reasoner import reason_bq_sql_plan

    with patch("ml.rag.chatbot.bq_sql_reasoner.llm_chat_complete", return_value=""):
        with patch.dict("os.environ", {"RAG_BQ_REASONER_RETRIES": "1"}):
            plan = reason_bq_sql_plan(
                "What was maize yield in Kenya in 2020?",
                decomposition={
                    "geography": ["Kenya"],
                    "entities": ["maize"],
                    "time_end": "2020-12-31",
                },
                task_mode="chat",
            )
    assert plan.get("skip_bq") is False or plan.get("rationale", "").startswith("ontology")
    if not plan.get("skip_bq"):
        assert "stg_faostat_production" in (plan.get("selected_tables") or [])
        assert "ontology_fallback" in str(plan.get("rationale") or "")


def test_investor_analytical_plan_multi_table() -> None:
    from ml.rag.chatbot.analytical_bq_plan import build_analytical_bq_plan

    known = {
        "stg_faostat_production",
        "stg_faostat_trade",
        "stg_africa_gdp_ppp",
        "stg_faostat_investment_asti",
        "stg_fews_market_prices",
        "stg_fews_food_security",
    }
    q = "Which African country is best for agricultural investment?"
    plan = build_analytical_bq_plan(
        q,
        decomposition={"africa_default": True, "entities": ["Africa"]},
        known_tables=known,
    )
    assert plan is not None
    assert plan["skip_bq"] is False
    assert len(plan["selected_tables"]) >= 3
    assert plan.get("measure_id") == "investor_best_country" or "investor" in str(
        plan.get("rationale") or ""
    )
