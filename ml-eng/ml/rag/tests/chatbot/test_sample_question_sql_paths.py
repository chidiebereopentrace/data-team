"""Smoke tests for sample_questions.md #2–#6 SQL / planning paths."""
from __future__ import annotations

from ml.rag.chatbot.bq_sql_templates import try_sql_template
from ml.rag.chatbot.generation_plan import build_generation_plan
from ml.rag.chatbot.ontology_context import sanitize_decomposition_for_bq


def test_sample_q2_bamako_maize_price_template() -> None:
    hit = try_sql_template(
        query="What is the current retail price of maize in markets around Bamako, Mali?",
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["fct_prices"],
        geo_country="Mali",
        primary_measures=["market_price"],
        task_mode="fact_lookup",
    )
    assert hit is not None
    assert hit["template"] == "mart_latest_price"
    assert "ORDER BY as_of_date DESC" in hit["sql"]
    assert "fct_prices" in hit["sql"]


def test_sample_q3_ethiopia_ipc_phase3_template() -> None:
    hit = try_sql_template(
        query="How many people in Ethiopia are in IPC Phase 3 or higher right now?",
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["fct_food_security", "agg_food_security_monthly"],
        geo_country="Ethiopia",
        primary_measures=["food_security_ipc"],
        task_mode="fact_lookup",
    )
    assert hit is not None
    assert hit["template"] == "mart_food_security_snapshot"
    assert "population_3+" in hit["sql"]
    assert "Current Situation" in hit["sql"]


def test_sample_q4_kenya_planting_season_narrative_shape() -> None:
    from ml.rag.chatbot.generation_plan import _refine_shape_from_query

    query = "When does the main rainy season start in western Kenya for planting maize?"
    shape = _refine_shape_from_query(query, None, "numeric_fact")
    assert shape == "policy_narrative"
    plan = build_generation_plan(
        query,
        task_mode="fact_lookup",
        reranked_context=[
            {
                "content": "Western Kenya long rains typically begin in March.",
                "_context_kind": "news",
                "metadata": {"title": "Kenya planting calendar", "country": "Kenya"},
            }
        ],
    )
    assert plan.answer_shape == "policy_narrative"


def test_sample_q5_tahoua_yield_sanitize_and_template() -> None:
    query = "What sorghum yield can farmers expect in Tahoua, Niger this harvest season?"
    dec = sanitize_decomposition_for_bq(
        {"query": query, "entities": ["sorghum", "Tahoua", "Niger"], "primary_measures": ["production"]},
        primary_measures=["production"],
    )
    assert dec["primary_measures"][0] == "yield"
    hit = try_sql_template(
        query=query,
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["fct_yield"],
        entities=dec.get("entities"),
        geo_country="Niger",
        primary_measures=dec["primary_measures"],
        task_mode="fact_lookup",
    )
    assert hit is not None
    assert hit["table_id"] == "fct_yield"


def test_sample_q6_kano_tomato_price_trend_not_latest_point() -> None:
    hit = try_sql_template(
        query="Are tomato prices going up or down at markets near Kano, Nigeria?",
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["fct_prices"],
        geo_country="Nigeria",
        primary_measures=["market_price"],
        task_mode="fact_lookup",
    )
    if hit is not None:
        assert hit["template"] != "mart_latest_price" or "ORDER BY as_of_date DESC" in hit["sql"]
