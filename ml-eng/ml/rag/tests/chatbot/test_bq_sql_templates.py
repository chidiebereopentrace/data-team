"""Tests for mart SQL template fallback."""
from __future__ import annotations

from ml.rag.chatbot.bq_sql_templates import (
    build_mart_food_security_sql,
    build_mart_point_fact_sql,
    match_mart_country_rank,
    match_mart_country_series,
    match_mart_food_security_snapshot,
    match_mart_point_fact,
    try_sql_template,
)


def test_match_mart_country_rank_africa_2020() -> None:
    assert match_mart_country_rank(
        query="which country in africa had the highest agricultural production in 2020",
        selected_tables={"fct_production"},
        time_start="2020-01-01",
        time_end="2020-12-31",
    )


def test_match_mart_country_rank_requires_mart_table() -> None:
    assert not match_mart_country_rank(
        query="highest agricultural production in Africa 2020",
        selected_tables={"dim_geography"},
    )


def test_match_mart_point_fact_nigeria_maize() -> None:
    assert match_mart_point_fact(
        query="What was maize production in Nigeria in 2022?",
        selected_tables={"fct_production", "agg_production_annual"},
        time_start="2022-01-01",
        time_end="2022-12-31",
        geo_country="Nigeria",
    )


def test_build_mart_point_fact_sql_uses_nga() -> None:
    sql = build_mart_point_fact_sql(
        project_id="proj",
        dataset="mart_dev",
        table_id="fct_production",
        country_labels=["Nigeria"],
        product_name="Maize",
        year=2022,
        blob="maize production Nigeria 2022",
        limit=5,
    )
    assert "stg_" not in sql
    assert "fct_production" in sql
    assert "NGA" in sql
    assert "2022" in sql
    assert "Maize" in sql or "product_key" in sql


def test_try_sql_template_mart_country_rank() -> None:
    hit = try_sql_template(
        query="highest agricultural production in Africa 2020",
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["fct_production"],
        time_start="2020-01-01",
        time_end="2020-12-31",
    )
    assert hit is not None
    assert hit["template"] == "mart_country_rank"
    assert hit["year"] == 2020
    assert "stg_" not in hit["sql"]
    assert "SUM(" in hit["sql"] or "total" in hit["sql"]


def test_try_sql_template_mart_point_fact() -> None:
    hit = try_sql_template(
        query="What was maize production in Nigeria in 2022?",
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["fct_production"],
        geo_country="Nigeria",
        time_start="2022-01-01",
        time_end="2022-12-31",
    )
    assert hit is not None
    assert hit["template"] == "mart_point_fact"
    assert "stg_" not in hit["sql"]
    assert "NGA" in hit["sql"]


def test_try_sql_template_mart_country_series() -> None:
    hit = try_sql_template(
        query="Export maize production data for Nigeria as a CSV",
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["fct_production"],
        geo_country="Nigeria",
    )
    assert hit is not None
    assert hit["template"] == "mart_country_series"
    assert "stg_" not in hit["sql"]
    assert "NGA" in hit["sql"] or "Nigeria" in hit["sql"]


def test_match_mart_country_series_rejects_multi_country() -> None:
    assert not match_mart_country_series(
        query="maize production series for Sahel countries",
        selected_tables={"fct_production"},
        entities=["Maize"],
        geo_countries=["Mali", "Niger", "Burkina Faso", "Chad"],
    )


def test_mart_food_security_snapshot() -> None:
    assert match_mart_food_security_snapshot(
        query="IPC food security crisis in Ethiopia",
        selected_tables={"fct_food_security"},
    )
    sql = build_mart_food_security_sql(
        project_id="proj",
        dataset="mart_dev",
        table_id="fct_food_security",
        year=None,
        countries=["Ethiopia"],
        blob="food security population",
    )
    assert "stg_" not in sql
    assert "fct_food_security" in sql
    assert "ETH" in sql or "Ethiopia" in sql
