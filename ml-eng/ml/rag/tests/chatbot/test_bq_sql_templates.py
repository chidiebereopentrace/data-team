"""Tests for FAOSTAT ranking SQL template fallback."""
from __future__ import annotations

from ml.rag.chatbot.bq_sql_templates import (
    build_faostat_country_rank_sql,
    match_faostat_country_rank,
    match_faostat_crop_rank,
    match_faostat_price_rank,
    match_fews_food_security,
    try_sql_template,
)


def test_match_faostat_country_rank_africa_2020() -> None:
    assert match_faostat_country_rank(
        query="which country in africa had the highest agricultural production in 2020",
        selected_tables={"stg_faostat_production"},
        time_start="2020-01-01",
        time_end="2020-12-31",
    )


def test_match_faostat_country_rank_requires_table() -> None:
    assert not match_faostat_country_rank(
        query="highest agricultural production in Africa 2020",
        selected_tables={"stg_yield_raw_data"},
    )


def test_match_faostat_country_rank_rejects_unrelated() -> None:
    assert not match_faostat_country_rank(
        query="what is the soil pH in Nakuru",
        selected_tables={"stg_faostat_production"},
        time_start="2020-01-01",
    )


def test_match_faostat_crop_rank_maize() -> None:
    assert match_faostat_crop_rank(
        query="which country in africa produced the most maize in 2020",
        selected_tables={"stg_faostat_production"},
        time_start="2020-01-01",
        time_end="2020-12-31",
    )
    # Crop-specific should win over generic country rank.
    assert not match_faostat_country_rank(
        query="which country in africa produced the most maize in 2020",
        selected_tables={"stg_faostat_production"},
        time_start="2020-01-01",
    )


def test_match_faostat_price_and_fews() -> None:
    assert match_faostat_price_rank(
        query="highest maize producer prices in Africa 2019",
        selected_tables={"stg_faostat_prices"},
        time_start="2019-01-01",
    )
    assert match_fews_food_security(
        query="which regions had the worst IPC food security crisis in 2022",
        selected_tables={"stg_fews_food_security"},
        time_start="2022-01-01",
    )


def test_build_faostat_country_rank_sql() -> None:
    sql = build_faostat_country_rank_sql(
        project_id="proj",
        dataset="staging_dev",
        year=2020,
        limit=10,
    )
    assert "stg_faostat_production" in sql
    assert "year = 2020" in sql
    assert "GROUP BY country_name" in sql
    assert "ORDER BY total DESC" in sql
    assert "dim_geography" not in sql


def test_try_sql_template_returns_payload() -> None:
    hit = try_sql_template(
        query="highest agricultural production in Africa 2020",
        project_id="proj",
        dataset="staging_dev",
        selected_tables=["stg_faostat_production"],
        time_start="2020-01-01",
        time_end="2020-12-31",
    )
    assert hit is not None
    assert hit["template"] == "faostat_country_rank"
    assert hit["year"] == 2020
    assert "SUM(value)" in hit["sql"]


def test_try_sql_template_crop_and_price() -> None:
    crop = try_sql_template(
        query="highest maize production in Africa 2020",
        project_id="proj",
        dataset="staging_dev",
        selected_tables=["stg_faostat_production"],
        time_start="2020-01-01",
    )
    assert crop is not None
    assert crop["template"] == "faostat_crop_rank"
    assert "maize" in crop["sql"].lower()

    price = try_sql_template(
        query="highest producer prices in Africa 2019",
        project_id="proj",
        dataset="staging_dev",
        selected_tables=["stg_faostat_prices"],
        time_start="2019-01-01",
    )
    assert price is not None
    assert price["template"] == "faostat_price_rank"
