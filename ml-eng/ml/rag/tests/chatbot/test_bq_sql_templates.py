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
from ml.rag.chatbot.bq_sql_validate import (
    validate_required_metric_filters,
    validate_sql_table_allowlist,
    validate_sql_value_samples,
)
from ml.rag.chatbot.bq_table_schema_yaml import (
    compile_product_filter_sql,
    resolve_dictionary_label,
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
        year=2022,
        blob="maize production Nigeria 2022",
        limit=1,
    )
    assert "stg_" not in sql
    assert "fct_production" in sql
    assert "NGA" in sql
    assert "2022" in sql
    assert "dim_product" in sql
    assert "product_name = 'Maize'" in sql
    assert "product_key = 'Maize'" not in sql
    assert "source_key" in sql
    assert "production_grain" in sql
    assert "element = 'Production'" in sql
    assert "metric = 'production_production_physical'" in sql
    assert "LIMIT 1" in sql


def test_compile_semantic_filter_fct_production_maize() -> None:
    sql, labels = compile_product_filter_sql(
        "fct_production",
        project_id="proj",
        dataset="mart_dev",
        blob="maize production Nigeria 2022",
    )
    assert labels == ["Maize"]
    assert "dim_product" in sql
    assert "product_name = 'Maize'" in sql
    assert "product_key = 'Maize'" not in sql


def test_compile_semantic_filter_agg_direct_product_name() -> None:
    sql, labels = compile_product_filter_sql(
        "agg_production_annual",
        project_id="proj",
        dataset="mart_dev",
        blob="maize production Nigeria 2022",
    )
    assert labels == ["Maize"]
    assert "product_name = 'Maize'" in sql
    assert "dim_product" not in sql


def test_resolve_dictionary_label_unknown_crop_returns_none() -> None:
    assert (
        resolve_dictionary_label(column="product_name", blob="xyzunknowncrop production")
        is None
    )


def test_compile_product_filter_fct_trade_maize() -> None:
    sql, labels = compile_product_filter_sql(
        "fct_trade",
        project_id="proj",
        dataset="mart_dev",
        blob="rice export Nigeria 2022",
        labels=["Rice"],
    )
    assert labels == ["Rice"]
    assert "dim_product" in sql
    assert "product_key IN (SELECT product_key" in sql
    assert "product_name = 'Rice'" in sql
    assert "product_name IN" not in sql


def test_mart_point_fact_sql_passes_validation() -> None:
    sql = build_mart_point_fact_sql(
        project_id="proj",
        dataset="mart_dev",
        table_id="fct_production",
        country_labels=["Nigeria"],
        year=2022,
        blob="maize production Nigeria 2022",
    )
    selected = {"fct_production"}
    assert validate_sql_table_allowlist(sql, selected) is None
    assert validate_sql_value_samples(sql, selected) is None
    assert validate_required_metric_filters(sql, selected) is None


def test_mart_point_fact_prefers_fct_production() -> None:
    hit = try_sql_template(
        query="What was maize production in Nigeria in 2022?",
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["agg_production_annual", "fct_production"],
        geo_country="Nigeria",
        time_start="2022-01-01",
        time_end="2022-12-31",
    )
    assert hit is not None
    assert hit["template"] == "mart_point_fact"
    assert hit["table_id"] == "fct_production"
    assert "fct_production" in hit["sql"]
    assert "dim_product" in hit["sql"]
    assert "product_name = 'Maize'" in hit["sql"]
    assert "production_grain" in hit["sql"]
    assert "source_key" in hit["sql"]
    assert "LIMIT 1" in hit["sql"]


def test_mart_point_fact_sql_includes_lineage_cols() -> None:
    sql = build_mart_point_fact_sql(
        project_id="proj",
        dataset="mart_dev",
        table_id="agg_production_annual",
        country_labels=["Nigeria"],
        year=2022,
        blob="maize production Nigeria 2022",
    )
    assert "source_key" in sql
    assert "source_name" in sql
    assert "product_name = 'Maize'" in sql
    assert "total_production_qty" in sql or "production_qty" in sql
    assert "ORDER BY record_count DESC" in sql
    assert "LIMIT 1" in sql


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
