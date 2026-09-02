"""Tests for mart SQL template fallback."""
from __future__ import annotations

from ml.rag.chatbot.bq_sql_templates import (
    build_mart_food_security_sql,
    build_mart_latest_price_sql,
    build_mart_point_fact_sql,
    build_mart_season_climate_sql,
    match_mart_country_rank,
    match_mart_country_series,
    match_mart_food_security_snapshot,
    match_mart_latest_price,
    match_mart_point_fact,
    match_mart_regional_panel,
    match_mart_season_climate,
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


def test_mart_point_fact_routes_to_agg_for_single_country() -> None:
    hit = try_sql_template(
        query="What was maize production in Nigeria in 2022?",
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["agg_production_annual", "fct_production"],
        geo_country="Nigeria",
        time_start="2022-01-01",
        time_end="2022-12-31",
        primary_measures=["production"],
    )
    assert hit is not None
    assert hit["template"] == "mart_point_fact"
    assert hit["table_id"] in ("agg_production_country_year", "agg_production_annual", "fct_production")
    assert hit["table_id"] in hit["sql"]
    assert "product_name = 'Maize'" in hit["sql"]
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


def test_match_mart_latest_price_current_retail() -> None:
    assert match_mart_latest_price(
        query="What is the current retail price of maize in Bamako, Mali?",
        selected_tables={"fct_prices"},
        entities=["maize", "Bamako", "Mali"],
        geo_country="Mali",
    )


def test_build_mart_latest_price_sql_orders_by_as_of_date() -> None:
    sql = build_mart_latest_price_sql(
        project_id="proj",
        dataset="mart_dev",
        table_id="fct_prices",
        country_labels=["Mali"],
        blob="current retail price maize Bamako",
        primary_measures=["market_price"],
        query="current retail price maize Bamako Mali",
    )
    assert "fct_prices" in sql
    assert "ORDER BY as_of_date DESC" in sql
    assert "MLI" in sql


def test_build_mart_food_security_ipc_phase3_sql() -> None:
    sql = build_mart_food_security_sql(
        project_id="proj",
        dataset="mart_dev",
        table_id="fct_food_security",
        year=2024,
        countries=["Ethiopia"],
        blob="How many people in Ethiopia are in IPC Phase 3 or higher right now?",
    )
    assert "population_3+" in sql
    assert "measure_type = 'population'" in sql
    assert "Current Situation" in sql


def test_match_mart_regional_panel_west_africa() -> None:
    assert match_mart_regional_panel(
        query="trend of production maize and rice across West Africa in 2022",
        selected_tables={"fct_production"},
        time_start="2022-01-01",
        time_end="2022-12-31",
        geo_countries=["Ghana", "Nigeria", "Mali"],
    )


def test_sanitize_yield_over_production_intent() -> None:
    from ml.rag.chatbot.ontology_context import sanitize_decomposition_for_bq

    dec = sanitize_decomposition_for_bq(
        {
            "query": "What was sorghum yield in Tahoua, Niger this season?",
            "entities": ["sorghum", "Tahoua", "Niger"],
            "primary_measures": ["production"],
        },
        primary_measures=["production"],
    )
    assert dec["primary_measures"][0] == "yield"


def test_match_mart_season_climate_western_kenya() -> None:
    assert match_mart_season_climate(
        query="When does the main rainy season start in western Kenya for planting maize?",
        selected_tables={"fct_yield", "fct_climate"},
        entities=["maize", "Kenya"],
        geo_country="Kenya",
    )


def test_build_mart_season_climate_sql_uses_fct_yield() -> None:
    sql = build_mart_season_climate_sql(
        project_id="proj",
        dataset="mart_dev",
        table_id="fct_yield",
        country_labels=["Kenya"],
        blob="rainy season planting maize western Kenya",
        primary_measures=["yield"],
        query="When does the main rainy season start in western Kenya for planting maize?",
    )
    assert "fct_yield" in sql
    assert "KEN" in sql
