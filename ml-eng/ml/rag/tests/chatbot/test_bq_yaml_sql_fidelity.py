"""Tests for YAML column allowlist and value-sample SQL validation."""
from __future__ import annotations

from ml.rag.chatbot.bq_sql_validate import (
    validate_required_metric_filters,
    validate_sql_column_allowlist,
    validate_sql_value_samples,
)
from ml.rag.chatbot.bq_table_schema_yaml import columns_for_tables, value_samples_for_tables
from ml.rag.retrievers.bq_retriever import _SCHEMA_FILTER_GUIDE


def test_columns_for_tables_faostat() -> None:
    cols = columns_for_tables({"stg_faostat_production"})
    assert "stg_faostat_production" in cols
    names = cols["stg_faostat_production"]
    assert "country_name" in names
    assert "product_name" in names
    assert "area" not in names
    assert "item" not in {n.lower() for n in names} or "item_code" in names


def test_value_samples_for_tables_element() -> None:
    samples = value_samples_for_tables({"stg_faostat_production"})
    by_col = samples["stg_faostat_production"]
    assert "Production" in by_col["element"]
    assert "product_name" in by_col
    assert "Maize" in by_col["product_name"]


def test_value_samples_include_fews_price_type() -> None:
    samples = value_samples_for_tables({"stg_fews_market_prices"})
    by_col = samples["stg_fews_market_prices"]
    assert "Retail" in by_col["price_type"]
    assert "market_name" in by_col


def test_column_allowlist_rejects_bronze_area() -> None:
    sql = (
        "SELECT area, product_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2020 GROUP BY area, product_name LIMIT 5"
    )
    err = validate_sql_column_allowlist(sql)
    assert err is not None
    assert "area" in err.lower()


def test_column_allowlist_accepts_opentrace_prod_fqn() -> None:
    sql = (
        "SELECT * FROM `opentrace-prod-5ga4.staging_dev.stg_faostat_production` "
        "WHERE country_name = 'Nigeria' AND product_name = 'Maize' "
        "AND element = 'Production' ORDER BY year"
    )
    assert validate_sql_column_allowlist(sql) is None


def test_column_allowlist_accepts_yaml_names() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2020 AND element = 'Production' "
        "GROUP BY country_name ORDER BY total DESC LIMIT 5"
    )
    assert validate_sql_column_allowlist(sql) is None


def test_column_allowlist_accepts_fews_country() -> None:
    sql = (
        "SELECT country, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_fews_market_prices` "
        "WHERE year = 2020 GROUP BY country LIMIT 5"
    )
    assert validate_sql_column_allowlist(sql) is None


def test_required_metric_filters_rejects_missing_element() -> None:
    sql = (
        "SELECT * FROM `opentrace-prod-5ga4.staging_dev.stg_faostat_production` "
        "WHERE country_name = 'Nigeria' AND product_name = 'Maize' ORDER BY year"
    )
    err = validate_required_metric_filters(sql)
    assert err is not None
    assert "element" in err.lower()


def test_required_metric_filters_accepts_element() -> None:
    sql = (
        "SELECT year, value, unit FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE country_name = 'Nigeria' AND product_name = 'Maize' "
        "AND element = 'Production' ORDER BY year"
    )
    assert validate_required_metric_filters(sql) is None


def test_required_metric_filters_ignores_unreferenced_plan_companions() -> None:
    """Plan may select ASTI + production; SQL only querying production must not demand indicator."""
    sql = (
        "SELECT year AS year, value, unit, product_name, country_name, element "
        "FROM `opentrace-prod-5ga4.staging_dev.stg_faostat_production` "
        "WHERE country_name = 'Nigeria' AND element = 'Production' "
        "ORDER BY year LIMIT 100"
    )
    plan_tables = {"stg_faostat_production", "stg_faostat_investment_asti"}
    assert validate_required_metric_filters(sql, plan_tables) is None
    assert validate_sql_column_allowlist(sql, plan_tables) is None
    assert validate_sql_value_samples(sql, plan_tables) is None


def test_required_metric_filters_rejects_missing_price_type() -> None:
    sql = (
        "SELECT year, value FROM `proj.staging_dev.stg_fews_market_prices` "
        "WHERE country = 'Nigeria' AND product_name = 'Maize' ORDER BY year"
    )
    err = validate_required_metric_filters(sql)
    assert err is not None
    assert "price_type" in err.lower()


def test_value_samples_rejects_bad_element() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2020 AND element = 'prod' "
        "GROUP BY country_name LIMIT 5"
    )
    err = validate_sql_value_samples(sql)
    assert err is not None
    assert "prod" in err


def test_value_samples_accepts_production() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2020 AND element = 'Production' "
        "GROUP BY country_name LIMIT 5"
    )
    assert validate_sql_value_samples(sql) is None


def test_value_samples_ignores_like() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2020 AND LOWER(element) LIKE '%prod%' "
        "GROUP BY country_name LIMIT 5"
    )
    assert validate_sql_value_samples(sql) is None


def test_schema_filter_guide_no_bronze_area_item_as_primary() -> None:
    guide = _SCHEMA_FILTER_GUIDE.lower()
    assert "columns block" in guide
    assert "never invent bronze/raw" in guide or "bronze/raw" in guide
    # Must not list bare area/item as recommended filter columns.
    assert "product / crop: product, product_name, item" not in guide
    assert "metric discriminator" in guide
