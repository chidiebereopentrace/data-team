"""Tests for YAML column allowlist and value-sample SQL validation."""
from __future__ import annotations

from ml.rag.chatbot.bq_sql_validate import (
    broaden_empty_sql_once,
    inject_missing_metric_filters,
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


def test_inject_missing_element_production() -> None:
    sql = (
        "SELECT * FROM `opentrace-prod-5ga4.staging_dev.stg_faostat_production` "
        "WHERE country_name = 'Nigeria' AND product_name = 'Maize' ORDER BY year"
    )
    fixed, notes = inject_missing_metric_filters(
        sql,
        query="maize production Nigeria 2022",
    )
    assert any("element='Production'" in n for n in notes)
    assert "element = 'Production'" in fixed
    assert validate_required_metric_filters(fixed) is None
    assert "ORDER BY year" in fixed
    assert "product_name = 'Maize'" in fixed
    assert "country_name = 'Nigeria'" in fixed


def test_inject_keeps_string_literals_before_group_by() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2022 AND country_name = 'Nigeria' AND product_name = 'Maize' "
        "GROUP BY country_name LIMIT 5"
    )
    fixed, notes = inject_missing_metric_filters(
        sql,
        query="maize production Nigeria 2022",
    )
    assert notes
    assert "element = 'Production'" in fixed
    assert "product_name = 'Maize'" in fixed
    assert "country_name = 'Nigeria'" in fixed
    assert "GROUP BY country_name" in fixed
    assert "product_nam " not in fixed


def test_inject_missing_element_prefers_yield() -> None:
    sql = (
        "SELECT country_name, value FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE country_name = 'Kenya' AND year = 2020"
    )
    fixed, notes = inject_missing_metric_filters(sql, query="maize yield Kenya 2020")
    assert any("element='Yield'" in n for n in notes)
    assert "element = 'Yield'" in fixed
    assert validate_required_metric_filters(fixed) is None


def test_inject_missing_price_type_retail() -> None:
    sql = (
        "SELECT year, value FROM `proj.staging_dev.stg_fews_market_prices` "
        "WHERE country = 'Ethiopia' AND product_name = 'Maize' ORDER BY year"
    )
    fixed, notes = inject_missing_metric_filters(
        sql,
        query="retail maize price Ethiopia",
    )
    assert any("price_type='Retail'" in n for n in notes)
    assert "price_type = 'Retail'" in fixed
    assert validate_required_metric_filters(fixed) is None


def test_inject_missing_skips_when_element_present() -> None:
    sql = (
        "SELECT year, value FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE country_name = 'Nigeria' AND product_name = 'Maize' "
        "AND element = 'Production' ORDER BY year"
    )
    fixed, notes = inject_missing_metric_filters(sql, query="maize production Nigeria")
    assert notes == []
    assert fixed == sql


def test_inject_does_not_fix_bad_element_label() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2020 AND element = 'Prod' "
        "GROUP BY country_name LIMIT 5"
    )
    fixed, notes = inject_missing_metric_filters(sql, query="maize production")
    assert notes == []
    assert "element = 'Prod'" in fixed
    err = validate_sql_value_samples(fixed)
    assert err is not None
    assert "Prod" in err or "prod" in err.lower()


def test_broaden_empty_sql_widens_single_year() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2022 AND element = 'Production' AND product_name = 'Maize' "
        "GROUP BY country_name LIMIT 5"
    )
    out = broaden_empty_sql_once(sql, crop_required=True)
    assert out is not None
    assert "year BETWEEN 2021 AND 2023" in out
    assert "product_name = 'Maize'" in out
    assert "stg_faostat_production" in out


def test_broaden_empty_sql_drops_optional_product() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2022 AND element = 'Production' AND product_name = 'Maize' "
        "GROUP BY country_name LIMIT 5"
    )
    out = broaden_empty_sql_once(sql, crop_required=False)
    assert out is not None
    assert "year BETWEEN 2021 AND 2023" in out
    assert "product_name = 'Maize'" not in out
    assert "dim_" not in out


def test_broaden_empty_sql_noop_without_year_or_product() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE element = 'Production' AND year BETWEEN 2020 AND 2022 "
        "GROUP BY country_name LIMIT 5"
    )
    assert broaden_empty_sql_once(sql, crop_required=True) is None
