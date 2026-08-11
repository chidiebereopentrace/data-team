"""Unit tests for BigQuery NL2SQL validation helpers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from ml.rag.chatbot.bq_sql_validate import (
    bare_table_ids_from_hints,
    dry_run_sql,
    referenced_stg_tables,
    referenced_tables,
    validate_sql_table_allowlist,
)


def test_referenced_stg_tables() -> None:
    sql = (
        "SELECT a.country_name FROM `proj.staging_dev.stg_faostat_production` a "
        "JOIN `proj.staging_dev.stg_africa_gdp_ppp` b ON a.country_name = b.country_name"
    )
    refs = referenced_stg_tables(sql)
    assert refs == {"stg_faostat_production", "stg_africa_gdp_ppp"}


def test_referenced_tables_includes_dim() -> None:
    sql = (
        "SELECT country_name FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE area_code_m49 IN ("
        "SELECT country_code FROM `proj.staging_dev.dim_geography` "
        "WHERE country_name = 'Africa')"
    )
    refs = referenced_tables(sql)
    assert "stg_faostat_production" in refs
    assert "dim_geography" in refs


def test_validate_sql_table_allowlist_rejects_dim_geography() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2020 AND area_code_m49 IN ("
        "SELECT country_code FROM `proj.staging_dev.dim_geography` "
        "WHERE country_name = 'Africa') "
        "GROUP BY country_name ORDER BY total DESC"
    )
    err = validate_sql_table_allowlist(sql, {"stg_faostat_production"})
    assert err is not None
    assert "dim_geography" in err


def test_validate_sql_table_allowlist_rejects_extra() -> None:
    sql = "SELECT * FROM `proj.staging_dev.stg_africa_gdp_ppp` LIMIT 5"
    err = validate_sql_table_allowlist(sql, {"stg_faostat_production"})
    assert err is not None
    assert "stg_africa_gdp_ppp" in err


def test_validate_sql_table_allowlist_allows_selected() -> None:
    sql = (
        "SELECT country_name, SUM(value) FROM `proj.staging_dev.stg_faostat_production` "
        "GROUP BY country_name LIMIT 5"
    )
    assert validate_sql_table_allowlist(sql, {"stg_faostat_production"}) is None


def test_bare_table_ids_from_hints() -> None:
    hints = [
        "Table: opentrace-prod.staging_dev.stg_faostat_production\nColumns:\n- country_name",
        "Table: staging_dev.stg_yield_raw_data",
    ]
    ids = bare_table_ids_from_hints(hints)
    assert "stg_faostat_production" in ids
    assert "stg_yield_raw_data" in ids


def test_dry_run_sql_success() -> None:
    client = MagicMock()
    with patch("google.cloud.bigquery.QueryJobConfig"):
        assert dry_run_sql(client, "SELECT 1") is None
    client.query.assert_called_once()


def test_dry_run_sql_failure_returns_message() -> None:
    client = MagicMock()
    client.query.side_effect = Exception("Unrecognized name: country")
    err = dry_run_sql(client, "SELECT country FROM t")
    assert err is not None
    assert "country" in err
