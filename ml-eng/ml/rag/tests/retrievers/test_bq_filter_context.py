"""Unit tests for BQ NL-to-SQL helpers."""
from __future__ import annotations

import re

from ml.rag.chatbot.query_decomposer import _extract_countries
from ml.rag.retrievers.bq_retriever import (
    _continental_scope_hint,
    _extract_single_select,
    _format_query_constraints,
    _parse_sql_queries,
    _rewrite_faostat_country_ident,
    _validate_sql,
)


def test_nigeria_not_niger() -> None:
    assert _extract_countries("agriculture in nigeria") == ["Nigeria"]
    assert "Niger" not in _extract_countries("products nigeria produces")


def test_format_query_constraints() -> None:
    block = _format_query_constraints(
        geo_country="Nigeria",
        time_start="2013-01-01",
        time_end="2022-12-31",
        entities=["agricultural products"],
        domains=["economy"],
    )
    assert "REQUIRED country" in block
    assert "2013" in block
    assert "country_name" in block
    assert "GROUP BY country when comparing" not in block


def test_format_query_constraints_multi_geo_prefers_country_name() -> None:
    block = _format_query_constraints(
        geo_country=None,
        geo_countries=["Kenya", "Nigeria"],
        time_start=None,
        time_end=None,
        entities=[],
        domains=None,
    )
    assert "country_name" in block
    assert "Columns block" in block
    assert "GROUP BY country when comparing" not in block


def test_rewrite_faostat_bare_country() -> None:
    sql = (
        "SELECT country, product_name, MAX(value) AS max_production "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE country = 'Nigeria' GROUP BY country, product_name LIMIT 10"
    )
    out = _rewrite_faostat_country_ident(sql)
    assert "country_name" in out
    assert re.search(r"(?<![A-Za-z0-9_])country(?![A-Za-z0-9_])", out) is None
    # Already-correct identifiers stay intact.
    assert "country_name" in out


def test_rewrite_skips_non_faostat() -> None:
    sql = (
        "SELECT country, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_fews_market_prices` "
        "WHERE country = 'Kenya' GROUP BY country LIMIT 5"
    )
    assert _rewrite_faostat_country_ident(sql) == sql


def test_validate_sql_rewrites_faostat_bare_country() -> None:
    sql = (
        "SELECT country, product_name, MAX(value) AS max_production "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2020 GROUP BY country, product_name"
    )
    validated = _validate_sql(sql, {"staging_dev"}, 10)
    assert validated is not None
    assert "country_name" in validated
    assert re.search(r"(?<![A-Za-z0-9_])country(?![A-Za-z0-9_])", validated) is None
    assert "LIMIT 10" in validated.upper()


def test_validate_sql_preserves_fews_country() -> None:
    sql = (
        "SELECT country, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_fews_market_prices` "
        "WHERE year = 2020 GROUP BY country LIMIT 5"
    )
    validated = _validate_sql(sql, {"staging_dev"}, 5)
    assert validated is not None
    assert "GROUP BY country" in validated
    assert "country_name" not in validated


def test_parse_sql_queries() -> None:
    raw = (
        "SELECT * FROM `proj.ds.t1` WHERE country_name = 'Nigeria' LIMIT 5\n"
        "---QUERY---\n"
        "SELECT year, gdp FROM `proj.ds.t2` WHERE country_name = 'Nigeria' LIMIT 5"
    )
    queries = _parse_sql_queries(raw, 10)
    assert len(queries) == 2
    assert "t1" in queries[0]
    assert "t2" in queries[1]


def test_extract_single_select_strips_inst_tokens() -> None:
    raw = "[INST] You are a BigQuery expert... [/INST] SELECT * FROM `proj.ds.t1` LIMIT 5"
    sql = _extract_single_select(raw)
    assert sql.startswith("SELECT")


def test_extract_single_select_prose_returns_empty() -> None:
    raw = "Here is a helpful explanation of the query you asked."
    assert _extract_single_select(raw) == ""


def test_extract_single_select_plain_select_unchanged() -> None:
    raw = "SELECT country, year FROM `proj.ds.t1` LIMIT 10"
    assert _extract_single_select(raw).startswith("SELECT")


def test_continental_scope_hint_africa_query() -> None:
    hint = _continental_scope_hint(
        "which country in africa had the highest agricultural production in 2020",
        None,
    )
    assert hint is not None
    assert "country_name" in hint
    assert "Africa" in hint


def test_format_query_constraints_africa_scope() -> None:
    block = _format_query_constraints(
        geo_country=None,
        time_start="2020-01-01",
        time_end="2020-12-31",
        entities=["agricultural production"],
        domains=["agriculture"],
        query="highest agricultural production in Africa 2020",
    )
    assert "CONTINENTAL/REGIONAL scope" in block
    assert "2020" in block


def test_format_query_constraints_uses_integer_years() -> None:
    block = _format_query_constraints(
        geo_country="Ghana",
        time_start="2022-01-01",
        time_end="2022-12-31",
        entities=["maize"],
        domains=None,
    )
    assert "year BETWEEN 2022 AND 2022" in block
    assert "2022-01-01" not in block
    assert "2022-12-31" not in block


def test_format_query_constraints_skips_continental_hint_when_countries_expanded() -> None:
    block = _format_query_constraints(
        geo_country=None,
        geo_countries=["Nigeria", "Ghana", "Senegal", "Mali"],
        time_start="2022-01-01",
        time_end="2022-12-31",
        entities=["maize", "rice"],
        domains=["agriculture"],
        query="production and trade maize and rice across west africa in 2022",
    )
    assert "CONTINENTAL/REGIONAL scope" not in block
    assert "Nigeria" in block
    assert "year BETWEEN 2022 AND 2022" in block
