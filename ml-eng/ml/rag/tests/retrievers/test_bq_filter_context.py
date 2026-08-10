"""Unit tests for BQ NL-to-SQL helpers."""
from __future__ import annotations

from ml.rag.chatbot.query_decomposer import _extract_countries
from ml.rag.retrievers.bq_retriever import (
    _continental_scope_hint,
    _extract_single_select,
    _format_query_constraints,
    _parse_sql_queries,
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
