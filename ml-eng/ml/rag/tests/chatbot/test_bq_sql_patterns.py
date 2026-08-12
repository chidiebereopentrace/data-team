"""Tests for structured SQL pattern builders and YAML join fragments."""
from __future__ import annotations

import re

from ml.rag.chatbot.bq_sql_patterns import (
    build_rank_by_sum_sql,
    build_share_of_total_sql,
    build_time_series_sql,
    build_yoy_delta_sql,
    normalize_pattern_name,
    try_sql_pattern,
)
from ml.rag.helpers.staging_semantic_relationships import (
    documented_join_pairs,
    format_join_fragments_for_nl2sql,
)
from ml.rag.retrievers.bq_retriever import _validate_sql


def test_normalize_pattern_unknown_to_custom() -> None:
    assert normalize_pattern_name("rank_by_sum") == "rank_by_sum"
    assert normalize_pattern_name("weird") == "custom"
    assert normalize_pattern_name(None) == "custom"


def test_rank_by_sum_builder() -> None:
    sql = build_rank_by_sum_sql(
        project_id="proj",
        dataset="staging_dev",
        table_id="stg_faostat_production",
        year=2020,
        product_name="Maize",
        limit=10,
    )
    assert sql.upper().startswith("SELECT")
    assert "stg_faostat_production" in sql
    assert "year = 2020" in sql
    assert "GROUP BY country_name" in sql
    assert "ORDER BY total DESC" in sql
    assert _validate_sql(sql, {"staging_dev"}, 10) is not None


def test_time_series_builder() -> None:
    sql = build_time_series_sql(
        project_id="proj",
        dataset="staging_dev",
        table_id="stg_faostat_production",
        year=2020,
        product_name="Maize",
    )
    assert "GROUP BY year" in sql or "GROUP BY harvest_year" in sql or "AS year" in sql
    assert "ORDER BY" in sql
    assert _validate_sql(sql, {"staging_dev"}, 10) is not None


def test_yoy_and_share_builders_allowlisted() -> None:
    yoy = build_yoy_delta_sql(
        project_id="proj",
        dataset="staging_dev",
        table_id="stg_faostat_production",
        year=2020,
        product_name="Maize",
    )
    share = build_share_of_total_sql(
        project_id="proj",
        dataset="staging_dev",
        table_id="stg_faostat_production",
        year=2020,
        product_name="Maize",
    )
    assert yoy.upper().lstrip().startswith("WITH")
    assert share.upper().lstrip().startswith("WITH")
    assert _validate_sql(yoy, {"staging_dev"}, 10) is not None
    assert _validate_sql(share, {"staging_dev"}, 10) is not None
    assert "yoy_delta" in yoy
    assert "share" in share


def test_try_sql_pattern_rank() -> None:
    hit = try_sql_pattern(
        {
            "pattern": "rank_by_sum",
            "tables": ["stg_faostat_production"],
            "metric": "value",
            "grain": ["country_name"],
            "filters": "maize",
        },
        project_id="proj",
        dataset="staging_dev",
        query="highest maize production in Africa 2020",
        time_start="2020-01-01",
        limit=5,
    )
    assert hit is not None
    assert hit["pattern"] == "rank_by_sum"
    assert "Maize" in hit["sql"] or "maize" in hit["sql"].lower()


def test_try_sql_pattern_maps_bare_country_grain() -> None:
    hit = try_sql_pattern(
        {
            "pattern": "rank_by_sum",
            "tables": ["stg_faostat_production"],
            "metric": "value",
            "grain": ["country"],
            "filters": "maize",
        },
        project_id="proj",
        dataset="staging_dev",
        query="highest maize production in Africa 2020",
        time_start="2020-01-01",
        limit=5,
    )
    assert hit is not None
    sql = hit["sql"]
    assert "country_name" in sql
    assert "GROUP BY country_name" in sql
    # Bare identifier country must not appear (country_name is fine).
    assert re.search(r"(?<![A-Za-z0-9_])country(?![A-Za-z0-9_])", sql) is None
    assert _validate_sql(sql, {"staging_dev"}, 5) is not None


def test_try_sql_pattern_fews_keeps_country_grain() -> None:
    hit = try_sql_pattern(
        {
            "pattern": "rank_by_sum",
            "tables": ["stg_fews_market_prices"],
            "metric": "value",
            "grain": ["country"],
        },
        project_id="proj",
        dataset="staging_dev",
        query="highest maize prices 2020",
        time_start="2020-01-01",
        limit=5,
    )
    assert hit is not None
    assert "GROUP BY country" in hit["sql"]
    assert "country_name" not in hit["sql"]


def test_try_sql_pattern_custom_skipped() -> None:
    assert (
        try_sql_pattern(
            {"pattern": "custom", "tables": ["stg_faostat_production"]},
            project_id="proj",
            dataset="staging_dev",
            query="anything 2020",
            time_start="2020-01-01",
        )
        is None
    )


def test_join_fragments_related_pair() -> None:
    text = format_join_fragments_for_nl2sql(
        ["stg_faostat_production", "stg_yield_raw_data"]
    )
    assert "JOIN fragments" in text
    assert "stg_faostat_production" in text
    assert "stg_yield_raw_data" in text
    assert " ON " in text
    pairs = documented_join_pairs(["stg_faostat_production", "stg_yield_raw_data"])
    assert pairs
    assert all(p.get("on") for p in pairs)


def test_join_fragments_unrelated_no_invented_on() -> None:
    text = format_join_fragments_for_nl2sql(
        ["stg_faostat_production", "stg_nakuru_air_quality"]
    )
    assert "SEPARATE SELECT" in text.upper() or "separate SELECT" in text
    assert " ON " not in text or "none documented" in text.lower()
    assert documented_join_pairs(["stg_faostat_production", "stg_nakuru_air_quality"]) == []
