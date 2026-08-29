"""Tests for structured SQL pattern builders and YAML join fragments."""
from __future__ import annotations

import re

from ml.rag.chatbot.analytical_bq_plan import (
    build_analytical_bq_plan,
    build_food_security_bq_plan,
)
from ml.rag.chatbot.bq_sql_patterns import (
    build_rank_by_sum_sql,
    build_share_of_total_sql,
    build_time_series_sql,
    build_yoy_delta_sql,
    normalize_pattern_name,
    try_sql_pattern,
    try_sql_patterns,
)
from ml.rag.chatbot.bq_table_schema_yaml import value_samples_for_mart_tables, value_samples_for_tables
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


def test_try_sql_patterns_compiles_production_and_trade() -> None:
    hits = try_sql_patterns(
        [
            {
                "pattern": "rank_by_sum",
                "tables": ["stg_faostat_production"],
                "metric": "value",
                "grain": ["country_name"],
                "filters": "element='Production'; maize",
            },
            {
                "pattern": "rank_by_sum",
                "tables": ["stg_faostat_trade"],
                "metric": "value",
                "grain": ["country_name"],
                "filters": "export; maize",
            },
            {
                "pattern": "custom",
                "tables": ["stg_ilri_household_food_security"],
            },
        ],
        project_id="proj",
        dataset="staging_dev",
        query="maize production and trade west africa 2022",
        time_start="2022-01-01",
        geo_countries=["Nigeria", "Ghana", "Senegal"],
    )
    assert len(hits) == 2
    tables = {h["table_id"] for h in hits}
    assert tables == {"stg_faostat_production", "stg_faostat_trade"}
    prod = next(h for h in hits if h["table_id"] == "stg_faostat_production")
    trade = next(h for h in hits if h["table_id"] == "stg_faostat_trade")
    samples = value_samples_for_tables({"stg_faostat_production", "stg_faostat_trade"})
    assert prod["product_name"] in samples["stg_faostat_production"]["product_name"]
    assert trade["product_name"] in samples["stg_faostat_trade"]["product_name"]
    assert "Maize" in prod["sql"]
    assert "Maize (corn)" in trade["sql"]
    assert "country_name IN (" in prod["sql"]
    assert "Nigeria" in prod["sql"]
    assert "Ghana" in prod["sql"]
    assert "Export quantity" in trade["sql"]


def test_try_sql_pattern_denies_survey_and_corridor() -> None:
    for tid in ("stg_ilri_household_food_security", "stg_fews_cross_border_trade"):
        hit = try_sql_pattern(
            {"pattern": "rank_by_sum", "tables": [tid], "metric": "value"},
            project_id="proj",
            dataset="staging_dev",
            query="maize 2020",
            time_start="2020-01-01",
        )
        assert hit is None


def test_try_sql_pattern_gdp_uses_yaml_measure_and_avg() -> None:
    hit = try_sql_pattern(
        {
            "pattern": "rank_by_sum",
            "tables": ["stg_africa_gdp_ppp"],
            "metric": "value",
            "grain": ["country_name"],
        },
        project_id="proj",
        dataset="staging_dev",
        query="highest GDP per capita 2020",
        time_start="2020-01-01",
        geo_countries=["Nigeria", "Ghana"],
    )
    assert hit is not None
    sql = hit["sql"]
    assert "stg_africa_gdp_ppp" in sql
    assert "observation_year = 2020" in sql
    assert "gdp_per_capita_ppp" in sql
    assert "AVG(gdp_per_capita_ppp)" in sql
    assert "SUM(value)" not in sql
    assert "country_name IN (" in sql


def test_analytical_plan_tags_trade_and_series_patterns() -> None:
    plan = build_analytical_bq_plan(
        "West Africa maize production and trade report",
        decomposition={
            "geography": ["Nigeria", "Ghana", "Senegal"],
            "time_start": "2022-01-01",
            "time_end": "2022-12-31",
            "entities": ["maize"],
        },
        known_tables={"fct_production", "fct_trade", "fct_economics"},
    )
    assert plan is not None
    by_notes = {i["notes"]: i["pattern"] for i in plan["query_intents"]}
    assert by_notes["analytical_products_by_country"] == "rank_by_sum"
    assert by_notes["analytical_series_endpoints"] == "time_series"
    assert by_notes["analytical_trade_export"] == "rank_by_sum"
    assert by_notes["analytical_trade_import"] == "rank_by_sum"


def test_food_security_plan_keeps_ipc_and_ilri_custom() -> None:
    plan = build_food_security_bq_plan(
        "Assess food security risk across the Sahel",
        decomposition={
            "geography": ["Mali", "Niger", "Burkina Faso"],
            "time_start": "2020-01-01",
            "time_end": "2025-12-31",
        },
        known_tables={
            "fct_food_security",
            "fct_household",
            "fct_production",
        },
    )
    assert plan is not None
    by_table = {
        (i.get("tables") or [""])[0]: i["pattern"] for i in plan["query_intents"]
    }
    assert by_table["fct_food_security"] == "custom"
    assert by_table["fct_household"] == "custom"


def test_try_sql_patterns_kenya_nigeria_maize_series() -> None:
    hits = try_sql_patterns(
        [
            {
                "pattern": "time_series",
                "tables": ["fct_production"],
                "metric": "value",
                "grain": ["country_iso3", "year"],
                "filters": "production_grain='physical'; maize",
            },
            {
                "pattern": "rank_by_sum",
                "tables": ["fct_production"],
                "metric": "value",
                "grain": ["country_iso3"],
                "filters": "production_grain='physical'; maize",
            },
        ],
        project_id="proj",
        dataset="mart_dev",
        query=(
            "Compare maize production in Kenya and Nigeria over the last five years, "
            "and give me a CSV export of the figures."
        ),
        entities=["maize"],
        time_start="2021-01-01",
        time_end="2026-12-31",
        geo_countries=["Kenya", "Nigeria"],
    )
    assert hits
    sqls = " ".join(h["sql"] for h in hits)
    assert "fct_production" in sqls
    assert "Kenya" in sqls or "KEN" in sqls
    assert "Nigeria" in sqls or "NGA" in sqls
    assert "country_iso3 IN (" in sqls
    samples = value_samples_for_mart_tables({"fct_production"})
    product_col = "product_name" if "product_name" in samples.get("fct_production", {}) else "product_key"
    if product_col in samples.get("fct_production", {}) and hits[0].get("product_name"):
        assert hits[0]["product_name"] in samples["fct_production"][product_col]


def test_time_series_honors_country_grain() -> None:
    sql = build_time_series_sql(
        project_id="proj",
        dataset="staging_dev",
        table_id="stg_faostat_production",
        year=2022,
        products=["Maize", "Rice"],
        grain=["country_name", "year"],
        element="Production",
        blob="element='Production'; maize rice",
    )
    assert "GROUP BY country_name, year" in sql
    assert "country_name" in sql.split("GROUP BY")[0]
    assert _validate_sql(sql, {"staging_dev"}, 10) is not None


def test_time_series_mart_uses_as_of_date_range() -> None:
    sql = build_time_series_sql(
        project_id="proj",
        dataset="mart_dev",
        table_id="fct_hdi",
        year=2024,
        grain=["country_iso3", "year"],
        time_start="2020-01-01",
        time_end="2024-12-31",
    )
    assert "as_of_date BETWEEN DATE '2020-01-01' AND DATE '2024-12-31'" in sql
    assert _validate_sql(sql, {"mart_dev"}, 10) is not None


def test_try_sql_patterns_west_africa_maize_rice_production_trade() -> None:
    from ml.rag.chatbot.geo_regions import expand_regions_in_decomposition

    query = (
        "give me a docx file report of the trend of production and trade maize and rice "
        "across west africa in 2022"
    )
    dec = expand_regions_in_decomposition(
        {
            "geography": [],
            "entities": ["maize", "rice"],
            "time_start": "2022-01-01",
            "time_end": "2022-12-31",
        },
        query,
    )
    countries = [str(g) for g in (dec.get("geography") or [])]
    assert "Nigeria" in countries
    assert "Senegal" in countries
    plan = build_analytical_bq_plan(
        query,
        decomposition=dec,
        known_tables={"fct_production", "fct_trade"},
    )
    assert plan is not None
    hits = try_sql_patterns(
        plan["query_intents"],
        project_id="proj",
        dataset="mart_dev",
        query=query,
        entities=["maize", "rice"],
        time_start="2022-01-01",
        time_end="2022-12-31",
        geo_countries=countries,
    )
    assert hits
    sqls = " ".join(h["sql"] for h in hits)
    assert "country_iso3 IN (" in sqls
    assert "Nigeria" in sqls or "NGA" in sqls
    assert "Senegal" in sqls or "SEN" in sqls
    assert "fct_production" in sqls or "fct_trade" in sqls
    assert "'West Africa'" not in sqls
    assert any(
        "GROUP BY country_iso3, year" in h["sql"]
        or "GROUP BY country_iso3, product_key, year" in h["sql"]
        for h in hits
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
