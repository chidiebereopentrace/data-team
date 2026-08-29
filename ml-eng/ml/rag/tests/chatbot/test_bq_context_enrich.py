"""Tests for YAML-driven BQ context enrichment."""
from __future__ import annotations

from ml.rag.chatbot.bq_context_enrich import (
    _table_from_sql,
    enrich_bq_results,
    format_row_prose,
    resolve_row_semantics,
    _resolve_unit,
)
from ml.rag.chatbot.bq_table_schema_yaml import (
    column_description,
    discriminator_columns,
    measure_columns,
    table_source_meta,
)
from ml.rag.chatbot.exports.tabular import rows_from_bq_results
from ml.rag.chatbot.generator import _build_prompt, is_ranking_numeric_query, is_numeric_data_query, pin_bq_context_first
from ml.rag.chatbot import reranker as R


def _bq_item(row: dict, *, sql: str, **meta_extra) -> dict:
    meta = {
        "sql": sql,
        "sql_source": "template",
        "template": "faostat_production_rank",
        **meta_extra,
    }
    meta.update(row)
    return {
        "content": str(row),
        "source": "bigquery",
        "metadata": meta,
    }


def test_yaml_helpers_faostat_production() -> None:
    table = "stg_faostat_production"
    assert "element" in discriminator_columns(table)
    assert "value" in measure_columns(table)
    desc = column_description(table, "element")
    assert "Production" in desc or "element" in desc.lower() or len(desc) > 10
    meta = table_source_meta(table)
    assert meta["table_id"] == table
    assert "production" in meta.get("description", "").lower() or meta.get("source_domain")


def test_faostat_production_element_semantics() -> None:
    row = {
        "country_name": "South Africa",
        "product_name": "Maize",
        "element": "Production",
        "year": 2020,
        "unit": "tonnes",
        "value": 45678901,
    }
    sem = resolve_row_semantics(
        row,
        table_id="stg_faostat_production",
        sql="SELECT country_name, SUM(value) AS total FROM `proj.staging_dev.stg_faostat_production`",
    )
    assert "production" in sem["measure_label"].lower()
    assert sem["unit"] == "tonnes"
    assert sem["geo"] == "South Africa"
    assert "yield" in " ".join(sem["not_this"]).lower()


def test_faostat_yield_not_production() -> None:
    row = {
        "country_name": "Kenya",
        "product_name": "Maize",
        "element": "Yield",
        "year": 2020,
        "unit": "hg/ha",
        "value": 2500,
    }
    sem = resolve_row_semantics(row, table_id="stg_faostat_production")
    assert "yield" in sem["measure_label"].lower()
    assert sem["unit"] == "hg/ha"
    assert any("production" in x.lower() for x in sem["not_this"])


def test_fews_food_security_population() -> None:
    row = {
        "country": "Ethiopia",
        "measure_type": "population",
        "phase_code": "3+",
        "phase_name": "Phase 3 and above",
        "scenario_name": "Current Situation",
        "year": 2024,
        "month": 3,
        "value": 1234567,
    }
    sem = resolve_row_semantics(row, table_id="stg_fews_food_security")
    assert "people" in sem["unit"].lower()
    assert "population" in sem["measure_label"].lower() or "food-insecure" in sem["measure_label"].lower()
    assert any("production" in x.lower() for x in sem["not_this"])


def test_fews_food_security_classification() -> None:
    row = {
        "country": "Ethiopia",
        "measure_type": "classification",
        "classification_scale": "IPC 3.1",
        "scenario_name": "Current Situation",
        "year": 2024,
        "month": 3,
        "value": 3,
    }
    sem = resolve_row_semantics(row, table_id="stg_fews_food_security")
    assert "classification" in sem["measure_label"].lower()
    assert any("population" in x.lower() for x in sem["not_this"])


def test_fews_market_prices_retail() -> None:
    row = {
        "country": "Kenya",
        "market_name": "Nairobi, Wakulima",
        "product_name": "Maize Grain (White)",
        "price_type": "Retail",
        "currency": "KES",
        "unit": "kg",
        "year": 2024,
        "month": 1,
        "value": 45.2,
    }
    sem = resolve_row_semantics(row, table_id="stg_fews_market_prices")
    assert "retail" in sem["measure_label"].lower()
    assert "KES/kg" in sem["unit"]
    assert any("production" in x.lower() for x in sem["not_this"])


def test_gdp_ppp_semantics() -> None:
    row = {
        "country_name": "Nigeria",
        "observation_year": 2020,
        "gdp_per_capita_ppp": 5432.1,
    }
    sem = resolve_row_semantics(row, table_id="stg_africa_gdp_ppp")
    assert "gdp" in sem["measure_label"].lower()
    assert "ppp" in sem["unit"].lower()
    assert any("agricultural" in x.lower() or "production" in x.lower() for x in sem["not_this"])


def test_yield_raw_data_columns() -> None:
    sem_yield = resolve_row_semantics(
        {"country": "Ghana", "product": "Maize", "harvest_year": 2019, "yield": 2.1},
        table_id="stg_yield_raw_data",
    )
    assert "yield" in sem_yield["measure_label"].lower()
    sem_prod = resolve_row_semantics(
        {"country": "Ghana", "product": "Maize", "harvest_year": 2019, "production": 1000},
        table_id="stg_yield_raw_data",
    )
    assert "production" in sem_prod["measure_label"].lower()
    sem_area = resolve_row_semantics(
        {"country": "Ghana", "product": "Maize", "harvest_year": 2019, "area": 500},
        table_id="stg_yield_raw_data",
    )
    assert "area" in sem_area["measure_label"].lower()


def test_enrich_single_row_prose() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `opentrace-prod-5ga4.staging_dev.stg_faostat_production` "
        "WHERE element = 'Production' AND year = 2020"
    )
    item = _bq_item(
        {"country_name": "South Africa", "total": 45678901, "element": "Production", "year": 2020, "unit": "tonnes"},
        sql=sql,
    )
    out = enrich_bq_results([item], query="maize production in South Africa 2020", plan={"selected_tables": ["stg_faostat_production"]})
    assert len(out) == 1
    content = out[0]["content"]
    assert "FAOSTAT" in content
    assert "OpenTrace statistical series" not in content
    assert "stg_faostat_production" not in content
    assert "production" in content.lower()
    assert "tonnes" in content.lower()
    assert "NOT" in content
    assert "Provenance:" not in content
    assert "sql_source=" not in content
    assert isinstance(out[0]["metadata"].get("raw_row"), dict)
    assert out[0]["metadata"].get("value_semantics")


def test_format_row_prose_fallback_omits_table_id() -> None:
    prose = format_row_prose(
        {
            "table_id": "stg_yield_raw_data",
            "source_domain": "",
            "measure_label": "Yield",
            "measure_value": 2.1,
            "unit": "t/ha",
            "geo": "Senegal",
            "time": "2020",
            "discriminators": {},
            "not_this": [],
            "table_description": "",
            "grain": "",
        }
    )
    assert prose.startswith("OpenTrace agricultural data")
    assert "stg_yield_raw_data" not in prose
    assert "FAOSTAT" not in prose


def test_format_row_prose_agg_yield_raw_data_shows_fews_net() -> None:
    prose = format_row_prose(
        {
            "table_id": "agg_production_annual",
            "source_domain": "production",
            "source_name": "yield_raw_data",
            "measure_label": "Crop production volume",
            "measure_value": 12_000_000,
            "unit": "tonnes",
            "geo": "Nigeria",
            "time": "2022",
            "discriminators": {
                "country_name": "Nigeria",
                "product_name": "Maize",
                "source_name": "yield_raw_data",
            },
            "not_this": [],
            "table_description": "agg production annual",
            "grain": "",
        }
    )
    assert prose.startswith("FEWS NET")
    assert "yield_raw_data" not in prose
    assert "source_name=" not in prose
    assert "Nigeria" in prose


def test_format_row_prose_includes_trend_line() -> None:
    prose = format_row_prose(
        {
            "table_id": "fct_production",
            "source_domain": "production",
            "measure_label": "Production",
            "measure_value": 1000,
            "unit": "tonnes",
            "geo": "Kenya",
            "time": "2020",
            "discriminators": {},
            "not_this": [],
            "table_description": "",
            "grain": "",
        },
        direction="increasing",
        magnitude=20.0,
    )
    assert "Trend: increasing (+20% change)" in prose


def test_ranking_consolidation() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE element = 'Production' AND year = 2020 GROUP BY country_name ORDER BY total DESC LIMIT 3"
    )
    items = [
        _bq_item({"country_name": "South Africa", "total": 100, "element": "Production", "year": 2020, "unit": "tonnes"}, sql=sql),
        _bq_item({"country_name": "Nigeria", "total": 80, "element": "Production", "year": 2020, "unit": "tonnes"}, sql=sql),
        _bq_item({"country_name": "Ethiopia", "total": 60, "element": "Production", "year": 2020, "unit": "tonnes"}, sql=sql),
    ]
    query = "which country in africa had the highest agricultural production in 2020"
    assert is_ranking_numeric_query(query)
    out = enrich_bq_results(items, query=query, plan={"selected_tables": ["stg_faostat_production"]})
    assert len(out) == 1
    assert out[0]["metadata"].get("bq_enrichment") == "ranked_table"
    ranked = out[0]["metadata"].get("ranked_rows")
    assert isinstance(ranked, list) and len(ranked) == 3
    assert ranked[0]["label"] == "South Africa"
    content = out[0]["content"]
    assert "Ranked results (highest first):" in content
    assert "Ranked OpenTrace structured results" not in content
    assert "Authoritative for which-country" not in content
    assert "Provenance:" not in content
    assert "sql_source=" not in content
    assert "1. South Africa" in content


def test_ranking_consolidation_stamps_acf_metadata() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE element = 'Production' AND year = 2020 GROUP BY country_name ORDER BY total DESC LIMIT 3"
    )
    items = [
        _bq_item({"country_name": "South Africa", "total": 100, "element": "Production", "unit": "tonnes"}, sql=sql),
        _bq_item({"country_name": "Nigeria", "total": 80, "element": "Production", "unit": "tonnes"}, sql=sql),
        _bq_item({"country_name": "Ethiopia", "total": 60, "element": "Production", "unit": "tonnes"}, sql=sql),
    ]
    query = "which country in africa had the highest agricultural production in 2020"
    out = enrich_bq_results(
        items,
        query=query,
        plan={"selected_tables": ["stg_faostat_production"]},
        decomposition={"time_start": "2020-01-01", "time_end": "2020-12-31"},
    )
    meta = out[0]["metadata"]
    assert meta.get("as_of_date") == "2020-01-01"
    assert meta.get("year") == 2020
    assert meta.get("metric")
    assert meta.get("geo_country_primary") == "South Africa"
    assert meta.get("coverage_strength") == 0.3


def test_single_row_enrich_stamps_year_from_sql() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE element = 'Production' AND year = 2020"
    )
    item = _bq_item(
        {"country_name": "Nigeria", "total": 45678901, "element": "Production", "unit": "tonnes"},
        sql=sql,
    )
    out = enrich_bq_results([item], query="Nigeria production 2020", plan={"selected_tables": ["stg_faostat_production"]})
    meta = out[0]["metadata"]
    assert meta.get("year") == 2020
    assert meta.get("as_of_date") == "2020-01-01"


def test_rows_from_bq_results_enriched_raw_row() -> None:
    enriched = {
        "source": "bigquery",
        "content": "prose not a dict",
        "metadata": {
            "raw_row": {"country_name": "Senegal", "value": 99},
            "value_semantics": {"measure_label": "Production", "unit": "tonnes"},
        },
    }
    rows = rows_from_bq_results([enriched])
    assert len(rows) == 1
    assert rows[0]["country_name"] == "Senegal"
    assert rows[0]["measure_label"] == "Production"


def test_rows_from_bq_results_ranked_rows() -> None:
    enriched = {
        "source": "bigquery",
        "content": "ranked table prose",
        "metadata": {
            "bq_enrichment": "ranked_table",
            "ranked_rows": [
                {
                    "rank": 1,
                    "label": "South Africa",
                    "value": 100,
                    "unit": "tonnes",
                    "measure_label": "Production",
                    "raw_row": {"country_name": "South Africa", "total": 100},
                }
            ],
        },
    }
    rows = rows_from_bq_results([enriched])
    assert len(rows) == 1
    assert rows[0]["country_name"] == "South Africa"
    assert rows[0]["rank"] == 1


def test_build_prompt_structured_bq_numeric_available_guard() -> None:
    messages = _build_prompt(
        "how much maize did Nigeria produce in 2020?",
        context_block="[Structured data] production row",
        structured_bq_numeric_available=True,
    )
    sys_msg = messages[0]["content"]
    assert "CRITICAL" in sys_msg
    assert "statistical figures" in sys_msg.lower()
    assert "authoritative" in sys_msg.lower()
    assert "structured bigquery" not in sys_msg.lower()


def test_build_prompt_structured_bq_comparative_available_guard() -> None:
    messages = _build_prompt(
        "what drove millet policy in Senegal?",
        context_block="[Structured data] production row",
        structured_bq_comparative_available=True,
    )
    sys_msg = messages[0]["content"]
    assert "comparative statistics" in sys_msg.lower()
    assert "policy drivers" in sys_msg.lower()
    assert "authoritative" not in sys_msg.lower()
    assert "CRITICAL" not in sys_msg


def test_reranker_ranked_table_extra_boost() -> None:
    item = {
        "_context_kind": "bigquery",
        "metadata": {"bq_enrichment": "ranked_table"},
    }
    boost = R._item_source_boost(item)
    assert boost >= 0.12 + 0.15


def test_reranker_numeric_query_boosts_bq_over_policy() -> None:
    bq = {"_context_kind": "bigquery", "metadata": {"value_semantics": {"unit": "tonnes"}}}
    policy = {"_context_kind": "policy", "metadata": {}}
    bq_boost = R._item_source_boost(bq, numeric_query=True)
    policy_boost = R._item_source_boost(policy, numeric_query=True)
    assert bq_boost > policy_boost


def test_reranker_comparative_query_boosts_bq_over_policy() -> None:
    bq = {"_context_kind": "bigquery", "metadata": {"value_semantics": {"unit": "tonnes"}}}
    policy = {"_context_kind": "policy", "metadata": {}}
    bq_boost = R._item_source_boost(bq, comparative_query=True)
    policy_boost = R._item_source_boost(policy, comparative_query=True)
    assert bq_boost > policy_boost
    numeric_bq_boost = R._item_source_boost(bq, numeric_query=True)
    assert numeric_bq_boost > bq_boost


def test_node_merge_puts_bq_first_for_ranking() -> None:
    from ml.rag.chatbot import graph as graph_mod

    state = {
        "query": "which country had the highest production in 2020",
        "bq_results": [
            {
                "content": "OpenTrace structured data ranked",
                "source": "bigquery",
                "metadata": {"sql": "SELECT 1"},
            }
        ],
        "vector_news_results": [{"content": "news chunk", "metadata": {"title": "t"}, "score": 0.9}],
        "vector_academic_papers_results": [],
        "vector_policies_results": [],
        "vector_public_reports_results": [],
        "vector_formation_results": [],
        "vector_ota_results": [],
    }
    merged = graph_mod.node_merge(state)["merged_context"]
    assert merged[0]["_context_kind"] == "bigquery"
    assert merged[1]["_context_kind"] == "news"


def test_table_from_sql_detects_mart() -> None:
    sql = (
        "SELECT country_iso3, SUM(value) AS total "
        "FROM `opentrace-prod-5ga4.mart_dev.fct_production` "
        "WHERE production_grain = 'physical' AND year = 2020"
    )
    assert _table_from_sql(sql) == "fct_production"


def test_fct_production_element_semantics() -> None:
    row = {
        "country_iso3": "ZAF",
        "product_key": "maize",
        "element": "Production",
        "production_grain": "physical",
        "year": 2020,
        "unit": "tonnes",
        "value": 45678901,
    }
    sem = resolve_row_semantics(
        row,
        table_id="fct_production",
        sql=(
            "SELECT country_iso3, SUM(value) AS total "
            "FROM `proj.mart_dev.fct_production`"
        ),
    )
    assert "production" in sem["measure_label"].lower()
    assert sem["unit"] == "tonnes"
    assert sem["geo"] == "ZAF"
    assert "yield" in " ".join(sem["not_this"]).lower()


def test_resolve_unit_production_qty_tonnes() -> None:
    assert (
        _resolve_unit(
            "agg_production_annual",
            {"total_production_qty": 963000000},
            "total_production_qty",
        )
        == "tonnes"
    )


def test_enrich_consolidates_duplicate_point_fact_rows() -> None:
    sql = (
        "SELECT country_iso3, product_key, year, total_production_qty, source_key "
        "FROM `proj.mart_dev.agg_production_annual` "
        "WHERE country_iso3 = 'NGA' AND year = 2022 LIMIT 1"
    )
    items = [
        _bq_item(
            {
                "country_iso3": "NGA",
                "product_key": "maize",
                "year": 2022,
                "total_production_qty": 963000000,
                "source_key": "other_src",
                "record_count": 1,
            },
            sql=sql,
            template="mart_point_fact",
        ),
        _bq_item(
            {
                "country_iso3": "NGA",
                "product_key": "maize",
                "year": 2022,
                "total_production_qty": 404000000,
                "source_key": "faostat_production_nga",
                "record_count": 50,
                "tier": 2,
            },
            sql=sql,
            template="mart_point_fact",
        ),
        _bq_item(
            {
                "country_iso3": "NGA",
                "product_key": "maize",
                "year": 2022,
                "total_production_qty": 887000000,
                "source_key": "legacy_src",
                "record_count": 10,
            },
            sql=sql,
            template="mart_point_fact",
        ),
    ]
    out = enrich_bq_results(
        items,
        query="What was maize production in Nigeria in 2022?",
        plan={"selected_tables": ["agg_production_annual"], "task_mode": "fact_lookup"},
    )
    assert len(out) == 1
    assert out[0]["metadata"].get("bq_enrichment") == "point_fact"
    assert out[0]["metadata"].get("source_key") == "faostat_production_nga"
    assert out[0]["metadata"].get("value_conflict") is True
    assert "tonnes" in out[0]["content"].lower()


def test_rank_consolidation_uses_country_iso3() -> None:
    sql = (
        "SELECT country_iso3, SUM(value) AS total "
        "FROM `proj.mart_dev.fct_production` "
        "WHERE element = 'Production' AND year = 2020 GROUP BY country_iso3"
    )
    items = [
        _bq_item({"country_iso3": "NGA", "total": 100.0, "element": "Production", "year": 2020}, sql=sql),
        _bq_item({"country_iso3": "ZAF", "total": 200.0, "element": "Production", "year": 2020}, sql=sql),
    ]
    out = enrich_bq_results(
        items,
        query="which country had the highest production in 2020",
        plan={"selected_tables": ["fct_production"]},
    )
    assert len(out) == 1
    ranked = out[0]["metadata"].get("ranked_rows") or []
    assert ranked[0]["label"] == "ZAF"
    assert out[0]["metadata"].get("bq_enrichment") == "ranked_table"
