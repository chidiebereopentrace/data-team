"""Shape catalog: every country, year shape, and warehouse indicator is a slot."""
from __future__ import annotations

import re

from ml.rag.chatbot.agri_measure_ontology import MEASURES
from ml.rag.chatbot.bq_sql_validate import validate_required_metric_filters
from ml.rag.chatbot.bq_table_schema_yaml import compile_table_bind_contract, year_column
from ml.rag.chatbot.compile_sql_from_bind import compile_sql_from_bind
from ml.rag.chatbot.query_forms import (
    QUERY_FORMS,
    YEAR_SHAPES,
    african_country_names,
    compile_form,
    region_names,
    report_indicator_ids,
    warehouse_indicator_ids,
)

_AGGREGATE_SQL = (
    "SELECT SUM(price_avg) AS price_avg "
    "FROM `proj.mart_dev.agg_prices_country_month` "
    "WHERE country_name = 'Kenya' AND year BETWEEN 2011 AND 2014 "
    "GROUP BY year"
)


def test_catalog_has_a_hundred_forms_and_the_full_inventory() -> None:
    assert len(QUERY_FORMS) >= 100
    assert len({form.template for form in QUERY_FORMS}) == len(QUERY_FORMS)
    assert set(YEAR_SHAPES) == {"single_year", "span", "latest"}
    assert len(african_country_names()) >= 50
    assert any(name.lower() == "west africa" for name in region_names())
    indicators = set(warehouse_indicator_ids())
    for indicator in (
        "production",
        "yield",
        "trade",
        "market_price",
        "food_security_ipc",
        "climate",
        "hdi",
        "gdp",
        "employment_share",
    ):
        assert indicator in indicators
    reports = set(report_indicator_ids())
    assert "news_briefing" in reports
    assert "research_synthesis" in reports


def test_compiled_forms_follow_the_shape_across_the_inventory() -> None:
    countries = african_country_names()
    indicators = warehouse_indicator_ids()
    checked = 0
    for index, form in enumerate(QUERY_FORMS):
        if form.shape == "report" or index % 5:
            continue
        indicator = indicators[index % len(indicators)]
        country = countries[index % len(countries)]
        table_id = _table_for(indicator)
        if not table_id:
            continue
        sql = compile_form(
            form,
            table_id=table_id,
            country=country,
            indicator=indicator,
            year=2011 + (index % 5),
            year_end=2015 + (index % 4),
        )
        if not sql:
            continue
        checked += 1
        if form.shape == "series" and year_column(table_id):
            assert "BETWEEN" in sql.upper()
            assert "ORDER BY" in sql
            assert re.search(r"\bLIMIT\s+1\b", sql, re.IGNORECASE) is None
        elif form.year_shape == "single_year":
            assert re.search(r"\bLIMIT\s+(?:1|50)\b", sql, re.IGNORECASE)
        if form.named_discriminator:
            assert "=" in sql
        else:
            assert validate_required_metric_filters(sql, {table_id}) is None
    assert checked >= 8


def test_span_does_not_collapse_to_the_end_year() -> None:
    start, end = 2011, 2014
    contract = compile_table_bind_contract(
        "agg_prices_country_month",
        facets={
            "geography": ["Kenya"],
            "time_start": f"{start}-01-01",
            "time_end": f"{end}-12-31",
            "entities": ["maize"],
            "primary_measures": ["market_price"],
            "job": "trend",
            "pattern": "time_series",
        },
        query=f"price trend of maize in kenya from {start} till {end}",
        country_labels=["Kenya"],
    )
    assert contract.time_sql
    assert "BETWEEN" in contract.time_sql
    assert f"= {end}" not in contract.time_sql
    sql = compile_sql_from_bind(contract, job="trend", pattern="time_series")
    assert sql is not None
    assert f"BETWEEN {start} AND {end}" in sql
    assert "price_source" in sql.split("FROM", 1)[0]
    assert "price_source =" not in sql.lower()
    assert "ORDER BY" in sql
    assert re.search(r"\bLIMIT\s+1\b", sql) is None
    assert validate_required_metric_filters(sql, {"agg_prices_country_month"}) is None


def test_single_year_stays_a_point_and_orders_the_finer_grain() -> None:
    contract = compile_table_bind_contract(
        "agg_prices_country_month",
        facets={
            "geography": ["Kenya"],
            "time_start": "2016-01-01",
            "time_end": "2016-12-31",
            "entities": ["maize"],
            "primary_measures": ["market_price"],
            "job": "fact",
        },
        query="what is the price of maize in kenya in 2016",
        country_labels=["Kenya"],
    )
    sql = compile_sql_from_bind(contract, job="fact")
    assert sql is not None
    assert "year = 2016" in sql
    assert "LIMIT 50" in sql
    assert "ORDER BY month DESC" in sql or "ORDER BY as_of_date DESC" in sql
    assert "product_name" in sql
    assert validate_required_metric_filters(sql, {"agg_prices_country_month"}) is None


def test_aggregate_without_a_discriminator_still_fails_validation() -> None:
    assert validate_required_metric_filters(_AGGREGATE_SQL, {"agg_prices_country_month"}) is not None


def _table_for(indicator: str) -> str | None:
    tables = MEASURES[indicator].candidate_tables
    return tables[0] if tables else None


def test_grain_columns_prefer_explicit_yaml_then_parsed_grain() -> None:
    from ml.rag.chatbot.bq_table_schema_yaml import grain_columns

    assert grain_columns("agg_production_country_year") == [
        "country_iso3",
        "product_key",
        "year",
        "source_key",
    ]
    assert "geo_key" in grain_columns("agg_production_annual")
    assert "trade_flow" in grain_columns("fct_trade")


def test_point_sql_labels_hash_keys_and_raises_limit_when_grain_is_open() -> None:
    contract = compile_table_bind_contract(
        "agg_production_country_year",
        facets={
            "geography": ["Kenya"],
            "time_start": "2020-01-01",
            "time_end": "2020-12-31",
            "primary_measures": ["production"],
            "job": "fact",
        },
        query="what is the production in Kenya in 2020",
        country_labels=["Kenya"],
        project_id="proj",
        dataset="mart_dev",
    )
    sql = compile_sql_from_bind(contract, project_id="proj", dataset="mart_dev", job="fact")
    assert sql is not None
    assert "dim_product" in sql
    assert "product_name" in sql
    assert "organisation_name" in sql
    assert "product_key" not in sql.split("FROM", 1)[0]
    assert "LIMIT 50" in sql


def test_unlabeled_figures_catch_a_missing_grain_column() -> None:
    from ml.rag.chatbot.query_form_admission import unlabeled_figures

    rows = [
        {"country_name": "Kenya", "year": 2020, "production_qty": 1.0},
        {"country_name": "Kenya", "year": 2020, "production_qty": 2.0},
    ]
    assert unlabeled_figures(rows, ["production_qty"])
    rows[1]["product_name"] = "Maize"
    rows[0]["product_name"] = "Rice"
    assert unlabeled_figures(rows, ["production_qty"]) is None


def test_rank_sql_groups_by_country_and_orders_the_measure() -> None:
    contract = compile_table_bind_contract(
        "agg_production_country_year",
        facets={
            "time_start": "2020-01-01",
            "time_end": "2020-12-31",
            "entities": ["maize"],
            "primary_measures": ["production"],
            "job": "rank",
        },
        query="which country produced the most maize in 2020",
        project_id="proj",
        dataset="mart_dev",
    )
    sql = compile_sql_from_bind(contract, project_id="proj", dataset="mart_dev", job="rank")
    assert sql is not None
    assert "GROUP BY" in sql
    assert "ORDER BY" in sql and "DESC" in sql
    assert "production_qty" in sql


def test_column_units_attach_to_agg_production_rows() -> None:
    from ml.rag.chatbot.bq_table_schema_yaml import with_column_units

    row = with_column_units(
        {"production_qty": 3789000.0, "area_harvested": 2135741.0},
        {"agg_production_country_year"},
    )
    assert row["production_qty_unit"] == "t"
    assert row["area_harvested_unit"] == "ha"
