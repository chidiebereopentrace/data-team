"""Bind SQL stays executable: predicate parsing, statement allowlist, latest row, sample tokens."""
from __future__ import annotations

from ml.rag.chatbot.bq_sql_validate import (
    inject_bind_contract_filters,
    inject_missing_metric_filters,
    validate_required_metric_filters,
    validate_sql_column_allowlist,
)
from ml.rag.chatbot.bq_table_schema_yaml import (
    compile_measure_filters,
    default_discriminator_value,
    match_value_samples,
    value_samples_for_mart_tables,
)
from ml.rag.chatbot.compile_sql_from_bind import compile_sql_from_bind

_AGG_SQL = (
    "SELECT price_avg, common_unit_price_avg, market_count "
    "FROM `opentrace-prod-5ga4.mart_dev.agg_prices_country_month` "
    "WHERE country_name = 'Kenya' AND product_key IN ("
    "SELECT product_key FROM `opentrace-prod-5ga4.mart_dev.dim_product` "
    "WHERE product_name = 'Maize') AND price_type = 'Retail' LIMIT 1"
)
_FCT_SQL = (
    "SELECT value FROM `opentrace-prod-5ga4.mart_dev.fct_prices` "
    "WHERE country_iso3 = 'KEN' AND product_key IN ("
    "SELECT product_key FROM `opentrace-prod-5ga4.mart_dev.dim_product` "
    "WHERE product_name = 'Maize') "
    "AND metric = 'price_retail_maize_flour' AND price_type = 'Retail' LIMIT 1"
)
_PLAN_TABLES = {"agg_prices_country_month", "fct_prices"}
_BINDS = {
    "agg_prices_country_month": {
        "table_id": "agg_prices_country_month",
        "geo_column": "country_name",
        "geo_literals": ["Kenya"],
        "measure_filters": [["price_type", "Retail"]],
        "required_filters_sql": (
            "AND country_name = 'Kenya' AND product_key IN ("
            "SELECT product_key FROM `opentrace-prod-5ga4.mart_dev.dim_product` "
            "WHERE product_name = 'Maize') AND price_type = 'Retail'"
        ),
    },
    "fct_prices": {
        "table_id": "fct_prices",
        "geo_column": "country_iso3",
        "geo_literals": ["KEN"],
        "measure_filters": [
            ["metric", "price_retail_maize_flour"],
            ["price_type", "Retail"],
        ],
        "required_filters_sql": (
            "AND country_iso3 = 'KEN' AND product_key IN ("
            "SELECT product_key FROM `opentrace-prod-5ga4.mart_dev.dim_product` "
            "WHERE product_name = 'Maize') AND metric = 'price_retail_maize_flour' "
            "AND price_type = 'Retail'"
        ),
    },
}


def _prepare(sql: str) -> str:
    injected, _notes = inject_bind_contract_filters(sql, _BINDS)
    prepared, _metric_notes = inject_missing_metric_filters(
        injected,
        _PLAN_TABLES,
        query="what is the price of maize in kenya",
    )
    return prepared


def test_multi_contract_injection_keeps_each_statement_on_its_own_columns() -> None:
    agg = _prepare(_AGG_SQL)
    fct = _prepare(_FCT_SQL)

    for sql in (agg, fct):
        assert " AND cou " not in f" {sql} "
        assert " AND pro " not in f" {sql} "
        assert validate_sql_column_allowlist(sql, _PLAN_TABLES) is None

    assert "metric" not in agg
    assert "country_name" not in fct
    assert "price_source = 'wfp'" not in agg
    assert "price_source = 'wfp'" not in fct


def test_between_predicate_stays_intact() -> None:
    sql = (
        "SELECT value FROM `proj.mart_dev.fct_prices` "
        "WHERE country_iso3 = 'KEN' LIMIT 1"
    )
    binds = {
        "fct_prices": {
            "table_id": "fct_prices",
            "geo_column": "country_iso3",
            "geo_literals": ["KEN"],
            "required_filters_sql": "AND year BETWEEN 2020 AND 2024",
        }
    }
    injected, _notes = inject_bind_contract_filters(sql, binds)
    assert "year BETWEEN 2020 AND 2024" in injected
    assert "year BETWEEN 2020'" not in injected


def test_column_allowlist_rejects_plan_companion_columns() -> None:
    sql = (
        "SELECT country_name, value FROM `proj.mart_dev.fct_prices` "
        "WHERE country_iso3 = 'KEN' AND price_type = 'Retail' LIMIT 1"
    )
    err = validate_sql_column_allowlist(sql, {"fct_prices", "agg_prices_country_month"})
    assert err is not None
    assert "country_name" in err


def test_undated_price_bind_orders_by_latest_grain() -> None:
    sql = compile_sql_from_bind(
        {
            "table_id": "agg_prices_country_month",
            "measure_columns": ["price_avg", "common_unit_price_avg"],
            "required_filters_sql": "country_name = 'Kenya' AND price_type = 'Retail'",
        },
        project_id="proj",
        dataset="mart_dev",
    )
    assert sql is not None
    assert "year" in sql
    assert "month" in sql
    assert "price_type" in sql
    assert "price_source" in sql
    assert "ORDER BY year DESC, month DESC" in sql
    assert validate_required_metric_filters(sql, {"agg_prices_country_month"}) is None
    assert validate_sql_column_allowlist(sql, {"agg_prices_country_month", "fct_prices"}) is None


def test_year_equality_bind_stays_a_single_year() -> None:
    sql = compile_sql_from_bind(
        {
            "table_id": "agg_production_country_year",
            "measure_columns": ["production_qty"],
            "required_filters_sql": "country_iso3 = 'KEN' AND year = 2016",
        },
        project_id="proj",
        dataset="mart_dev",
    )
    assert sql is not None
    assert "year = 2016" in sql
    assert "ORDER BY" not in sql


def test_price_sample_follows_tokens_in_the_question() -> None:
    samples = value_samples_for_mart_tables({"fct_prices"})["fct_prices"]["metric"]
    maize = match_value_samples("what is the price of maize in kenya", samples)
    flour = match_value_samples("what is the price of maize flour in kenya", samples)
    assert maize
    assert flour
    assert maize[0] == "price_retail_maize_-_retail"
    assert "flour" in flour[0]
    assert (
        default_discriminator_value(
            "price_source",
            ["fews", "faostat", "wfp"],
            query="what is the price of maize in kenya",
        )
        is None
    )


def test_unnamed_price_type_is_selected_not_pinned() -> None:
    filters = compile_measure_filters(
        "agg_prices_country_month",
        measure_blob_text="whats the price trend of rice in ghana from 2020 till 2023",
        primary_measures=["market_price"],
    )
    assert all(col.lower() != "price_type" for col, _val in filters)
    sql = compile_sql_from_bind(
        {
            "table_id": "agg_prices_country_month",
            "measure_columns": ["price_avg"],
            "required_filters_sql": "country_name = 'Ghana' AND year BETWEEN 2020 AND 2023",
            "time_sql": "year BETWEEN 2020 AND 2023",
            "filter_provenance": {"time": "question"},
        },
        project_id="proj",
        dataset="mart_dev",
        job="trend",
    )
    assert sql is not None
    assert "price_type = 'Retail'" not in sql
    assert "price_type" in sql.split("FROM", 1)[0]
    assert "BETWEEN 2020 AND 2023" in sql
    assert "SUM(" not in sql.upper()


def test_named_retail_still_pins_price_type() -> None:
    filters = compile_measure_filters(
        "agg_prices_country_month",
        measure_blob_text="retail rice price in ghana",
        primary_measures=["market_price"],
    )
    assert ("price_type", "Retail") in filters
