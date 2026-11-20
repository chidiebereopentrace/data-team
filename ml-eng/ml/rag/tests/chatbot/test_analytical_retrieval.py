"""A request is one operation, compiled for every class and serving table it matches."""

from __future__ import annotations

import pytest

from ml.rag.chatbot.bq_context_enrich import resolve_row_semantics
from ml.rag.chatbot.bq_sql_validate import validate_sql_table_allowlist
from ml.rag.chatbot.bq_table_schema_yaml import (
    compile_measure_filters,
    compile_table_bind_contract,
    compile_time_filter_sql,
    value_samples_for_table,
    year_column,
)
from ml.rag.chatbot.class_supervisor import compile_supervisor_plan
from ml.rag.chatbot.composer import is_sentinel_row
from ml.rag.chatbot.compile_sql_from_bind import (
    RESULT_ROW_BUDGET,
    compile_operation,
    compile_sql_from_bind,
    requested_operation,
)
from ml.rag.chatbot.measure_vocab import accept_names
from ml.rag.chatbot.retrieval_contract import choose_agg_vs_fact
from ml.rag.chatbot.value_index import complete_enum
from ml.rag.chatbot.coverage_ledger import operation_supported, serving_tables
from ml.rag.helpers.mart_semantic_relationships import relationships_for


@pytest.mark.parametrize(
    ("query", "operation"),
    [
        ("maize production in Kenya in 2020", "lookup"),
        ("latest maize production in Kenya", "latest"),
        ("maize production trend in Kenya since 2015", "series"),
        ("cassava production in Ghana from 2015 till date", "series"),
        ("rice production in Nigeria since 2015", "series"),
        ("compare maize production in Kenya and Ghana", "compare"),
        ("which country has the highest maize production", "rank"),
        ("maize production by crop in Kenya", "categorical"),
    ],
)
def test_requested_operation_reads_the_question_once(query: str, operation: str) -> None:
    assert requested_operation(query) == operation


def test_two_named_classes_both_run() -> None:
    plan = compile_supervisor_plan(
        "Compare agricultural employment and greenhouse gas emissions in Kenya and Ghana"
    )
    codes = set(plan.classes) | set(plan.secondary)
    assert "EL" in codes
    assert "ENV" in codes


def test_time_key_matches_the_literal_year_and_dim_time() -> None:
    assert year_column("fct_land_use") == "time_key"
    sql = compile_time_filter_sql(
        "fct_land_use", year=2020, project_id="p", dataset="mart_dev"
    )
    assert "time_key = 2020" in sql
    assert "dim_time" in sql
    assert "WHERE year = 2020" in sql


def test_year_column_stays_an_equality() -> None:
    assert year_column("fct_employment") == "year"
    sql = compile_time_filter_sql(
        "fct_employment", year=2020, project_id="p", dataset="mart_dev"
    )
    assert sql.strip() == "AND year = 2020"
    assert "dim_time" not in sql


def test_monthly_food_security_planned_year_filter_uses_allowed_dimension() -> None:
    table = "agg_food_security_monthly"
    where = compile_time_filter_sql(table, year=2020, project_id="p", dataset="mart_dev")
    assert "dim_time" in where
    sql = f"SELECT time_key FROM `p.mart_dev.{table}` WHERE TRUE {where}"
    assert validate_sql_table_allowlist(sql, {table}) is None


def test_as_of_date_window_is_unchanged() -> None:
    sql = compile_time_filter_sql(
        "fct_air_quality",
        time_start="2020-01-01",
        time_end="2020-12-31",
        project_id="p",
        dataset="mart_dev",
    )
    assert "as_of_date BETWEEN DATE '2020-01-01' AND DATE '2020-12-31'" in sql
    assert "dim_time" not in sql


def _contract(table: str, operation: str):
    geos = ["Kenya", "Ghana"] if operation == "compare" else ["Kenya"]
    contract = compile_table_bind_contract(
        table,
        facets={"geography": geos, "time_start": "2020-01-01", "time_end": "2020-12-31"},
        query=f"{operation} value in Kenya",
        country_labels=["KEN", "GHA"] if operation == "compare" else ["KEN"],
    )
    return contract


@pytest.mark.parametrize("operation", ["compare", "series", "rank"])
def test_supported_operation_compiles_with_declared_dimensions(operation: str) -> None:
    table = next(
        (
            name
            for name in serving_tables()
            if name.startswith(("fct_", "agg_")) and operation_supported(name, operation)[0]
        ),
        None,
    )
    assert table, operation
    compiled = compile_operation(
        _contract(table, operation),
        operation=operation,  # type: ignore[arg-type]
        project_id="p",
        dataset="mart_dev",
        limit=1,
    )
    assert compiled is not None and compiled.sql, compiled.refusal if compiled else "no compile"
    sql = compiled.sql or ""
    upper = sql.upper()
    if operation == "rank":
        assert "ORDER BY" in upper and "DESC" in upper
    else:
        assert f"LIMIT {RESULT_ROW_BUDGET + 1}" in upper
    declared = [
        str(spec.get("table") or "")
        for spec in (relationships_for(table).get("joins_with") or [])
        if isinstance(spec, dict) and str(spec.get("table") or "").startswith(("dim_", "bridge_"))
    ]
    if declared:
        assert any(dim in sql for dim in declared)


def test_name_list_is_kept_only_when_it_matches_the_distinct_count() -> None:
    assert accept_names(["a", "b"], 2) == ["a", "b"]
    assert accept_names(["a"], 2) is None
    assert accept_names([], 0) is None


def test_employment_name_past_the_yaml_cap_is_bound() -> None:
    preview = set(value_samples_for_table("fct_employment").get("indicator") or [])
    held = complete_enum("fct_employment", "indicator")
    extra = [name for name in held if name not in preview]
    assert extra
    filters = compile_measure_filters("fct_employment", measure_blob_text=extra[0])
    bound = [value for col, value in filters if col == "indicator"]
    assert bound
    assert bound[0] not in preview


def test_wide_fact_aliases_the_named_column() -> None:
    contract = compile_table_bind_contract(
        "fct_household",
        facets={"geography": ["Kenya"]},
        query="what is the farm income of households in Kenya",
        country_labels=["KEN"],
        project_id="p",
        dataset="mart_dev",
    )
    assert contract.measure_columns == ["farm_income"]
    sql = compile_sql_from_bind(contract, project_id="p", dataset="mart_dev")
    assert sql is not None
    assert "farm_income AS value" in sql


def test_country_question_uses_the_national_rollup() -> None:
    routed = choose_agg_vs_fact(
        "fct_employment",
        query="agricultural employment in Kenya",
        multi_country=False,
        year_hint="2020",
        single_country=True,
    )
    assert routed == "agg_employment_country_year"


def test_fnid_fact_is_not_summed_into_a_country_total() -> None:
    contract = compile_table_bind_contract(
        "fct_yield",
        facets={"geography": ["Kenya"], "time_start": "2020-01-01", "time_end": "2020-12-31"},
        query="maize yield in Kenya in 2020",
        country_labels=["KEN"],
        project_id="p",
        dataset="mart_dev",
    )
    assert "geo_level = 'country'" not in contract.required_filters_sql.lower()
    sql = compile_sql_from_bind(contract, project_id="p", dataset="mart_dev")
    assert sql is not None
    assert "SUM(" not in sql.upper()


def test_named_month_and_season_are_filtered() -> None:
    month_sql = compile_time_filter_sql(
        "fct_food_security",
        year=2020,
        query="food security in Kenya in March 2020",
        project_id="p",
        dataset="mart_dev",
    )
    assert "month = 3" in month_sql
    assert "2020" in month_sql
    season_sql = compile_time_filter_sql(
        "fct_yield",
        year=2020,
        query="maize yield in the long rains",
        project_id="p",
        dataset="mart_dev",
    )
    assert "dim_season" in season_sql
    assert "Long" in season_sql


def test_measure_label_uses_the_bound_name_or_the_column_description() -> None:
    bound = resolve_row_semantics(
        {"value": 1.0},
        table_id="fct_employment",
        sql="SELECT value FROM t WHERE indicator = 'Total employment in agrifood systems (AFS)'",
    )
    assert "agrifood" in str(bound["measure_label"]).lower()
    named = resolve_row_semantics(
        {"value": 10.0},
        table_id="fct_household",
        sql="SELECT farm_income AS value FROM t",
    )
    assert str(named["measure_label"]).lower() == "farm income"


def test_empty_result_stays_out_of_evidence() -> None:
    assert is_sentinel_row({"metadata": {"status": "empty_result"}, "content": "[BQ empty_result]"})


def test_open_span_compiles_between_start_and_coverage() -> None:
    from ml.rag.chatbot.active_request import extract_turn_slots

    slots = extract_turn_slots("cassava production in Ghana from 2015 till date")
    assert "2015" in str(slots.get("time_phrase"))
    assert "till date" in str(slots.get("time_phrase"))
    contract = compile_table_bind_contract(
        "agg_production_country_year",
        facets={
            "geography": ["Ghana"],
            "entities": ["cassava"],
            "time_start": "2015-01-01",
            "time_end": "2026-12-31",
            "primary_measures": ["production"],
        },
        query="what is the production trend of cassava in ghana from 2015 till date",
        country_labels=["GHA"],
        project_id="p",
        dataset="mart_dev",
    )
    assert "BETWEEN 2015 AND 2024" in contract.required_filters_sql or "BETWEEN 2015 AND 2024" in (
        contract.time_sql or ""
    )
    assert "year = 2015" not in contract.required_filters_sql
    assert "Cassava, fresh" in contract.required_filters_sql


def test_west_africa_does_not_bind_admin_2() -> None:
    from ml.rag.chatbot.bq_table_schema_yaml import match_subnational_geo_labels
    from ml.rag.chatbot.geo_regions import detect_regions_in_text, expand_regions_in_decomposition

    assert match_subnational_geo_labels("rice production for all west african countries") == []
    assert "west africa" in detect_regions_in_text("list rice for west african countries")
    out = expand_regions_in_decomposition({"geography": []}, "rice for west african countries")
    assert "Ghana" in out.get("geography", [])
    assert "Nigeria" in out.get("geography", [])


def test_african_countries_panel_expands() -> None:
    from ml.rag.chatbot.continental_scope import wants_africa_panel_scope
    from ml.rag.chatbot.geo_regions import expand_regions_in_decomposition

    assert wants_africa_panel_scope("food price in Nigeria compared with other african countries")
    out = expand_regions_in_decomposition(
        {"geography": ["Nigeria"]},
        "food price in Nigeria compared with other african countries since 2023",
    )
    assert len(out.get("geography") or []) > 10


def test_gdp_and_food_security_are_two_series() -> None:
    plan = compile_supervisor_plan(
        "how does the gdp of nigeria affect the food security from 2020 till date"
    )
    codes = set(plan.classes) | set(plan.secondary)
    assert "EL" in codes
    assert "FS" in codes
    assert "HDI" not in codes
