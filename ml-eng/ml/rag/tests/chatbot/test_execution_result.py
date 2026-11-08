"""Typed execution results, result-shape contract and filter provenance gate (fixture truth only)."""

from __future__ import annotations

from ml.rag.chatbot.acf_metadata import stamp_temporal_direction
from ml.rag.chatbot.bq_execute_state import (
    NON_EVIDENCE_ROW_STATUSES,
    evidence_id_for_sql,
    execution_results_from_items,
    execution_status,
)
from ml.rag.chatbot.bq_table_schema_yaml import TableBindContract
from ml.rag.chatbot.compile_sql_from_bind import compile_sql_from_bind
from ml.rag.chatbot.result_shape import check_result_shape

COMPARE = {
    "table_id": "fct_production",
    "geo_column": "country_iso3",
    "geo_literals": ["GHA", "NGA"],
    "measure_columns": ["value"],
}


def test_missing_compare_member_is_reported() -> None:
    rows = [{"country_iso3": "GHA", "year": 2020, "value": 1.0}]
    shape = check_result_shape(rows, COMPARE)
    assert not shape.complete
    assert shape.missing_members == {"country_iso3": ["NGA"]}


def test_rows_repeating_a_grain_key_are_counted() -> None:
    rows = [
        {"country_iso3": "GHA", "year": 2020, "value": 1.0},
        {"country_iso3": "GHA", "year": 2020, "value": 2.0},
        {"country_iso3": "NGA", "year": 2020, "value": 3.0},
    ]
    assert check_result_shape(rows, COMPARE).duplicate_grain_rows == 1


def test_complete_shape() -> None:
    rows = [
        {"country_iso3": "GHA", "year": 2020, "value": 1.0},
        {"country_iso3": "NGA", "year": 2020, "value": 3.0},
    ]
    assert check_result_shape(rows, COMPARE).complete


def test_statuses_map_onto_the_typed_vocabulary() -> None:
    assert execution_status("empty_result") == "empty_ok"
    assert execution_status("execution_error") == "exec_failed"
    assert execution_status("partial") == "partial"
    assert "partial" not in NON_EVIDENCE_ROW_STATUSES
    assert {"empty_result", "result_too_large", "bq_timeout"} <= NON_EVIDENCE_ROW_STATUSES


def test_items_group_into_results_with_stable_ids_and_hashes() -> None:
    sql = "SELECT country_iso3, year, value FROM t WHERE year = 2020"
    items = [
        {"source": "bigquery", "metadata": {"sql": sql, "status": "ok", "job_id": "j1", "country_iso3": "GHA", "value": 1.0}},
        {"source": "bigquery", "metadata": {"sql": sql, "status": "ok", "job_id": "j1", "country_iso3": "NGA", "value": 3.0}},
        {"source": "bigquery", "metadata": {"sql": "SELECT 2", "status": "empty_result"}},
        {"source": "news", "metadata": {"sql": "ignored"}},
    ]
    results = execution_results_from_items(items)
    assert [r.status for r in results] == ["ok", "empty_ok"]
    ok = results[0]
    assert ok.evidence_id == evidence_id_for_sql(sql)
    assert len(ok.rows) == 2 and ok.rowset_hash
    reordered = execution_results_from_items([items[1], items[0]])[0]
    assert reordered.rowset_hash == ok.rowset_hash
    assert not results[1].is_evidence and results[1].rows == ()


def test_derived_change_carries_its_source_evidence() -> None:
    out = stamp_temporal_direction({"total_curr": 120, "total_prev": 100, "evidence_id": "bq:abc"})
    assert out["magnitude"] is not None
    assert out["derived_from"] == ["bq:abc"]


def test_filters_without_provenance_are_refused() -> None:
    contract = TableBindContract(
        table_id="fct_production",
        geo_column="country_iso3",
        geo_literals=["GHA"],
        required_filters_sql="country_iso3 = 'GHA' AND admin_1_name = 'Ségou'",
        measure_columns=["value"],
        filter_provenance={"geo": "question"},
    )
    assert contract.unproven_filters() == ["subnational"]
    assert compile_sql_from_bind(contract, project_id="p", dataset="mart_dev") is None
