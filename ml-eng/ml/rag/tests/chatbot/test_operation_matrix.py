"""Every class compiles every operation its ledger declares, through the registry-driven compiler.

Requests are fixture requests (Kenya / Ghana, 2015-2020); nothing here asserts warehouse values.
"""

from __future__ import annotations

from typing import Any

import pytest

from ml.rag.chatbot.bq_sql_validate import validate_sql_column_allowlist
from ml.rag.chatbot.bq_table_schema_yaml import (
    TableBindContract,
    compile_table_bind_contract,
    measure_columns,
    measure_columns_mart,
    year_column,
)
from ml.rag.chatbot.compile_sql_from_bind import RESULT_ROW_BUDGET, compile_operation
from ml.rag.chatbot.coverage_ledger import OPERATIONS, load_ledger
from ml.rag.chatbot.sql_ast_validate import parse_select

# operation -> (geographies, iso3 labels, time_start, time_end, job)
_REQUEST: dict[str, tuple[list[str], list[str], str | None, str | None, str]] = {
    "lookup": (["Kenya"], ["KEN"], "2020-01-01", "2020-12-31", "fact"),
    "latest": (["Kenya"], ["KEN"], None, None, "fact"),
    "series": (["Kenya"], ["KEN"], "2015-01-01", "2020-12-31", "trend"),
    "compare": (["Kenya", "Ghana"], ["KEN", "GHA"], "2020-01-01", "2020-12-31", "compare"),
    "rank": ([], [], "2020-01-01", "2020-12-31", "rank"),
    "categorical": (["Kenya"], ["KEN"], "2020-01-01", "2020-12-31", "breakdown"),
}


def _cases() -> list[Any]:
    ledger = load_ledger()
    objects = ledger.get("objects") or {}
    cases = []
    for code, cls in sorted((ledger.get("classes") or {}).items()):
        for table in cls.get("serving_tables") or []:
            entry = objects.get(table) or {}
            if entry.get("readiness") not in ("fixture_verified", "integration_verified"):
                continue
            for op in OPERATIONS:
                if (entry.get("operations") or {}).get(op, {}).get("supported"):
                    cases.append(pytest.param(table, op, id=f"{code}-{table}-{op}"))
    return cases


def _contract(table: str, op: str) -> TableBindContract:
    geos, isos, start, end, _job = _REQUEST[op]
    facets = {"geography": geos, "time_start": start, "time_end": end}
    when = f" from {start[:4]} to {end[:4]}" if start and end and start[:4] != end[:4] else (f" in {start[:4]}" if start else "")
    contract = compile_table_bind_contract(
        table,
        facets=facets,
        query=f"value in {' and '.join(geos) or 'Africa'}{when}",
        country_labels=isos,
    )
    if not contract.measure_columns:
        contract = TableBindContract.from_dict(
            {**contract.to_dict(), "measure_columns": measure_columns_mart(table) or measure_columns(table)}
        ) or contract
    return contract


def test_every_class_has_operations_under_test() -> None:
    covered = {str(c.id).split("-")[0] for c in _cases()}
    assert covered == set(load_ledger()["classes"])


@pytest.mark.parametrize(("table", "op"), _cases())
def test_declared_operation_compiles_and_passes_the_gates(table: str, op: str) -> None:
    contract = _contract(table, op)
    compiled = compile_operation(contract, job=_REQUEST[op][4], project_id="p", dataset="mart_dev")
    assert compiled is not None
    assert compiled.sql, compiled.refusal
    sql = compiled.sql
    _root, err = parse_select(sql)
    assert err is None, err
    assert validate_sql_column_allowlist(sql) is None
    upper = sql.upper()
    if op in ("series", "compare") and year_column(table):
        assert f"LIMIT {RESULT_ROW_BUDGET + 1}" in upper
    if op == "rank":
        assert "DESC" in upper
    if op == "compare" and contract.geo_literals:
        assert all(iso in sql for iso in ("KEN", "GHA")) or all(n in sql for n in ("Kenya", "Ghana"))
    if op == "categorical":
        assert "GROUP BY" in upper


def test_unsupported_operation_is_refused_with_the_ledger_reason() -> None:
    contract = _contract("agg_hdi_latest", "series")
    compiled = compile_operation(contract, job="trend", project_id="p", dataset="mart_dev")
    assert compiled is not None and compiled.sql is None
    assert compiled.operation == "series"
    assert "not supported on agg_hdi_latest" in compiled.refusal
