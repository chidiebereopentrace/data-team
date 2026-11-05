"""BigQuery execute-state flags shared across graph, gap messages, and web routing."""
from __future__ import annotations

from typing import Any


def collect_pre_queries(
    plan: dict[str, Any] | None,
    *,
    state_queries: list[Any] | None = None,
    bq_sql_debug: list[Any] | None = None,
) -> list[str]:
    """SQL strings that prove a warehouse attempt — plan, post-retrieve, or debug.

    Bind-first planned path often leaves ``plan.bq_sql_queries`` empty until
    retrieve runs; flags must use post-retrieve SQL / debug rows.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: Any) -> None:
        text = str(raw or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)

    plan_dict = plan if isinstance(plan, dict) else {}
    for q in plan_dict.get("bq_sql_queries") or []:
        _add(q)
    for q in state_queries or []:
        _add(q)
    for row in bq_sql_debug or []:
        if isinstance(row, dict):
            _add(row.get("sql"))
    return out


def plan_indicates_warehouse_attempt(plan: dict[str, Any] | None) -> bool:
    """True when the BQ plan committed to warehouse work (bind or SQL)."""
    if not isinstance(plan, dict) or plan.get("skip_bq"):
        return False
    if plan.get("bind_contracts"):
        return True
    if str(plan.get("plan_source") or "").strip() == "class_engine":
        return True
    if plan.get("bq_sql_queries") or plan.get("selected_tables") or plan.get("engine_results"):
        return True
    if plan.get("query_intents"):
        return True
    if str(plan.get("retrieve_mode") or "").strip() == "planned":
        return True
    return False


def bq_execute_flags(
    bq_sql_debug: list[dict[str, Any]],
    *,
    pre_queries: list[str],
    usable_bq: bool,
    compile_error: bool = False,
    warehouse_attempted: bool = False,
) -> dict[str, bool]:
    debug = [d for d in bq_sql_debug if isinstance(d, dict)]
    has_engine_sql = bool(pre_queries) or (
        warehouse_attempted and any(str(d.get("sql") or "").strip() or d.get("job_id") for d in debug)
    )
    # Bind path with retrieve executed but no SQL string yet still counts as attempt
    # when debug rows exist (validation_failed / empty / timeout statuses).
    if warehouse_attempted and not has_engine_sql and debug:
        has_engine_sql = True
    any_job = any(d.get("job_id") for d in debug)
    any_timeout = any(str(d.get("status") or "") == "timeout" for d in debug)
    any_validation_failed = any(
        str(d.get("status") or "") == "validation_failed" for d in debug
    )
    never_executed = (
        has_engine_sql
        and not any_job
        and not usable_bq
        and not any_validation_failed
    )
    timed_out = has_engine_sql and any_timeout and any_job
    empty = has_engine_sql and any_job and not usable_bq and not any_timeout
    validation_failed = has_engine_sql and any_validation_failed and not any_job
    compile_err = compile_error and not usable_bq and not has_engine_sql
    return {
        "structured_bq_timed_out": timed_out,
        "structured_bq_never_executed": never_executed,
        "structured_bq_empty": empty,
        "structured_bq_validation_failed": validation_failed,
        "structured_bq_compile_error": compile_err,
        "structured_bq_unavailable": (
            not usable_bq
            and not never_executed
            and not timed_out
            and not empty
            and not validation_failed
            and not compile_err
        ),
    }


__all__ = [
    "bq_execute_flags",
    "collect_pre_queries",
    "plan_indicates_warehouse_attempt",
]


def warehouse_was_attempted(state: dict[str, Any]) -> bool:
    plan = state.get("bq_sql_plan")
    if plan_indicates_warehouse_attempt(plan if isinstance(plan, dict) else None):
        return True
    return bool(state.get("bq_sql_queries") or state.get("bq_sql_debug"))
