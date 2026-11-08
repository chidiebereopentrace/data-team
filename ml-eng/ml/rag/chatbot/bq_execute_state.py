"""BigQuery execute-state flags and typed execution results shared across graph, generation and routing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

ExecutionStatus = Literal[
    "clarify",
    "validation_failed",
    "timeout",
    "exec_failed",
    "empty_ok",
    "partial",
    "result_too_large",
    "ok",
]

# Row-item metadata status (as emitted by the retriever) -> typed execution status.
_ROW_STATUS: dict[str, ExecutionStatus] = {
    "ok": "ok",
    "partial": "partial",
    "empty_result": "empty_ok",
    "result_too_large": "result_too_large",
    "validation_failed": "validation_failed",
    "no_valid_sql": "validation_failed",
    "compile_error": "validation_failed",
    "timeout": "timeout",
    "bq_timeout": "timeout",
    "execution_error": "exec_failed",
    "bq_error": "exec_failed",
    "no_project": "exec_failed",
    "clarify": "clarify",
}

EVIDENCE_STATUSES: frozenset[ExecutionStatus] = frozenset({"ok", "partial"})

NON_EVIDENCE_ROW_STATUSES = frozenset(
    raw for raw, typed in _ROW_STATUS.items() if typed not in EVIDENCE_STATUSES
)


def execution_status(row_status: str) -> ExecutionStatus:
    """Typed status for a retriever row status; rows without a status are data rows."""
    return _ROW_STATUS.get((row_status or "ok").strip().lower(), "ok")


def evidence_id_for_sql(sql: str) -> str:
    return "bq:" + hashlib.sha256(" ".join((sql or "").split()).encode()).hexdigest()[:12]


def rowset_hash(rows: list[dict[str, Any]]) -> str:
    canonical = sorted(json.dumps(r, sort_keys=True, default=str) for r in rows)
    return hashlib.sha256("\n".join(canonical).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ExecutionResult:
    """One executed (or refused) warehouse query and the rows it proved."""

    evidence_id: str
    status: ExecutionStatus
    sql: str
    rows: tuple[dict[str, Any], ...] = ()
    rowset_hash: str = ""
    job_id: str | None = None
    detail: str = ""
    result_shape: dict[str, Any] = field(default_factory=dict)

    @property
    def is_evidence(self) -> bool:
        return self.status in EVIDENCE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "status": self.status,
            "sql": self.sql,
            "row_count": len(self.rows),
            "rowset_hash": self.rowset_hash,
            "job_id": self.job_id,
            "detail": self.detail,
            "result_shape": dict(self.result_shape),
        }


_ROW_META_KEYS = frozenset(
    {
        "sql", "sql_index", "sql_count", "sql_source", "nl2sql_model", "status", "prep_error",
        "job_id", "bytes_processed", "bq_ms", "row_count", "result_shape", "evidence_id",
        "subquestion_id", "slot_id", "template", "pattern", "coverage_note", "value_semantics",
    }
)


def execution_results_from_items(items: list[dict[str, Any]]) -> list[ExecutionResult]:
    """Group retriever BigQuery items by SQL into typed results (one per query)."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items or []:
        if not isinstance(item, dict) or str(item.get("source") or "") != "bigquery":
            continue
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        sql = str(meta.get("sql") or "").strip()
        if sql:
            grouped.setdefault(sql, []).append(meta)
    out: list[ExecutionResult] = []
    for sql, metas in grouped.items():
        status = execution_status(str(metas[0].get("status") or ""))
        rows = (
            [{k: v for k, v in m.items() if k not in _ROW_META_KEYS} for m in metas]
            if status in EVIDENCE_STATUSES
            else []
        )
        shape = metas[0].get("result_shape")
        out.append(
            ExecutionResult(
                evidence_id=evidence_id_for_sql(sql),
                status=status,
                sql=sql,
                rows=tuple(rows),
                rowset_hash=rowset_hash(rows) if rows else "",
                job_id=metas[0].get("job_id"),
                detail=str(metas[0].get("prep_error") or ""),
                result_shape=shape if isinstance(shape, dict) else {},
            )
        )
    return out


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


_EXPLAINED_MISS_STATUSES = NON_EVIDENCE_ROW_STATUSES


def ensure_planned_attempt_debug(
    plan: dict[str, Any] | None,
    bq_sql_debug: list[dict[str, Any]],
    *,
    usable_bq: bool,
) -> list[dict[str, Any]]:
    """Leave a debug row when a warehouse plan never started a BigQuery job.

    The row carries compiled SQL when the plan has it, otherwise status
    ``compile_error``. An empty result or a job id already explains the attempt.
    """
    if usable_bq or not plan_indicates_warehouse_attempt(plan):
        return bq_sql_debug
    debug = [row for row in bq_sql_debug if isinstance(row, dict)]
    if any(row.get("job_id") for row in debug):
        return bq_sql_debug
    if any(
        str(row.get("status") or "") in _EXPLAINED_MISS_STATUSES
        or str(row.get("sql") or "").strip()
        for row in debug
    ):
        return bq_sql_debug
    sql = ""
    planned = (plan or {}).get("bq_sql_queries")
    if isinstance(planned, list):
        for raw in planned:
            text = str(raw or "").strip()
            if text:
                sql = text
                break
    row: dict[str, Any] = {
        "sql": sql,
        "status": "compile_error",
        "prep_error": "Warehouse intent was planned but no BigQuery job was started",
        "sql_source": str((plan or {}).get("plan_source") or "planned"),
    }
    return [*bq_sql_debug, row]


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
    "EVIDENCE_STATUSES",
    "ExecutionResult",
    "ExecutionStatus",
    "NON_EVIDENCE_ROW_STATUSES",
    "bq_execute_flags",
    "collect_pre_queries",
    "ensure_planned_attempt_debug",
    "evidence_id_for_sql",
    "execution_results_from_items",
    "execution_status",
    "plan_indicates_warehouse_attempt",
    "rowset_hash",
]


def warehouse_was_attempted(state: dict[str, Any]) -> bool:
    plan = state.get("bq_sql_plan")
    if plan_indicates_warehouse_attempt(plan if isinstance(plan, dict) else None):
        return True
    return bool(state.get("bq_sql_queries") or state.get("bq_sql_debug"))
