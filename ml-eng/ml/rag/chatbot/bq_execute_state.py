"""BigQuery execute-state flags shared across graph, gap messages, and web routing."""
from __future__ import annotations

from typing import Any


def bq_execute_flags(
    bq_sql_debug: list[dict[str, Any]],
    *,
    pre_queries: list[str],
    usable_bq: bool,
) -> dict[str, bool]:
    debug = [d for d in bq_sql_debug if isinstance(d, dict)]
    has_engine_sql = bool(pre_queries)
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
    return {
        "structured_bq_timed_out": timed_out,
        "structured_bq_never_executed": never_executed,
        "structured_bq_empty": empty,
        "structured_bq_validation_failed": validation_failed,
        "structured_bq_unavailable": (
            not usable_bq
            and not never_executed
            and not timed_out
            and not empty
            and not validation_failed
        ),
    }


__all__ = ["bq_execute_flags"]
