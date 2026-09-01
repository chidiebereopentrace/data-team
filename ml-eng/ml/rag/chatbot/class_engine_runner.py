"""Run class engines for supervisor plan (plan-only, no execute)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ml.rag.chatbot.class_engines.registry import engine_for_class
from ml.rag.chatbot.class_engines.base import EngineResult
from ml.rag.chatbot.class_supervisor import SupervisorPlan
from ml.rag.chatbot.schema_card import load_schema_card


def _max_concurrent_nl2sql() -> int:
    try:
        return max(1, min(int(os.environ.get("RAG_CLASS_ENGINE_CONCURRENCY", "2") or 2), 4))
    except ValueError:
        return 2


def run_class_engines(
    query: str,
    *,
    supervisor_plan: SupervisorPlan,
    facets: dict[str, Any],
) -> list[EngineResult]:
    """Plan SQL for each class; max 2 concurrent, remainder deferred."""
    codes = list(supervisor_plan.classes) + list(supervisor_plan.secondary)
    if not codes:
        return []
    max_workers = _max_concurrent_nl2sql()
    active = codes[:max_workers]
    deferred = codes[max_workers:]
    results: list[EngineResult] = []

    def _run(code: str) -> EngineResult:
        engine = engine_for_class(code)
        card = load_schema_card(code) or {}
        return engine.run_plan(query, facets=facets, card=card)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(active))) as pool:
        futs = {pool.submit(_run, c): c for c in active}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as exc:
                code = futs[fut]
                results.append(
                    EngineResult(
                        class_code=code,
                        status="planner_error",
                        table_id="",
                        sql=None,
                        caveats=[str(exc)[:200]],
                    )
                )

    for code in deferred:
        results.append(
            EngineResult(
                class_code=code,
                status="deferred",
                table_id="",
                sql=None,
                caveats=[f"deferred_pending_slot_cap_{max_workers}"],
            )
        )
    return results


def _iter_sql_entries(er: EngineResult) -> list[dict[str, Any]]:
    if er.sql_plans:
        return list(er.sql_plans)
    if er.sql and er.table_id:
        return [
            {
                "table_id": er.table_id,
                "sql": er.sql,
                "value_hits": er.value_hits,
                "status": er.status,
            }
        ]
    return []


def engine_results_to_bq_plan(
    results: list[EngineResult],
    *,
    rationale: str = "class_engines",
) -> dict[str, Any]:
    """Convert engine plans into bq_sql_plan shape for retrieve node."""
    selected: list[str] = []
    intents: list[dict[str, Any]] = []
    sql_queries: list[str] = []
    debug: list[dict[str, Any]] = []
    value_hits_all: dict[str, Any] = {}

    for er in results:
        if er.status == "deferred":
            debug.append({**er.to_dict(), "sql": None})
            continue
        entries = _iter_sql_entries(er)
        if not entries and er.status != "planned":
            debug.append({**er.to_dict(), "sql": er.sql})
            continue
        for entry in entries:
            table_id = str(entry.get("table_id") or er.table_id or "")
            sql = entry.get("sql")
            sub_status = str(entry.get("status") or er.status)
            sub_hits = entry.get("value_hits") if isinstance(entry.get("value_hits"), dict) else er.value_hits
            if table_id and table_id not in selected:
                selected.append(table_id)
            if sql:
                sql_queries.append(str(sql))
                debug.append(
                    {
                        "sql": sql,
                        "status": sub_status,
                        "class_code": er.class_code,
                        "table_id": table_id,
                        "value_hits": sub_hits,
                        "sql_source": "engine",
                        "family_id": entry.get("family_id"),
                        "role": entry.get("role"),
                    }
                )
            if sql and table_id:
                intents.append(
                    {
                        "goal": f"{er.class_code} from {table_id}",
                        "tables": [table_id],
                        "filters": "",
                        "notes": er.class_code,
                        "pattern": "custom",
                        "metric": "value",
                        "grain": [],
                        "order_by": "year DESC",
                    }
                )
        value_hits_all[er.class_code] = er.value_hits

    return {
        "selected_tables": selected,
        "query_intents": intents,
        "skip_bq": not sql_queries,
        "rationale": rationale,
        "engine_results": [r.to_dict() for r in results],
        "bq_sql_queries": sql_queries,
        "bq_sql_debug": debug,
        "value_hits": value_hits_all,
        "sql_source": "engine" if sql_queries else None,
    }


__all__ = ["run_class_engines", "engine_results_to_bq_plan"]
