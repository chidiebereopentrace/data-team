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
        return max(1, min(int(os.environ.get("RAG_CLASS_ENGINE_CONCURRENCY", "4") or 4), 6))
    except ValueError:
        return 4


def run_class_engines(
    query: str,
    *,
    supervisor_plan: SupervisorPlan,
    facets: dict[str, Any],
) -> list[EngineResult]:
    """Plan SQL for each class; all classes run (no silent deferral)."""
    codes = list(supervisor_plan.classes) + list(supervisor_plan.secondary)
    if not codes:
        return []
    max_workers = min(_max_concurrent_nl2sql(), len(codes))
    results: list[EngineResult] = []

    def _run(code: str) -> EngineResult:
        engine = engine_for_class(code)
        card = load_schema_card(code) or {}
        return engine.run_plan(query, facets=facets, card=card)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_run, c): c for c in codes}
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
    order = {c: i for i, c in enumerate(codes)}
    results.sort(key=lambda r: order.get(r.class_code, 999))
    return results


def engine_results_to_bq_plan(
    results: list[EngineResult],
    *,
    rationale: str = "class_engines",
) -> dict[str, Any]:
    """Convert engine plans into bq_sql_plan for template/pattern/NL2SQL retrieve."""
    selected: list[str] = []
    intents: list[dict[str, Any]] = []
    table_hints: list[str] = []
    bind_contracts: dict[str, Any] = {}
    debug: list[dict[str, Any]] = []
    value_hits_all: dict[str, Any] = {}
    hints_truncated = False
    has_planned = False

    for er in results:
        if er.status == "deferred":
            debug.append({**er.to_dict(), "sql": None})
            continue

        value_hits_all[er.class_code] = er.value_hits
        table_id = str(er.table_id or "").strip()

        if er.bind_contract and table_id:
            bind_contracts[table_id] = er.bind_contract
        for hint in er.table_hints or []:
            if hint and hint not in table_hints:
                table_hints.append(hint)
        hints_truncated = hints_truncated or bool(er.hints_truncated)

        for intent in er.query_intents or []:
            if isinstance(intent, dict):
                intents.append(intent)
                for t in intent.get("tables") or []:
                    tid = str(t).strip().split(".")[-1]
                    if tid and tid not in selected:
                        selected.append(tid)

        if table_id and table_id not in selected:
            selected.append(table_id)

        if er.status == "planned":
            has_planned = True
            debug.append(
                {
                    **er.to_dict(),
                    "sql": None,
                    "status": "planned",
                    "sql_source": "bind_contract",
                }
            )
        elif er.status not in ("ready",) or er.sql:
            debug.append({**er.to_dict(), "sql": er.sql})

    skip_bq = not selected and not has_planned and not intents

    return {
        "selected_tables": selected,
        "query_intents": intents,
        "skip_bq": skip_bq,
        "rationale": rationale,
        "engine_results": [r.to_dict() for r in results],
        "bq_sql_queries": [],
        "bq_sql_debug": debug,
        "value_hits": value_hits_all,
        "table_hints": table_hints,
        "hints_truncated": hints_truncated,
        "bind_contracts": bind_contracts,
        "sql_source": "bind_contract" if has_planned else None,
        "nl2sql_fallback": has_planned,
    }


__all__ = ["run_class_engines", "engine_results_to_bq_plan"]
