"""Typed warehouse gap messages — never generic insufficient when SQL was planned."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.bq_execute_state import bq_execute_flags
from ml.rag.chatbot.empty_answer_templates import empty_answer_for_contract
from ml.rag.chatbot.generator import filter_context_items, is_usable_structured_bq_row, normalize_context_kind
from ml.rag.chatbot.turn_contract import TurnContract

_NARRATIVE_KINDS = frozenset(
    {
        "news",
        "academic",
        "policy",
        "public_report",
        "ota_insight",
        "formation",
        "web_search",
        "web_wikipedia",
    }
)


def _scope_hint(decomposition: dict[str, Any] | None) -> str:
    if not isinstance(decomposition, dict):
        return "the requested filter"
    geo = decomposition.get("geography") or decomposition.get("countries") or []
    if isinstance(geo, list) and geo:
        places = ", ".join(str(g) for g in geo[:6])
        if len(geo) > 6:
            places = f"{places}, …"
        ts = str(decomposition.get("time_start") or "")[:4]
        te = str(decomposition.get("time_end") or "")[:4]
        if ts and te:
            return f"{places} × {ts}–{te}"
        return places
    ts = str(decomposition.get("time_start") or "")[:4]
    te = str(decomposition.get("time_end") or "")[:4]
    if ts and te:
        return f"{ts}–{te}"
    return "the requested filter"


def typed_bq_gap_answer(
    *,
    flag: str,
    decomposition: dict[str, Any] | None = None,
    turn_contract: dict[str, Any] | None = None,
    prep_error: str = "",
) -> str:
    scope = _scope_hint(decomposition)
    if flag == "structured_bq_timed_out":
        return (
            f"OpenTrace warehouse queries were submitted for {scope} but did not "
            "return rows within the time limit. Try a narrower geography, a single year, "
            "or fewer breakdown dimensions."
        )
    if flag == "structured_bq_empty":
        return (
            f"OpenTrace warehouse returned no rows for {scope} with the filters applied. "
            "This is a scoped empty result — not evidence that the data does not exist elsewhere."
        )
    if flag == "structured_bq_validation_failed":
        detail = f" Detail: {prep_error[:200]}." if prep_error else ""
        return (
            f"OpenTrace warehouse SQL for {scope} failed validation before execution.{detail} "
            "Try narrowing geography, time, or entity filters."
        )
    if flag == "structured_bq_never_executed":
        return (
            f"OpenTrace warehouse SQL was compiled for {scope} but was not submitted to "
            "the warehouse. This is a retrieval failure — not proof that sources are unavailable."
        )
    if isinstance(turn_contract, dict):
        tc = TurnContract.from_dict(turn_contract)
        rendered = empty_answer_for_contract(tc, query="", category="")
        if rendered:
            return rendered
    return (
        f"OpenTrace warehouse did not return usable structured rows for {scope}. "
        "Narrow geography, time, or entity filters and try again."
    )


def first_prep_error(bq_sql_debug: list[dict[str, Any]]) -> str:
    for row in bq_sql_debug:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "") == "validation_failed":
            prep = row.get("prep_error")
            if prep:
                return str(prep)
    for row in bq_sql_debug:
        if not isinstance(row, dict):
            continue
        prep = row.get("prep_error")
        if prep:
            return str(prep)
    return ""


def warehouse_was_attempted(state: dict[str, Any]) -> bool:
    plan = state.get("bq_sql_plan")
    if isinstance(plan, dict):
        if plan.get("bq_sql_queries") or plan.get("selected_tables") or plan.get("engine_results"):
            return True
        if not plan.get("skip_bq"):
            return bool(plan.get("query_intents"))
    if state.get("bq_sql_queries") or state.get("bq_sql_debug"):
        return True
    for row in state.get("bq_sql_debug") or []:
        if isinstance(row, dict) and str(row.get("sql") or "").strip():
            return True
    return False


def has_usable_narrative_context(context_items: list[dict[str, Any]]) -> bool:
    for item in filter_context_items(context_items):
        if normalize_context_kind(item) in _NARRATIVE_KINDS:
            return True
    return False


def warehouse_blocks_web(state: dict[str, Any]) -> bool:
    """Block web only to cover missing job_id / validation failure — not every BQ attempt."""
    plan = state.get("bq_sql_plan") if isinstance(state.get("bq_sql_plan"), dict) else {}
    pre_queries = list(plan.get("bq_sql_queries") or state.get("bq_sql_queries") or [])
    if not pre_queries:
        return False
    bq_debug = [
        d
        for d in list(state.get("bq_sql_debug") or []) + list(plan.get("bq_sql_debug") or [])
        if isinstance(d, dict)
    ]
    bq_results = state.get("bq_results") or []
    usable_bq = any(is_usable_structured_bq_row(r) for r in bq_results if isinstance(r, dict))
    flags = bq_execute_flags(bq_debug, pre_queries=pre_queries, usable_bq=usable_bq)
    if flags.get("structured_bq_never_executed") or flags.get("structured_bq_validation_failed"):
        return True
    if flags.get("structured_bq_timed_out") and not any(d.get("job_id") for d in bq_debug):
        return True
    if flags.get("structured_bq_empty"):
        return False
    if flags.get("structured_bq_timed_out") and usable_bq:
        return False
    return False


def should_hard_return_bq_gap(
    *,
    exec_flags: dict[str, bool],
    pre_queries: list[str],
    usable_bq: bool,
    context_items: list[dict[str, Any]],
    is_numeric_job: bool,
) -> bool:
    """Hard-return typed BQ gap only when no structured rows and no narrative escape hatch."""
    if not pre_queries or usable_bq:
        return False
    if not any(
        exec_flags.get(k)
        for k in (
            "structured_bq_timed_out",
            "structured_bq_validation_failed",
            "structured_bq_empty",
            "structured_bq_never_executed",
        )
    ):
        return False
    if exec_flags.get("structured_bq_empty") and not is_numeric_job:
        return False
    if has_usable_narrative_context(context_items):
        return False
    return True


def forbid_generic_insufficient(state: dict[str, Any]) -> bool:
    sp = state.get("supervisor_plan")
    if isinstance(sp, dict) and sp.get("forbid_insufficient_without_attempt"):
        return True
    plan = state.get("bq_sql_plan")
    if isinstance(plan, dict):
        sp2 = plan.get("supervisor_plan")
        if isinstance(sp2, dict) and sp2.get("forbid_insufficient_without_attempt"):
            return True
    return warehouse_was_attempted(state)
