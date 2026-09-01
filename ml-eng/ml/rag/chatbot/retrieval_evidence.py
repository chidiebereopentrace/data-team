"""Shared retrieval-evidence helpers (no policy / gap-message dependencies)."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.generator import filter_context_items, normalize_context_kind

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


__all__ = ["has_usable_narrative_context", "warehouse_was_attempted"]
