"""Normalized BQ SQL plan metadata: plan source and retrieve mode."""
from __future__ import annotations

from typing import Any, Literal

PlanSource = Literal[
    "turn_requirements",
    "class_engine",
    "slot_reasoner",
    "analytical",
    "retrieval_contract",
    "compile_error",
    "engine",
]
RetrieveMode = Literal["planned", "legacy"]


def derive_retrieve_mode(plan: dict[str, Any]) -> RetrieveMode:
    """Infer planned (NL2SQL-only) vs legacy (template/pattern) retrieve path."""
    bind = plan.get("bind_contracts")
    if isinstance(bind, dict) and bind:
        return "planned"
    if plan.get("nl2sql_fallback"):
        return "planned"
    src = str(plan.get("sql_source") or "").strip().lower()
    if src == "bind_contract":
        return "planned"
    if src == "compile_error" and plan.get("nl2sql_fallback"):
        return "planned"
    if str(plan.get("plan_source") or "").strip() == "class_engine":
        return "planned"
    return "legacy"


def normalize_bq_plan(
    plan: dict[str, Any],
    *,
    plan_source: PlanSource | None = None,
) -> dict[str, Any]:
    """Fill plan_source / retrieve_mode without dropping existing plan keys."""
    out = dict(plan)
    if plan_source is not None:
        out["plan_source"] = plan_source
    elif not out.get("plan_source"):
        src = str(out.get("sql_source") or "").strip().lower()
        if src == "engine":
            out["plan_source"] = "engine"
        elif out.get("slot_path"):
            out["plan_source"] = "slot_reasoner"
        elif out.get("analytical_mode"):
            out["plan_source"] = "analytical"
        elif out.get("compile_error"):
            out["plan_source"] = "compile_error"
    if not out.get("retrieve_mode"):
        out["retrieve_mode"] = derive_retrieve_mode(out)
    return out


__all__ = [
    "PlanSource",
    "RetrieveMode",
    "derive_retrieve_mode",
    "normalize_bq_plan",
]
