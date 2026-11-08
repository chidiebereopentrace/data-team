"""Single coverage retry after merge/rerank when required slots are missing."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.facet_coverage import missing_required_facets
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.web_policy import primary_class_from_state


def _decomposition(state: dict[str, Any]) -> dict[str, Any]:
    dec: dict[str, Any] = {}
    raw = state.get("decomposition")
    if isinstance(raw, dict):
        dec = raw
    return dec


def _as_dict(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        out = value
    return out


def required_geos(state: dict[str, Any]) -> list[str]:
    dec = _decomposition(state)
    geos: list[str] = []
    raw_geos = dec.get("geography")
    if not isinstance(raw_geos, list):
        return geos
    for g in raw_geos:
        s = str(g).strip()
        if s:
            geos.append(s)
    return geos


def required_entity_families(state: dict[str, Any]) -> list[str]:
    dec = _decomposition(state)
    out: list[str] = []
    for key in ("entities", "primary_measures"):
        raw = dec.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            s = str(item).strip().lower()
            if s:
                out.append(s)
    return list(dict.fromkeys(out))


def _blob_text(state: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "bq_results",
        "reranked_context",
        "merged_context",
        "vector_results",
    ):
        raw = state.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or ""))
    return "\n".join(parts).casefold()


def slots_missing_from_evidence(state: dict[str, Any]) -> bool:
    """True when a requested geography, or every requested entity, is absent."""
    if not _blob_text(state).strip():
        return True
    return bool(missing_required_facets(state))


def should_coverage_retry(state: dict[str, Any]) -> bool:
    if int(state.get("coverage_retry") or 0) >= 1:
        return False
    if state.get("early_short_circuit") or state.get("skipped_retrieval"):
        return False
    route = str(state.get("route_candidate") or "")
    if route and route != "full_rag":
        return False
    return slots_missing_from_evidence(state)


def should_retry_bq(state: dict[str, Any]) -> bool:
    """Second BQ only when measurable path and first job empty/timeout."""
    plan = _as_dict(state.get("bq_sql_plan"))
    if plan.get("skip_bq"):
        return False
    rows = state.get("bq_results")
    if isinstance(rows, list) and rows:
        return False
    debug = state.get("bq_sql_debug")
    if (
        not (isinstance(debug, list) and debug)
        and not plan.get("selected_tables")
        and not plan.get("bind_contracts")
    ):
        return False
    # Prefer card warehouse_role when present
    code = primary_class_from_state(state)
    card = load_schema_card(code) if code else None
    role = str((card or {}).get("warehouse_role") or "measurable").strip().lower()
    return role in ("measurable", "")


__all__ = [
    "required_entity_families",
    "required_geos",
    "should_coverage_retry",
    "should_retry_bq",
    "slots_missing_from_evidence",
]
