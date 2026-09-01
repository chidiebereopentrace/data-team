"""Card-driven empty-retrieval policy — generate_weak vs typed_gap."""
from __future__ import annotations

import os
from typing import Any, Literal

from ml.rag.chatbot.retrieval_evidence import has_usable_narrative_context, warehouse_was_attempted
from ml.rag.chatbot.generator import filter_context_items, is_usable_structured_bq_row
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.turn_contract import NON_RAG_JOBS, TurnContract

EmptyPolicy = Literal["typed_gap", "narrative_ok", "web_ok", "generate_weak"]

PRODUCT_DEFAULT_EMPTY_POLICY: EmptyPolicy = "generate_weak"

_BROKEN_RETRIEVE_FLAGS = frozenset(
    {
        "structured_bq_never_executed",
        "structured_bq_validation_failed",
        "structured_bq_compile_error",
    }
)


def parse_empty_policy(raw: str | None) -> EmptyPolicy:
    val = (raw or "").strip().lower()
    if val in ("typed_gap", "narrative_ok", "web_ok", "generate_weak"):
        return val  # type: ignore[return-value]
    return PRODUCT_DEFAULT_EMPTY_POLICY


def empty_policy_for_class(class_code: str) -> EmptyPolicy:
    card = load_schema_card(class_code)
    if not card:
        return PRODUCT_DEFAULT_EMPTY_POLICY
    return parse_empty_policy(str(card.get("empty_policy") or ""))


def primary_class_from_state(state: dict[str, Any]) -> str:
    sp = state.get("supervisor_plan")
    if isinstance(sp, dict):
        classes = sp.get("classes")
        if isinstance(classes, list) and classes:
            return str(classes[0]).strip().upper()
    mid = str(state.get("measure_id") or "").strip()
    if mid:
        from ml.rag.chatbot.agri_measure_ontology import MEASURES

        spec = MEASURES.get(mid)
        if spec and spec.indicator_classes:
            return str(spec.indicator_classes[0]).strip().upper()
    return ""


def resolve_empty_policy(state: dict[str, Any]) -> EmptyPolicy:
    task_mode = str(state.get("task_mode") or "").strip().lower()
    if task_mode in ("help", "social", "clarify", "chat"):
        return "typed_gap"
    code = primary_class_from_state(state)
    if code:
        return empty_policy_for_class(code)
    return PRODUCT_DEFAULT_EMPTY_POLICY


def retrieval_was_executed(state: dict[str, Any]) -> bool:
    if warehouse_was_attempted(state):
        return True
    vector_keys = (
        "vector_news_results",
        "vector_academic_papers_results",
        "vector_policies_results",
        "vector_public_reports_results",
        "vector_ota_results",
        "vector_formation_results",
    )
    for key in vector_keys:
        if key in state:
            return True
    if state.get("web_results") is not None:
        return True
    if state.get("merged_context") is not None or state.get("reranked_context") is not None:
        return True
    return False


def has_usable_evidence(
    *,
    context_items: list[dict[str, Any]],
    bq_results: list[dict[str, Any]] | None = None,
) -> bool:
    if any(is_usable_structured_bq_row(r) for r in (bq_results or []) if isinstance(r, dict)):
        return True
    usable = filter_context_items(context_items or [])
    if has_usable_narrative_context(usable):
        return True
    for item in usable:
        if str(item.get("_context_kind") or "").lower() == "bigquery":
            return True
    return False


def execute_miss_flag(exec_flags: dict[str, bool]) -> str:
    for key in (
        "structured_bq_timed_out",
        "structured_bq_empty",
        "structured_bq_never_executed",
        "structured_bq_validation_failed",
        "structured_bq_compile_error",
        "structured_bq_unavailable",
    ):
        if exec_flags.get(key):
            return key.replace("structured_bq_", "")
    return "empty_result"


def broken_retrieve(exec_flags: dict[str, bool]) -> bool:
    return any(exec_flags.get(k) for k in _BROKEN_RETRIEVE_FLAGS)


def weak_after_retrieval_failure_enabled() -> bool:
    return os.environ.get("RAG_WEAK_AFTER_RETRIEVAL_FAILURE", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def should_generate_weak(
    state: dict[str, Any],
    *,
    policy: EmptyPolicy | None = None,
    context_items: list[dict[str, Any]] | None = None,
    exec_flags: dict[str, bool] | None = None,
    turn_contract: TurnContract | None = None,
) -> bool:
    pol = policy or resolve_empty_policy(state)
    if pol != "generate_weak":
        return False
    tc = turn_contract
    if tc is None:
        raw = state.get("turn_contract")
        if isinstance(raw, dict):
            tc = TurnContract.from_dict(raw)
    if tc is not None:
        if tc.job in NON_RAG_JOBS or tc.serve_status == "clarify":
            return False
    if not retrieval_was_executed(state):
        return False
    flags = exec_flags or {}
    if broken_retrieve(flags) and not weak_after_retrieval_failure_enabled():
        return False
    bq_results = state.get("bq_results") if isinstance(state.get("bq_results"), list) else []
    if has_usable_evidence(context_items=context_items or [], bq_results=bq_results):
        return False
    return True


def build_filter_miss_block(
    *,
    query: str,
    decomposition: dict[str, Any] | None,
    turn_contract: TurnContract | None,
    exec_flags: dict[str, bool] | None = None,
    class_code: str = "",
) -> str:
    dec = decomposition if isinstance(decomposition, dict) else {}
    lines = ["[No federated evidence for this filter]"]
    geo = dec.get("geography") or dec.get("expanded_regions") or []
    if isinstance(geo, list) and geo:
        lines.append(f"Geography: {', '.join(str(g) for g in geo[:8])}")
    elif dec.get("africa_default") or dec.get("africa_panel"):
        lines.append("Geography: Africa (continental scope)")
    entities = dec.get("entities")
    if isinstance(entities, list) and entities:
        lines.append(f"Entities: {', '.join(str(e) for e in entities[:8])}")
    ts = str(dec.get("time_start") or "")[:10]
    te = str(dec.get("time_end") or "")[:10]
    if ts or te:
        lines.append(f"Time: {ts or '?'} – {te or '?'}")
    if turn_contract is not None:
        lines.append(f"Measure: {turn_contract.measure_id or 'unknown'}")
        lines.append(f"Job: {turn_contract.job}")
        lines.append(f"Geo grain: {turn_contract.geo_grain}")
    if class_code:
        lines.append(f"Indicator class: {class_code}")
    status = execute_miss_flag(exec_flags or {})
    lines.append(f"Execute status: {status}")
    lines.append(f"Question: {(query or '').strip()[:400]}")
    return "\n".join(lines)


__all__ = [
    "EmptyPolicy",
    "PRODUCT_DEFAULT_EMPTY_POLICY",
    "build_filter_miss_block",
    "broken_retrieve",
    "empty_policy_for_class",
    "execute_miss_flag",
    "has_usable_evidence",
    "parse_empty_policy",
    "primary_class_from_state",
    "resolve_empty_policy",
    "retrieval_was_executed",
    "should_generate_weak",
    "weak_after_retrieval_failure_enabled",
]
