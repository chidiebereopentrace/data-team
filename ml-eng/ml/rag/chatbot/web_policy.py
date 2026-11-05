"""Card-driven web policy: may web results stand in for an empty warehouse job?"""
from __future__ import annotations

from typing import Any, Literal

from ml.rag.chatbot.agri_measure_ontology import MEASURES
from ml.rag.chatbot.schema_card import load_schema_card

WebPolicy = Literal["never", "if_empty_job", "companion"]

PRODUCT_DEFAULT_WEB_POLICY: WebPolicy = "never"


def parse_web_policy(raw: str | None) -> WebPolicy:
    val = (raw or "").strip().lower()
    if val in ("never", "if_empty_job", "companion"):
        return val  # type: ignore[return-value]
    return PRODUCT_DEFAULT_WEB_POLICY


def web_policy_for_class(class_code: str) -> WebPolicy:
    card = load_schema_card(class_code)
    if not card:
        return PRODUCT_DEFAULT_WEB_POLICY
    return parse_web_policy(str(card.get("web_policy") or ""))


def primary_class_from_state(state: dict[str, Any]) -> str:
    sp = state.get("supervisor_plan")
    if isinstance(sp, dict):
        classes = sp.get("classes")
        if isinstance(classes, list) and classes:
            return str(classes[0]).strip().upper()
    spec = MEASURES.get(str(state.get("measure_id") or "").strip())
    if spec and spec.indicator_classes:
        return str(spec.indicator_classes[0]).strip().upper()
    return ""


def resolve_web_policy(state: dict[str, Any]) -> WebPolicy:
    code = primary_class_from_state(state)
    if code:
        return web_policy_for_class(code)
    return PRODUCT_DEFAULT_WEB_POLICY


__all__ = [
    "PRODUCT_DEFAULT_WEB_POLICY",
    "WebPolicy",
    "parse_web_policy",
    "primary_class_from_state",
    "resolve_web_policy",
    "web_policy_for_class",
]
