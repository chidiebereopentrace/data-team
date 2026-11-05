"""One answer to "what kind of evidence is this item?"."""
from __future__ import annotations

from typing import Any

WAREHOUSE = "bigquery"

_KIND_ALIASES = {
    "bigquery": WAREHOUSE,
    "bq": WAREHOUSE,
    "news": "news",
    "news_article": "news",
    "policy": "policy",
    "policies": "policy",
    "public": "public_report",
    "public_report": "public_report",
    "public_reports": "public_report",
    "academic": "academic",
    "academic_paper": "academic",
    "academic_papers": "academic",
    "research": "academic",
    "ota": "ota_insight",
    "ota_insight": "ota_insight",
    "formation": "formation",
    "wikipedia": "web",
    "web": "web",
    "web_search": "web",
    "web_wikipedia": "web",
    "tavily": "web",
}

NARRATIVE_KINDS = ("news", "policy", "public_report", "academic", "ota_insight", "formation", "web")


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("metadata")
    return raw if isinstance(raw, dict) else {}


def source_kind(item: dict[str, Any]) -> str:
    meta = _metadata(item)
    raw = str(
        item.get("_context_kind")
        or item.get("source")
        or meta.get("context_kind")
        or meta.get("source_kind")
        or meta.get("doc_kind")
        or ""
    ).strip().lower()
    kind = _KIND_ALIASES.get(raw, raw)
    if kind:
        return kind
    if str(item.get("content") or "").startswith("[Structured data]"):
        return WAREHOUSE
    return "other"


def is_warehouse_item(item: dict[str, Any]) -> bool:
    return source_kind(item) == WAREHOUSE


def is_document_item(item: dict[str, Any]) -> bool:
    return source_kind(item) in NARRATIVE_KINDS


__all__ = [
    "NARRATIVE_KINDS",
    "WAREHOUSE",
    "is_document_item",
    "is_warehouse_item",
    "source_kind",
]
