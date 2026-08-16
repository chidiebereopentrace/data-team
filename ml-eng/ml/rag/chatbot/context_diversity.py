"""Diversity-aware packing of reranked context across corpora / source kinds."""
from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from typing import Any

_KIND_ALIASES = {
    "bigquery": "bigquery",
    "bq": "bigquery",
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
    "ota": "ota_insight",
    "ota_insight": "ota_insight",
    "formation": "formation",
    "wikipedia": "web",
    "web": "web",
    "web_search": "web",
}

_NARRATIVE_KINDS = (
    "news",
    "policy",
    "public_report",
    "academic",
    "ota_insight",
    "formation",
    "web",
)


def _item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("metadata")
    return raw if isinstance(raw, dict) else {}


def normalize_context_kind(item: dict[str, Any]) -> str:
    raw = str(
        item.get("_context_kind")
        or item.get("source")
        or _item_metadata(item).get("doc_kind")
        or ""
    ).strip().lower()
    return _KIND_ALIASES.get(raw, raw or "other")


def _dedupe_key(item: dict[str, Any]) -> str:
    meta = _item_metadata(item)
    for key in ("url", "source_url", "canonical_url", "doi"):
        val = str(meta.get(key) or "").strip().lower()
        if val:
            return f"url:{val}"
    title = str(meta.get("title") or meta.get("headline") or "").strip().lower()
    title = re.sub(r"\s+", " ", title)
    if len(title) >= 12:
        return f"title:{title[:160]}"
    text = str(item.get("content") or item.get("text") or "").strip().lower()
    text = re.sub(r"\s+", " ", text)[:240]
    if text:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"text:{digest}"
    return f"id:{id(item)}"


def dedupe_context_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop near-duplicate chunks (same URL/title/content fingerprint); keep first."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items or []:
        key = _dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _pack_size_default(*, has_bq: bool, task_mode: str) -> int:
    mode = (task_mode or "chat").strip().lower()
    try:
        base = int(os.environ.get("RAG_DIVERSIFY_PACK_SIZE", "14") or 14)
    except ValueError:
        base = 14
    base = max(6, min(base, 24))
    if not has_bq and mode in ("chat", "briefing"):
        try:
            soft = int(os.environ.get("RAG_DIVERSIFY_PACK_SIZE_NO_BQ", str(base)) or base)
        except ValueError:
            soft = base
        return max(6, min(soft, base))
    return base


def diversify_context_pack(
    items: list[dict[str, Any]],
    *,
    top_k: int | None = None,
    task_mode: str = "chat",
    bq_max: int | None = None,
    per_kind_min: int | None = None,
) -> list[dict[str, Any]]:
    """
    Pack scored/reranked items with source-kind diversity.

    Pins usable BigQuery rows first (up to ``bq_max``), ensures a minimum from
    each narrative corpus that returned hits, then fills by original order
    (assumed score-ranked). Dedupes before packing.
    """
    ranked = dedupe_context_items(list(items or []))
    if not ranked:
        return []

    has_bq = any(normalize_context_kind(i) == "bigquery" for i in ranked)
    limit = int(top_k) if top_k and top_k > 0 else _pack_size_default(has_bq=has_bq, task_mode=task_mode)
    try:
        bq_cap = int(bq_max) if bq_max is not None else int(os.environ.get("RAG_DIVERSIFY_BQ_MAX", "4") or 4)
    except ValueError:
        bq_cap = 4
    bq_cap = max(1, min(bq_cap, limit))
    try:
        kind_min = (
            int(per_kind_min)
            if per_kind_min is not None
            else int(os.environ.get("RAG_DIVERSIFY_PER_KIND_MIN", "2") or 2)
        )
    except ValueError:
        kind_min = 2
    kind_min = max(1, min(kind_min, 4))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ranked:
        buckets[normalize_context_kind(item)].append(item)

    packed: list[dict[str, Any]] = []
    used: set[int] = set()

    def _take(item: dict[str, Any]) -> None:
        ident = id(item)
        if ident in used or len(packed) >= limit:
            return
        used.add(ident)
        packed.append(item)

    for item in buckets.get("bigquery") or []:
        if sum(1 for p in packed if normalize_context_kind(p) == "bigquery") >= bq_cap:
            break
        _take(item)

    narrative_present = [k for k in _NARRATIVE_KINDS if buckets.get(k)]
    # Round-robin minima across narrative kinds that have hits.
    for _ in range(kind_min):
        progress = False
        for kind in narrative_present:
            if len(packed) >= limit:
                break
            for item in buckets[kind]:
                if id(item) in used:
                    continue
                _take(item)
                progress = True
                break
        if not progress:
            break

    for item in ranked:
        if len(packed) >= limit:
            break
        _take(item)

    return packed


__all__ = [
    "dedupe_context_items",
    "diversify_context_pack",
    "normalize_context_kind",
]
