"""Job-typed context packing — zero passages on fail closed."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.generator import is_usable_context_item, is_usable_structured_bq_row
from ml.rag.chatbot.turn_contract import NUMERIC_JOBS, TurnContract

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PLACE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def _is_bq_item(item: dict[str, Any]) -> bool:
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    kind = str(meta.get("context_kind") or meta.get("source_kind") or "").lower()
    if kind == "bigquery":
        return True
    content = str(item.get("content") or "")
    return content.startswith("[Structured data]")


def _is_narrative_item(item: dict[str, Any]) -> bool:
    if _is_bq_item(item):
        return False
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    kind = str(meta.get("context_kind") or meta.get("source_kind") or "").lower()
    return kind not in ("bigquery", "web")


def _years_in_item(item: dict[str, Any]) -> set[int]:
    text = str(item.get("content") or "")
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for key in ("year", "harvest_year", "as_of_date", "publication_year", "period_start", "period_end"):
        val = meta.get(key)
        if val is not None:
            m = _YEAR_RE.search(str(val))
            if m:
                text += f" {m.group(0)}"
    pub = str(meta.get("published_at") or "")[:10]
    if pub[:4].isdigit():
        text += f" {pub[:4]}"
    return {int(m.group(0)) for m in _YEAR_RE.finditer(text)}


def _requested_year_window(contract: TurnContract) -> tuple[int | None, int | None]:
    ts = contract.time_spec.start
    te = contract.time_spec.end
    y0 = int(ts[:4]) if ts and len(ts) >= 4 and ts[:4].isdigit() else None
    y1 = int(te[:4]) if te and len(te) >= 4 and te[:4].isdigit() else None
    return y0, y1


def _chunk_overlaps_window(item: dict[str, Any], y0: int | None, y1: int | None) -> bool:
    years = _years_in_item(item)
    if not years:
        return False
    if y0 is not None and y1 is not None:
        return any(y0 <= y <= y1 for y in years)
    if y0 is not None:
        return any(y >= y0 for y in years)
    if y1 is not None:
        return any(y <= y1 for y in years)
    return True


def _filter_vector_by_time(items: list[dict[str, Any]], contract: TurnContract) -> list[dict[str, Any]]:
    if not contract.time_spec.hard_filter:
        return items
    y0, y1 = _requested_year_window(contract)
    if y0 is None and y1 is None:
        return items
    narrative = [it for it in items if _is_narrative_item(it)]
    bq = [it for it in items if not _is_narrative_item(it)]
    filtered = [it for it in narrative if _chunk_overlaps_window(it, y0, y1)]
    return bq + filtered


def _filter_trend_rows(items: list[dict[str, Any]], contract: TurnContract) -> list[dict[str, Any]]:
    bq = [it for it in items if is_usable_structured_bq_row(it)]
    y0, y1 = _requested_year_window(contract)
    if y0 is None and y1 is None:
        return sorted(bq, key=lambda it: max(_years_in_item(it) or {0}), reverse=True)
    filtered: list[dict[str, Any]] = []
    for it in bq:
        years = _years_in_item(it)
        if not years:
            filtered.append(it)
            continue
        if y0 is not None and y1 is not None:
            if any(y0 <= y <= y1 for y in years):
                filtered.append(it)
        elif y0 is not None and any(y >= y0 for y in years):
            filtered.append(it)
        elif y1 is not None and any(y <= y1 for y in years):
            filtered.append(it)
    return filtered or bq


def _filter_list_rows(items: list[dict[str, Any]], contract: TurnContract) -> list[dict[str, Any]]:
    bq = [it for it in items if is_usable_structured_bq_row(it)]
    grain = contract.geo_grain
    out: list[dict[str, Any]] = []
    for it in bq:
        meta = it.get("metadata") if isinstance(it.get("metadata"), dict) else {}
        text = str(it.get("content") or "")
        place_keys = (
            "admin2", "admin_2", "district", "region", "admin1", "admin_1",
            "place_name", "fnid", "country_name",
        )
        if any(meta.get(k) for k in place_keys):
            out.append(it)
            continue
        if grain in ("admin2", "admin1") and _PLACE_RE.search(text):
            out.append(it)
    return out or bq


def _filter_breakdown_rows(items: list[dict[str, Any]], contract: TurnContract) -> list[dict[str, Any]]:
    bq = [it for it in items if is_usable_structured_bq_row(it)]
    if "sex" not in contract.breakdown:
        return bq
    sex_rows: list[dict[str, Any]] = []
    for it in bq:
        meta = item_meta(it)
        text = str(it.get("content") or "").lower()
        sex_val = str(meta.get("sex") or "").lower()
        if sex_val in ("male", "female", "m", "f") or "male" in text or "female" in text:
            sex_rows.append(it)
    return sex_rows or bq


def item_meta(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _cap_narrative(items: list[dict[str, Any]], max_news: int = 2) -> list[dict[str, Any]]:
    narrative = [it for it in items if _is_narrative_item(it)]
    news = [
        it for it in narrative
        if str(item_meta(it).get("context_kind") or "").lower() in ("news", "public_report", "policy")
    ]
    other = [it for it in narrative if it not in news]
    return news[:max_news] + other[: max(0, max_news - len(news[:max_news]))]


def _has_bq_timeout(items: list[dict[str, Any]]) -> bool:
    for it in items:
        meta = item_meta(it)
        if str(meta.get("status") or "").strip().lower() == "bq_timeout":
            return True
    return False


def should_zero_pack(
    contract: TurnContract | None,
    *,
    bq_failed: bool = False,
    context_items: list[dict[str, Any]] | None = None,
) -> bool:
    if contract is None:
        return False
    if _has_bq_timeout(context_items or []):
        return True
    if contract.vector_policy == "fallback_only" and contract.is_fail_closed():
        return False
    if contract.is_fail_closed():
        if contract.job not in NUMERIC_JOBS:
            return False
        return True
    if bq_failed and contract.job in NUMERIC_JOBS:
        if contract.vector_policy == "companion" and any(
            _is_narrative_item(it) for it in (context_items or [])
        ):
            return False
        return True
    if contract.serve_status != "served" and contract.job in NUMERIC_JOBS:
        return True
    return False


def typed_context_pack(
    items: list[dict[str, Any]],
    contract: TurnContract | None,
    *,
    top_k: int | None = None,
    bq_failed: bool = False,
) -> list[dict[str, Any]]:
    usable = [it for it in items if is_usable_context_item(it)]
    usable = _filter_vector_by_time(usable, contract) if contract else usable
    if should_zero_pack(contract, bq_failed=bq_failed, context_items=usable):
        return []

    if contract is None:
        return usable[:top_k] if top_k else usable

    job = contract.job
    bq_items = [it for it in usable if _is_bq_item(it)]
    narrative = [it for it in usable if _is_narrative_item(it)]

    if contract.vector_policy == "fallback_only":
        packed = _cap_narrative(narrative, max_news=3)
        if top_k and len(packed) > top_k:
            packed = packed[:top_k]
        return packed

    if job in NUMERIC_JOBS:
        if job == "trend":
            packed = _filter_trend_rows(bq_items, contract)
        elif job == "list":
            packed = _filter_list_rows(bq_items, contract)
        elif job == "fact" and contract.breakdown:
            packed = _filter_breakdown_rows(bq_items, contract)
        else:
            packed = bq_items
        if contract.vector_policy == "companion" and narrative:
            packed = list(packed)
            packed.extend(_cap_narrative(narrative, max_news=2))
        if top_k and len(packed) > top_k:
            packed = packed[:top_k]
        return packed

    if job == "outlook":
        phase_rows = []
        for it in bq_items:
            meta = item_meta(it)
            if any(meta.get(k) for k in ("phase", "ipc_phase", "period", "season")):
                phase_rows.append(it)
        packed = phase_rows or bq_items
        packed.extend(_cap_narrative(narrative, max_news=2))
        if top_k and len(packed) > top_k:
            packed = packed[:top_k]
        return packed

    if job in ("diagnose", "brief"):
        packed = list(bq_items)
        packed.extend(_cap_narrative(narrative, max_news=2))
        if top_k and len(packed) > top_k:
            packed = packed[:top_k]
        return packed

    if top_k and len(usable) > top_k:
        return usable[:top_k]
    return usable
