"""Job-typed evidence packing: warehouse rows plus a document budget.

Rows answer the figure. Documents explain it, and carry the answer alone when
the warehouse misses. A warehouse failure never empties the pack: whatever
documents survive the place filter stay in it.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ml.rag.chatbot.evidence_kinds import is_document_item, is_warehouse_item
from ml.rag.chatbot.generator import is_usable_context_item
from ml.rag.chatbot.turn_contract import NUMERIC_JOBS, TurnContract

DOCUMENT_BUDGET = 6
RELEVANCE_FLOOR = 0.15

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PLACE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_PLACE_KEYS = (
    "admin2", "admin_2", "district", "region", "admin1", "admin_1",
    "place_name", "fnid", "country_name",
)


@dataclass
class PackResult:
    items: list[dict[str, Any]]
    dropped: dict[str, int] = field(default_factory=dict)


def item_meta(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _years_in_item(item: dict[str, Any]) -> set[int]:
    text = str(item.get("content") or "")
    meta = item_meta(item)
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


def _in_window(years: set[int], y0: int | None, y1: int | None) -> bool:
    if y0 is not None and y1 is not None:
        return any(y0 <= y <= y1 for y in years)
    if y0 is not None:
        return any(y >= y0 for y in years)
    if y1 is not None:
        return any(y <= y1 for y in years)
    return True


def _rank_documents_by_period(
    documents: list[dict[str, Any]], contract: TurnContract
) -> list[dict[str, Any]]:
    """In-window documents first, then undated, then out-of-window. Rerank order holds inside each band."""
    y0, y1 = _requested_year_window(contract)
    if y0 is None and y1 is None:
        return documents

    def band(item: dict[str, Any]) -> int:
        years = _years_in_item(item)
        if not years:
            return 1
        return 0 if _in_window(years, y0, y1) else 2

    return sorted(documents, key=band)


def _filter_trend_rows(rows: list[dict[str, Any]], contract: TurnContract) -> list[dict[str, Any]]:
    y0, y1 = _requested_year_window(contract)
    if y0 is None and y1 is None:
        return sorted(rows, key=lambda it: max(_years_in_item(it) or {0}), reverse=True)
    kept = [it for it in rows if not _years_in_item(it) or _in_window(_years_in_item(it), y0, y1)]
    return kept or rows


def _filter_list_rows(rows: list[dict[str, Any]], contract: TurnContract) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in rows:
        if any(item_meta(it).get(k) for k in _PLACE_KEYS) or contract.geo_grain in ("admin2", "admin1") and _PLACE_RE.search(str(it.get("content") or "")):
            out.append(it)
    return out or rows


def _filter_breakdown_rows(rows: list[dict[str, Any]], contract: TurnContract) -> list[dict[str, Any]]:
    if "sex" not in contract.breakdown:
        return rows
    sex_rows = []
    for it in rows:
        text = str(it.get("content") or "").lower()
        sex_val = str(item_meta(it).get("sex") or "").lower()
        if sex_val in ("male", "female", "m", "f") or "male" in text or "female" in text:
            sex_rows.append(it)
    return sex_rows or rows


def _filter_outlook_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase_rows = [
        it for it in rows
        if any(item_meta(it).get(k) for k in ("phase", "ipc_phase", "period", "season"))
    ]
    return phase_rows or rows


def _shape_rows(rows: list[dict[str, Any]], contract: TurnContract) -> list[dict[str, Any]]:
    if contract.job == "trend":
        return _filter_trend_rows(rows, contract)
    if contract.job == "list":
        return _filter_list_rows(rows, contract)
    if contract.job == "fact" and contract.breakdown:
        return _filter_breakdown_rows(rows, contract)
    if contract.job == "outlook":
        return _filter_outlook_rows(rows)
    return rows


def typed_context_pack(
    items: list[dict[str, Any]],
    contract: TurnContract | None,
    *,
    top_k: int | None = None,
) -> PackResult:
    dropped: Counter[str] = Counter()
    usable: list[dict[str, Any]] = []
    for it in items:
        if is_usable_context_item(it):
            usable.append(it)
        else:
            dropped["unusable"] += 1

    if contract is None:
        packed = usable[:top_k] if top_k else usable
        dropped["over_top_k"] += len(usable) - len(packed)
        return PackResult(packed, {k: v for k, v in dropped.items() if v})

    rows = [it for it in usable if is_warehouse_item(it)]
    documents = [it for it in usable if is_document_item(it)]
    other = [it for it in usable if it not in rows and it not in documents]

    shaped_rows = _shape_rows(rows, contract)
    dropped["row_shape"] += len(rows) - len(shaped_rows)

    ranked_docs = _rank_documents_by_period(documents, contract)
    scored_docs: list[dict[str, Any]] = []
    for it in ranked_docs:
        score = it.get("_rerank_score", it.get("score"))
        try:
            value = float(score) if score is not None else None
        except (TypeError, ValueError):
            value = None
        if value is not None and value < RELEVANCE_FLOOR:
            dropped["relevance_floor"] += 1
            continue
        scored_docs.append(it)
    budget_docs = scored_docs[:DOCUMENT_BUDGET]
    dropped["document_budget"] += len(scored_docs) - len(budget_docs)

    if contract.job in NUMERIC_JOBS:
        packed = [*shaped_rows, *budget_docs]
        dropped["non_evidence_kind"] += len(other)
    else:
        packed = [*shaped_rows, *budget_docs, *other]

    if top_k and len(packed) > top_k:
        dropped["over_top_k"] += len(packed) - top_k
        packed = packed[:top_k]
    return PackResult(packed, {k: v for k, v in dropped.items() if v})


__all__ = ["DOCUMENT_BUDGET", "PackResult", "RELEVANCE_FLOOR", "item_meta", "typed_context_pack"]
