"""Composer: dual-bag coverage law for heavy-path generation."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.generator import is_usable_context_item, is_usable_structured_bq_row
from ml.rag.chatbot.reasoner_plan import ReasonerPlan, SubQuestion


def is_sentinel_row(item: dict[str, Any]) -> bool:
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    status = str(meta.get("status") or item.get("status") or "").strip().lower()
    if status in ("no_valid_sql", "bq_timeout", "empty_result", "validation_failed"):
        return True
    val = item.get("value")
    if val is not None and str(val).strip().lower() in ("timeout", "timeout_s"):
        return True
    content = str(item.get("content") or "").lower()
    if "[bq no_valid_sql" in content or "yaml-only" in content:
        return True
    return False


def partition_bq_by_subquestion(
    bq_results: list[dict[str, Any]],
    reasoner: ReasonerPlan | None,
) -> dict[str, list[dict[str, Any]]]:
    """Group usable BQ rows by subquestion_id metadata or measure/table match."""
    out: dict[str, list[dict[str, Any]]] = {}
    if reasoner:
        for sq in reasoner.bq_subquestions():
            out[sq.id] = []

    unassigned: list[dict[str, Any]] = []
    for row in bq_results or []:
        if not is_usable_context_item(row) or is_sentinel_row(row):
            continue
        if not is_usable_structured_bq_row(row):
            continue
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        sid = str(meta.get("subquestion_id") or meta.get("slot_id") or "").strip()
        if sid and reasoner and sid in out:
            out[sid].append(row)
            continue
        blob = (str(row.get("content") or "") + " " + str(meta)).lower()
        matched = False
        if reasoner:
            for sq in reasoner.bq_subquestions():
                if sq.measure and sq.measure.replace("_", " ") in blob:
                    out[sq.id].append(row)
                    matched = True
                    break
                if any(t.replace("_", " ") in blob for t in sq.tables):
                    out[sq.id].append(row)
                    matched = True
                    break
        if not matched:
            unassigned.append(row)

    if unassigned:
        if reasoner and reasoner.bq_subquestions():
            primary_id = reasoner.bq_subquestions()[0].id
            out.setdefault(primary_id, []).extend(unassigned)
        else:
            out.setdefault("primary", []).extend(unassigned)
    return out


def slot_status(subquestion: SubQuestion, rows: list[dict[str, Any]]) -> str:
    if subquestion.status == "slot_unsupported":
        return "slot_unsupported"
    if rows:
        return "served"
    return "empty"


def coverage_lines(reasoner: ReasonerPlan, rows_by_slot: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Explicit miss lines for empty required slots."""
    lines: list[str] = []
    for sq in reasoner.subquestions:
        if sq.library != "bq" or not sq.required:
            continue
        bag = rows_by_slot.get(sq.id) or []
        st = slot_status(sq, bag)
        if st == "empty":
            lines.append(f"No structured OpenTrace rows for: {sq.nl}.")
        elif st == "slot_unsupported":
            lines.append(f"Warehouse slot not served for: {sq.nl}.")
    return lines


def composer_context_block(
    *,
    rows_by_slot: dict[str, list[dict[str, Any]]],
    passages: list[dict[str, Any]],
    reasoner: ReasonerPlan | None,
) -> str:
    parts: list[str] = []
    if reasoner:
        for sq in reasoner.subquestions:
            if sq.library == "bq":
                rows = rows_by_slot.get(sq.id) or []
                parts.append(f"=== BQ SLOT: {sq.id} ({sq.measure}) ===")
                if not rows:
                    parts.append(f"(empty — {sq.nl})")
                else:
                    for i, row in enumerate(rows[:12], 1):
                        parts.append(str(row.get("content") or "").strip() or f"row {i}")
            elif sq.library == "vector":
                parts.append(f"=== NARRATIVE SLOT: {sq.id} ===")
        parts.append("=== PASSAGES (narrative only; quantities from BQ slots) ===")
    for item in passages or []:
        if not is_usable_context_item(item) or is_sentinel_row(item):
            continue
        parts.append(str(item.get("content") or "").strip())
    return "\n\n".join(p for p in parts if p)


def composer_addendum(
    reasoner: ReasonerPlan,
    rows_by_slot: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        "COMPOSER COVERAGE (heavy path):",
        f"Job: {reasoner.job}. Shape: {reasoner.shape}. Geos: {len(reasoner.geos)}.",
        "For each required slot with rows, include that part in the answer (table or subsection).",
        "For empty required slots, state one explicit miss line — do not invent numbers.",
        "No number unless in BQ rows or quoted passage. No place name unless in bags.",
    ]
    misses = coverage_lines(reasoner, rows_by_slot)
    for m in misses[:6]:
        lines.append(m)
    geo_label = ", ".join(reasoner.geos[:12])
    if len(reasoner.geos) > 12:
        geo_label += f", … ({len(reasoner.geos)} total)"
    if geo_label:
        lines.append(f"Full geography scope: {geo_label}.")
    return "\n".join(lines)[:1200]
