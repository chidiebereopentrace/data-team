"""Plan enricher: bundles + entities → retrieval subquestions.

NOT query_enricher.py (memory rewrite for embeddings).
Tables resolve from agri_measure_ontology.get_measure — not hardcoded here.
"""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import get_measure
from ml.rag.chatbot.intent_bundles import (
    MatchedBundle,
    bundle_required_measures,
    has_bundle,
    match_intent_bundles,
)
from ml.rag.chatbot.reasoner_plan import SubQuestion

_JUNK_ENTITIES = frozenset(
    {
        "agriculture",
        "agricultural",
        "development",
        "policy",
        "policies",
        "research",
        "data",
        "information",
        "report",
        "analysis",
    }
)

_MEASURE_ID_ALIASES: dict[str, str] = {
    "employment": "employment_share",
    "food_security": "food_security_ipc",
}

_TRADE_RE = re.compile(r"\b(trade|export|import|imports|exports)\b", re.I)
_PRICE_RE = re.compile(r"\b(price|prices|volatility|market)\b", re.I)
_LIVESTOCK_RE = re.compile(r"\b(livestock|cattle|goat|poultry|dairy)\b", re.I)
_ADMIN_RE = re.compile(r"\b(district|districts|admin1|admin2|regions?\b)", re.I)
_REPORT_MULTI_GEO_RE = re.compile(r"\b(country by country|country-by-country)\b", re.I)


def _canonical_measure_id(measure: str) -> str:
    mid = (measure or "").strip().lower()
    return _MEASURE_ID_ALIASES.get(mid, mid)


def _tables_for_measure(measure_id: str, known: set[str]) -> tuple[str, ...]:
    spec = get_measure(_canonical_measure_id(measure_id))
    if spec is None:
        return ()
    return tuple(str(t).split(".")[-1].lower() for t in spec.candidate_tables if str(t).strip())


def _filter_entities(entities: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for e in entities:
        key = e.strip().lower()
        if not key or key in _JUNK_ENTITIES:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(e.strip())
    return out


def _geo_label(geos: tuple[str, ...]) -> str:
    if not geos:
        return "requested geography"
    if len(geos) <= 4:
        return ", ".join(geos)
    return f"{len(geos)} countries/regions"


def _time_label(time_start: str, time_end: str) -> str:
    y0 = (time_start or "")[:4]
    y1 = (time_end or "")[:4]
    if y0 and y1 and y0 != y1:
        return f"{y0}–{y1}"
    if y1:
        return y1
    if y0:
        return y0
    return "latest available"


def _slot(
    slot_id: str,
    nl: str,
    measure: str,
    *,
    required: bool = True,
    library: str = "bq",
    known_tables: set[str] | None = None,
) -> SubQuestion:
    mid = _canonical_measure_id(measure)
    tables = _tables_for_measure(mid, known_tables or set())
    return SubQuestion(
        id=slot_id,
        nl=nl,
        measure=mid,
        required=required,
        library=library,  # type: ignore[arg-type]
        tables=tables,
    )


def _apply_known_tables(slot: SubQuestion, known: set[str]) -> SubQuestion:
    if slot.library != "bq" or not slot.tables:
        return slot
    filtered = tuple(t for t in slot.tables if not known or t in known)
    if known and not filtered:
        return SubQuestion(
            id=slot.id,
            nl=slot.nl,
            measure=slot.measure,
            required=slot.required,
            library=slot.library,
            tables=slot.tables,
            status="slot_unsupported",
        )
    return SubQuestion(
        id=slot.id,
        nl=slot.nl,
        measure=slot.measure,
        required=slot.required,
        library=slot.library,
        tables=filtered or slot.tables,
        status=slot.status,
    )


def _add_measure_slot(
    slots: list[SubQuestion],
    seen_ids: set[str],
    *,
    slot_id: str,
    nl: str,
    measure: str,
    required: bool,
    library: str,
    known: set[str],
) -> None:
    if slot_id in seen_ids:
        return
    slot = _apply_known_tables(
        _slot(slot_id, nl, measure, required=required, library=library, known_tables=known),
        known,
    )
    seen_ids.add(slot_id)
    slots.append(slot)


def enrich_subquestions(
    *,
    query: str,
    decomposition: dict[str, Any] | None,
    job: str,
    plan_type: str,
    geos: tuple[str, ...],
    geo_grain: str,
    time_start: str,
    time_end: str,
    entities: list[str] | None = None,
    breakdown: list[str] | None = None,
    known_tables: set[str] | None = None,
    matched_bundles: tuple[MatchedBundle, ...] | None = None,
    depth: str = "full",
) -> tuple[SubQuestion, ...]:
    """Deterministic bundles + checklist → subquestions[]."""
    q = query or ""
    dec = decomposition if isinstance(decomposition, dict) else {}
    blob = " ".join(
        [
            q,
            " ".join(str(e) for e in (entities or [])),
            " ".join(str(e) for e in (dec.get("entities") or [])),
        ]
    ).lower()
    geo_label = _geo_label(geos)
    time_label = _time_label(time_start, time_end)
    known = known_tables or set()
    light = (depth or "full").strip().lower() == "light"

    bundles = matched_bundles
    if bundles is None:
        bundles = match_intent_bundles(q, dec, breakdown=list(breakdown or []))

    slots: list[SubQuestion] = []
    seen_ids: set[str] = set()

    # Bundle-driven required measures (authoritative multi-hit).
    for mid in bundle_required_measures(bundles):
        sid = {
            "production": "prod_panel",
            "trade": "trade_panel",
            "food_balance": "fbs_panel",
            "protected_area": "protected_panel",
            "employment_share": "employment_sex",
        }.get(mid, f"{mid}_panel")
        _add_measure_slot(
            slots,
            seen_ids,
            slot_id=sid,
            nl=f"{mid.replace('_', ' ').title()} for {geo_label}, {time_label}",
            measure=mid,
            required=True,
            library="bq",
            known=known,
        )

    # Outlook overlay — vector narrative; optional unless job=outlook.
    if has_bundle(bundles, "outlook_overlay") or job == "outlook":
        _add_measure_slot(
            slots,
            seen_ids,
            slot_id="outlook_narrative",
            nl=f"Food security outlook and IPC/FEWS assessments for {geo_label}, {time_label}",
            measure="food_security_ipc",
            required=job == "outlook",
            library="vector",
            known=known,
        )

    if not light:
        # Regional report without explicit activities bundle.
        if (
            not has_bundle(bundles, "agricultural_activities")
            and job in ("report", "compare", "synthesis")
            and len(geos) >= 2
        ):
            for mid, sid in (("production", "prod_panel"), ("trade", "trade_panel")):
                _add_measure_slot(
                    slots,
                    seen_ids,
                    slot_id=sid,
                    nl=f"{'Production' if mid == 'production' else 'Trade'} by {geo_label}, {time_label}",
                    measure=mid,
                    required=True,
                    library="bq",
                    known=known,
                )

        if _TRADE_RE.search(blob) and not any(s.id == "trade_panel" for s in slots):
            _add_measure_slot(
                slots,
                seen_ids,
                slot_id="trade_panel",
                nl=f"Trade volumes by {geo_label}, {time_label}",
                measure="trade",
                required=job in ("compare", "rank", "report"),
                library="bq",
                known=known,
            )

        pt = (plan_type or "").strip().lower()
        if _PRICE_RE.search(blob) or pt in ("agribusiness", "agribusinesses", "integrated"):
            _add_measure_slot(
                slots,
                seen_ids,
                slot_id="price_panel",
                nl=f"Staple prices / volatility by {geo_label}, {time_label}",
                measure="market_price",
                required=_PRICE_RE.search(blob) is not None,
                library="bq",
                known=known,
            )

        if _LIVESTOCK_RE.search(blob):
            _add_measure_slot(
                slots,
                seen_ids,
                slot_id="livestock_panel",
                nl=f"Livestock indicators by {geo_label}, {time_label}",
                measure="livestock",
                required=False,
                library="bq",
                known=known,
            )

        if _ADMIN_RE.search(blob) and job == "list":
            _add_measure_slot(
                slots,
                seen_ids,
                slot_id="admin_list",
                nl=f"Sub-national indicators at {geo_grain} for {geo_label}, {time_label}",
                measure="climate",
                required=True,
                library="bq",
                known=known,
            )

        if job in ("diagnosis", "brief", "report", "synthesis") and not any(
            s.library == "vector" for s in slots
        ):
            _add_measure_slot(
                slots,
                seen_ids,
                slot_id="narrative_context",
                nl=f"Policy and research context for {geo_label}, {time_label}",
                measure="research_meta",
                required=False,
                library="vector",
                known=known,
            )

    if not slots:
        pm = dec.get("primary_measures")
        measure = ""
        if isinstance(pm, list) and pm:
            measure = str(pm[0]).strip()
        if not measure:
            measure = "production"
        _add_measure_slot(
            slots,
            seen_ids,
            slot_id=f"{_canonical_measure_id(measure)}_primary",
            nl=f"{measure.replace('_', ' ').title()} for {geo_label}, {time_label}",
            measure=measure,
            required=True,
            library="bq",
            known=known,
        )

    return tuple(slots)


def primary_measure_from_slots(subquestions: tuple[SubQuestion, ...]) -> str:
    for sq in subquestions:
        if sq.library == "bq" and sq.required and sq.measure:
            return sq.measure
    for sq in subquestions:
        if sq.library == "bq" and sq.measure:
            return sq.measure
    return ""
