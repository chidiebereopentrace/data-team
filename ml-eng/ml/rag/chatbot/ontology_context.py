"""Unified ontology context for BQ compile, reason, validate, and UX."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import MEASURES, MeasureHit, resolve_measures
from ml.rag.chatbot.mart_indicator_classes import class_for_query, facts_for_classes


@dataclass
class IndicatorClassContext:
    code: str
    name: str
    primary_facts: list[str] = field(default_factory=list)
    example_claims: list[str] = field(default_factory=list)
    do_not_mix_notes: list[str] = field(default_factory=list)


@dataclass
class OntologyContext:
    """Single contract object: measure + indicator classes + decomposition slots."""

    query: str
    primary_measures: list[str] = field(default_factory=list)
    measure_hits: list[MeasureHit] = field(default_factory=list)
    indicator_classes: list[IndicatorClassContext] = field(default_factory=list)
    geography: list[str] = field(default_factory=list)
    time_start: str = ""
    time_end: str = ""
    crop_required: bool = True
    geography_required: bool = True
    candidate_tables: list[str] = field(default_factory=list)
    filter_hints: list[str] = field(default_factory=list)

    def to_reasoner_block(self) -> str:
        lines = ["Ontology contract (MUST honor in SQL filters):"]
        if self.primary_measures:
            lines.append(f"- primary_measures: {', '.join(self.primary_measures)}")
        for ic in self.indicator_classes[:4]:
            facts = ", ".join(ic.primary_facts[:4]) or "-"
            lines.append(f"- indicator {ic.code} ({ic.name}): tables {facts}")
            for claim in ic.example_claims[:2]:
                lines.append(f"  example: {claim}")
        for hint in self.filter_hints[:6]:
            lines.append(f"- filter_hint: {hint}")
        if self.geography:
            lines.append(f"- geography: {', '.join(self.geography[:8])}")
        if self.time_start or self.time_end:
            lines.append(f"- time: {self.time_start or '?'} .. {self.time_end or '?'}")
        return "\n".join(lines)


def _load_class_context(code: str) -> IndicatorClassContext | None:
    from ml.rag.chatbot.mart_indicator_classes import _class_spec

    spec = _class_spec(code)
    if not spec:
        return None
    claims: list[str] = []
    for fam in spec.get("families") or []:
        if isinstance(fam, dict):
            claim = str(fam.get("example_claim") or "").strip()
            if claim:
                claims.append(claim)
    notes: list[str] = []
    for pair in spec.get("do_not_mix") or []:
        if isinstance(pair, dict):
            reason = str(pair.get("reason") or "").strip()
            if reason:
                notes.append(reason)
    return IndicatorClassContext(
        code=code.upper(),
        name=str(spec.get("name") or code),
        primary_facts=[str(t).split(".")[-1] for t in (spec.get("primary_facts") or [])],
        example_claims=claims,
        do_not_mix_notes=notes,
    )


def build_ontology_context(
    query: str,
    decomposition: dict[str, Any] | None = None,
) -> OntologyContext:
    """Merge measure ontology, indicator classes, and decomposition facets."""
    dec = dict(decomposition or {})
    hits = resolve_measures(query, dec)
    pm: list[str] = []
    if hits:
        pm.append(hits[0].measure.id)
    declared = dec.get("primary_measures")
    if isinstance(declared, list):
        for m in declared:
            mid = str(m).strip().lower()
            if mid and mid not in pm:
                pm.append(mid)

    ic_codes = class_for_query(query)
    ic_contexts = [c for code in ic_codes if (c := _load_class_context(code))]

    geo_raw = dec.get("geography")
    geography = [str(g).strip() for g in geo_raw if str(g).strip()] if isinstance(geo_raw, list) else []

    tables: list[str] = []
    seen: set[str] = set()
    if hits:
        for tid in hits[0].measure.candidate_tables:
            bare = str(tid).split(".")[-1].lower()
            if bare and bare not in seen:
                seen.add(bare)
                tables.append(bare)
    for code in ic_codes:
        for tid in facts_for_classes([code]):
            bare = str(tid).split(".")[-1].lower()
            if bare and bare not in seen:
                seen.add(bare)
                tables.append(bare)

    hints: list[str] = []
    if hits:
        hint = str(hits[0].measure.filter_hints or "").strip()
        if hint:
            hints.append(hint)

    crop_required = hits[0].measure.crop_required if hits else True
    geography_required = hits[0].measure.geography_required if hits else True

    return OntologyContext(
        query=query,
        primary_measures=pm,
        measure_hits=hits,
        indicator_classes=ic_contexts,
        geography=geography,
        time_start=str(dec.get("time_start") or "")[:10],
        time_end=str(dec.get("time_end") or "")[:10],
        crop_required=crop_required,
        geography_required=geography_required,
        candidate_tables=tables,
        filter_hints=hints,
    )


_CONFLICTING_ENTITIES: dict[str, frozenset[str]] = {
    "production": frozenset({"yield", "climate", "rainfall", "temperature", "ipc", "food security"}),
    "yield": frozenset({"production", "trade", "export", "import"}),
}


def sanitize_decomposition_for_bq(
    decomposition: dict[str, Any] | None,
    *,
    primary_measures: list[str] | None = None,
) -> dict[str, Any]:
    """
    Strip decomposer noise from entities/domains when primary_measures is set.

    Keeps geography/time intact; measure_blob path uses primary_measures not entity spam.
    """
    dec = dict(decomposition or {})
    pm = [str(m).strip().lower() for m in (primary_measures or dec.get("primary_measures") or []) if str(m).strip()]
    if pm:
        dec["primary_measures"] = pm
    if not pm:
        return dec

    primary = pm[0]
    drop = _CONFLICTING_ENTITIES.get(primary, frozenset())

    entities = dec.get("entities")
    if isinstance(entities, list):
        cleaned = [str(e).strip() for e in entities if str(e).strip()]
        cleaned = [e for e in cleaned if e.lower() not in drop and e.lower() != primary]
        if primary not in {e.lower() for e in cleaned}:
            cleaned.insert(0, primary)
        dec["entities"] = cleaned

    domains = dec.get("domains")
    if isinstance(domains, list) and primary == "production":
        dec["domains"] = [
            d for d in domains
            if str(d).strip()
            and "climate" not in str(d).lower()
            and "rainfall" not in str(d).lower()
            and str(d).lower() not in ("yield",)
        ]

    return dec


def list_indicator_class_contexts(max_classes: int = 12) -> list[IndicatorClassContext]:
    """Load indicator class contexts for UX catalog rendering."""
    out: list[IndicatorClassContext] = []
    for code in class_for_query("production trade prices food security climate"):
        if len(out) >= max_classes:
            break
        ctx = _load_class_context(code)
        if ctx and ctx.code not in {c.code for c in out}:
            out.append(ctx)
    from ml.rag.chatbot.mart_indicator_classes import all_class_codes

    for code in all_class_codes():
        if len(out) >= max_classes:
            break
        ctx = _load_class_context(code)
        if ctx and ctx.code not in {c.code for c in out}:
            out.append(ctx)
    return out


__all__ = [
    "IndicatorClassContext",
    "OntologyContext",
    "build_ontology_context",
    "list_indicator_class_contexts",
    "sanitize_decomposition_for_bq",
]
