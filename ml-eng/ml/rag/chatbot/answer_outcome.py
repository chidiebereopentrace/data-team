"""One decision for every turn: answer from rows, documents, both, or name the gap.

Every exit that used to return a canned refusal now asks ``decide_outcome``.
A turn is a gap only when neither a warehouse row nor a document survived
packing. A gap names the requested scope and distinguishes incomplete work
from a completed search that did not verify a finding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ml.rag.chatbot.agri_measure_ontology import MEASURES
from ml.rag.chatbot.analytical_intent import is_document_lookup
from ml.rag.chatbot.answer_language import detect_answer_language, insufficient_context_answer
from ml.rag.chatbot.evidence_kinds import is_document_item, is_warehouse_item
from ml.rag.chatbot.geo_iso3 import iso3_to_canonical_name, resolve_geography_iso3
from ml.rag.chatbot.turn_contract import TurnContract

OutcomeKind = Literal["rows", "rows_and_documents", "documents_only", "gap"]

DOCUMENTS_ONLY_INSTRUCTION = (
    "OpenTrace warehouse figures were not retrieved for this question. "
    "Answer only from the sources in the context. "
    "Do not invent totals, yields, prices, or a time series. "
    "Say plainly that the figures come from reports, not the OpenTrace warehouse."
)


@dataclass(frozen=True)
class AnswerOutcome:
    kind: OutcomeKind
    rows: int
    documents: int

    @property
    def is_gap(self) -> bool:
        return self.kind == "gap"

    @property
    def documents_only(self) -> bool:
        return self.kind == "documents_only"


def decide_outcome(context_items: list[dict[str, Any]]) -> AnswerOutcome:
    """Classify packed, usable evidence. Callers filter unusable items first."""
    rows = sum(1 for item in context_items if is_warehouse_item(item))
    documents = sum(1 for item in context_items if is_document_item(item))
    if rows and documents:
        kind: OutcomeKind = "rows_and_documents"
    elif rows:
        kind = "rows"
    elif documents:
        kind = "documents_only"
    else:
        kind = "gap"
    return AnswerOutcome(kind=kind, rows=rows, documents=documents)


def _indicator_label(measure_id: str | None) -> str:
    spec = MEASURES.get(str(measure_id or ""))
    if spec is None:
        return "data"
    worded = [str(a) for a in spec.aliases if "_" not in str(a)]
    return worded[0] if worded else spec.id.replace("_", " ")


_FILTER_LABELS = {"profile": "your profile", "ui": "the screen filter"}


def _requested_period(decomposition: dict[str, Any]) -> str:
    ts = str(decomposition.get("time_start") or decomposition.get("filtered_time_start") or "")[:4]
    te = str(decomposition.get("time_end") or decomposition.get("filtered_time_end") or "")[:4]
    if ts.isdigit() and te.isdigit():
        period = f"in {ts}" if ts == te else f"for {ts}–{te}"
    elif ts.isdigit():
        period = f"from {ts}"
    elif te.isdigit():
        period = f"up to {te}"
    else:
        return ""
    label = _FILTER_LABELS.get(str(decomposition.get("time_source") or ""))
    return f"{period} (set by {label})" if label else period


def _requested_places(decomposition: dict[str, Any]) -> tuple[list[str], str]:
    """Places the turn is scoped to, and who set them when it was not the question."""
    geography = [
        str(g)
        for g in (
            decomposition.get("geography")
            or decomposition.get("countries")
            or decomposition.get("filtered_geography")
            or []
        )
        if str(g).strip()
    ]
    return geography, _FILTER_LABELS.get(str(decomposition.get("geography_source") or ""), "")


def gap_answer(
    query: str,
    decomposition: dict[str, Any] | None,
    contract: TurnContract | None,
    *,
    incomplete_source: Literal["warehouse", "documents", "both"] | None = None,
) -> str:
    dec = decomposition if isinstance(decomposition, dict) else {}
    if is_document_lookup(query):
        status = (
            "The document search did not complete."
            if incomplete_source in ("documents", "both") else
            "The search did not return a verified passage from that document."
        )
        return (
            "I could not verify the requested document's answer to your question. "
            + status + " This does not establish whether the document covers or omits the topic."
        )
    measure_id = (contract.measure_id if contract else None) or next(
        iter(dec.get("primary_measures") or []), None
    )
    geography, place_setter = _requested_places(dec)
    iso3s = resolve_geography_iso3(query, geography=geography)
    lead = ""
    if detect_answer_language(query) not in ("en", ""):
        lead = insufficient_context_answer(query=query) + "\n\n"
    if not measure_id and not iso3s and incomplete_source is None:
        return insufficient_context_answer(query=query)

    indicator = _indicator_label(measure_id)
    place = ", ".join(iso3_to_canonical_name(code) or code for code in iso3s[:3]) or ", ".join(geography[:3])
    if place and place_setter:
        place = f"{place} (set by {place_setter})"
    period = _requested_period(dec)
    scope = " ".join(part for part in (f"for {place}" if place else "", period) if part)
    if incomplete_source == "both":
        return lead + f"OpenTrace could not verify evidence {scope} because the data query and document search did not complete."
    if incomplete_source:
        figures = f"{indicator} figures" if measure_id else "the requested figures"
        detail = f"{figures} {scope}".strip()
        check = "data query" if incomplete_source == "warehouse" else "document search"
        return lead + f"OpenTrace could not verify {detail} because the {check} did not complete."
    lines = [f"{indicator} figures {scope} are not confirmed by the sources checked.".replace("  ", " ")]
    lines.append("That does not mean the data does not exist elsewhere.")
    return lead + " ".join(lines)


__all__ = [
    "DOCUMENTS_ONLY_INSTRUCTION",
    "AnswerOutcome",
    "OutcomeKind",
    "decide_outcome",
    "gap_answer",
]
