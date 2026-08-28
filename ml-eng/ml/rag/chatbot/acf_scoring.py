"""
ADZA Confidence Framework (ACF) — Path B facade over the open-trace ``acf`` library.

Module name is ``acf_scoring`` (not ``acf``) so pytest path insertion cannot shadow
the installed open-trace package.

Scores an answer from the **cited** evidence set only (not full retrieval context).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from acf import from_payload, from_row, score_evidence
from acf.aggregate.evidence import ExtractedClaim
from acf.enums import ClaimLevel, QuestionType

from ml.rag.chatbot.acf_metadata import context_item_to_acf_record
from ml.rag.chatbot.acf_question import classify_acf_question

logger = logging.getLogger(__name__)

ACFBandLiteral = str  # very_strong | strong | moderate | limited | low | no_evidence


@dataclass(frozen=True)
class ACFResult:
    """API-facing ACF signal (0–100 Path B)."""

    band: str
    band_label: str
    score: int
    explanation: str
    note: str  # alias of explanation for older call sites
    components: dict[str, Any] | None = None
    applied_ceiling: str | None = None
    config_version: str | None = None
    claim_level: str | None = None
    question_type: str | None = None


def curated_product_acf(*, explanation: str | None = None) -> ACFResult:
    """Fixed signal for meta/product KB short-circuits (not triangulation)."""
    expl = explanation or (
        "This response is from the OpenTrace product knowledge base "
        "(curated, not multi-source evidence triangulation)."
    )
    return ACFResult(
        band="strong",
        band_label="Strong confidence",
        score=90,
        explanation=expl,
        note=expl,
        components=None,
        applied_ceiling=None,
        config_version="curated_v1",
        claim_level=None,
        question_type=None,
    )


def no_evidence_acf(*, explanation: str | None = None) -> ACFResult:
    """Empty or unadaptable cited set."""
    expl = explanation or (
        "No scorably tagged OpenTrace evidence was cited for this answer."
    )
    return ACFResult(
        band="no_evidence",
        band_label="No evidence",
        score=0,
        explanation=expl,
        note=expl,
        components=None,
        applied_ceiling=None,
        config_version=None,
        claim_level=None,
        question_type=None,
    )


def _components_to_dict(components: dict[str, Any] | None) -> dict[str, Any] | None:
    if not components:
        return None
    out: dict[str, Any] = {}
    for name, comp in components.items():
        try:
            out[name] = {
                "name": getattr(comp, "name", name),
                "value": float(getattr(comp, "value", 0.0)),
                "breakdown": dict(getattr(comp, "breakdown", {}) or {}),
            }
        except Exception:
            out[name] = {"name": name, "value": None, "breakdown": {}}
    return out


def _item_from_source_ref(ref: Any) -> dict[str, Any] | None:
    """Accept SourceRef dataclass or raw context dict."""
    if ref is None:
        return None
    if isinstance(ref, dict):
        return ref
    item = getattr(ref, "item", None)
    return item if isinstance(item, dict) else None


def adapt_cited_claims(cited_items: list[Any]) -> list[ExtractedClaim]:
    """Map cited context items / SourceRefs to ExtractedClaim (skip failures)."""
    claims: list[ExtractedClaim] = []
    for ref in cited_items or []:
        item = _item_from_source_ref(ref)
        if not item:
            continue
        record = context_item_to_acf_record(item)
        if not record:
            continue
        try:
            # BQ-shaped rows prefer from_row (region field); vectors use from_payload.
            kind = str(
                item.get("_context_kind") or item.get("source") or ""
            ).strip().lower()
            if kind in ("bigquery", "structured_data"):
                geo_field = "place_scope" if record.get("place_scope") is not None else "region"
                claims.append(from_row(record, geo_field=geo_field))
            else:
                claims.append(from_payload(record))
        except Exception as exc:
            logger.debug("ACF adapt skipped claim: %s", exc)
            continue
    return claims


def score_cited_evidence(
    cited_items: list[Any],
    *,
    query: str = "",
    decomposition: dict[str, Any] | None = None,
    claim_level: ClaimLevel | str | None = None,
    question_type: QuestionType | str | None = None,
    reference_date: date | None = None,
) -> ACFResult:
    """
    Path B entry: adapt cited sources → ``score_evidence``.

    Parameters
    ----------
    cited_items:
        Cited ``SourceRef``s or context dicts (not the full retrieval set).
    """
    qclass = classify_acf_question(query, decomposition, reference_date=reference_date)
    cl = claim_level
    qt = question_type
    if isinstance(cl, str):
        cl = ClaimLevel(cl)
    if isinstance(qt, str):
        qt = QuestionType(qt)
    if cl is None:
        cl = qclass.claim_level
    if qt is None:
        qt = qclass.question_type

    claims = adapt_cited_claims(cited_items)
    if not claims:
        result = no_evidence_acf()
        return ACFResult(
            band=result.band,
            band_label=result.band_label,
            score=result.score,
            explanation=result.explanation,
            note=result.note,
            claim_level=cl.value,
            question_type=qt.value,
        )

    try:
        lib = score_evidence(
            claims,
            claim_level=cl,
            question_type=qt,
            reference_date=reference_date,
        )
    except Exception as exc:
        logger.warning("score_evidence failed: %s", exc)
        return no_evidence_acf(explanation=f"ACF scoring failed: {exc}")

    ceiling = None
    if lib.applied_ceiling is not None:
        rule = getattr(lib.applied_ceiling, "rule", None)
        reason = getattr(lib.applied_ceiling, "reason", None)
        ceiling = str(rule or reason or lib.applied_ceiling)

    band = lib.band.value if hasattr(lib.band, "value") else str(lib.band)
    label = getattr(lib.band, "label", None) or band.replace("_", " ").title()
    expl = str(lib.explanation or "")

    return ACFResult(
        band=band,
        band_label=label,
        score=int(lib.score),
        explanation=expl,
        note=expl,
        components=_components_to_dict(lib.components),
        applied_ceiling=ceiling,
        config_version=str(lib.config_version) if lib.config_version else None,
        claim_level=cl.value,
        question_type=qt.value,
    )


def acf_result_to_state(result: ACFResult) -> dict[str, Any]:
    """Flatten ACFResult into graph state keys."""
    return {
        "acf_band": result.band,
        "acf_band_label": result.band_label,
        "acf_score": result.score,
        "acf_note": result.note,
        "acf_explanation": result.explanation,
        "acf_components": result.components,
        "acf_applied_ceiling": result.applied_ceiling,
        "acf_config_version": result.config_version,
        "acf_claim_level": result.claim_level,
        "acf_question_type": result.question_type,
    }
