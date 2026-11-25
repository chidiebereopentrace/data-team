"""RARR answer mode: draft from model knowledge, then revise it against retrieved evidence.

The draft runs beside the graph. The revision reads the evidence the graph already
retrieved, so retrieval itself is unchanged. The score combines the strength of the
cited evidence with the share of statements the evidence supports.

Every stage is recorded in ``out["rarr"]`` for the Streamlit inspector.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from ml.rag.chatbot.acf_scoring import _BAND_LABELS, ACFResult, acf_result_to_state, score_cited_evidence
from ml.rag.chatbot.answer_language import detect_answer_language, language_instruction
from ml.rag.chatbot.chat_memory import build_memory_prompt_block
from ml.rag.chatbot.citation_pack import extract_referenced_source_ids
from ml.rag.chatbot.generation_prompt import _current_date_block
from ml.rag.chatbot.generator import (
    _build_context_block,
    _chunk_max_chars,
    _context_max_chars,
    filter_context_items,
    pin_bq_context_first,
    referenced_citations,
)
from ml.rag.chatbot.plan_policy import model_for_plan, plan_generation_addendum
from ml.rag.chatbot.stage_budget import remaining_seconds
from ml.rag.llm_chat import llm_chat_complete

NO_EVIDENCE_NOTE = (
    "Source: no OpenTrace evidence was found to verify this answer. It comes from general "
    "model knowledge, so treat its figures as unverified."
)
_RETRY_PAUSE_S = 0.5
_RETRY_MIN_LEFT_S = 3.0
DRAFT_MAX_TOKENS = 1200
REVISE_MAX_TOKENS = 1600

_DRAFT_PROMPT = (
    "You are ADZA, an expert on agriculture across Africa: crops, livestock, fisheries, "
    "markets and prices, trade, climate and weather, soils, inputs, finance, policy, "
    "nutrition and food security. Answer the question fully and directly from your own "
    "knowledge. Give the specific figures, dates, places and names you know, and state "
    "the year a figure belongs to. Do not say you lack data and do not refuse; give your "
    "best answer. Do not cite sources."
)

_REVISE_PROMPT = (
    "You check a draft answer against numbered evidence and return the corrected answer.\n"
    "Rules:\n"
    "1. Keep every draft statement the evidence supports, and put the matching [Source N] "
    "marker at the end of that sentence.\n"
    "2. Where the evidence contradicts the draft, replace the draft statement with what the "
    "evidence says and cite it.\n"
    "3. Add key facts from the evidence that answer the question and the draft missed, "
    "with their markers.\n"
    "4. Keep draft statements the evidence does not cover only if they are general, "
    "non-quantified reasoning, without a marker. Delete any specific number, date, "
    "percentage, named company, or named policy the evidence does not cover -- do not keep "
    "it and do not label it (estimate); a guess tagged as an estimate is still a guess.\n"
    "5. Only use [Source N] numbers that appear in the evidence. Never cite a source for a "
    "statement it does not support.\n"
    "6. Return only the final answer, in the language and register of the draft. Do not "
    "mention the draft, the evidence check, or these rules, and do not add a sources list."
)

_STATEMENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_LIST_LEAD = re.compile(r"^\s*(?:[-*>#]+|\d+[.)])\s*")
_MARKER = re.compile(r"\[Source\s*\d+(?:\s*,\s*(?:Source\s*)?\d+)*\]", re.IGNORECASE)
_BAND_FLOORS: tuple[tuple[int, str], ...] = (
    (85, "very_strong"),
    (70, "strong"),
    (55, "moderate"),
    (40, "limited"),
)


@dataclass(frozen=True)
class Revision:
    answer: str
    citations: list[dict[str, Any]]
    acf: ACFResult
    statements: list[dict[str, Any]]
    trace: dict[str, Any]


def _timeout() -> float | None:
    left = remaining_seconds(None)
    return None if left is None else max(1.0, left)


def draft_answer(
    query: str,
    *,
    plan_type: str | None = None,
    conversation_summary: str = "",
    recent_turns: list[dict[str, str]] | None = None,
    reference_date: date | None = None,
) -> str:
    """Write a full answer from model knowledge alone. Return "" when the call fails."""
    system = (
        _DRAFT_PROMPT
        + "\n\n"
        + language_instruction(detect_answer_language(query), inline_citations=False)
        + plan_generation_addendum(plan_type)
        + _current_date_block(reference_date)
    )
    memory = build_memory_prompt_block(conversation_summary, recent_turns)
    user = f"{memory}\n\nQuestion: {query}" if memory else f"Question: {query}"
    return llm_chat_complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model_for_plan(plan_type),
        max_tokens=DRAFT_MAX_TOKENS,
        timeout_s=_timeout(),
        purpose="rarr_draft",
    ).strip()


def statements(answer: str) -> list[str]:
    """Split an answer into statements, ignoring headings and bare list markers."""
    out: list[str] = []
    for part in _STATEMENT_SPLIT.split(answer):
        text = _LIST_LEAD.sub("", part).strip()
        if len(_MARKER.sub("", text).strip()) < 12 or text.endswith(":"):
            continue
        out.append(text)
    return out


def statement_rows(answer: str, known_ids: set[int]) -> list[dict[str, Any]]:
    """One row per statement: the sources it cites and whether those sources exist."""
    rows: list[dict[str, Any]] = []
    for text in statements(answer):
        cited = extract_referenced_source_ids(text)
        rows.append(
            {
                "text": text,
                "sources": sorted(cited & known_ids),
                "unknown_sources": sorted(cited - known_ids),
                "verified": bool(cited & known_ids),
                "estimate": "(estimate)" in text.lower(),
            }
        )
    return rows


def _band(score: int) -> str:
    for floor, band in _BAND_FLOORS:
        if score >= floor:
            return band
    return "low"


def score_revision(
    rows: list[dict[str, Any]],
    cited_refs: list[Any],
    *,
    query: str,
    decomposition: dict[str, Any] | None,
    reference_date: date | None,
) -> ACFResult:
    """Evidence strength times the share of statements carrying a valid marker."""
    total = len(rows)
    verified = sum(1 for row in rows if row["verified"])
    if not cited_refs or verified == 0:
        explanation = (
            f"No OpenTrace evidence verifies any of the {total} statements; "
            "the answer comes from model knowledge."
        )
        return ACFResult(
            band="no_evidence",
            band_label=_BAND_LABELS["no_evidence"],
            score=0,
            explanation=explanation,
            note=explanation,
            components={"coverage": 0.0, "statements_total": total, "statements_verified": 0},
        )
    strength = score_cited_evidence(
        cited_refs, query=query, decomposition=decomposition, reference_date=reference_date
    )
    coverage = verified / total
    score = int(round(strength.score * coverage))
    band = _band(score)
    explanation = (
        f"{verified} of {total} statements are supported by cited OpenTrace evidence. "
        f"{strength.explanation}"
    )
    components = dict(strength.components or {})
    components.update(
        evidence_score=strength.score,
        evidence_band=strength.band,
        coverage=round(coverage, 3),
        statements_total=total,
        statements_verified=verified,
    )
    return ACFResult(
        band=band,
        band_label=_BAND_LABELS[band],
        score=score,
        explanation=explanation,
        note=explanation,
        components=components,
        applied_ceiling=strength.applied_ceiling,
        config_version=strength.config_version,
        claim_level=strength.claim_level,
        question_type=strength.question_type,
    )


def _revise_call(query: str, draft: str, context: str, plan_type: str | None) -> tuple[str, int]:
    """Call the reviser, retrying failed or empty replies while the request budget allows."""
    attempts = max(1, int(os.environ.get("RAG_RARR_REVISE_ATTEMPTS", "3") or 3))
    messages = [
        {"role": "system", "content": _REVISE_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nDraft answer:\n{draft}\n\nEvidence:\n{context}"},
    ]
    for attempt in range(1, attempts + 1):
        revised = llm_chat_complete(
            messages,
            model=model_for_plan(plan_type),
            max_tokens=REVISE_MAX_TOKENS,
            timeout_s=_timeout(),
            purpose="rarr_revise",
        ).strip()
        if revised:
            return revised, attempt
        left = remaining_seconds(None)
        if attempt == attempts or (left is not None and left < _RETRY_MIN_LEFT_S):
            return "", attempt
        time.sleep(_RETRY_PAUSE_S)
    return "", attempts


def _source_row(ref: Any) -> dict[str, Any]:
    item = ref.item if isinstance(ref.item, dict) else {}
    body = str(item.get("content") or item.get("text") or "")
    return {
        "id": ref.source_id,
        "citation": ref.citation_line,
        "kind": str(item.get("source") or item.get("_context_kind") or ""),
        "preview": body[:500],
    }


def revise_with_evidence(
    query: str,
    draft: str,
    state: dict[str, Any],
    *,
    plan_type: str | None = None,
    reference_date: date | None = None,
) -> Revision | None:
    """Correct the draft against the graph's reranked evidence.

    None when there is no draft, or when evidence exists but the reviser returned nothing.
    With no evidence at all, the draft is returned with a no-evidence note and score 0.
    """
    if not draft:
        return None
    items = pin_bq_context_first(filter_context_items(list(state.get("reranked_context") or [])))
    context, registry = _build_context_block(items, _context_max_chars(), _chunk_max_chars())
    trace: dict[str, Any] = {
        "evidence_sources": [_source_row(ref) for ref in registry],
        "evidence_chars": len(context),
        "revise_called": bool(registry),
        "revise_raw": "",
        "revise_ms": 0.0,
    }
    answer = draft
    if registry:
        started = time.perf_counter()
        revised, attempts = _revise_call(query, draft, context, plan_type)
        trace["revise_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        trace["revise_attempts"] = attempts
        trace["revise_raw"] = revised
        if not revised:
            # Evidence existed but the check failed (timeout or empty reply): never serve the
            # unchecked draft over the evidence answer.
            return None
        answer = revised
    rows = statement_rows(answer, {ref.source_id for ref in registry})
    cited_ids = {sid for row in rows for sid in row["sources"]}
    if not cited_ids:
        answer = f"{answer}\n\n{NO_EVIDENCE_NOTE}"
    cited_refs = [ref for ref in registry if ref.source_id in cited_ids]
    citations = referenced_citations(answer, cited_refs) if cited_refs else []
    acf = score_revision(
        rows,
        cited_refs,
        query=query,
        decomposition=state.get("decomposition") if isinstance(state.get("decomposition"), dict) else None,
        reference_date=reference_date,
    )
    return Revision(answer, citations, acf, rows, trace)


def apply_revision(out: dict[str, Any], stages: dict[str, Any], revision: Revision) -> None:
    """Make the revision the answer; keep every stage, and the evidence-only answer, in out["rarr"]."""
    components = revision.acf.components or {}
    out["rarr"] = {
        **stages,
        **revision.trace,
        "status": "revised",
        "statements": revision.statements,
        "evidence_score": components.get("evidence_score"),
        "evidence_band": components.get("evidence_band"),
        "coverage": components.get("coverage"),
        "evidence_answer": out.get("answer") or "",
        "evidence_citations": list(out.get("citations") or []),
        "evidence_acf": {k: v for k, v in out.items() if k.startswith("acf_")},
    }
    out["answer"] = revision.answer
    out["citations"] = revision.citations
    out.update(acf_result_to_state(revision.acf))
    out["answer_mode"] = "rarr"
