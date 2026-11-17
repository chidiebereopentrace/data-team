"""Admit scoped passages for explicit document requirements."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.agri_entities import CROP_ENTITY_RE
from ml.rag.chatbot.agri_measure_ontology import MEASURES, explicit_measure_ids
from ml.rag.chatbot.bq_table_schema_yaml import fold_accents
from ml.rag.chatbot.evidence_kinds import is_document_item
from ml.rag.chatbot.geo_iso3 import name_to_iso3
from ml.rag.chatbot.turn_contract import TurnContract

_GENERIC_TERMS = frozenset(
    "about after also and assessments before between both change changed cite compare could describe "
    "difference does effect effects explain explanation find findings from have impact impacted "
    "influence into passage relationship report reports research role said says show study studies that "
    "their them these this those through under versus what when where which with would years "
    "avec comment dans demande effet entre explique impact influence lien pour rapport quels selon".split()
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_FIGURE = r"\b\d+(?:[.,]\d+)?\s*"
_FIGURE_UNITS = {
    "employment_share": re.compile(_FIGURE + r"(?:%|per\s+cent\b|percent\b)", re.I),
    "yield": re.compile(
        _FIGURE + r"(?:(?:t|kg)\s*/\s*ha\b|(?:tonnes?|tons?|kilograms?)\s+per\s+hectare\b)", re.I,
    ),
}
_EXPLANATION = re.compile(r"\b(?:explain|effects?|impacts?|relationship|influence|effets?|lien)\b", re.I)
_LINK = re.compile(
    r"\b(?:affect|attribut|associat|caus|contribut|driv|effect|impact|increas|influenc|led|link|"
    r"reduc|decreas|raised|lowered|result|associ|contribu|augment|diminu)\w*\b",
    re.I,
)
_NAMED_ACTION_PLAN = re.compile(r"\b(?:what|how)\s+does\s+(.+?\baction\s+plan)\b", re.I)


def _terms(text: str) -> set[str]:
    return {
        token.replace("isation", "ization").removesuffix("s")
        for token in re.findall(r"[^\W\d_]{4,}", fold_accents(text).casefold())
        if token not in _GENERIC_TERMS
    }


def requirement_document_candidates(
    turn: TurnContract, items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require a matching specific subject and keep figures as separate candidates."""
    requested_ids = {fact.measure_id for fact in turn.facts}
    places = {code for geo in turn.geo if (code := name_to_iso3(geo))}
    place_terms = _terms(" ".join(turn.geo))
    topics = []
    for document in turn.documents:
        question = document.question
        indicators = set(explicit_measure_ids(question)) & requested_ids
        indicator_terms = {
            term for indicator in indicators for alias in MEASURES[indicator].aliases for term in _terms(alias)
        }
        crops = {match.group().casefold() for match in CROP_ENTITY_RE.finditer(question)}
        specific_terms = _terms(question) - place_terms - indicator_terms - _terms(" ".join(crops))
        title = _NAMED_ACTION_PLAN.search(turn.original_question or question)
        title_terms = _terms(title[1]) - place_terms if title else set()
        topics.append((question, indicators, crops, specific_terms - title_terms, title_terms))

    accepted: list[dict[str, Any]] = []
    for item in items:
        if not is_document_item(item):
            continue
        content = str(item.get("content") or item.get("text") or "")
        raw_meta = item.get("metadata")
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        country = name_to_iso3(str(
            meta.get("country_iso3") or meta.get("geo_country_primary") or meta.get("country_name") or ""
        ))
        if places and not (
            country in places or any(re.search(rf"\b{re.escape(geo)}\b", content, re.I) for geo in turn.geo)
        ):
            continue
        years = {int(year) for year in _YEAR.findall(content)}
        overlaps = any(
            period.start.year <= year <= period.end.year
            for period in turn.time_spec.windows for year in years
        )
        if turn.time_spec.windows and years and not overlaps:
            continue
        indicators = set(explicit_measure_ids(content)) & requested_ids
        crops_in_passage = {match.group().casefold() for match in CROP_ENTITY_RE.finditer(content)}
        terms = _terms(content)
        matched = []
        for question, required, crops, specific, title_terms in topics:
            title_text = str(meta.get("article_title") or meta.get("title") or "")
            if title_terms and not title_terms <= terms | _terms(title_text):
                continue
            if not required <= indicators or not crops <= crops_in_passage:
                continue
            if specific and not terms & specific or not specific and not required:
                continue
            if specific and _EXPLANATION.search(question) and not any(
                _LINK.search(sentence) and _terms(sentence) & specific
                and required <= set(explicit_measure_ids(sentence))
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", content)
            ):
                continue
            matched.append(question)
        fact_candidates = [
            mid for mid in sorted(indicators)
            if overlaps and mid in _FIGURE_UNITS and _FIGURE_UNITS[mid].search(content)
        ]
        if matched or fact_candidates:
            meta["document_requirements"] = matched
            meta["document_fact_candidates"] = fact_candidates
            accepted.append({**item, "metadata": meta})
    return accepted


def document_requirement_coverage(
    turn: TurnContract, items: list[dict[str, Any]],
    *, corpus_results: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    corpus_statuses = {
        result.get("status") for result in (corpus_results or {}).values() if isinstance(result, dict)
    }
    failed_statuses = corpus_statuses & {"error", "timeout", "busy"}
    if corpus_statuses and corpus_statuses == {"timeout"}:
        missing_status = "document_search_timed_out"
    elif corpus_statuses and corpus_statuses == failed_statuses:
        missing_status = "document_search_unavailable"
    elif failed_statuses:
        missing_status = "partial_document_search"
    else:
        missing_status = "no_matching_passage"
    counts = dict.fromkeys((doc.question for doc in turn.documents), 0)
    for item in items:
        meta = item.get("metadata")
        if not isinstance(meta, dict):
            continue
        questions = meta.get("document_requirements")
        if not isinstance(questions, list):
            continue
        for question in questions:
            if question in counts:
                counts[question] += 1
    return [
        {"question": question, "status": "candidate_available" if count else missing_status, "count": count}
        for question, count in counts.items()
    ]
