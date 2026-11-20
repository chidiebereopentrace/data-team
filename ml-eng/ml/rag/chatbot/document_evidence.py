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
    "improvement improvements improving changes trends implications "
    "influence into passage relationship report reports research role said says show study studies that "
    "their them these this those through under versus what when where which with would years "
    "avec comment dans demande effet entre explique impact influence lien pour rapport quels selon".split()
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_FIGURE = r"\b\d+(?:[.,]\d+)?\s*"
_FIGURE_UNITS = {
    "production": re.compile(r"\b\d[\d, .]*\s*(?:(?:million|thousand)\s+)?(?:tonnes?|tons?|kg|t)\b(?!\s*(?:/|per\s+hectare))", re.I),
    "employment_share": re.compile(_FIGURE + r"(?:%|per\s+cent\b|percent\b)", re.I),
    "yield": re.compile(
        _FIGURE + r"(?:(?:t|kg)\s*/\s*ha\b|(?:tonnes?|tons?|kilograms?)\s+per\s+hectare\b)", re.I,
    ),
}
_EXPLANATION = re.compile(r"\b(?:explain|effects?|impacts?|relationship|influence|effets?|lien)\b", re.I)
_LINK = re.compile(
    r"\b(?:affect|attribut|associat|caus|contribut|driv|effect|impact|improv|increas|influenc|led|link|"
    r"reduc|decreas|raised|lowered|result|associ|contribu|augment|diminu)\w*\b",
    re.I,
)
_NAMED_DOCUMENT = re.compile(
    r"\b(?:what|how)\s+does\s+(.+?\b(?:action\s+plan|policy|strategy|report)"
    r"(?:\s+\d{4}\s*[–—-]\s*\d{4})?)\s+(?:say|state|recommend|address|provide|cover|define)\b",
    re.I,
)


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
    windows = turn.time_spec.windows
    first_year = min((period.start.year for period in windows), default=0)
    last_year = max((period.end.year for period in windows), default=0)
    topics = []
    named = _NAMED_DOCUMENT.search(turn.original_question)
    title_terms = _terms(named[1]) - place_terms if named else set()
    edition = set(_YEAR.findall(named[1])) if named else set()
    for document in turn.documents:
        question = document.question
        indicators = set(explicit_measure_ids(question)) & requested_ids
        indicator_terms = {
            term for indicator in indicators for alias in MEASURES[indicator].aliases for term in _terms(alias)
        }
        crops = {match.group().casefold() for match in CROP_ENTITY_RE.finditer(question)}
        specific_terms = _terms(question) - place_terms - indicator_terms - _terms(" ".join(crops))
        topics.append((question, indicators, crops, specific_terms - title_terms, title_terms))

    accepted: list[dict[str, Any]] = []
    for item in items:
        if not is_document_item(item):
            continue
        content = str(item.get("content") or item.get("text") or "")
        raw_meta = item.get("metadata")
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        if named:
            title = str(meta.get("document_title") or meta.get("title") or meta.get("article_title") or "")
            # Mentions in the body of another report do not establish document identity.
            if not title_terms <= _terms(title) or not edition <= set(_YEAR.findall(title)):
                continue
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
            for period in windows for year in years
        )
        publication = str(meta.get("publication_year") or meta.get("report_year") or "")[:4]
        published_in_window = publication.isdigit() and first_year <= int(publication) <= last_year
        if windows and years and not any(first_year <= year <= last_year for year in years) and not published_in_window:
            continue
        indicators = set(explicit_measure_ids(content)) & requested_ids
        crops_in_passage = {match.group().casefold() for match in CROP_ENTITY_RE.finditer(content)}
        terms = _terms(content)
        matched = []
        for question, required, crops, specific, _title_terms in topics:
            if not required <= indicators or not crops <= crops_in_passage:
                continue
            if specific and not terms & specific or not specific and not required:
                continue
            if specific and _EXPLANATION.search(question):
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", content):
                    if not _LINK.search(sentence) or not _terms(sentence) & specific:
                        continue
                    sentence_indicators = set(explicit_measure_ids(sentence))
                    if required and required <= sentence_indicators:
                        break
                    if not required and (not requested_ids or requested_ids & sentence_indicators):
                        break
                else:
                    continue
            matched.append(question)
        fact_candidates = [
            mid for mid in sorted(indicators)
            if overlaps and mid in _FIGURE_UNITS and _FIGURE_UNITS[mid].search(content)
            and (mid != "production" or any(
                "production" in explicit_measure_ids(sentence)
                and _FIGURE_UNITS["production"].search(sentence)
                and CROP_ENTITY_RE.search(sentence)
                and any(period.start.year <= int(year) <= period.end.year
                        for year in _YEAR.findall(sentence) for period in windows)
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", content)
            ))
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
