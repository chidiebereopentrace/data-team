"""Rule-based claim_level + question_type classification for ACF Path B."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from acf.enums import ClaimLevel, QuestionType

_LOCAL_RE = re.compile(
    r"\b(ward|village|kebele|parish|settlement|community|locality|commune)\b",
    re.IGNORECASE,
)
_SUBNAT_RE = re.compile(
    r"\b(county|counties|district|province|region|regions|state|governorate|"
    r"lga|department|oblast|prefecture|municipality|sub-?national|adm1|admin\s*1)\b",
    re.IGNORECASE,
)
_TIME_SENSITIVE_RE = re.compile(
    r"\b(price|prices|this\s+(month|week|season)|current|today|latest|"
    r"right\s+now|now|recent|this\s+year|harvest\s+season|market\s+rate)\b",
    re.IGNORECASE,
)
_STRUCTURAL_RE = re.compile(
    r"\b(decade|decades|long[- ]term|over\s+the\s+years|historically|"
    r"trend\s+over|multi[- ]year|past\s+\d+\s+years|since\s+\d{4})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ACFQuestionClass:
    claim_level: ClaimLevel
    question_type: QuestionType


def _parse_iso_date(raw: Any) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def classify_acf_question(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    reference_date: date | None = None,
) -> ACFQuestionClass:
    """Classify claim_level and question_type from query + decomposition."""
    q = (query or "").strip()
    dec = decomposition if isinstance(decomposition, dict) else {}
    ref = reference_date or date.today()

    # --- claim_level ---
    geo_bits: list[str] = []
    geography = dec.get("geography")
    if isinstance(geography, list):
        geo_bits.extend(str(g) for g in geography if g)
    elif isinstance(geography, str) and geography.strip():
        geo_bits.append(geography.strip())
    entities = dec.get("entities")
    if isinstance(entities, list):
        geo_bits.extend(str(e) for e in entities if e)
    hay_geo = " ".join(geo_bits) + " " + q

    if _LOCAL_RE.search(hay_geo):
        claim_level = ClaimLevel.LOCAL
    elif _SUBNAT_RE.search(hay_geo):
        claim_level = ClaimLevel.SUB_NATIONAL
    else:
        claim_level = ClaimLevel.NATIONAL

    # --- question_type ---
    if _STRUCTURAL_RE.search(q):
        question_type = QuestionType.STRUCTURAL
    elif _TIME_SENSITIVE_RE.search(q):
        question_type = QuestionType.TIME_SENSITIVE
    else:
        t_start = _parse_iso_date(dec.get("time_start"))
        t_end = _parse_iso_date(dec.get("time_end"))
        # Narrow recent window → time_sensitive; multi-year span → structural
        if t_start and t_end and (t_end - t_start).days >= 365 * 3:
            question_type = QuestionType.STRUCTURAL
        elif t_start and (ref - t_start).days <= 366:
            question_type = QuestionType.TIME_SENSITIVE
        elif t_end and abs((ref - t_end).days) <= 180:
            question_type = QuestionType.TIME_SENSITIVE
        else:
            # Default: agricultural monitoring leans time-sensitive for price-ish domains
            domains = dec.get("domains") or []
            domain_text = " ".join(str(d) for d in domains).lower() if isinstance(domains, list) else str(domains).lower()
            if any(k in domain_text for k in ("price", "market", "weather", "rainfall", "drought")):
                question_type = QuestionType.TIME_SENSITIVE
            else:
                question_type = QuestionType.STRUCTURAL

    return ACFQuestionClass(claim_level=claim_level, question_type=question_type)


def classify_acf_question_values(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    reference_date: date | None = None,
) -> tuple[str, str]:
    """String values for graph state / Langfuse."""
    c = classify_acf_question(query, decomposition, reference_date=reference_date)
    return c.claim_level.value, c.question_type.value
