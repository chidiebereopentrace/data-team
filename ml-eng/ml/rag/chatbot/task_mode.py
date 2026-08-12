"""Unified task_mode router for full_rag specializations."""
from __future__ import annotations

import re
from typing import Any, Literal

from ml.rag.chatbot.analytical_intent import is_analytical_query
from ml.rag.chatbot.export_intent import detect_export_intent
from ml.rag.chatbot.generator import is_numeric_data_query, is_ranking_numeric_query
from ml.rag.chatbot.geo_regions import detect_regions_in_text

TaskMode = Literal[
    "clarify",
    "analytical",
    "fact_lookup",
    "briefing",
    "data_export_only",
    "chat",
]

_BRIEFING_RE = re.compile(
    r"\b("
    r"brief(?:\s+me)?|briefing|latest|headline|headlines|what'?s\s+new|"
    r"this\s+week|this\s+month|news\s+update|quick\s+update|situation\s+update"
    r")\b",
    re.IGNORECASE,
)

_EXPORT_ONLY_RE = re.compile(
    r"\b("
    r"just\s+(the\s+)?data|only\s+(the\s+)?data|give\s+me\s+(the\s+)?data|"
    r"no\s+(essay|analysis|narrative|prose|report\s+text)|"
    r"data\s+only|csv\s+only|export\s+only|table\s+only"
    r")\b",
    re.IGNORECASE,
)

_AGRI_RE = re.compile(
    r"\b("
    r"agricultur(?:e|al)|farming|crops?|commodit(?:y|ies)|production|"
    r"yield|livestock|cereal|maize|rice|cassava|sorghum|millet|wheat|"
    r"price|prices|faostat|food\s+security"
    r")\b",
    re.IGNORECASE,
)

_CROP_ENTITY_TERMS = frozenset(
    {
        "maize",
        "corn",
        "rice",
        "cassava",
        "sorghum",
        "millet",
        "wheat",
        "soybean",
        "soy",
        "cotton",
        "cocoa",
        "coffee",
        "tea",
        "sugarcane",
        "groundnut",
        "yam",
        "plantain",
        "banana",
        "livestock",
        "cattle",
        "poultry",
        "production",
        "yield",
        "price",
        "prices",
        "export",
        "imports",
        "gdp",
    }
)

_YEAR_ASK_RE = re.compile(
    r"\b(in|for|during)\s+(19|20)\d{2}\b|\b(19|20)\d{2}\b",
    re.IGNORECASE,
)


def _geo_list(decomposition: dict[str, Any] | None) -> list[str]:
    if not isinstance(decomposition, dict):
        return []
    geo = decomposition.get("geography")
    if not isinstance(geo, list):
        return []
    return [str(g).strip() for g in geo if str(g).strip()]


def _has_crop_or_commodity(query: str, decomposition: dict[str, Any] | None) -> bool:
    q = (query or "").lower()
    if any(term in q for term in _CROP_ENTITY_TERMS):
        return True
    if not isinstance(decomposition, dict):
        return False
    entities = decomposition.get("entities")
    if isinstance(entities, list):
        for ent in entities:
            text = str(ent or "").strip().lower()
            if any(term in text for term in _CROP_ENTITY_TERMS):
                return True
    return False


def _has_year(decomposition: dict[str, Any] | None, query: str) -> bool:
    if isinstance(decomposition, dict):
        if str(decomposition.get("time_start") or "").strip() or str(decomposition.get("time_end") or "").strip():
            return True
    return bool(_YEAR_ASK_RE.search(query or ""))


def needs_clarify(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    profile_country: str | None = None,
) -> bool:
    """Numeric/agri ask missing country and/or crop (year optional unless implied without a date)."""
    q = (query or "").strip()
    if not q:
        return False
    if is_analytical_query(q, decomposition):
        return False
    if not (is_numeric_data_query(q, decomposition) or _AGRI_RE.search(q)):
        return False

    has_region = bool(detect_regions_in_text(q))
    has_country = (
        bool(_geo_list(decomposition))
        or bool((profile_country or "").strip())
        or has_region
    )
    has_crop = _has_crop_or_commodity(q, decomposition)
    africa = bool(isinstance(decomposition, dict) and decomposition.get("africa_default"))

    # Continental ranking with crop can proceed without an explicit country.
    if is_ranking_numeric_query(q) and (africa or has_region) and has_crop:
        return False

    if not has_country or not has_crop:
        return True

    # Time-specific ask without any year/period in query or decomposition.
    if re.search(r"\b(that\s+year|which\s+year|what\s+year|in\s+what\s+year)\b", q, re.IGNORECASE):
        if not _has_year(decomposition, q):
            return True
    return False


def is_briefing_query(query: str, decomposition: dict[str, Any] | None = None) -> bool:
    q = (query or "").strip()
    if not q or not _BRIEFING_RE.search(q):
        return False
    if is_analytical_query(q, decomposition):
        return False
    return bool(_AGRI_RE.search(q) or _geo_list(decomposition) or detect_regions_in_text(q))


def is_data_export_only_query(query: str, decomposition: dict[str, Any] | None = None) -> bool:
    q = (query or "").strip()
    export = detect_export_intent(q)
    if not export:
        return False
    if is_analytical_query(q, decomposition):
        # Analytical report PDFs are not export-only.
        return False
    if _EXPORT_ONLY_RE.search(q):
        return True
    # Export without report/compare prose → caption + files.
    if export in ("csv", "chart") and not re.search(
        r"\b(explain|analy[sz]e|report|compar|why|brief)\b", q, re.IGNORECASE
    ):
        return True
    if export in ("pdf", "docx", "multi") and re.search(
        r"\b(data|table|csv|numbers|figures)\b", q, re.IGNORECASE
    ) and not re.search(r"\b(analytical|detailed\s+report|compar)\b", q, re.IGNORECASE):
        return True
    return False


def is_fact_lookup_query(query: str, decomposition: dict[str, Any] | None = None) -> bool:
    q = (query or "").strip()
    if not q or is_analytical_query(q, decomposition):
        return False
    # News briefings are not single-fact lookups even when a crop entity is present.
    if is_briefing_query(q, decomposition):
        return False
    if not (is_numeric_data_query(q, decomposition) or is_ranking_numeric_query(q)):
        return False
    geo = _geo_list(decomposition)
    if len(geo) >= 2:
        return False
    return True


def resolve_task_mode(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    profile_country: str | None = None,
) -> TaskMode:
    """Precedence: clarify → analytical → data_export_only → fact_lookup → briefing → chat."""
    q = (query or "").strip()
    if not q:
        return "chat"

    if needs_clarify(q, decomposition, profile_country=profile_country):
        return "clarify"
    if is_analytical_query(q, decomposition):
        return "analytical"
    if is_data_export_only_query(q, decomposition):
        return "data_export_only"
    if is_fact_lookup_query(q, decomposition):
        return "fact_lookup"
    if is_briefing_query(q, decomposition):
        return "briefing"
    return "chat"


def clarify_answer(query: str = "") -> str:
    """Deterministic prompt asking for missing slots."""
    _ = query
    return (
        "I need a bit more detail before I can pull OpenTrace structured data.\n\n"
        "Please include:\n"
        "- **Country** (or region such as West Africa)\n"
        "- **Crop or commodity** (e.g. maize, rice, cassava)\n"
        "- **Time period** when you care about a specific year or range (e.g. 2020 or 2015–2022)\n\n"
        "Example: “What was maize production in Nigeria in 2020?”"
    )


__all__ = [
    "TaskMode",
    "clarify_answer",
    "is_briefing_query",
    "is_data_export_only_query",
    "is_fact_lookup_query",
    "needs_clarify",
    "resolve_task_mode",
]
