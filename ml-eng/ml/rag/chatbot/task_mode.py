"""Unified task_mode router for full_rag specializations."""
from __future__ import annotations

import re
from typing import Any, Literal

from ml.rag.chatbot.agri_measure_ontology import (
    MeasureHit,
    effective_crop_required,
    resolve_measure,
    wants_africa_panel,
    wants_data_export_panel,
)
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
    "research",
    "chat",
]

_BRIEFING_RE = re.compile(
    r"\b("
    r"brief(?:\s+me)?|briefing|latest|headline|headlines|what'?s\s+new|"
    r"this\s+week|this\s+month|news\s+update|quick\s+update|situation\s+update|"
    r"\bnews\b"
    r")\b",
    re.IGNORECASE,
)

_RESEARCH_RE = re.compile(
    r"\b("
    r"what\s+does\s+research\s+say|literature\s+on|according\s+to\s+research|"
    r"studies\s+show|academic\s+research|peer[- ]reviewed|synthesis\s+of\s+research"
    r")\b",
    re.IGNORECASE,
)

_EXPORT_ONLY_RE = re.compile(
    r"\b("
    r"just\s+(the\s+)?data|only\s+(the\s+)?data|give\s+me\s+(the\s+)?data|"
    r"no\s+(essay|analysis|narrative|prose|report\s+text)|"
    r"data\s+only|csv\s+only|export\s+only|table\s+only|numbers?\s+only|copy\b"
    r")\b",
    re.IGNORECASE,
)

_AGRI_RE = re.compile(
    r"\b("
    r"agricultur(?:e|al)|farming|crops?|commodit(?:y|ies)|production|"
    r"yields?|livestock|cereal|maize|rice|cassava|sorghum|millet|wheat|"
    r"price|prices|faostat|food\s+security|ipc|rainfall|soil|gdp|export|import|"
    r"investment|climate|ndvi"
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
    if any(re.search(rf"\b{re.escape(term)}\b", q) for term in _CROP_ENTITY_TERMS):
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


def _measure_hit(query: str, decomposition: dict[str, Any] | None) -> MeasureHit | None:
    return resolve_measure(query, decomposition)


def needs_clarify(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    profile_country: str | None = None,
    measure_hit: MeasureHit | None = None,
) -> bool:
    """Slot-aware clarify using measure ontology (crop not always required)."""
    q = (query or "").strip()
    if not q:
        return False
    if is_analytical_query(q, decomposition):
        return False

    hit = measure_hit if measure_hit is not None else _measure_hit(q, decomposition)
    if hit and hit.measure.country_is_answer:
        return False
    if hit and hit.measure.id in ("news_briefing", "research_synthesis", "research_meta"):
        return False
    if isinstance(decomposition, dict) and (
        decomposition.get("africa_default") or decomposition.get("africa_panel")
    ):
        # Continental ranking / full panel: country is not a missing slot.
        crop_needed = effective_crop_required(hit) if hit else True
        if not crop_needed:
            return False
        if _has_crop_or_commodity(q, decomposition):
            return False
        # Panel/ranking with crop required but missing → clarify crop only (handled below).

    if not (is_numeric_data_query(q, decomposition) or _AGRI_RE.search(q)):
        return False

    has_region = bool(detect_regions_in_text(q))
    africa = bool(
        isinstance(decomposition, dict)
        and (decomposition.get("africa_default") or decomposition.get("africa_panel"))
    )
    has_country = (
        bool(_geo_list(decomposition))
        or bool((profile_country or "").strip())
        or has_region
        or africa
        or bool(hit and hit.measure.country_is_answer)
        or (hit is not None and not hit.measure.geography_required)
    )
    has_crop = _has_crop_or_commodity(q, decomposition)
    # Crop is required only when the measure says so, or for numeric/ranking asks
    # with no measure hit — not for soft opinion / agribusiness / food-security.
    if hit is not None:
        crop_required = bool(effective_crop_required(hit))
    else:
        crop_required = bool(
            is_numeric_data_query(q, decomposition) or is_ranking_numeric_query(q)
        )

    # Continental ranking with crop (when required) can proceed without an explicit country.
    if is_ranking_numeric_query(q) and (africa or has_region) and (has_crop or not crop_required):
        return False

    if wants_africa_panel(q) and (has_crop or not crop_required):
        return False

    if not has_country:
        return True
    if crop_required and not has_crop:
        return True

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
    hit = _measure_hit(q, decomposition)
    if hit and hit.measure.id == "news_briefing":
        return True
    return bool(_AGRI_RE.search(q) or _geo_list(decomposition) or detect_regions_in_text(q))


def is_research_query(query: str, decomposition: dict[str, Any] | None = None) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if _RESEARCH_RE.search(q):
        return True
    hit = _measure_hit(q, decomposition)
    return bool(hit and hit.measure.id in ("research_synthesis", "research_meta"))


def is_data_export_only_query(query: str, decomposition: dict[str, Any] | None = None) -> bool:
    q = (query or "").strip()
    # Africa panels / copy-numbers beat "past N years" analytical-report detection
    # unless the user explicitly asked for a report/analysis essay.
    if wants_data_export_panel(q) and not re.search(
        r"\b(analytical\s+report|detailed\s+report|explain\b|why\b|compar(?:e|ative)\s+analytics)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    export = detect_export_intent(q)
    if not export and not _EXPORT_ONLY_RE.search(q):
        return False
    if is_analytical_query(q, decomposition):
        return False
    if _EXPORT_ONLY_RE.search(q):
        return True
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
    if is_briefing_query(q, decomposition) or is_research_query(q, decomposition):
        return False
    if wants_africa_panel(q):
        return False
    # Regional multi-pillar assessments are briefing/analytical, not single-fact BQ.
    if detect_regions_in_text(q) and re.search(
        r"\b(assessment|assess\b|hunger|production.+prices|prices.+production|"
        r"food\s+security\s+risk|insecurit)\b",
        q,
        re.IGNORECASE,
    ):
        return False
    if not (is_numeric_data_query(q, decomposition) or is_ranking_numeric_query(q)):
        hit = _measure_hit(q, decomposition)
        if not (hit and hit.measure.default_task_mode == "fact_lookup"):
            return False
    geo = _geo_list(decomposition)
    # Expanded region lists should not block fact_lookup for single-country asks;
    # multi-country without africa flags → not a single fact.
    if len(geo) >= 2 and not (
        isinstance(decomposition, dict)
        and (decomposition.get("africa_default") or decomposition.get("expanded_regions"))
    ):
        return False
    return True


def resolve_task_mode(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    profile_country: str | None = None,
) -> TaskMode:
    """Precedence: clarify → analytical → data_export_only → fact_lookup → research → briefing → chat."""
    q = (query or "").strip()
    if not q:
        return "chat"

    hit = _measure_hit(q, decomposition)
    if hit and hit.measure.id == "investor_best_country":
        return "analytical"

    if needs_clarify(q, decomposition, profile_country=profile_country, measure_hit=hit):
        return "clarify"
    # Panels / numbers-only exports beat analytical "past N years" report detection.
    if is_data_export_only_query(q, decomposition):
        return "data_export_only"
    if is_analytical_query(q, decomposition) or (
        hit
        and hit.measure.default_task_mode == "analytical"
        and hit.measure.country_is_answer
    ):
        return "analytical"
    if is_fact_lookup_query(q, decomposition):
        return "fact_lookup"
    if is_research_query(q, decomposition):
        return "research"
    if is_briefing_query(q, decomposition):
        return "briefing"
    if hit and hit.measure.default_task_mode in (
        "fact_lookup",
        "briefing",
        "research",
        "analytical",
        "data_export_only",
    ):
        mode = hit.measure.default_task_mode
        if mode == "clarify":
            return "chat"
        return mode  # type: ignore[return-value]
    return "chat"


def clarify_answer(
    query: str = "",
    *,
    decomposition: dict[str, Any] | None = None,
    measure_hit: MeasureHit | None = None,
) -> str:
    """Slot-aware clarify prompt — never the shallow maize-Nigeria-only template."""
    q = (query or "").strip()
    hit = measure_hit if measure_hit is not None else (_measure_hit(q, decomposition) if q else None)
    missing: list[str] = []
    has_geo = bool(_geo_list(decomposition)) or bool(
        isinstance(decomposition, dict)
        and (decomposition.get("africa_default") or decomposition.get("africa_panel"))
    )
    has_crop = _has_crop_or_commodity(q, decomposition) if q else False
    crop_req = effective_crop_required(hit) if hit else True
    geo_req = True if hit is None else hit.measure.geography_required
    if hit and hit.measure.country_is_answer:
        geo_req = False

    if geo_req and not has_geo and not detect_regions_in_text(q):
        missing.append("**Geography** — a country, region (e.g. West Africa), or continental Africa panel")
    if crop_req and not has_crop:
        missing.append("**Crop or commodity** (e.g. maize, rice, coffee) — required for this measure")
    if re.search(r"\b(that\s+year|which\s+year)\b", q, re.IGNORECASE) and not _has_year(decomposition, q):
        missing.append("**Time period** (e.g. 2020 or 2018–2022)")

    measure_line = ""
    if hit:
        measure_line = f"Detected measure: `{hit.measure.id}`.\n\n"

    if not missing:
        missing = [
            "**Country or region** when the question is place-specific",
            "**Crop or commodity** when asking production, yield, prices, or trade",
            "**Time period** when you care about a specific year or range",
        ]

    examples = [
        "What was maize **yield** in Kenya in 2020?",
        "Retail maize prices in Ethiopia recently",
        "IPC food security outlook for Somalia right now",
        "Coffee exports from Uganda in 2019",
        "Maize yields for all African countries, past 5 years (numbers only)",
    ]
    if hit and hit.measure.id == "soil":
        examples = ["Soil organic matter / fertility indicators for Zambia", *examples[:2]]
    if hit and hit.measure.id == "climate":
        examples = ["Rainy season / rainfall patterns for Senegal", *examples[:2]]

    return (
        "I need a bit more detail before I can pull OpenTrace structured data.\n\n"
        f"{measure_line}"
        "Please include:\n"
        + "\n".join(f"- {m}" for m in missing)
        + "\n\nSuggested prompts:\n"
        + "\n".join(f"- “{e}”" for e in examples[:4])
    )


__all__ = [
    "TaskMode",
    "clarify_answer",
    "is_briefing_query",
    "is_data_export_only_query",
    "is_fact_lookup_query",
    "is_research_query",
    "needs_clarify",
    "resolve_task_mode",
]
