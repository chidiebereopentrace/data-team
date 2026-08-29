"""
Category segments for audience-aware generation (Ask ADZA backend contract).
Maps backend category enums to short system-prompt instructions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

# Public catalog for GET /v1/meta (id, label, description).
CATEGORIES: list[dict[str, str]] = [
    {
        "id": "Government",
        "label": "Government & Public Institutions",
        "description": (
            "Planning, policy design, and resource allocation: production trends, "
            "regional risks, and food security pressures without waiting months for reports."
        ),
    },
    {
        "id": "NGOs",
        "label": "Foundations, NGOs & Development Partners",
        "description": (
            "Priority regions, overlapping climate–nutrition–market risks, and program "
            "relevance using consistent, localized intelligence rather than fragmented data."
        ),
    },
    {
        "id": "Agribusinesses",
        "label": "Agribusinesses & Financial Institutions",
        "description": (
            "Production stability, market volatility, and regional risk exposure for sourcing, "
            "investment, and agricultural finance decisions."
        ),
    },
    {
        "id": "Farmers",
        "label": "Farmers, Cooperatives & Communities",
        "description": (
            "Clearer insights on rainfall, markets, and production trends via trusted framing—"
            "avoid raw tables and jargon; favor plain language and actionable takeaways."
        ),
    },
]

_CATEGORY_IDS = frozenset(c["id"] for c in CATEGORIES)

# Compact instructions appended to the generator system prompt.
_CATEGORY_INSTRUCTIONS: dict[str, str] = {
    "Government": (
        "Audience: government and public institutions (ministries, planners, agencies). "
        "Preferred framing: policy synthesis for planning and resource allocation — "
        "national/sub-national production, climate, food-security, and trade pressures. "
        "Answer shape: (1) planning takeaway in one sentence; (2) evidence by region or time "
        "with named indicators; (3) explicit uncertainty, data gaps, and what would change "
        "the recommendation. Prefer actionable monitoring cues over academic literature dumps. "
        "Jargon: policy and indicator language is fine; define uncommon metrics once; "
        "avoid farm-level how-to tips and vendor sales tone."
    ),
    "NGOs": (
        "Audience: development partners, foundations, and NGOs. "
        "Preferred framing: program-relevant geographic priorities and overlapping "
        "climate–nutrition–market–conflict risks with comparable regional angles. "
        "Answer shape: who/where is most affected; which risk overlap matters for targeting; "
        "what to monitor next; cite consistent units and geographies. "
        "Prefer operationally useful synthesis over long academic digressions. "
        "Jargon: keep accessible to program managers; avoid pure theory and "
        "commodity-trading jargon unless the question asks for it."
    ),
    "Agribusinesses": (
        "Audience: agribusiness and financial institutions. "
        "Preferred framing: production stability, price/volatility, trade and sourcing risk, "
        "and exposure that affects commercial or lending decisions. "
        "Answer shape: analytical brief — key signal first, then drivers and exposure, "
        "then practical implication for sourcing/investment when evidence supports it. "
        "When the question is investment attractiveness, rank or compare on multi-signal "
        "evidence (production/trade/GDP/prices/risk) and state confidence limits. "
        "Jargon: market and risk terms are fine; avoid pastoral storytelling and "
        "generic livelihood advice."
    ),
    "Farmers": (
        "Audience: farmers, cooperatives, and communities (often via intermediaries). "
        "Preferred framing: rainfall, markets, soil, and production/yield trends in plain, "
        "local terms — what it means for this season and next decisions. "
        "Answer shape: short actionable bullets — what is happening, what it means on the farm, "
        "and one clear next step when evidence supports it. Never write like a researcher: "
        "no literature reviews, no schema/SQL/table names, no dense citation stacks. "
        "Translate numbers into day-to-day meaning (prices, rain, yield bags/hectare). "
        "If structured data is missing, say so plainly and suggest the missing crop/place/year."
    ),
}

_INVESTOR_OVERLAY = (
    "Investor lens overlay: treat this as an agricultural investment decision brief. "
    "Lead with which countries/signals look relatively stronger on the available evidence; "
    "separate opportunity (production growth, trade, GDP, investment flows) from risk "
    "(price volatility, food-security/IPC pressure). Do not ask the user which country — "
    "country ranking is the answer. Crop is optional context, not a blocker."
)

_RECENCY_OVERLAY: dict[str, str] = {
    "live": (
        "Recency: prefer the most recent assessments, market prices, IPC/FEWS updates, "
        "and news. Call out when evidence is stale."
    ),
    "near_term": (
        "Recency: prefer recent years and latest available series; deprioritize outdated "
        "PDFs when fresher structured data exists."
    ),
    "point_in_time": (
        "Recency: answer for the specific year/period asked; do not substitute a different year "
        "without saying so."
    ),
    "historical_ok": "",
}


def valid_category_ids() -> frozenset[str]:
    return _CATEGORY_IDS


def is_valid_category(category: str) -> bool:
    return category.strip() in _CATEGORY_IDS


def instruction_for_category(
    category: str | None,
    *,
    measure_id: str | None = None,
    recency_tier: str | None = None,
) -> str:
    parts: list[str] = []
    if category:
        key = category.strip()
        base = _CATEGORY_INSTRUCTIONS.get(key, "")
        if base:
            parts.append(base)
    mid = (measure_id or "").strip()
    if mid == "investor_best_country" or (
        mid and "invest" in mid and category and category.strip() in ("Agribusinesses", "Farmers", "Government")
    ):
        parts.append(_INVESTOR_OVERLAY)
    # Farmers category still gets investor overlay only for investor measure.
    tier = (recency_tier or "").strip()
    overlay = _RECENCY_OVERLAY.get(tier, "")
    if overlay:
        parts.append(overlay)
    return " ".join(parts).strip()


# Backward-compat aliases for internal tooling (Streamlit) until migrated.
STAKEHOLDER_TYPES = CATEGORIES


def is_valid_stakeholder_type(stakeholder_type: str) -> bool:
    return is_valid_category(stakeholder_type)


def instruction_for_stakeholder(stakeholder_type: str | None) -> str:
    return instruction_for_category(stakeholder_type)


CategorySource = Literal["explicit", "query", "plan_type", "none"]

_PERSONA_PLAN_TYPES = frozenset({"Farmers", "NGOs", "Government", "Agribusinesses"})

_FARMER_CUES = re.compile(
    r"\b("
    r"farm(?:er|ing)?s?|cooperative|co-op|my field|plant(?:ing)?|harvest|this season|"
    r"crop rotation|what should i|on the farm|livestock feed|bags per|hectare|"
    r"rainfall forecast|when to plant|feed costs"
    r")\b",
    re.I,
)
_NGO_CUES = re.compile(
    r"\b("
    r"program|beneficiar|targeting|humanitarian|field program|development partner|"
    r"\bngo\b|foundation|affected population|intervention|programme monitoring"
    r")\b",
    re.I,
)
_GOV_CUES = re.compile(
    r"\b("
    r"policy|ministr|national plan|budget|indicator|sub-national|administration|"
    r"public institution|planning|resource allocation|government|ipc phase|"
    r"food security assessment|yoy|year-over-year"
    r")\b",
    re.I,
)
_AGB_CUES = re.compile(
    r"\b("
    r"sourcing|investment|portfolio|volatility|exposure|lending|commercial|"
    r"agribusiness|supply chain|procurement|market risk|best country to invest"
    r")\b",
    re.I,
)

_INFERENCE_MIN_SCORE = 2


def _intent_boost(decomposition: dict[str, Any] | None, scores: dict[str, int]) -> None:
    if not isinstance(decomposition, dict):
        return
    intent = str(decomposition.get("intent") or "").strip().lower()
    if intent == "decision_support":
        scores["Government"] = scores.get("Government", 0) + 1
    if intent == "locate":
        scores["NGOs"] = scores.get("NGOs", 0) + 1
    if intent == "compare":
        scores["Agribusinesses"] = scores.get("Agribusinesses", 0) + 1
        scores["Government"] = scores.get("Government", 0) + 1


def infer_category_from_query(
    query: str,
    decomposition: dict[str, Any] | None = None,
) -> str | None:
    """Heuristic keyword + intent inference → persona id or None."""
    blob = query or ""
    if isinstance(decomposition, dict):
        for key in ("entities", "topics"):
            val = decomposition.get(key)
            if isinstance(val, list):
                blob += " " + " ".join(str(x) for x in val)
    scores: dict[str, int] = {
        "Farmers": len(_FARMER_CUES.findall(blob)),
        "NGOs": len(_NGO_CUES.findall(blob)),
        "Government": len(_GOV_CUES.findall(blob)),
        "Agribusinesses": len(_AGB_CUES.findall(blob)),
    }
    _intent_boost(decomposition, scores)
    best_id = max(scores, key=lambda k: scores[k])
    if scores[best_id] < _INFERENCE_MIN_SCORE:
        return None
    runners = [k for k, v in scores.items() if v == scores[best_id] and v >= _INFERENCE_MIN_SCORE]
    if len(runners) > 1:
        return None
    return best_id


def resolve_effective_category(
    *,
    category: str | None,
    plan_type: str | None,
    query: str,
    decomposition: dict[str, Any] | None = None,
) -> tuple[str | None, CategorySource]:
    """Resolve persona for generation: explicit API category wins over query inference."""
    if category and is_valid_category(category):
        return category.strip(), "explicit"
    inferred = infer_category_from_query(query, decomposition)
    if inferred:
        return inferred, "query"
    pt = (plan_type or "").strip()
    if pt in _PERSONA_PLAN_TYPES:
        return pt, "plan_type"
    return None, "none"


StatDensity = Literal["low", "medium", "high"]
VocabularyLevel = Literal["plain", "operational", "policy", "market"]


@dataclass(frozen=True)
class PersonaProseRegister:
    vocabulary: VocabularyLevel
    stat_density: StatDensity
    allow_markdown_tables: bool
    prefer_inline_source_refs: bool
    translate_numbers_to_everyday: bool
    name_indicators_explicitly: bool
    use_bullet_layout: bool


_REGISTER_BY_CATEGORY: dict[str, PersonaProseRegister] = {
    "Government": PersonaProseRegister(
        vocabulary="policy",
        stat_density="high",
        allow_markdown_tables=True,
        prefer_inline_source_refs=True,
        translate_numbers_to_everyday=False,
        name_indicators_explicitly=True,
        use_bullet_layout=False,
    ),
    "NGOs": PersonaProseRegister(
        vocabulary="operational",
        stat_density="medium",
        allow_markdown_tables=True,
        prefer_inline_source_refs=False,
        translate_numbers_to_everyday=False,
        name_indicators_explicitly=True,
        use_bullet_layout=False,
    ),
    "Agribusinesses": PersonaProseRegister(
        vocabulary="market",
        stat_density="high",
        allow_markdown_tables=True,
        prefer_inline_source_refs=False,
        translate_numbers_to_everyday=False,
        name_indicators_explicitly=False,
        use_bullet_layout=False,
    ),
    "Farmers": PersonaProseRegister(
        vocabulary="plain",
        stat_density="low",
        allow_markdown_tables=False,
        prefer_inline_source_refs=False,
        translate_numbers_to_everyday=True,
        name_indicators_explicitly=False,
        use_bullet_layout=True,
    ),
}


def prose_register_for_persona(category: str | None) -> PersonaProseRegister | None:
    if not category:
        return None
    return _REGISTER_BY_CATEGORY.get(category.strip())


def prose_register_addendum(
    category: str | None,
    *,
    task_mode: str = "chat",
    answer_shape: str = "",
    inline_citations: bool = False,
) -> str:
    reg = prose_register_for_persona(category)
    if reg is None:
        return ""
    parts = [
        "PROSE REGISTER:",
        f"Vocabulary: {reg.vocabulary}.",
        f"Statistical detail: {reg.stat_density}.",
    ]
    if reg.allow_markdown_tables:
        parts.append(
            "Use markdown tables when comparing countries, ranks, or multi-year series "
            "if the Context supports it."
        )
    else:
        parts.append("Do not use markdown tables — use short plain sentences or bullets.")
    if reg.translate_numbers_to_everyday:
        parts.append(
            "Translate numbers into everyday farm meaning (prices, rain, bags/hectare); "
            "round where helpful."
        )
    if reg.name_indicators_explicitly:
        parts.append("Name indicators explicitly; spell out uncommon acronyms once.")
    else:
        parts.append("Avoid dense indicator acronyms unless unavoidable — explain in plain words.")
    if reg.prefer_inline_source_refs and inline_citations:
        parts.append("Inline [N] source markers in the body are appropriate for key figures.")
    elif not reg.prefer_inline_source_refs:
        parts.append(
            "Keep citations light in the body; the system appends a Sources section — "
            "do not stack dense inline reference markers."
        )
    if reg.use_bullet_layout and task_mode == "analytical":
        parts.append(
            "Use 3–6 plain bullet points instead of ## section headings for this audience."
        )
    if answer_shape in ("ranking", "comparison") and reg.stat_density == "low":
        parts.append("State the top finding in one sentence; skip detailed rank tables.")
    return " ".join(parts)


_SECTION_TITLE_MAP: dict[str, dict[str, str]] = {
    "Government": {
        "Key findings": "Planning takeaway",
        "Executive summary": "Planning takeaway",
        "Lead findings": "Planning takeaway",
        "Regional picture": "Regional picture",
        "Geographic comparison": "Regional picture",
        "Comparison": "Regional picture",
        "Trend summary": "Regional trend",
        "Trend": "Regional trend",
        "Drivers": "Policy relevance",
        "Drivers and context": "Policy relevance",
        "Program implications": "Policy relevance",
        "Market and production context": "Production and markets",
        "Production and trade": "Production and markets",
        "Investment signal": "Investment signal",
        "Food security risk": "Food security pressure",
        "Data notes": "Data notes",
        "Monitoring": "Monitoring cues",
        "Confidence and data notes": "Data notes",
    },
    "NGOs": {
        "Key findings": "Situation",
        "Executive summary": "Situation",
        "Lead findings": "Situation",
        "Regional picture": "Who is affected",
        "Geographic comparison": "Who is affected",
        "Comparison": "Geographic priorities",
        "Trend summary": "Situation trend",
        "Drivers": "Risk overlap",
        "Program implications": "Program implications",
        "Monitoring": "What to monitor",
        "Data notes": "Data notes",
    },
    "Agribusinesses": {
        "Key findings": "Market signal",
        "Executive summary": "Market signal",
        "Lead findings": "Market signal",
        "Regional picture": "Exposure",
        "Geographic comparison": "Exposure",
        "Comparison": "Comparative exposure",
        "Trend summary": "Trend signal",
        "Drivers": "Drivers",
        "Investment signal": "Opportunity",
        "Food security risk": "Downside risk",
        "Production and trade": "Supply fundamentals",
        "Data notes": "Confidence limits",
    },
}


def format_outline_for_persona(
    category: str | None,
    sections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Rename section titles for persona; return (sections, use_bullet_layout)."""
    reg = prose_register_for_persona(category)
    use_bullets = bool(reg and reg.use_bullet_layout)
    if not category or category.strip() not in _SECTION_TITLE_MAP:
        return sections, use_bullets
    title_map = _SECTION_TITLE_MAP[category.strip()]
    out: list[dict[str, Any]] = []
    for sec in sections:
        title = str(sec.get("title") or "").strip()
        mapped = title_map.get(title, title_map.get(title.lower(), title))
        out.append({**sec, "title": mapped})
    return out, use_bullets
