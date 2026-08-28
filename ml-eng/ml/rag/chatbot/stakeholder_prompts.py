"""
Category segments for audience-aware generation (Ask ADZA backend contract).
Maps backend category enums to short system-prompt instructions.
"""
from __future__ import annotations

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
