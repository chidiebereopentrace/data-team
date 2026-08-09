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
        "Audience: government and public institutions. "
        "Preferred framing: policy synthesis for planning and resource allocation — "
        "national/sub-national production, climate, and food-security pressures. "
        "Answer shape: lead with the planning takeaway; then evidence by region or time; "
        "end with explicit uncertainty, data gaps, and what would change the recommendation. "
        "Jargon: use policy and indicators language when needed; define uncommon metrics once; "
        "avoid farm-level how-to tips and vendor sales tone."
    ),
    "NGOs": (
        "Audience: development partners, foundations, and NGOs. "
        "Preferred framing: program-relevant geographic priorities and overlapping "
        "climate–nutrition–market risks with comparable regional angles. "
        "Answer shape: who/where is most affected; which risk overlap matters for targeting; "
        "what to monitor next. Prefer consistent units and geographies across claims. "
        "Jargon: keep accessible to program managers; avoid pure academic theory and "
        "commodity-trading jargon unless the question asks for it."
    ),
    "Agribusinesses": (
        "Audience: agribusiness and financial institutions. "
        "Preferred framing: production stability, price/volatility, trade and sourcing risk, "
        "and exposure that affects commercial or lending decisions. "
        "Answer shape: analytical brief — key signal first, then drivers and exposure, "
        "then practical implication for sourcing/investment when evidence supports it. "
        "Jargon: market and risk terms are fine; avoid pastoral storytelling and "
        "generic livelihood advice."
    ),
    "Farmers": (
        "Audience: farmers, cooperatives, and communities (often via intermediaries). "
        "Preferred framing: rainfall, markets, and production trends in plain, local terms. "
        "Answer shape: short actionable bullets or very short paragraphs — what is happening, "
        "what it means on the farm this season, and one clear next step when evidence supports it. "
        "Jargon rules: plain language only; no academic tone, no raw tables, no schema/SQL names; "
        "translate numbers into day-to-day meaning (prices, rain, yield)."
    ),
}


def valid_category_ids() -> frozenset[str]:
    return _CATEGORY_IDS


def is_valid_category(category: str) -> bool:
    return category.strip() in _CATEGORY_IDS


def instruction_for_category(category: str | None) -> str:
    if not category:
        return ""
    key = category.strip()
    return _CATEGORY_INSTRUCTIONS.get(key, "")


# Backward-compat aliases for internal tooling (Streamlit) until migrated.
STAKEHOLDER_TYPES = CATEGORIES


def is_valid_stakeholder_type(stakeholder_type: str) -> bool:
    return is_valid_category(stakeholder_type)


def instruction_for_stakeholder(stakeholder_type: str | None) -> str:
    return instruction_for_category(stakeholder_type)
