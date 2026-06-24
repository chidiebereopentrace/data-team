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
        "Audience: government and public institutions. Emphasize policy-relevant synthesis, "
        "regional risk and food security framing, and evidence suitable for planning and "
        "resource allocation. Be precise about uncertainty and data limits."
    ),
    "NGOs": (
        "Audience: development partners and foundations. Highlight geographic priorities, "
        "overlapping risks (climate, nutrition, markets), and how findings relate to program "
        "design and monitoring. Prefer consistent, comparable regional angles."
    ),
    "Agribusinesses": (
        "Audience: agribusiness and financial institutions. Focus on production stability, "
        "volatility, exposure, and practical implications for sourcing, investment, and "
        "ag finance. Keep tone analytical."
    ),
    "Farmers": (
        "Audience: farmers, cooperatives, and communities (often via intermediaries). Use plain "
        "language, short sentences, and concrete examples. Do not dump raw tables or technical "
        "schemas; translate numbers into what they mean day-to-day."
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
