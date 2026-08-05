"""Ask ADZA plan_type tier gates and category validation."""
from __future__ import annotations

import copy
import os
from typing import Any

from ml.rag.chatbot.stakeholder_prompts import (
    instruction_for_category,
    is_valid_category,
)

# Map URL path slugs → canonical plan_type IDs (for plan-scoped API routes).
PLAN_ROUTE_SLUGS: dict[str, str] = {
    "free": "Free",
    "farmers": "Farmers",
    "government": "Government",
    "ngos": "NGOs",
    "agribusinesses": "Agribusinesses",
    "integrated": "Integrated",
}

PLAN_TYPES: list[dict[str, str]] = [
    {
        "id": "Free",
        "label": "Free",
        "description": "Try Ask ADZA — explore with limited queries, one country at a time, top-line insights.",
    },
    {
        "id": "Farmers",
        "label": "Farmers, Cooperatives & Communities",
        "description": "Localized crop, rainfall, and market insights in plain language.",
    },
    {
        "id": "Government",
        "label": "Government & Public Institutions",
        "description": "National and sub-national production, climate, and food security patterns.",
    },
    {
        "id": "NGOs",
        "label": "Foundations, NGOs & Development Partners",
        "description": "Program monitoring and multi-region overlap analysis for development work.",
    },
    {
        "id": "Agribusinesses",
        "label": "Agribusinesses & Financial Institutions",
        "description": "Market volatility, sourcing risk, and cross-country comparison for commercial decisions.",
    },
    {
        "id": "Integrated",
        "label": "Integrated Account",
        "description": "Full cross-sector access; category lens selected per message.",
    },
]

_PLAN_TYPE_IDS = frozenset(p["id"] for p in PLAN_TYPES)

# Plans that allow multi-country retrieval and compare-style answers.
_CROSS_COUNTRY_PLANS = frozenset({"Agribusinesses", "Integrated"})


def valid_plan_type_ids() -> frozenset[str]:
    return _PLAN_TYPE_IDS


def is_valid_plan_type(plan_type: str) -> bool:
    return plan_type.strip() in _PLAN_TYPE_IDS


def allows_cross_country(plan_type: str | None) -> bool:
    return (plan_type or "").strip() in _CROSS_COUNTRY_PLANS


def plan_generation_addendum(plan_type: str | None) -> str:
    """Tier-specific system-prompt lines from the Ask ADZA pricing spec."""
    pt = (plan_type or "").strip()
    if pt == "Free":
        return (
            "Plan tier: Free explorer. Keep answers concise and top-line only — roughly 1–3 short "
            "paragraphs. Do not produce deep multi-year trend analysis or cross-country comparisons. "
            "Focus on one country at a time."
        )
    if pt == "Farmers":
        return (
            "Plan tier: Farmers. Emphasize localized, district-level framing where the context "
            "supports it. Crop-specific insights (yield, rainfall, prices). No multi-country "
            "comparison."
        )
    if pt == "Government":
        return (
            "Plan tier: Government. National and sub-national scope; historical trends and "
            "climate/production/food-security indicators are in scope. Do not compare across "
            "multiple countries in one answer."
        )
    if pt == "NGOs":
        return (
            "Plan tier: NGOs / development partners. Build on government-tier depth; highlight "
            "overlapping risks (climate × nutrition × markets) and program-relevant regional angles. "
            "No cross-country private-sector sourcing comparisons."
        )
    if pt == "Agribusinesses":
        return (
            "Plan tier: Agribusiness. Market dynamics, price volatility, and sourcing-region risk "
            "are in scope. Cross-country comparison is allowed when the context supports it."
        )
    if pt == "Integrated":
        return (
            "Plan tier: Integrated. Apply the selected category lens fully; no artificial "
            "single-country restriction beyond what the category and context require."
        )
    return ""


def _pick_single_country(
    geography: list[Any],
    profile_country: str | None,
) -> list[str]:
    country = (profile_country or "").strip()
    if country:
        return [country]
    for g in geography:
        s = str(g).strip()
        if s:
            return [s]
    return []


def apply_plan_decomposition_gates(
    decomposition: dict[str, Any],
    plan_type: str | None,
    profile_country: str | None = None,
) -> dict[str, Any]:
    """
    Clamp decomposition for plan-tier retrieval limits.

    When cross-country is disallowed: geography → one country; compare → descriptive.
    """
    if not plan_type or not isinstance(decomposition, dict):
        return decomposition

    out = copy.deepcopy(decomposition)
    if not allows_cross_country(plan_type):
        geo = out.get("geography")
        geo_list = geo if isinstance(geo, list) else []
        out["geography"] = _pick_single_country(geo_list, profile_country)
        if str(out.get("intent") or "").strip().lower() == "compare":
            out["intent"] = "descriptive"

    if plan_type.strip() == "Free":
        if str(out.get("intent") or "").strip().lower() == "compare":
            out["intent"] = "descriptive"

    return out


def default_category_for_plan(plan_type: str | None) -> str | None:
    """Return the natural default category for a plan tier, or None if no default.

    Used by plan-scoped session creation routes to bootstrap a session
    with the appropriate category persona when the caller doesn't specify one.
    """
    _defaults: dict[str, str | None] = {
        "Free": None,
        "Farmers": "Farmers",
        "Government": "Government",
        "NGOs": "NGOs",
        "Agribusinesses": "Agribusinesses",
        "Integrated": None,  # category selected per message by the Integrated plan
    }
    return _defaults.get((plan_type or "").strip())


def model_for_plan(plan_type: str | None) -> str | None:
    """Return the configured OpenRouter model ID for a plan tier (ML-041).

    Reads per-plan env vars; falls back to the tier default when unset.
    Returns None when plan_type is unknown — callers fall back to RAG_LLM_MODEL_ID.

    Default model assignment:
        Free           → meta-llama/llama-3.1-8b-instruct   (RAG_LLM_MODEL_FREE)
        Farmers        → meta-llama/llama-3.1-70b-instruct  (RAG_LLM_MODEL_FARMERS)
        Government     → meta-llama/llama-3.1-70b-instruct  (RAG_LLM_MODEL_GOVERNMENT)
        NGOs           → qwen/qwen-2.5-72b-instruct         (RAG_LLM_MODEL_NGOS)
        Agribusinesses → qwen/qwen-2.5-72b-instruct         (RAG_LLM_MODEL_AGRIBUSINESSES)
        Integrated     → thudm/glm-4-plus                   (RAG_LLM_MODEL_INTEGRATED)
    """
    pt = (plan_type or "").strip()
    _env_keys: dict[str, str] = {
        "Free": "RAG_LLM_MODEL_FREE",
        "Farmers": "RAG_LLM_MODEL_FARMERS",
        "Government": "RAG_LLM_MODEL_GOVERNMENT",
        "NGOs": "RAG_LLM_MODEL_NGOS",
        "Agribusinesses": "RAG_LLM_MODEL_AGRIBUSINESSES",
        "Integrated": "RAG_LLM_MODEL_INTEGRATED",
    }
    _defaults: dict[str, str] = {
        "Free": "meta-llama/llama-3.1-8b-instruct",
        "Farmers": "meta-llama/llama-3.1-70b-instruct",
        "Government": "meta-llama/llama-3.1-70b-instruct",
        "NGOs": "qwen/qwen-2.5-72b-instruct",
        "Agribusinesses": "qwen/qwen-2.5-72b-instruct",
        "Integrated": "thudm/glm-4-plus",
    }
    if not pt or pt not in _env_keys:
        return None  # unknown plan — caller falls back to RAG_LLM_MODEL_ID
    env_key = _env_keys[pt]
    default = _defaults[pt]
    return os.environ.get(env_key, default).strip() or default


__all__ = [
    "PLAN_ROUTE_SLUGS",
    "PLAN_TYPES",
    "allows_cross_country",
    "apply_plan_decomposition_gates",
    "default_category_for_plan",
    "instruction_for_category",
    "is_valid_category",
    "is_valid_plan_type",
    "model_for_plan",
    "plan_generation_addendum",
    "valid_plan_type_ids",
]
