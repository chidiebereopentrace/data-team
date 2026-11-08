"""Ask ADZA plan_type tier gates and category validation."""
from __future__ import annotations

import copy
import os
from typing import Any

from ml.rag.chatbot.stakeholder_prompts import (
    instruction_for_category,
    is_valid_category,
)
from ml.rag.llm_chat import DEFAULT_LLM_MODEL_ID

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

# Plans whose dedicated API routes may produce downloadable exports.
_EXPORT_PLANS = frozenset({"Agribusinesses", "Integrated"})


def valid_plan_type_ids() -> frozenset[str]:
    return _PLAN_TYPE_IDS


def is_valid_plan_type(plan_type: str) -> bool:
    return plan_type.strip() in _PLAN_TYPE_IDS


def allows_cross_country(plan_type: str | None) -> bool:
    """Static tier set. Retrieval gates use ``plan_allows_cross_country``."""
    return (plan_type or "").strip() in _CROSS_COUNTRY_PLANS


def single_country_policy_enabled() -> bool:
    """True only when ``ADZA_SINGLE_COUNTRY_POLICY`` is explicitly on. Default off."""
    return os.getenv("ADZA_SINGLE_COUNTRY_POLICY", "false").strip().lower() == "true"


def plan_allows_cross_country(plan_type: str | None) -> bool:
    """Whether this turn may keep every country the query named.

    The flag defaults off, so every plan keeps query geography. When it is on,
    only Agribusinesses and Integrated stay multi-country.
    """
    if not single_country_policy_enabled():
        return True
    return allows_cross_country(plan_type)


def allows_export(plan_type: str | None) -> bool:
    """Whether export builders may run (defense-in-depth; route also gates via export_enabled)."""
    return (plan_type or "").strip() in _EXPORT_PLANS


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
            "single-country restriction beyond what the category and context require. For "
            "cross-sector questions, synthesize how relevant factors interact instead of "
            "listing stakeholder perspectives independently. When the evidence supports it, "
            "explain the pathway from drivers to outcomes and implications, citing each factual "
            "link. Distinguish directly sourced findings from reasoned interpretation, and do "
            "not present correlation or a plausible mechanism as established causation."
        )
    return ""


def _pick_single_country(
    geography: list[Any],
    profile_country: str | None,
) -> list[str]:
    """Query country wins. Profile is only the fallback when the query named none."""
    for g in geography:
        s = str(g).strip()
        if s:
            return [s]
    country = (profile_country or "").strip()
    if country:
        return [country]
    return []


def apply_plan_decomposition_gates(
    decomposition: dict[str, Any],
    plan_type: str | None,
    profile_country: str | None = None,
) -> dict[str, Any]:
    """
    Clamp decomposition for plan-tier retrieval limits.

    When cross-country is disallowed: geography → one country; compare → descriptive.
    Preserve africa_default / africa_panel / expanded regional panels (do not amputate
    West Africa → Benin).
    """
    if not plan_type or not isinstance(decomposition, dict):
        return decomposition

    out = copy.deepcopy(decomposition)
    preserve_multi = bool(
        out.get("africa_default")
        or out.get("africa_panel")
        or out.get("expanded_regions")
    )
    if not plan_allows_cross_country(plan_type) and not preserve_multi:
        # Lazy import: geo_policy imports this module for the feature flag.
        from ml.rag.chatbot.geo_policy import sanitize_profile_country

        geo = out.get("geography")
        geo_list = geo if isinstance(geo, list) else []
        had_query_geo = any(str(g).strip() for g in geo_list)
        safe_profile = sanitize_profile_country(profile_country)
        out["geography"] = _pick_single_country(geo_list, safe_profile)
        if out["geography"] and not had_query_geo:
            out["geography_source"] = "profile"
        elif had_query_geo:
            out["geography_source"] = "query"
        if str(out.get("intent") or "").strip().lower() == "compare":
            out["intent"] = "descriptive"

    if plan_type.strip() == "Free" and not preserve_multi:
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

    Reads per-plan env vars; falls back to the shared default when unset.
    Returns None when plan_type is unknown — callers fall back to RAG_LLM_MODEL_ID.

    All plan tiers default to the same chat model (``DEFAULT_LLM_MODEL_ID``).
    Per-plan env overrides still work when explicitly set.
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
    if not pt or pt not in _env_keys:
        return None  # unknown plan — caller falls back to RAG_LLM_MODEL_ID
    env_key = _env_keys[pt]
    return os.environ.get(env_key, DEFAULT_LLM_MODEL_ID).strip() or DEFAULT_LLM_MODEL_ID


__all__ = [
    "PLAN_ROUTE_SLUGS",
    "PLAN_TYPES",
    "allows_cross_country",
    "allows_export",
    "plan_allows_cross_country",
    "single_country_policy_enabled",
    "apply_plan_decomposition_gates",
    "default_category_for_plan",
    "instruction_for_category",
    "is_valid_category",
    "is_valid_plan_type",
    "model_for_plan",
    "plan_generation_addendum",
    "valid_plan_type_ids",
]
