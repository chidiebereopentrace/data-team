"""Retrieval geography policy: profile country applies only for farmers_communities."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.query_decomposer import normalize_geography_for_filter

FARMER_STAKEHOLDER = "farmers_communities"


def profile_country_for_retrieval(
    stakeholder_type: str | None,
    user_profile: dict[str, Any] | None,
) -> str:
    """
    Return profile country for retrieval filtering, or empty string.

    Only ``farmers_communities`` may use ``user_profile.country`` as a retrieval
    geo filter; other personas rely on query decomposition geography only.
    """
    if (stakeholder_type or "").strip() != FARMER_STAKEHOLDER:
        return ""
    if not user_profile or not isinstance(user_profile, dict):
        return ""
    country = str(user_profile.get("country") or "").strip()
    if not country:
        return ""
    normalized = normalize_geography_for_filter([country])
    return normalized[0] if normalized else ""


def effective_geo_override(
    stakeholder_type: str | None,
    user_profile: dict[str, Any] | None,
) -> str:
    """Geo override passed into graph state for retrieval nodes."""
    return profile_country_for_retrieval(stakeholder_type, user_profile)
