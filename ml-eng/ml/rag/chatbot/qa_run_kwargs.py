"""Pure helpers for Streamlit QA run kwargs.

Plan (commercial tier) and Persona (answer audience) are separate choices; a blank field
stays blank so the shared core sees exactly what the API would send.
"""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.plan_policy import allows_export
from ml.rag.chatbot.time_resolver import utc_today


def build_run_kwargs(
    *,
    news_top_k: int,
    academic_top_k: int,
    bq_top_k: int,
    ota_top_k: int,
    rerank_top_k: int,
    plan_type: str,
    category: str,
    profile_country: str,
    t_start: str,
    t_end: str,
    prior_summary: str,
    prior_recent: list[dict[str, str]],
    active_request: dict[str, Any] | None = None,
    preset_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build run_rag kwargs from the QA sidebar; blank Plan or Persona is passed as absent."""
    kwargs: dict[str, Any] = {
        "news_top_k": int(news_top_k),
        "academic_top_k": int(academic_top_k),
        "bq_top_k": int(bq_top_k),
        "ota_top_k": int(ota_top_k),
        "rerank_top_k": int(rerank_top_k),
    }
    overrides = preset_overrides or {}
    pt = str(overrides.get("plan_type") or plan_type or "").strip() or None
    cat = str(overrides.get("category") or category or "").strip() or None
    profile = overrides.get("user_profile") if isinstance(overrides.get("user_profile"), dict) else {}

    if pt:
        kwargs["plan_type"] = pt
    if cat:
        kwargs["category"] = cat
    kwargs["export_enabled"] = allows_export(pt)
    country = profile.get("country") or (profile_country.strip() or None)
    if pt or cat or country:
        kwargs["user_profile"] = {
            "country": country,
            "plan_type": profile.get("plan_type") or pt,
            "category": profile.get("category") or cat,
        }
    if t_start.strip():
        kwargs["time_start_override"] = t_start.strip()[:10]
    if t_end.strip():
        kwargs["time_end_override"] = t_end.strip()[:10]
    kwargs["reference_date"] = utc_today().isoformat()
    if prior_summary.strip() or prior_recent:
        kwargs["conversation_summary"] = prior_summary
        kwargs["recent_turns"] = prior_recent
    if isinstance(active_request, dict):
        kwargs["active_request"] = active_request
    return kwargs
