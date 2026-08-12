"""Pure helpers for Streamlit QA run kwargs (persona defaults)."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.plan_policy import allows_export


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
    preset_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build run_rag kwargs with Farmers persona defaults when UI fields are blank."""
    kwargs: dict[str, Any] = {
        "news_top_k": int(news_top_k),
        "academic_top_k": int(academic_top_k),
        "bq_top_k": int(bq_top_k),
        "ota_top_k": int(ota_top_k),
        "rerank_top_k": int(rerank_top_k),
    }
    overrides = preset_overrides or {}
    # Defaults keep a persona lens on every QA turn (avoid plan_type/category blanks).
    pt = str(overrides.get("plan_type") or plan_type or "Farmers").strip() or "Farmers"
    cat = str(overrides.get("category") or category or "Farmers").strip() or "Farmers"
    profile = overrides.get("user_profile") if isinstance(overrides.get("user_profile"), dict) else None

    kwargs["plan_type"] = pt
    kwargs["category"] = cat
    kwargs["export_enabled"] = allows_export(pt)
    if profile:
        kwargs["user_profile"] = {
            "country": profile.get("country") or (profile_country.strip() or None),
            "plan_type": profile.get("plan_type") or pt,
            "category": profile.get("category") or cat,
        }
    else:
        kwargs["user_profile"] = {
            "country": profile_country.strip() or None,
            "plan_type": pt,
            "category": cat,
        }
    if t_start.strip():
        kwargs["time_start_override"] = t_start.strip()[:10]
    if t_end.strip():
        kwargs["time_end_override"] = t_end.strip()[:10]
    if prior_summary.strip() or prior_recent:
        kwargs["conversation_summary"] = prior_summary
        kwargs["recent_turns"] = prior_recent
    return kwargs
