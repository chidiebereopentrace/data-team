"""API / request_context profile country cleaning before session persist (§6)."""
from __future__ import annotations

import pytest

from ml.rag.request_context import resolve_request_context


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("NG", "NGA"),
        ("Nigeria", "NGA"),
        ("Nigéria", "NGA"),
    ],
)
def test_request_context_cleans_iso2_and_names(raw: str, expected: str) -> None:
    ctx = resolve_request_context(
        user_profile={"country": raw, "plan_type": "Farmers", "category": "Farmers"},
    )
    assert ctx.user_profile is not None
    assert ctx.user_profile.get("country") == expected


@pytest.mark.parametrize("raw", ["Albania", "US", "Unknown"])
def test_request_context_drops_non_african(raw: str) -> None:
    ctx = resolve_request_context(
        user_profile={"country": raw, "plan_type": "Farmers", "category": "Farmers"},
    )
    assert ctx.user_profile is not None
    assert ctx.user_profile.get("country") is None


def test_session_persist_receives_cleaned_iso3() -> None:
    """API persist path uses cleaned profile country (same as request_context output)."""
    ctx = resolve_request_context(
        user_profile={"country": "NG", "plan_type": "Farmers", "category": "Farmers"},
    )
    profile = ctx.user_profile or {}
    # Mirrors app/api.py: country=str(profile.get("country") or "").strip() or None
    persisted_country = str(profile.get("country") or "").strip() or None
    assert persisted_country == "NGA"
