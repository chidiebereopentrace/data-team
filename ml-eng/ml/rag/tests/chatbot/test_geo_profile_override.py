"""Profile country must not replace a country the query names."""

from __future__ import annotations

import pytest

from ml.rag.chatbot.agri_entities import CROP_ENTITY_RE
from ml.rag.chatbot.answer_outcome import gap_answer
from ml.rag.chatbot.geo_iso3 import name_to_iso3
from ml.rag.chatbot.geo_policy import (
    apply_profile_geography_default,
    clear_profile_ui_retrieval_scope,
    effective_geo_override,
    sanitize_profile_country,
)
from ml.rag.chatbot.graph import _apply_ui_scope_overrides
from ml.rag.chatbot.plan_policy import apply_plan_decomposition_gates, plan_allows_cross_country
from ml.rag.chatbot.query_decomposer import (
    _extract_countries,
    resolve_retrieval_geographies,
)
from ml.rag.chatbot.query_normalize import normalize_query_text

_QUERY = "What is the current state of casssava in Nigeria"


@pytest.mark.parametrize("plan", ["Farmers", "Integrated", "Agribusinesses"])
def test_albania_profile_does_not_override_nigeria_query(plan: str) -> None:
    normalized = normalize_query_text(_QUERY)
    assert CROP_ENTITY_RE.search(normalized)
    assert "cassava" in normalized.lower()
    countries = _extract_countries(normalized)
    assert countries == ["Nigeria"]
    assert sanitize_profile_country("Albania") is None

    dec = apply_plan_decomposition_gates(
        {"intent": "descriptive", "geography": countries},
        plan,
        "Albania",
    )
    dec = apply_profile_geography_default(dec, "Albania")
    assert dec["geography"] == ["Nigeria"]
    assert dec.get("geography_source") == "query"

    geos = resolve_retrieval_geographies(geo_override="Albania", geography=dec["geography"])
    assert geos == ["Nigeria"]
    assert name_to_iso3(geos[0]) == "NGA"

    message = gap_answer(normalized, dec, None)
    assert "Albania" not in message
    assert "Nigeria" in message
    assert effective_geo_override(plan, {"country": "Albania"}) == ""


def test_empty_query_geography_uses_sanitized_kenya_profile() -> None:
    dec = apply_profile_geography_default({"geography": []}, "Kenya")
    assert dec["geography"] == ["KEN"]
    assert dec["geography_source"] == "profile"
    message = gap_answer("cassava prices", dec, None)
    assert "Kenya (set by your profile)" in message


@pytest.mark.parametrize("raw", ["Albania", "France", "US", "", "  ", None])
def test_non_african_profile_country_is_dropped(raw: str | None) -> None:
    assert sanitize_profile_country(raw) is None


def test_cassava_typo_is_a_crop_after_normalize() -> None:
    normalized = normalize_query_text("casssava yields")
    assert "cassava" in normalized
    assert CROP_ENTITY_RE.search(normalized)
    assert normalize_query_text("what this means for maize") == "what this means for maize"
    assert normalize_query_text("planting season for maize") == "planting season for maize"


def test_farmers_does_not_force_single_country_by_default() -> None:
    assert plan_allows_cross_country("Farmers") is True
    out = apply_plan_decomposition_gates(
        {"intent": "compare", "geography": ["Nigeria", "Ghana"]},
        "Farmers",
        "Kenya",
    )
    assert out["geography"] == ["Nigeria", "Ghana"]
    assert out["intent"] == "compare"


def test_current_resolves_a_window_and_ignores_screen_dates() -> None:
    from datetime import date

    from ml.rag.chatbot.time_resolver import resolve_relative_time

    ref = date(2026, 10, 14)
    windows = resolve_relative_time(_QUERY, ref)
    assert len(windows) == 1
    assert windows[0].start == date(2026, 1, 1)
    assert windows[0].end == ref
    assert windows[0].strict is False
    out = _apply_ui_scope_overrides(
        {"geography": ["Nigeria"], "geography_source": "query"},
        {"query": _QUERY, "time_start_override": "2024-04-05", "time_end_override": "2024-04-05", "geo_override": "Albania", "reference_date": ref.isoformat()},
    )
    assert out["geography"] == ["Nigeria"]
    assert "time_start" not in out
    assert "Albania" not in str(out.get("geography"))


def test_ui_geo_fills_only_when_query_geography_is_empty() -> None:
    filled = _apply_ui_scope_overrides(
        {"geography": []},
        {"query": "cassava prices", "geo_override": "Kenya"},
    )
    assert filled["geography"] == ["Kenya"]
    assert filled["geography_source"] == "ui"


def test_cleared_profile_scope_keeps_query_country_and_labels_profile() -> None:
    cleared = clear_profile_ui_retrieval_scope(
        {
            "geography": ["Kenya"],
            "geography_source": "profile",
            "time_start": "2024-04-05",
            "time_source": "ui",
        }
    )
    assert cleared["geography"] == []
    assert cleared["filtered_geography"] == ["Kenya"]
    assert cleared["time_start"] == ""
    assert cleared["filtered_time_start"] == "2024-04-05"
    message = gap_answer("cassava", cleared, None)
    assert "Kenya (set by your profile)" in message
    assert "2024 (set by the screen filter)" in message
