"""Unit tests for farmer-only profile country retrieval policy."""

from __future__ import annotations

from ml.rag.chatbot.geo_policy import (
    FARMER_STAKEHOLDER,
    effective_geo_override,
    profile_country_for_retrieval,
)


def test_farmers_profile_country_applied() -> None:
    assert (
        profile_country_for_retrieval(
            FARMER_STAKEHOLDER,
            {"country": "Nigeria"},
        )
        == "Nigeria"
    )
    assert effective_geo_override(FARMER_STAKEHOLDER, {"country": "Nigeria"}) == "Nigeria"


def test_non_farmer_profile_country_ignored() -> None:
    profile = {"country": "Nigeria"}
    assert profile_country_for_retrieval("government_public", profile) == ""
    assert profile_country_for_retrieval("private_sector", profile) == ""


def test_missing_profile_returns_empty() -> None:
    assert profile_country_for_retrieval(FARMER_STAKEHOLDER, None) == ""
    assert profile_country_for_retrieval(FARMER_STAKEHOLDER, {}) == ""
    assert profile_country_for_retrieval(FARMER_STAKEHOLDER, {"country": ""}) == ""


def test_region_token_in_profile_filtered() -> None:
    assert profile_country_for_retrieval(FARMER_STAKEHOLDER, {"country": "Africa"}) == ""
    assert profile_country_for_retrieval(FARMER_STAKEHOLDER, {"country": "West Africa"}) == ""
