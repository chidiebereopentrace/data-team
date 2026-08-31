"""Tests for intent bundle matching (multi-measure turns)."""
from __future__ import annotations

from ml.rag.chatbot.intent_bundles import (
    bundle_required_measures,
    bundles_block_primary,
    has_bundle,
    match_intent_bundles,
)


def test_agricultural_activities_requires_production_and_trade() -> None:
    bundles = match_intent_bundles(
        "Prepare an agricultural activities report for West Africa country by country.",
        {"entities": ["food security"]},
    )
    assert has_bundle(bundles, "agricultural_activities")
    req = bundle_required_measures(bundles)
    assert "production" in req
    assert "trade" in req


def test_activities_plus_food_security_does_not_make_fs_primary() -> None:
    bundles = match_intent_bundles(
        "Agricultural activities and food security outlook for West Africa.",
    )
    assert has_bundle(bundles, "agricultural_activities")
    assert bundles_block_primary("food_security_ipc", bundles)


def test_protected_area_bundle() -> None:
    bundles = match_intent_bundles("Compare WDPA terrestrial protected area in Kenya and Uganda.")
    assert has_bundle(bundles, "protected_area")
    assert bundle_required_measures(bundles) == ("protected_area",)


def test_food_balance_panel_bundle() -> None:
    bundles = match_intent_bundles("Food balance and import dependency for Senegal.")
    assert has_bundle(bundles, "food_balance_panel")
    req = bundle_required_measures(bundles)
    assert "production" in req and "trade" in req and "food_balance" in req
