"""Tests for bundle primary measure routing."""
from __future__ import annotations

from ml.rag.chatbot.intent_bundles import (
    bundle_primary_measure,
    bundle_primary_measures,
    match_intent_bundles,
)


def test_food_balance_panel_primary_is_food_balance_not_production() -> None:
    q = "What share of Ghana wheat domestic supply was imported?"
    bundles = match_intent_bundles(q, {})
    assert bundle_primary_measure(bundles, q) == "food_balance"
    assert bundle_primary_measures(bundles, q)[0] == "food_balance"
