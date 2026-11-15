"""Tests for query normalization."""
from __future__ import annotations

import re

from ml.rag.chatbot.query_normalize import normalize_query_text


def test_prize_typo_becomes_price() -> None:
    assert "price" in normalize_query_text("what is the prize of maize in nigeria").lower()
    assert "prize" not in normalize_query_text("what is the prize of maize").lower()


def test_od_becomes_of() -> None:
    assert " of " in normalize_query_text("price od maize")


def test_country_typo_repairs_niggeria_and_keeps_niger() -> None:
    fixed = normalize_query_text("cassava production in niggeria from 2015")
    assert "nigeria" in fixed.lower()
    assert "niggeria" not in fixed.lower()
    kept = normalize_query_text("millet price in Niger in 2023")
    assert re.search(r"\bniger\b", kept, re.I)
    assert "nigeria" not in kept.lower()
