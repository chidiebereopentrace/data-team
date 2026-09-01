"""Tests for query normalization."""
from __future__ import annotations

from ml.rag.chatbot.query_normalize import normalize_query_text


def test_prize_typo_becomes_price() -> None:
    assert "price" in normalize_query_text("what is the prize of maize in nigeria").lower()
    assert "prize" not in normalize_query_text("what is the prize of maize").lower()


def test_od_becomes_of() -> None:
    assert " of " in normalize_query_text("price od maize")
