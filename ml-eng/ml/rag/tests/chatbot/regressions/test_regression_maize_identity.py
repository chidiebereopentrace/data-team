"""Maize is not maize flour; producer price is not retail price."""
from __future__ import annotations

from ml.rag.chatbot.bq_table_schema_yaml import match_value_samples

PRODUCTS = ["Maize flour", "Maize", "Rice"]
PRICE_TYPES = [
    "Retail",
    "Wholesale",
    "Producer Price Index (2014-2016 = 100)",
    "Producer Price (LCU/tonne)",
    "Value",
]


def test_maize_binds_the_grain_not_the_flour() -> None:
    assert match_value_samples("price of maize in Kenya", PRODUCTS) == ["Maize"]


def test_maize_flour_binds_the_flour() -> None:
    assert match_value_samples("maize flour price in Kenya", PRODUCTS) == ["Maize flour"]


def test_producer_price_is_not_retail_or_an_index() -> None:
    hits = match_value_samples("what producer price do maize farmers get in Kenya", PRICE_TYPES)
    assert hits == ["Producer Price (LCU/tonne)"]


def test_retail_price_binds_retail() -> None:
    assert match_value_samples("retail price of maize in Nairobi", PRICE_TYPES) == ["Retail"]


def test_unqualified_price_does_not_guess_a_price_type() -> None:
    assert match_value_samples("maize price in Kenya", PRICE_TYPES) == []
