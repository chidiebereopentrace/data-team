"""Shared crop and commodity vocabulary for slot detection."""
from __future__ import annotations

import re

CROP_COMMODITY_TERMS: frozenset[str] = frozenset(
    {
        "maize",
        "corn",
        "rice",
        "cassava",
        "sorghum",
        "millet",
        "wheat",
        "soybean",
        "soy",
        "cotton",
        "cocoa",
        "coffee",
        "tea",
        "sugarcane",
        "groundnut",
        "yam",
        "plantain",
        "banana",
        "cowpea",
        "beans",
        "tomato",
        "onion",
        "potato",
        "livestock",
        "cattle",
        "goat",
        "sheep",
        "poultry",
        "fish",
    }
)

CROP_ENTITY_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(CROP_COMMODITY_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def query_has_crop_or_commodity(query: str, decomposition: dict | None = None) -> bool:
    q = (query or "").lower()
    if CROP_ENTITY_RE.search(q):
        return True
    if not isinstance(decomposition, dict):
        return False
    entities = decomposition.get("entities")
    if isinstance(entities, list):
        for ent in entities:
            text = str(ent or "").strip().lower()
            if any(term in text for term in CROP_COMMODITY_TERMS):
                return True
    return False


__all__ = [
    "CROP_COMMODITY_TERMS",
    "CROP_ENTITY_RE",
    "query_has_crop_or_commodity",
]
