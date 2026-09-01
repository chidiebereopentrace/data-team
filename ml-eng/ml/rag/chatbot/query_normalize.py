"""Deterministic query normalization before facet enrichment and measure resolution."""
from __future__ import annotations

import re

# Curated agronomy / market typos — word-boundary safe, not open-ended fuzzy search.
_WORD_TYPO_MAP: dict[str, str] = {
    "prize": "price",
    "pric": "price",
    "pricese": "prices",
    "hiw": "how",
}

_PHRASE_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwhat\s+is\s+the\s+prize\b", re.I), "what is the price"),
    (re.compile(r"\bwetin\s+be\s+the\s+prize\b", re.I), "wetin be the price"),
)


def normalize_query_text(query: str) -> str:
    """Apply curated typo and phrase fixes; returns stripped query."""
    q = (query or "").strip()
    if not q:
        return q
    for pattern, repl in _PHRASE_FIXES:
        q = pattern.sub(repl, q)
    for wrong, right in _WORD_TYPO_MAP.items():
        q = re.sub(rf"\b{re.escape(wrong)}\b", right, q, flags=re.I)
    q = re.sub(r"\bod\b", "of", q, flags=re.I)
    return q.strip()


__all__ = ["normalize_query_text"]
