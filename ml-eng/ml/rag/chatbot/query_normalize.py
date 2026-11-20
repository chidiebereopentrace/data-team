"""Deterministic query normalization before facet enrichment and measure resolution."""
from __future__ import annotations

import re
from functools import lru_cache

from ml.rag.chatbot.agri_entities import CROP_COMMODITY_TERMS

# Curated agronomy / market typos — word-boundary safe, not open-ended fuzzy search.
_WORD_TYPO_MAP: dict[str, str] = {
    "prize": "price",
    "pric": "price",
    "pricese": "prices",
    "hiw": "how",
    "casssava": "cassava",
    "cassva": "cassava",
    "casava": "cassava",
}

_PHRASE_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwhat\s+is\s+the\s+prize\b", re.I), "what is the price"),
    (re.compile(r"\bwetin\s+be\s+the\s+prize\b", re.I), "wetin be the price"),
)

_TOKEN_RE = re.compile(r"\b[A-Za-z]{5,}\b")


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        curr = [i]
        for j, right_ch in enumerate(right, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (left_ch != right_ch)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


@lru_cache(maxsize=1)
def _country_aliases() -> tuple[tuple[str, str], ...]:
    """Folded alias → display country name. Single-word aliases only for token repair."""
    from ml.rag.chatbot.bq_table_schema_yaml import _AFRICA_COUNTRY_ISO3

    out: list[tuple[str, str]] = []
    for name in _AFRICA_COUNTRY_ISO3:
        folded = name.casefold()
        if " " in folded or "-" in folded or "'" in folded:
            continue
        out.append((folded, name))
    return tuple(out)


@lru_cache(maxsize=1)
def _known_country_tokens() -> frozenset[str]:
    return frozenset(alias for alias, _ in _country_aliases())


def _fuzzy_crop(token: str) -> str | None:
    """One crop within a small edit distance, and only when the prefix agrees.

    The prefix guard keeps ordinary words (``means``) from becoming crops (``beans``).
    """
    folded = token.casefold()
    if len(folded) < 5 or folded in CROP_COMMODITY_TERMS:
        return None
    prefix = folded[:3]
    limit = 1 if len(folded) < 7 else 2
    hits: list[str] = []
    for term in CROP_COMMODITY_TERMS:
        if len(term) < 5 or not term.startswith(prefix):
            continue
        distance = _edit_distance(folded, term)
        if distance <= 0 or distance > limit:
            continue
        # Distance 2 must be a prefix typo (extra letters), not a different word
        # such as planting → plantain.
        if distance == 2 and not (folded.startswith(term) or term.startswith(folded)):
            continue
        hits.append(term)
        if len(hits) > 1:
            return None
    if len(hits) == 1:
        return hits[0]
    return None


def _fuzzy_country(token: str) -> str | None:
    """One-edit country repair. A token that is already a country stays that country."""
    folded = token.casefold()
    if len(folded) < 5 or folded in _known_country_tokens():
        return None
    prefix = folded[:3]
    hits: list[str] = []
    for alias, display in _country_aliases():
        if len(alias) < 5 or not alias.startswith(prefix):
            continue
        distance = _edit_distance(folded, alias)
        if distance != 1:
            continue
        hits.append(display)
        if len(hits) > 1:
            return None
    if len(hits) == 1:
        return hits[0]
    return None


def _apply_fuzzy_tokens(query: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        word = match.group(0)
        country = _fuzzy_country(word)
        if country:
            if word[0].isupper():
                return country
            return country.casefold()
        crop = _fuzzy_crop(word)
        return crop if crop else word

    return _TOKEN_RE.sub(_replace, query)


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
    q = _apply_fuzzy_tokens(q)
    return q.strip()


def indicator_query_text(query: str) -> str:
    """Treat word-separating dashes alike for indicator matching, leaving dates intact."""
    return re.sub(r"(?<=[^\W\d_])[-‐‑‒–—](?=[^\W\d_])", " ", normalize_query_text(query))


__all__ = ["normalize_query_text", "indicator_query_text"]
