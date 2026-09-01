"""Continental Africa scope detection — single source for ranking/panel/count phrasing."""
from __future__ import annotations

import re

CONTINENTAL_RANK_RE = re.compile(
    r"\b("
    r"which\s+country|which\s+countries|which\s+african\s+countr(?:y|ies)|"
    r"how\s+many\s+countr(?:y|ies)|"
    r"number\s+of\s+countr(?:y|ies)|count\s+of\s+countr(?:y|ies)|"
    r"highest|lowest|top\s+\d+|bottom\s+\d+|"
    r"rank(?:ing|ed)?|largest|smallest|biggest|best|worst|"
    r"most\s+(?:produced|production)|least\s+(?:produced|production)"
    r")\b",
    re.IGNORECASE,
)

CONTINENTAL_PANEL_RE = re.compile(
    r"\b("
    r"all\s+african\s+countr(?:y|ies)|"
    r"every\s+african\s+countr(?:y|ies)|"
    r"by\s+african\s+countr(?:y|ies)|"
    r"across\s+(?:all\s+)?african\s+countr(?:y|ies)|"
    r"for\s+all\s+african\s+countr(?:y|ies)|"
    r"african\s+countr(?:y|ies)\s+panel|"
    r"each\s+african\s+countr(?:y|ies)"
    r")\b",
    re.IGNORECASE,
)

CONTINENTAL_COUNT_RE = re.compile(
    r"\b("
    r"how\s+many\s+countr(?:y|ies)|"
    r"number\s+of\s+countr(?:y|ies)|count\s+of\s+countr(?:y|ies)"
    r")\b",
    re.IGNORECASE,
)

RANKING_QUERY_RE = re.compile(
    r"\b("
    r"highest|lowest|top\s+\d+|bottom\s+\d+|"
    r"which\s+(?:\w+\s+){0,3}countr(?:y|ies)|"
    r"how\s+many\s+countr(?:y|ies)|"
    r"number\s+of\s+countr(?:y|ies)|count\s+of\s+countr(?:y|ies)|"
    r"rank(?:ing|ed)?|most\s+(?:produced|production)|"
    r"produces?\s+the\s+most|the\s+most\s+\w+|"
    r"least\s+(?:produced|production)|"
    r"largest|smallest|biggest"
    r")\b",
    re.IGNORECASE,
)


def is_continental_count_query(query: str) -> bool:
    return bool(CONTINENTAL_COUNT_RE.search(query or ""))


def is_continental_rank_query(query: str) -> bool:
    return bool(CONTINENTAL_RANK_RE.search(query or ""))


def wants_africa_panel_scope(query: str) -> bool:
    """True when the user wants values for every African country (~54-country panel)."""
    return bool(CONTINENTAL_PANEL_RE.search(query or ""))


def wants_africa_default_scope(query: str, *, extract_countries: bool) -> bool:
    """
    True for unscoped which-country / ranking / count questions.

    ``extract_countries`` must return True when a named country appears in the query.
    """
    q = (query or "").strip()
    if not q:
        return False
    if wants_africa_panel_scope(q):
        return False
    if extract_countries:
        return False
    if not CONTINENTAL_RANK_RE.search(q):
        return False
    return True


def decomposition_has_africa_scope(decomposition: dict | None) -> bool:
    if not isinstance(decomposition, dict):
        return False
    if decomposition.get("africa_default") or decomposition.get("africa_panel"):
        return True
    expanded = decomposition.get("expanded_regions")
    if isinstance(expanded, list):
        for region in expanded:
            if str(region).strip().lower() in ("africa", "african"):
                return True
    return False


__all__ = [
    "CONTINENTAL_COUNT_RE",
    "CONTINENTAL_PANEL_RE",
    "CONTINENTAL_RANK_RE",
    "RANKING_QUERY_RE",
    "decomposition_has_africa_scope",
    "is_continental_count_query",
    "is_continental_rank_query",
    "wants_africa_default_scope",
    "wants_africa_panel_scope",
]
