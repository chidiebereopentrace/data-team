"""African region → country expansion for analytical / BQ retrieval."""
from __future__ import annotations

import re
from typing import Any

# Member lists use names that match FAOSTAT / news geo filters in query_decomposer.
REGION_COUNTRIES: dict[str, tuple[str, ...]] = {
    "west africa": (
        "Benin",
        "Burkina Faso",
        "Cabo Verde",
        "Cote d'Ivoire",
        "Gambia",
        "Ghana",
        "Guinea",
        "Guinea-Bissau",
        "Liberia",
        "Mali",
        "Mauritania",
        "Niger",
        "Nigeria",
        "Senegal",
        "Sierra Leone",
        "Togo",
    ),
    "ecowas": (
        "Benin",
        "Burkina Faso",
        "Cabo Verde",
        "Cote d'Ivoire",
        "Gambia",
        "Ghana",
        "Guinea",
        "Guinea-Bissau",
        "Liberia",
        "Mali",
        "Niger",
        "Nigeria",
        "Senegal",
        "Sierra Leone",
        "Togo",
    ),
    "east africa": (
        "Burundi",
        "Comoros",
        "Djibouti",
        "Eritrea",
        "Ethiopia",
        "Kenya",
        "Madagascar",
        "Malawi",
        "Mauritius",
        "Mozambique",
        "Rwanda",
        "Seychelles",
        "Somalia",
        "South Sudan",
        "Tanzania",
        "Uganda",
        "Zambia",
        "Zimbabwe",
    ),
    "southern africa": (
        "Angola",
        "Botswana",
        "Eswatini",
        "Lesotho",
        "Malawi",
        "Mozambique",
        "Namibia",
        "South Africa",
        "Zambia",
        "Zimbabwe",
    ),
    "north africa": (
        "Algeria",
        "Egypt",
        "Libya",
        "Morocco",
        "Sudan",
        "Tunisia",
    ),
    "sahel": (
        "Burkina Faso",
        "Chad",
        "Mali",
        "Mauritania",
        "Niger",
        "Senegal",
        "Sudan",
    ),
    "central africa": (
        "Cameroon",
        "Central African Republic",
        "Chad",
        "Congo",
        "Democratic Republic of the Congo",
        "Equatorial Guinea",
        "Gabon",
    ),
}

# Longer aliases first for substring matching.
_REGION_ALIASES: tuple[tuple[str, str], ...] = (
    ("sub-saharan africa", "west africa"),  # soft default; prefer explicit regions
    ("sub saharan africa", "west africa"),
    ("west africa", "west africa"),
    ("western africa", "west africa"),
    ("east africa", "east africa"),
    ("eastern africa", "east africa"),
    ("southern africa", "southern africa"),
    ("north africa", "north africa"),
    ("northern africa", "north africa"),
    ("central africa", "central africa"),
    ("ecowas", "ecowas"),
    ("sahel", "sahel"),
)


def detect_regions_in_text(text: str) -> list[str]:
    """Return canonical region keys found in text (order preserved, unique)."""
    q = (text or "").lower()
    found: list[str] = []
    seen: set[str] = set()
    for alias, key in _REGION_ALIASES:
        if key == "west africa" and alias.startswith("sub"):
            # Only use SSA fallback when no other region is present.
            continue
        if re.search(rf"\b{re.escape(alias)}\b", q) and key not in seen:
            seen.add(key)
            found.append(key)
    if not found and re.search(r"\bsub[-\s]?saharan\s+africa\b", q):
        found.append("west africa")
    return found


def countries_for_regions(region_keys: list[str], *, max_countries: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in region_keys:
        for c in REGION_COUNTRIES.get(key, ()):
            if c not in seen:
                seen.add(c)
                out.append(c)
            if max_countries is not None and len(out) >= max_countries:
                return out
    return out


def expand_regions_in_decomposition(
    decomposition: dict[str, Any],
    query: str,
    *,
    max_countries: int | None = None,
) -> dict[str, Any]:
    """
    Replace region labels with member countries in decomposition.geography.

    Scans the user query and existing geography/entities for region tokens.
    """
    if not isinstance(decomposition, dict):
        return decomposition

    texts = [query or ""]
    for key in ("geography", "entities"):
        raw = decomposition.get(key)
        if isinstance(raw, list):
            texts.extend(str(x) for x in raw)

    blob = " ".join(texts)
    regions = detect_regions_in_text(blob)
    if not regions:
        return decomposition

    countries = countries_for_regions(regions, max_countries=max_countries)
    if not countries:
        return decomposition

    out = dict(decomposition)
    existing = out.get("geography")
    existing_list = [str(g).strip() for g in existing] if isinstance(existing, list) else []
    # Drop bare region labels from geography; keep explicit countries.
    region_labels = {k for k in REGION_COUNTRIES} | {a for a, _ in _REGION_ALIASES}
    kept = [g for g in existing_list if g and g.lower() not in region_labels]
    merged: list[str] = []
    seen: set[str] = set()
    for c in kept + countries:
        if c not in seen:
            seen.add(c)
            merged.append(c)
        if max_countries is not None and len(merged) >= max_countries:
            break
    out["geography"] = merged
    out["expanded_regions"] = regions
    return out


__all__ = [
    "REGION_COUNTRIES",
    "countries_for_regions",
    "detect_regions_in_text",
    "expand_regions_in_decomposition",
]
