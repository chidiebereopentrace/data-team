"""Synthetic facet combinations for property tests."""
from __future__ import annotations

import itertools
from typing import Any, Iterator

COUNTRIES = ("Kenya", "Uganda", "Senegal", "Malawi", "Ghana")
CROPS = ("rice", "maize", "millet", "sorghum")
MEASURES = ("production", "yield", "employment_share", "market_price", "food_security_ipc")
JOBS = ("fact", "trend", "rank", "list", "compare")
GEO_GRAINS = ("country", "admin2", "admin1")
TIME_GRAINS = ("year", "year_range", "latest")
HEAVY_PLAN_TYPES = ("government", "agribusiness", "integrated")
WEST_AFRICA_GEOS = ("Nigeria", "Ghana", "Senegal", "Mali", "Burkina Faso", "Niger")


def synthetic_numeric_facets(*, limit: int = 200) -> Iterator[dict[str, Any]]:
    """Yield synthetic measure×geo×time×job tuples (not natural-language questions)."""
    combos = itertools.product(COUNTRIES, CROPS, MEASURES, JOBS, GEO_GRAINS, TIME_GRAINS)
    for idx, (country, crop, measure, job, geo_grain, time_grain) in enumerate(combos):
        if idx >= limit:
            break
        breakdown: list[str] = []
        if measure == "employment_share" and idx % 3 == 0:
            breakdown = ["sex"]
        yield {
            "country": country,
            "crop": crop,
            "measure_id": measure,
            "job": job,
            "geo_grain": geo_grain,
            "time_grain": time_grain,
            "breakdown": breakdown,
        }


def facet_to_query(facet: dict[str, Any]) -> str:
    country = facet["country"]
    crop = facet.get("crop") or ""
    measure = facet["measure_id"]
    job = facet["job"]
    geo_grain = facet["geo_grain"]
    if measure == "employment_share":
        base = f"What share of employment is in agriculture in {country}"
        if facet.get("breakdown"):
            base += " by men and women"
        if facet.get("time_grain") == "latest":
            base += " (latest available)"
        return base + "?"
    if job == "trend":
        return f"How has {crop} production in {country} changed over the last 5 years?"
    if job == "rank":
        return f"Which countries produce the most {crop}?"
    if job == "list" and geo_grain == "admin2":
        return f"Which districts in {country} had the highest {crop} yield in 2023?"
    if job == "fact":
        return f"What was {crop} production in {country} in 2022?"
    return f"Compare {crop} production in {country} versus neighbours"


def synthetic_heavy_reasoner_facets(*, limit: int = 24) -> Iterator[dict[str, Any]]:
    """Synthetic facets for Global Reasoner property tests (T1–T10)."""
    templates: list[dict[str, Any]] = [
        {
            "id": "agri_activities_panel",
            "plan_type": "government",
            "job": "report",
            "geos": list(WEST_AFRICA_GEOS[:6]),
            "query": (
                "Prepare an agricultural activities report for West Africa country by country, "
                "including production and trade, 2015 to latest."
            ),
        },
        {
            "id": "protected_wdpa",
            "plan_type": "integrated",
            "job": "compare",
            "geos": ["Kenya", "Uganda", "Tanzania"],
            "query": "Compare terrestrial protected area (WDPA) coverage in Kenya, Uganda, and Tanzania.",
        },
        {
            "id": "trend_last_years",
            "plan_type": "agribusiness",
            "job": "trend",
            "geos": ["Nigeria"],
            "query": "How has maize production in Nigeria changed over the last 5 years?",
        },
        {
            "id": "food_balance",
            "plan_type": "government",
            "job": "compare",
            "geos": ["Senegal", "Mali"],
            "query": (
                "Compare food balance and import dependency for Senegal and Mali — "
                "production vs imports vs consumption."
            ),
        },
        {
            "id": "employment_sex",
            "plan_type": "government",
            "job": "fact",
            "geos": ["Ghana"],
            "query": "What share of agricultural employment in Ghana by men and women?",
            "breakdown": ["sex"],
        },
        {
            "id": "outlook_ipc",
            "plan_type": "government",
            "job": "outlook",
            "geos": ["Somalia"],
            "query": "Food security outlook and IPC phase for Somalia lean season 2024.",
        },
        {
            "id": "farmers_simple_fact",
            "plan_type": "farmers",
            "job": "fact",
            "geos": ["Kenya"],
            "query": "What was maize production in Kenya in 2022?",
        },
    ]
    for idx, tpl in enumerate(templates):
        if idx >= limit:
            break
        yield tpl


def heavy_facet_decomposition(facet: dict[str, Any]) -> dict[str, Any]:
    return {
        "geography": list(facet.get("geos") or []),
        "entities": list(facet.get("entities") or []),
        "intent": facet.get("job") or "fact",
        "time_start": facet.get("time_start") or "",
        "time_end": facet.get("time_end") or "",
    }
