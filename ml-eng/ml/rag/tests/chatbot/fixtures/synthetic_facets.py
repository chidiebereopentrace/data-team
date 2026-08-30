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
