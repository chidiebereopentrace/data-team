"""Yam GHA/NGA "lately": both countries, the Yams identity, a covered window, no stray place.

Coverage years below are FIXTURE TRUTH chosen for the test, not observed warehouse values.
"""
from __future__ import annotations

import pytest

from ml.rag.chatbot import time_policy
from ml.rag.chatbot.bind_planning import compile_bind_for_request
from ml.rag.chatbot.compile_sql_from_bind import RESULT_ROW_BUDGET, compile_sql_from_bind
from ml.rag.chatbot.mart_coverage import Coverage, CoverageEntry

QUESTION = "How has yam production in Ghana and Nigeria changed lately?"
TABLE = "agg_production_country_year"


def _fixture_coverage() -> Coverage:
    return Coverage(
        entries=[
            CoverageEntry("fct_production", "country_iso3", "GHA", "Ghana", "GHA", 1961, 2023),
            CoverageEntry("fct_production", "country_iso3", "NGA", "Nigeria", "NGA", 1961, 2024),
        ]
    )


@pytest.fixture()
def facets() -> dict:
    return {
        "geography": ["Ghana", "Nigeria"],
        "entities": ["yam", "Ségou"],
        "time_windows": [{"start": "2025-11-07", "end": "2026-11-07", "strict": False}],
        "time_phrase": "lately",
        "time_start": "2025-11-07",
        "time_end": "2026-11-07",
        "job": "trend",
    }


@pytest.fixture()
def contract(monkeypatch: pytest.MonkeyPatch, facets: dict):
    monkeypatch.setattr(time_policy, "load_coverage", _fixture_coverage)
    return compile_bind_for_request(
        TABLE,
        facets=facets,
        query=QUESTION,
        country_labels=["GHA", "NGA"],
        product_labels=["Yams"],
    )


def test_binds_both_countries_and_the_yams_identity(contract) -> None:
    sql = contract.required_filters_sql
    assert "country_iso3 IN ('GHA', 'NGA')" in sql
    assert "product_name = 'Yams'" in sql


def test_relative_phrase_reads_the_latest_common_years_of_the_parent_fact(contract) -> None:
    assert "year BETWEEN 2019 AND 2023" in contract.required_filters_sql
    assert "2026" not in contract.required_filters_sql
    assert contract.time_policy["coverage_table"] == "fct_production"
    assert contract.filter_provenance["time"] == "policy:relative_latest_5_common_years"
    assert "2019-2023" in contract.time_policy["disclosure"]


def test_entities_never_become_places(contract) -> None:
    sql = contract.required_filters_sql
    assert "Ségou" not in sql and "Segou" not in sql
    assert "admin_1" not in sql and "city_name" not in sql
    assert "subnational" not in contract.filter_provenance


def test_series_fetches_the_complete_shape(contract) -> None:
    sql = compile_sql_from_bind(contract, job="trend") or ""
    assert f"LIMIT {RESULT_ROW_BUDGET + 1}" in sql
    assert "LIMIT 10)" not in sql


def test_unknown_coverage_drops_the_calendar_window(monkeypatch: pytest.MonkeyPatch, facets: dict) -> None:
    monkeypatch.setattr(time_policy, "load_coverage", lambda: Coverage())
    contract = compile_bind_for_request(
        TABLE, facets=facets, query=QUESTION, country_labels=["GHA", "NGA"], product_labels=["Yams"]
    )
    assert "year" not in contract.required_filters_sql
    assert contract.time_policy["status"] == "no_coverage"


def test_explicit_years_are_never_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time_policy, "load_coverage", _fixture_coverage)
    explicit = {
        "geography": ["Ghana", "Nigeria"],
        "time_windows": [{"start": "2010-01-01", "end": "2012-12-31", "strict": True}],
        "time_start": "2010-01-01",
        "time_end": "2012-12-31",
    }
    contract = compile_bind_for_request(
        TABLE, facets=explicit, query="yam production in Ghana and Nigeria 2010 to 2012",
        country_labels=["GHA", "NGA"], product_labels=["Yams"],
    )
    assert "year BETWEEN 2010 AND 2012" in contract.required_filters_sql
    assert contract.time_policy == {}
