"""Unit tests for query decomposition heuristics."""
from __future__ import annotations

from datetime import date

from ml.rag.chatbot.query_decomposer import (
    _extract_countries,
    _extract_year_range,
    decompose_query,
    should_use_llm_decompose,
)


def test_nigeria_not_niger() -> None:
    assert _extract_countries("agriculture in nigeria") == ["Nigeria"]
    assert "Niger" not in _extract_countries("products nigeria produces")


def test_since_year_till_now_open_range() -> None:
    q = "whats the trend of rice production in nigeria since 2015 till now"
    ts, te = _extract_year_range(q)
    assert ts == "2015-01-01"
    assert te == date.today().isoformat()


def test_decompose_since_till_now_not_single_calendar_year() -> None:
    q = "whats the trend of rice production in nigeria since 2015 till now"
    dec = decompose_query(q)
    assert dec["geography"] == ["Nigeria"]
    assert dec["time_start"] == "2015-01-01"
    assert dec["time_end"] == date.today().isoformat()
    assert dec["time_end"] != "2015-12-31"


def test_in_year_stays_calendar_year() -> None:
    ts, te = _extract_year_range("crop yields in kenya in 2019")
    assert ts == "2019-01-01"
    assert te == "2019-12-31"


def test_should_skip_llm_for_simple_fact_lookup() -> None:
    assert should_use_llm_decompose("maize production in Kenya 2020") is False


def test_should_use_llm_for_compare() -> None:
    assert should_use_llm_decompose("compare maize yields Kenya vs Nigeria 2020") is True


def test_two_explicit_years() -> None:
    ts, te = _extract_year_range("trends from 2013 to 2022 in senegal")
    assert ts == "2013-01-01"
    assert te == "2022-12-31"


def test_past_n_years() -> None:
    ts, te = _extract_year_range("yields over the past 5 years in ghana")
    assert ts is not None and te is not None
    assert ts[:4] == str(date.today().year - 5)


if __name__ == "__main__":
    test_nigeria_not_niger()
    test_since_year_till_now_open_range()
    test_decompose_since_till_now_not_single_calendar_year()
    test_in_year_stays_calendar_year()
    test_two_explicit_years()
    test_past_n_years()
    print("ok")
