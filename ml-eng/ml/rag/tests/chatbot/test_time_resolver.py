"""Phrase → calendar window. Reference dates are fixed; nothing reads the clock."""
from __future__ import annotations

from datetime import date

import pytest

from ml.rag.chatbot.time_resolver import (
    SeasonRow,
    resolve_relative_time,
    set_season_calendar,
    time_gap_sentence,
)

R = date(2026, 10, 14)

_KENYA_SEASONS = (
    SeasonRow("KEN", "Long rains", 3, 5),
    SeasonRow("KEN", "Short rains", 10, 12),
)


@pytest.fixture
def kenya_seasons() -> None:
    set_season_calendar(_KENYA_SEASONS)
    yield
    set_season_calendar(None)


def _one(text: str, ref: date = R, country: str | None = None):
    windows = resolve_relative_time(text, ref, country)
    assert len(windows) == 1
    return windows[0]


@pytest.mark.parametrize(
    ("text", "start", "end", "grain"),
    [
        ("prices today", date(2026, 10, 14), date(2026, 10, 14), "day"),
        ("prices yesterday", date(2026, 10, 13), date(2026, 10, 13), "day"),
        ("aujourd'hui", date(2026, 10, 14), date(2026, 10, 14), "day"),
        ("hoje", date(2026, 10, 14), date(2026, 10, 14), "day"),
        ("this week", date(2026, 10, 12), date(2026, 10, 14), "week"),
        ("last week", date(2026, 10, 5), date(2026, 10, 11), "week"),
        ("this month", date(2026, 10, 1), date(2026, 10, 14), "month"),
        ("this year", date(2026, 1, 1), date(2026, 10, 14), "year"),
        ("cette année", date(2026, 1, 1), date(2026, 10, 14), "year"),
        ("este ano", date(2026, 1, 1), date(2026, 10, 14), "year"),
        ("last year", date(2025, 1, 1), date(2025, 12, 31), "year"),
        ("l'année dernière", date(2025, 1, 1), date(2025, 12, 31), "year"),
        ("ano passado", date(2025, 1, 1), date(2025, 12, 31), "year"),
        ("past 5 years", date(2021, 10, 14), date(2026, 10, 14), "range"),
        ("since 2015", date(2015, 1, 1), date(2026, 10, 14), "range"),
        ("since March 2020", date(2020, 3, 1), date(2026, 10, 14), "range"),
        ("in March", date(2026, 3, 1), date(2026, 3, 31), "month"),
        ("Q1 2025", date(2025, 1, 1), date(2025, 3, 31), "range"),
        ("H1 2025", date(2025, 1, 1), date(2025, 6, 30), "range"),
        ("first half of 2025", date(2025, 1, 1), date(2025, 6, 30), "range"),
        ("2020-2024", date(2020, 1, 1), date(2024, 12, 31), "range"),
        ("between 2020 and 2024", date(2020, 1, 1), date(2024, 12, 31), "range"),
        ("trends from 2013 to 2022 in senegal", date(2013, 1, 1), date(2022, 12, 31), "range"),
        ("crop yields in kenya in 2019", date(2019, 1, 1), date(2019, 12, 31), "year"),
    ],
)
def test_phrase_table(text: str, start: date, end: date, grain: str) -> None:
    window = _one(text)
    assert window.start == start
    assert window.end == end
    assert window.grain == grain


def test_last_month_on_new_years_day() -> None:
    window = _one("last month", date(2026, 1, 1))
    assert window.start == date(2025, 12, 1)
    assert window.end == date(2025, 12, 31)


def test_last_year_on_leap_day() -> None:
    window = _one("last year", date(2028, 2, 29))
    assert window.start == date(2027, 1, 1)
    assert window.end == date(2027, 12, 31)


def test_in_march_before_march() -> None:
    window = _one("in March", date(2026, 1, 15))
    assert window.start == date(2025, 3, 1)
    assert window.end == date(2025, 3, 31)


def test_next_year_is_a_forecast() -> None:
    window = _one("maize prices next year")
    assert window.start == date(2027, 1, 1)
    assert window.end == date(2027, 12, 31)
    assert window.forecast is True


@pytest.mark.parametrize("query", ["maize prices right now", "how much is a kilo of beans in Nigeria now?"])
def test_prices_now_are_ninety_days(query: str) -> None:
    window = _one(query)
    assert window.start == date(2026, 7, 16)
    assert window.end == R
    assert window.strict is False


def test_latest_yield_asks_for_the_current_year() -> None:
    window = _one("latest maize yield in Kenya")
    assert window.start == date(2026, 1, 1)
    assert window.end == R
    assert window.from_report is False


def test_latest_report_is_marked_for_the_cited_year() -> None:
    window = _one("latest maize yield report")
    assert window.from_report is True


def test_this_season_in_kenya_is_the_short_rains(kenya_seasons: None) -> None:
    window = _one("this season", country="KEN")
    assert window.grain == "season"
    assert window.start == date(2026, 10, 1)
    assert window.end == date(2026, 12, 31)


def test_this_season_without_a_country_is_six_months(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_warehouse_lookup():
        pytest.fail("A season question without a country must not load the warehouse calendar")

    monkeypatch.setattr("ml.rag.chatbot.time_resolver._season_rows", unexpected_warehouse_lookup)
    window = _one("this season")
    assert window.start == date(2026, 4, 14)
    assert window.end == R


def test_this_season_unknown_country_is_six_months(kenya_seasons: None) -> None:
    window = _one("this season", country="TZA")
    assert window.start == date(2026, 4, 14)
    assert window.end == R
    assert window.strict is False
    assert window.grain == "range"


def test_comparison_keeps_both_windows() -> None:
    windows = resolve_relative_time("maize prices in 2024 and last month", R)
    assert len(windows) == 2
    assert (windows[0].start, windows[0].end) == (date(2024, 1, 1), date(2024, 12, 31))
    assert (windows[1].start, windows[1].end) == (date(2026, 9, 1), date(2026, 9, 30))


def test_comparison_fills_retrieval_envelope_on_decompose() -> None:
    from ml.rag.chatbot.query_decomposer import decompose_query

    out = decompose_query(
        "maize prices in 2024 and last month",
        use_llm=False,
        reference_date=R,
    )
    assert out["intent"] == "compare"
    assert len(out["time_windows"]) == 2
    assert out["time_start"] == "2024-01-01"
    assert out["time_end"] == "2026-09-30"


def test_named_year_drops_a_relative_phrase() -> None:
    window = _one("maize prices last month in 2024")
    assert window.start == date(2024, 1, 1)
    assert window.end == date(2024, 12, 31)


def test_question_with_no_time_has_no_window() -> None:
    assert resolve_relative_time("maize production in Kenya", R) == []


def test_dateparser_covers_a_phrase_our_rules_miss() -> None:
    pytest.importorskip("dateparser")
    window = _one("prices on 15/09/2024")
    assert window.start == date(2024, 9, 15)
    assert window.end == date(2024, 9, 15)


def test_gap_sentence_names_both_ranges() -> None:
    sentence = time_gap_sentence(
        "last month",
        (date(2026, 9, 1), date(2026, 9, 30)),
        (date(2024, 1, 1), date(2024, 12, 31)),
    )
    assert "last month (2026-09-01–2026-09-30)" in sentence
    assert "2024-01-01–2024-12-31" in sentence


def test_screen_dates_apply_only_when_the_question_names_no_time() -> None:
    from ml.rag.chatbot.decompose_context import DecomposeContext
    from ml.rag.chatbot.query_decomposer import decompose_query

    ref = date(2026, 10, 14)
    bare = DecomposeContext(
        query="maize production in Kenya",
        time_start_override="2020-01-01",
        time_end_override="2020-12-31",
        today_iso=ref.isoformat(),
    )
    out = decompose_query(
        "maize production in Kenya",
        use_llm=False,
        context=bare,
        reference_date=ref,
    )
    assert out["time_start"] == "2020-01-01"
    assert out["time_end"] == "2020-12-31"
    assert out["time_source"] == "ui"

    dated = DecomposeContext(
        query="maize prices last month",
        time_start_override="2020-01-01",
        time_end_override="2020-12-31",
        today_iso=ref.isoformat(),
    )
    replaced = decompose_query(
        "maize prices last month",
        use_llm=False,
        context=dated,
        reference_date=ref,
    )
    assert replaced["time_start"] == "2026-09-01"
    assert replaced["time_end"] == "2026-09-30"
    assert replaced["time_source"] == "query"


def test_answer_prefixes_the_gap_when_evidence_is_outside_the_window() -> None:
    from ml.rag.chatbot.generator import _prepend_time_gap

    text = _prepend_time_gap(
        "Maize was 200.",
        {
            "time_forecast": False,
            "time_fallback": {
                "requested": {
                    "phrase": "last month",
                    "start": "2026-09-01",
                    "end": "2026-09-30",
                },
                "used": {"start": "2024-01-01", "end": "2024-12-31"},
            },
        },
    )
    assert text.startswith("You asked about last month (2026-09-01–2026-09-30).")
    assert "2024-01-01–2024-12-31" in text
    assert "Maize was 200." in text
