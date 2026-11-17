"""Calendar windows for relative time phrases in a Question.

The reference date is an argument. This module does not read the clock.
Screen dates are not parsed here.
"""
from __future__ import annotations

import calendar
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Literal, cast

logger = logging.getLogger(__name__)

Grain = Literal["day", "week", "month", "season", "year", "range"]

_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "janvier": 1, "janeiro": 1,
    "february": 2, "feb": 2, "fevrier": 2, "février": 2, "fevereiro": 2,
    "march": 3, "mar": 3, "mars": 3, "marco": 3, "março": 3,
    "april": 4, "apr": 4, "avril": 4, "abril": 4,
    "may": 5, "mai": 5, "maio": 5,
    "june": 6, "jun": 6, "juin": 6, "junho": 6,
    "july": 7, "jul": 7, "juillet": 7, "julho": 7,
    "august": 8, "aug": 8, "aout": 8, "août": 8, "agosto": 8,
    "september": 9, "sep": 9, "sept": 9, "septembre": 9, "setembro": 9,
    "october": 10, "oct": 10, "octobre": 10, "outubro": 10,
    "november": 11, "nov": 11, "novembre": 11, "novembro": 11,
    "december": 12, "dec": 12, "decembre": 12, "décembre": 12, "dezembro": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_YEAR = r"(?:19|20)\d{2}"
_COMPARE_SPLIT = re.compile(
    r"\s+(?:and|vs\.?|versus|compared\s+to|compared\s+with)\s+",
    re.IGNORECASE,
)
_RELATIVE_HINT = re.compile(
    r"\b(?:"
    r"today|yesterday|this\s+week|last\s+week|this\s+month|last\s+month|"
    r"this\s+year|last\s+year|next\s+year|this\s+season|current\s+season|"
    r"right\s+now|as\s+of\s+now|currently|current|recently|recent|lately|latest|"
    r"now|aujourd'hui|aujourdhui|hier|hoje|ontem|"
    r"cette\s+ann[eé]e|l['’]ann[eé]e\s+derni[eè]re|este\s+ano|ano\s+passado|"
    r"cette\s+semaine|esta\s+semana|ce\s+mois|este\s+m[eê]s|"
    r"le\s+mois\s+dernier|m[eê]s\s+passado"
    r")\b"
    r"|\b(?:past|last)\s+\d{1,2}\s+(?:days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(rf"\b({_YEAR})\b")
_SINCE_RE = re.compile(r"\b(?:since|depuis|desde)\b", re.IGNORECASE)
_DAILY_SUBJECT = re.compile(
    r"\b(?:prices?|news|weather|rainfall|temperature|market)\b",
    re.IGNORECASE,
)
_REPORT_SUBJECT = re.compile(
    r"\b(?:reports?|publications?|figures)\b",
    re.IGNORECASE,
)
_DATE_TOKEN = re.compile(
    rf"\d|{_MONTH_PATTERN}",
    re.IGNORECASE,
)


def utc_today() -> date:
    """UTC calendar date for a request edge. The resolver itself does not call this."""
    return datetime.now(timezone.utc).date()


@dataclass(frozen=True)
class TimeWindow:
    start: date
    end: date
    grain: Grain
    phrase: str
    source: Literal["query"] = "query"
    strict: bool = True
    forecast: bool = False
    from_report: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> TimeWindow | None:
        start = parse_iso_date(str(raw.get("start") or ""))
        end = parse_iso_date(str(raw.get("end") or ""))
        grain = raw.get("grain")
        if start is None or end is None or start > end:
            return None
        if grain not in ("day", "week", "month", "season", "year", "range"):
            return None
        return cls(
            start, end, cast(Grain, grain), str(raw.get("phrase") or ""),
            strict=raw.get("strict") is True,
            forecast=raw.get("forecast") is True,
            from_report=raw.get("from_report") is True,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "grain": self.grain,
            "phrase": self.phrase,
            "source": self.source,
            "strict": self.strict,
            "forecast": self.forecast,
            "from_report": self.from_report,
        }


def resolve_relative_time(
    text: str,
    ref_date: date,
    country_iso3: str | None = None,
) -> list[TimeWindow]:
    """Windows named by the question. Empty when the question names no time."""
    raw = text or ""
    # "between 2020 and 2024" is one span. "in 2024 and last month" is two windows.
    if not re.search(r"\b(?:between|from|entre)\b", raw, re.IGNORECASE):
        for left, right in _comparison_sides(raw):
            first = _resolve_one(left, ref_date, country_iso3)
            second = _resolve_one(right, ref_date, country_iso3)
            if first is not None and second is not None:
                return [first, second]
    one = _resolve_one(raw, ref_date, country_iso3)
    return [one] if one is not None else []


def retrieval_envelope(windows: list[TimeWindow]) -> tuple[date, date] | None:
    """Single filter span that includes every Compared window. None when empty."""
    if not windows:
        return None
    return min(w.start for w in windows), max(w.end for w in windows)


def time_gap_sentence(
    phrase: str,
    requested: tuple[date, date],
    used: tuple[date, date],
    *,
    forecast: bool = False,
) -> str:
    """The sentence an answer uses when the cited evidence is outside the window."""
    asked = f"{requested[0].isoformat()}–{requested[1].isoformat()}"
    got = f"{used[0].isoformat()}–{used[1].isoformat()}"
    lead = f"You asked about {phrase} ({asked}). "
    if forecast:
        lead += "That period has not happened. "
    return f"{lead}The most recent OpenTrace data is from {got}; here is that."


def parse_iso_date(raw: str) -> date | None:
    text = (raw or "").strip()[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _resolve_one(text: str, ref: date, country_iso3: str | None) -> TimeWindow | None:
    window = (
        _between(text, ref)
        or _year_dash(text)
        or _since(text, ref)
        or _quarter_or_half(text, ref)
        or _named_month(text, ref)
        or _named_year_drops_relative(text)
        or _fixed_phrase(text, ref, country_iso3)
        or _bare_year(text)
        or _dateparser_window(text, ref)
    )
    return window


def _comparison_sides(text: str) -> list[tuple[str, str]]:
    sides: list[tuple[str, str]] = []
    for match in _COMPARE_SPLIT.finditer(text or ""):
        left = text[: match.start()].strip(" ,;")
        right = text[match.end() :].strip(" ,;")
        if left and right:
            sides.append((left, right))
    return sides


def _window(
    start: date,
    end: date,
    grain: Grain,
    phrase: str,
    *,
    strict: bool = True,
    forecast: bool = False,
    from_report: bool = False,
) -> TimeWindow:
    if end < start:
        start, end = end, start
    return TimeWindow(
        start=start,
        end=end,
        grain=grain,
        phrase=phrase.strip(),
        strict=strict,
        forecast=forecast,
        from_report=from_report,
    )


def _shift_years(day: date, years: int) -> date:
    year = day.year + years
    return date(year, day.month, min(day.day, calendar.monthrange(year, day.month)[1]))


def _shift_months(day: date, months: int) -> date:
    index = day.month - 1 + months
    year = day.year + index // 12
    month = index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _full_year(year: int, phrase: str) -> TimeWindow:
    return _window(date(year, 1, 1), date(year, 12, 31), "year", phrase)


def _most_recent_month(month: int, ref: date) -> tuple[date, date]:
    year = ref.year if (ref.year, month) <= (ref.year, ref.month) else ref.year - 1
    if month > ref.month:
        year = ref.year - 1
    return date(year, month, 1), _month_end(year, month)


def _between(text: str, ref: date) -> TimeWindow | None:
    match = re.search(
        rf"\b(?:between|from|entre)\s+(.+?)\s+(?:and|to|et|e)\s+(.+?)(?:[.?]|$)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    start = _bound(match.group(1), ref, is_end=False)
    end = _bound(match.group(2), ref, is_end=True)
    if start is None or end is None:
        return None
    phrase = match.group(0).strip(" .?")
    return _window(start, end, "range", phrase)


def _bound(fragment: str, ref: date, *, is_end: bool) -> date | None:
    text = fragment.strip(" .?")
    # "2022 in senegal" is still the year 2022. Trailing place words must not change the day.
    year_lead = re.match(rf"({_YEAR})\b", text)
    if year_lead and not re.search(rf"\b({_MONTH_PATTERN})\b", text, re.IGNORECASE):
        year = int(year_lead.group(1))
        return date(year, 12, 31) if is_end else date(year, 1, 1)
    month = _named_month(text, ref)
    if month is not None:
        return month.end if is_end else month.start
    parsed = _dateparser_window(text, ref)
    if parsed is not None:
        return parsed.end if is_end else parsed.start
    return None


def _year_dash(text: str) -> TimeWindow | None:
    match = re.search(rf"\b({_YEAR})\s*[-–]\s*({_YEAR})\b", text)
    if not match:
        return None
    y0, y1 = int(match.group(1)), int(match.group(2))
    return _window(date(y0, 1, 1), date(y1, 12, 31), "range", match.group(0))


def _since(text: str, ref: date) -> TimeWindow | None:
    match = re.search(
        rf"\b(?:since|depuis|desde|from|starting)\s+(?:the\s+)?(?:year\s+)?"
        rf"(?:({_MONTH_PATTERN})\s+)?({_YEAR})\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    if re.search(r"\b(?:to|and|until|till)\s+(?:the\s+)?(?:year\s+)?(?:19|20)\d{2}\b", text, re.IGNORECASE):
        return None
    year = int(match.group(2))
    month_name = match.group(1)
    start = date(year, _MONTHS[month_name.casefold()], 1) if month_name else date(year, 1, 1)
    return _window(start, ref, "range", match.group(0).strip())


def _quarter_or_half(text: str, ref: date) -> TimeWindow | None:
    half = re.search(
        rf"\b(?:(first|1st|second|2nd)\s+half|(premi[eè]re|deuxi[eè]me)\s+moiti[eé])"
        rf"(?:\s+of)?(?:\s+({_YEAR}))?\b",
        text,
        re.IGNORECASE,
    )
    if half:
        word = (half.group(1) or half.group(2) or "").casefold()
        which = 1 if word.startswith("f") or word.startswith("1") or word.startswith("p") else 2
        year = int(half.group(3)) if half.group(3) else _recent_half_year(which, ref)
        return _half_window(which, year, half.group(0).strip())
    short = re.search(rf"\b([Hh])([12])(?:\s+({_YEAR}))?\b", text)
    if short:
        which = int(short.group(2))
        year = int(short.group(3)) if short.group(3) else _recent_half_year(which, ref)
        return _half_window(which, year, short.group(0).strip())
    quarter = re.search(rf"\b[Qq]([1-4])(?:\s+({_YEAR}))?\b", text)
    if not quarter:
        return None
    number = int(quarter.group(1))
    year = int(quarter.group(2)) if quarter.group(2) else _recent_quarter_year(number, ref)
    start_month = 3 * (number - 1) + 1
    return _window(
        date(year, start_month, 1),
        _month_end(year, start_month + 2),
        "range",
        quarter.group(0).strip(),
    )


def _recent_quarter_year(number: int, ref: date) -> int:
    start_month = 3 * (number - 1) + 1
    return ref.year if start_month <= ref.month else ref.year - 1


def _recent_half_year(which: int, ref: date) -> int:
    start_month = 1 if which == 1 else 7
    return ref.year if start_month <= ref.month else ref.year - 1


def _half_window(which: int, year: int, phrase: str) -> TimeWindow:
    if which == 1:
        return _window(date(year, 1, 1), date(year, 6, 30), "range", phrase)
    return _window(date(year, 7, 1), date(year, 12, 31), "range", phrase)


def _named_month(text: str, ref: date) -> TimeWindow | None:
    match = re.search(
        rf"\b(?:in|during|for|en|em|de)\s+({_MONTH_PATTERN})(?:\s+({_YEAR}))?\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        match = re.search(rf"\b({_MONTH_PATTERN})\s+({_YEAR})\b", text, re.IGNORECASE)
        if not match:
            return None
    month = _MONTHS[match.group(1).casefold()]
    if match.lastindex and match.lastindex >= 2 and match.group(2):
        year = int(match.group(2))
        start, end = date(year, month, 1), _month_end(year, month)
    else:
        start, end = _most_recent_month(month, ref)
    return _window(start, end, "month", match.group(0).strip())


def _named_year_drops_relative(text: str) -> TimeWindow | None:
    """A year written beside a relative phrase is the only window, except comparisons and since."""
    if not _RELATIVE_HINT.search(text) or _SINCE_RE.search(text):
        return None
    years = [int(item) for item in _YEAR_RE.findall(text)]
    if len(years) != 1:
        return None
    explicit = re.search(rf"\b(?:in|during|for)\s+({_YEAR})\b", text, re.IGNORECASE)
    phrase = explicit.group(0) if explicit else str(years[0])
    return _full_year(years[0], phrase)


def _fixed_phrase(text: str, ref: date, country_iso3: str | None) -> TimeWindow | None:
    folded = text.casefold()
    rules: tuple[tuple[str, object], ...] = (
        (r"\b(?:today|aujourd['’]?hui|aujourdhui|hoje)\b", lambda m: _window(ref, ref, "day", m.group(0))),
        (r"\b(?:yesterday|hier|ontem)\b", lambda m: _window(ref - timedelta(days=1), ref - timedelta(days=1), "day", m.group(0))),
        (r"\b(?:this\s+week|cette\s+semaine|esta\s+semana)\b", lambda m: _this_week(ref, m.group(0))),
        (r"\b(?:last\s+week|la\s+semaine\s+derni[eè]re|semana\s+passada)\b", lambda m: _last_week(ref, m.group(0))),
        (r"\b(?:this\s+month|ce\s+mois(?:-ci)?|este\s+m[eê]s)\b", lambda m: _window(date(ref.year, ref.month, 1), ref, "month", m.group(0))),
        (r"\b(?:last\s+month|le\s+mois\s+dernier|m[eê]s\s+passado)\b", lambda m: _last_month(ref, m.group(0))),
        (r"\b(?:this\s+year|cette\s+ann[eé]e|este\s+ano)\b", lambda m: _window(date(ref.year, 1, 1), ref, "year", m.group(0))),
        (r"\b(?:last\s+year|l['’]ann[eé]e\s+derni[eè]re|ano\s+passado)\b", lambda m: _full_year(ref.year - 1, m.group(0))),
        (r"\b(?:next\s+year|l['’]ann[eé]e\s+prochaine|ano\s+que\s+vem)\b", lambda m: _window(date(ref.year + 1, 1, 1), date(ref.year + 1, 12, 31), "year", m.group(0), forecast=True)),
        (r"\b(?:past|last)\s+(\d{1,2})\s+(days?|weeks?|months?|years?)\b", lambda m: _last_n(ref, m)),
        (r"\b(?:this|current)\s+season\b", lambda m: _season(ref, country_iso3, m.group(0))),
    )
    for pattern, build in rules:
        match = re.search(pattern, folded, re.IGNORECASE)
        if match:
            built = build(match)  # type: ignore[operator]
            if isinstance(built, TimeWindow):
                return built
    recency = re.search(
        r"\b(?:right\s+now|as\s+of\s+now|currently|current|recently|recent|lately|latest|now)\b",
        folded,
        re.IGNORECASE,
    )
    if recency:
        return _recency(text, ref, recency.group(0))
    return None


def _this_week(ref: date, phrase: str) -> TimeWindow:
    monday = ref - timedelta(days=ref.weekday())
    return _window(monday, ref, "week", phrase)


def _last_week(ref: date, phrase: str) -> TimeWindow:
    this_monday = ref - timedelta(days=ref.weekday())
    start = this_monday - timedelta(days=7)
    end = this_monday - timedelta(days=1)
    return _window(start, end, "week", phrase)


def _last_month(ref: date, phrase: str) -> TimeWindow:
    first_this = date(ref.year, ref.month, 1)
    end = first_this - timedelta(days=1)
    return _window(date(end.year, end.month, 1), end, "month", phrase)


def _last_n(ref: date, match: re.Match[str]) -> TimeWindow:
    count = max(1, min(50, int(match.group(1))))
    unit = match.group(2).casefold()
    if unit.startswith("day"):
        start = ref - timedelta(days=count)
        grain: Grain = "day"
    elif unit.startswith("week"):
        start = ref - timedelta(days=7 * count)
        grain = "week"
    elif unit.startswith("month"):
        start = _shift_months(ref, -count)
        grain = "month"
    else:
        start = _shift_years(ref, -count)
        grain = "year"
    return _window(start, ref, "range" if grain == "year" else grain, match.group(0))


def _recency(text: str, ref: date, phrase: str) -> TimeWindow:
    if _REPORT_SUBJECT.search(text):
        return _window(
            date(ref.year, 1, 1),
            ref,
            "year",
            phrase,
            strict=False,
            from_report=True,
        )
    if _DAILY_SUBJECT.search(text):
        return _window(ref - timedelta(days=90), ref, "range", phrase, strict=False)
    return _window(date(ref.year, 1, 1), ref, "year", phrase, strict=False)


def _bare_year(text: str) -> TimeWindow | None:
    # Leave day-level dates (15/09/2024, 2024-09-15) for the dateparser fallback.
    cleaned = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b", " ", text or "")
    cleaned = re.sub(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", " ", cleaned)
    years = [int(item) for item in _YEAR_RE.findall(cleaned)]
    if not years:
        return None
    if len(years) >= 2:
        return _window(date(min(years), 1, 1), date(max(years), 12, 31), "range", f"{min(years)}-{max(years)}")
    explicit = re.search(rf"\b(?:in|during|for)\s+({_YEAR})\b", cleaned, re.IGNORECASE)
    phrase = explicit.group(0) if explicit else str(years[0])
    return _full_year(years[0], phrase)


@dataclass(frozen=True)
class SeasonRow:
    """One country × season_name row from dim_season."""

    country_iso3: str
    season_name: str
    start_month: int
    end_month: int
    crosses_year: bool = False
    is_off_season: bool = False


_season_calendar_override: tuple[SeasonRow, ...] | None = None


def set_season_calendar(rows: Sequence[SeasonRow] | None) -> None:
    """Inject season rows for tests. ``None`` restores the dim_season loader."""
    global _season_calendar_override
    _season_calendar_override = tuple(rows) if rows is not None else None
    _season_rows_from_bq.cache_clear()


def _season_rows() -> tuple[SeasonRow, ...]:
    if _season_calendar_override is not None:
        return _season_calendar_override
    return _season_rows_from_bq()


@lru_cache(maxsize=1)
def _season_rows_from_bq() -> tuple[SeasonRow, ...]:
    """Cached read of mart_dev.dim_season. Empty when BQ is unavailable."""
    project = (os.environ.get("BQ_PROJECT") or "").strip()
    if not project:
        return ()
    dataset = (os.environ.get("BQ_DATASET_GOLD") or "mart_dev").strip() or "mart_dev"
    sql = (
        f"SELECT country, season_name, start_month, end_month, "
        f"crosses_year, is_off_season "
        f"FROM `{project}.{dataset}.dim_season`"
    )
    try:
        from google.cloud import bigquery

        from ml.rag.chatbot.geo_iso3 import name_to_iso3

        client = bigquery.Client(project=project)
        raw_rows = list(client.query(sql).result())
    except Exception:
        logger.debug("dim_season load failed; season windows will soft-fallback", exc_info=True)
        return ()
    out: list[SeasonRow] = []
    for raw in raw_rows:
        iso = name_to_iso3(str(getattr(raw, "country", "") or ""))
        if not iso:
            continue
        try:
            start_month = int(getattr(raw, "start_month"))
            end_month = int(getattr(raw, "end_month"))
        except (TypeError, ValueError):
            continue
        if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
            continue
        crosses = bool(getattr(raw, "crosses_year", False)) or start_month > end_month
        out.append(
            SeasonRow(
                country_iso3=iso,
                season_name=str(getattr(raw, "season_name", "") or "").strip(),
                start_month=start_month,
                end_month=end_month,
                crosses_year=crosses,
                is_off_season=bool(getattr(raw, "is_off_season", False)),
            )
        )
    return tuple(out)


def _month_in_season(month: int, row: SeasonRow) -> bool:
    if row.crosses_year or row.start_month > row.end_month:
        return month >= row.start_month or month <= row.end_month
    return row.start_month <= month <= row.end_month


def _season_span(row: SeasonRow, ref: date) -> tuple[date, date]:
    if not (row.crosses_year or row.start_month > row.end_month):
        return date(ref.year, row.start_month, 1), _month_end(ref.year, row.end_month)
    if ref.month >= row.start_month:
        return date(ref.year, row.start_month, 1), _month_end(ref.year + 1, row.end_month)
    return date(ref.year - 1, row.start_month, 1), _month_end(ref.year, row.end_month)


def _season_end_before(row: SeasonRow, ref: date) -> date:
    end = _season_span(row, ref)[1]
    if end >= ref:
        previous = date(ref.year - 1, ref.month, min(ref.day, 28))
        end = _season_span(row, previous)[1]
        if end >= ref:
            end = _month_end(ref.year - 1, row.end_month)
    return end


def _pick_season(rows: list[SeasonRow]) -> SeasonRow:
    preferred = [
        row
        for row in rows
        if not row.is_off_season and "off" not in row.season_name.casefold()
    ]
    return (preferred or rows)[0]


def _season(ref: date, country_iso3: str | None, phrase: str) -> TimeWindow:
    iso = (country_iso3 or "").strip().upper()
    rows = [row for row in _season_rows() if row.country_iso3 == iso]
    if not rows:
        return _window(_shift_months(ref, -6), ref, "range", phrase, strict=False)
    containing = [row for row in rows if _month_in_season(ref.month, row)]
    if containing:
        chosen = _pick_season(containing)
        start, end = _season_span(chosen, ref)
        return _window(start, end, "season", phrase)
    ended = max(rows, key=lambda row: _season_end_before(row, ref))
    end = _season_end_before(ended, ref)
    start = _season_span(ended, end)[0]
    return _window(start, end, "season", phrase)


def _dateparser_window(text: str, ref: date) -> TimeWindow | None:
    """Recognize a phrase our rules did not claim. Settings pin the reference date."""
    try:
        from dateparser.search import search_dates
    except ImportError:
        return None
    found = search_dates(
        text,
        languages=["en", "fr", "pt"],
        settings={
            "RELATIVE_BASE": datetime(ref.year, ref.month, ref.day),
            "PREFER_DATES_FROM": "past",
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )
    if not found:
        return None
    useful: list[tuple[str, date]] = []
    for fragment, parsed in found:
        frag = (fragment or "").strip()
        if not _DATE_TOKEN.search(frag):
            continue
        day = parsed.date() if isinstance(parsed, datetime) else parsed
        if isinstance(day, date):
            useful.append((frag, day))
    if not useful:
        return None
    phrase = useful[0][0] if len(useful) == 1 else text.strip()[:80]
    days = [item[1] for item in useful]
    if len(days) == 1:
        return _window(days[0], days[0], "day", phrase)
    return _window(min(days), max(days), "range", phrase)
