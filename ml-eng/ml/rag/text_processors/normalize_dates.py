"""
Normalize free-form publication / as-of date strings to ISO ``YYYY-MM-DD``.

Accepts ISO and RFC-822 / email dates. Rejects truncated or unparseable values
(omit over invent — e.g. ``Fri, 25 Se`` → empty).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import Any


def normalize_to_iso_date(raw: Any) -> str | None:
    """
    Return ``YYYY-MM-DD`` when ``raw`` is a parseable date; otherwise ``None``.

    Supported forms:
    - ``date`` / ``datetime`` instances
    - ISO ``YYYY-MM-DD``, ``YYYY-MM-DDTHH:MM…``, ``YYYY-MM``
    - bare year ``YYYY`` → ``YYYY-01-01``
    - RFC-822 / email dates via ``email.utils.parsedate_to_datetime``
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()

    text = str(raw).strip().strip("'").strip('"')
    if not text:
        return None

    # ISO date prefix (possibly datetime)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        try:
            date.fromisoformat(text)
            return text
        except ValueError:
            return None
    if re.match(r"^\d{4}-\d{2}-\d{2}[ T]", text):
        candidate = text[:10]
        try:
            date.fromisoformat(candidate)
            return candidate
        except ValueError:
            return None

    # YYYY-MM
    m = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if m:
        try:
            month = int(m.group(2))
            if 1 <= month <= 12:
                return f"{m.group(1)}-{m.group(2)}-01"
        except ValueError:
            pass
        return None

    # year only
    if re.fullmatch(r"\d{4}", text):
        return f"{text}-01-01"

    # fromisoformat with Z / offset
    iso_try = text
    if iso_try.endswith("Z") and "T" in iso_try:
        iso_try = iso_try[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(iso_try).date().isoformat()
    except Exception:
        pass

    # Embedded ISO date substring
    embedded = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if embedded:
        try:
            date.fromisoformat(embedded.group(1))
            return embedded.group(1)
        except ValueError:
            pass

    # RFC-822 / email dates (e.g. Fri, 25 Sep 2024 12:00:00 GMT)
    try:
        dt = parsedate_to_datetime(text)
        if dt is not None:
            return dt.date().isoformat()
    except (TypeError, ValueError, IndexError, OverflowError):
        pass

    return None


def normalize_published_at(value: str) -> str:
    """News front-matter helper: ISO date string or empty (never garbage)."""
    parsed = normalize_to_iso_date(value)
    return parsed or ""
