"""One price template over fct_prices: product keys x country x time, labelled from the mart.

Every filter literal comes from the price vocabulary built off live BigQuery
(``price_vocab.py``): product_key, price_source, price_type, metric. Output rows
always carry product_name, price_type, price_source, unit, currency, the market
count and the observation dates, grouped so sources, units and currencies never blend.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ml.rag.chatbot.geo_iso3 import iso3_to_canonical_name, name_to_iso3
from ml.rag.chatbot.price_vocab import (
    PricePlace,
    PriceSeries,
    choose_price_source,
    choose_price_type,
    region_place,
    resolve_price_place,
    resolve_price_products,
)

_ISO3_RE = re.compile(r"^[A-Z]{3}$")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PRICE_RE = re.compile(
    r"\b(price|prices|priced|cost|costs|how much|expensive|cheap|retail|wholesale|farm[- ]?gate|volatile|volatility)\b",
    re.IGNORECASE,
)
_ROW_LIMIT = 60


@dataclass(frozen=True)
class PriceWindow:
    """Single year, inclusive year range, or latest month held per series."""

    year_start: int | None
    year_end: int | None

    @property
    def latest(self) -> bool:
        return self.year_start is None and self.year_end is None

    @property
    def multi_year(self) -> bool:
        return self.year_start is not None and self.year_end is not None and self.year_end > self.year_start


def is_price_question(text: str) -> bool:
    return bool(_PRICE_RE.search(text or ""))


def to_iso3(labels: list[str]) -> list[str]:
    out: list[str] = []
    for raw in labels:
        text = str(raw or "").strip()
        code = text.upper() if _ISO3_RE.match(text) else (name_to_iso3(text) or "")
        if code and code not in out:
            out.append(code)
    return out


def price_window(time_start: str | None, time_end: str | None, query: str) -> PriceWindow:
    years: list[int] = []
    for raw in (time_start, time_end):
        m = _YEAR_RE.search(str(raw or ""))
        if m:
            years.append(int(m.group(0)))
    if not years:
        years = [int(m.group(0)) for m in _YEAR_RE.finditer(query or "")]
    if not years:
        return PriceWindow(None, None)
    return PriceWindow(min(years), max(years))


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _in(values: list[str]) -> str:
    return ", ".join(_lit(v) for v in sorted(set(values)))


def build_price_sql(
    *,
    project_id: str,
    dataset: str,
    iso3s: list[str],
    series: list[PriceSeries],
    price_source: str,
    price_type: str | None,
    window: PriceWindow,
    places: list[PricePlace] | None = None,
) -> str:
    """Code-owned price SQL. ``series`` are vocabulary rows already scoped to ``iso3s``.

    ``places`` are the named market and/or its admin-1 region, narrowest first. Rows
    are then split by place tier (market, region, rest of the country) so the
    answer gets the local figure when it exists and the national context always.
    """
    keys = sorted({s.product_key for s in series})
    metrics = sorted({m for s in series if price_source in s.price_sources for m in s.metrics})
    filters = [
        f"country_iso3 IN ({_in(iso3s)})",
        f"product_key IN ({_in(keys)})",
        f"price_source = {_lit(price_source)}",
    ]
    scope_inner = scope_outer = scope_group = scope_order = ""
    scope_partition = ""
    if places:
        rest = f"Rest of {iso3_to_canonical_name(iso3s[0]) or iso3s[0]}"
        whens = " ".join(
            f"WHEN market_key IN ({_in(list(p.market_keys))}) THEN {_lit(p.name)}" for p in places
        )
        scope_expr = f"CASE {whens} ELSE {_lit(rest)} END"
        label_col = "market_name" if places[0].level == "market" else "admin_1"
        scope_inner = f"{scope_expr} AS {label_col}, "
        scope_outer = f"b.{label_col}, "
        scope_group = f", b.{label_col}"
        scope_order = f"STARTS_WITH(b.{label_col}, 'Rest of '), "
        scope_partition = f", {scope_expr}"
    if price_type:
        filters.append(f"price_type = {_lit(price_type)}")
    if metrics:
        filters.append(f"metric IN ({_in(metrics)})")
    qualify = ""
    if window.latest:
        floor = min(int(s.last_date[:4]) for s in series if s.last_date[:4].isdigit()) - 1
        filters.append(f"year >= {floor}")
        qualify = (
            "QUALIFY DATE_TRUNC(as_of_date, MONTH) = MAX(DATE_TRUNC(as_of_date, MONTH)) OVER "
            f"(PARTITION BY country_iso3, product_key, price_type, unit, currency{scope_partition}) "
        )
    elif window.multi_year:
        filters.append(f"year BETWEEN {window.year_start} AND {window.year_end}")
    else:
        filters.append(f"year = {window.year_start}")
    year_col = "b.year, " if window.multi_year else ""
    fqn = f"`{project_id}.{dataset}"
    return (
        f"SELECT b.country_iso3, {scope_outer}d.product_name, b.price_type, b.price_source, b.unit, b.currency, {year_col}"
        "AVG(b.value) AS avg_price, MIN(b.value) AS min_price, "
        "MAX(b.value) AS max_price, COUNT(DISTINCT b.market_key) AS market_count, "
        "MIN(b.as_of_date) AS period_start, MAX(b.as_of_date) AS latest_date "
        "FROM ("
        f"SELECT country_iso3, {scope_inner}product_key, price_type, price_source, unit, currency, "
        "year, as_of_date, market_key, value "
        f"FROM {fqn}.fct_prices` "
        f"WHERE {' AND '.join(filters)} "
        f"{qualify}"
        f") b JOIN {fqn}.dim_product` d ON d.product_key = b.product_key "
        f"GROUP BY b.country_iso3{scope_group}, d.product_name, b.price_type, b.price_source, b.unit, b.currency"
        f"{', b.year' if window.multi_year else ''} "
        f"ORDER BY {'b.year, ' if window.multi_year else ''}{scope_order}market_count DESC "
        f"LIMIT {_ROW_LIMIT}"
    )


def _covers(series: PriceSeries, window: PriceWindow) -> bool:
    first, last = series.first_date[:4], series.last_date[:4]
    if not (first.isdigit() and last.isdigit()):
        return False
    return int(first) <= (window.year_end or 0) and int(last) >= (window.year_start or 0)


def _gap_message(product_text: str, iso3s: list[str], closest: list[str]) -> str:
    places = ", ".join(iso3_to_canonical_name(c) or c for c in iso3s)
    line = f"No warehouse price series for the requested product in {places}."
    if closest:
        line += " Closest products with price data there: " + "; ".join(closest) + "."
    return line


def plan_price_template(
    *,
    query: str,
    blob: str,
    country_labels: list[str],
    time_start: str | None,
    time_end: str | None,
    project_id: str,
    dataset: str,
) -> dict[str, Any] | None:
    """Template hit for a price question with at least one country, or None.

    Returns ``{"gap_message": ...}`` with empty SQL when the question names a
    product the mart does not hold for the place (never an unfiltered query).
    """
    iso3s = to_iso3(country_labels)
    if not iso3s or not is_price_question(blob):
        return None
    series: list[PriceSeries] = []
    closest: list[str] = []
    named = False
    for code in iso3s:
        res = resolve_price_products(blob, code)
        named = named or res.named
        series.extend(res.series)
        closest.extend(n for n in res.closest if n not in closest)
    if not series:
        if not named:
            return None
        return {
            "sql": "",
            "template": "price_gap",
            "gap_message": _gap_message(blob, iso3s, closest[:3]),
            "countries": iso3s,
            "table_id": "fct_prices",
        }
    places, place_note = _places_for(query, iso3s, series)
    notes = [place_note] if place_note else []
    window = price_window(time_start, time_end, query)
    if not window.latest:
        in_window = [s for s in series if _covers(s, window)]
        if in_window:
            series = in_window
        else:
            span = f"{window.year_start}" + ("" if window.year_end == window.year_start else f"-{window.year_end}")
            latest_year = max(s.last_date[:4] for s in series)
            notes.append(f"No price rows for {span}; latest available month shown (data runs to {latest_year}).")
            window = PriceWindow(None, None)
    source = choose_price_source(blob, series)
    if not source:
        return None
    scoped = [s for s in series if source in s.price_sources]
    price_type = choose_price_type(blob, scoped, source)
    if price_type is None:
        price_type = choose_price_type("", scoped, source)
        notes.append(f"The requested price type is not held for this product; {price_type} prices shown.")
    if window.latest:
        freshest = max(s.last_date for s in scoped)
        cutoff = f"{int(freshest[:4]) - 1}{freshest[4:]}"
        scoped = [s for s in scoped if s.last_date >= cutoff]
    if len(iso3s) > 1 and len(currencies := sorted({c for s in scoped for c in s.currencies})) > 1:
        notes.append(
            f"Prices are in {', '.join(currencies)}; without an exchange rate no country's price "
            "can be called higher or lower than another's."
        )
    return {
        "sql": build_price_sql(
            project_id=project_id,
            dataset=dataset,
            iso3s=sorted({s.country_iso3 for s in scoped}),
            series=scoped,
            price_source=source,
            price_type=price_type,
            window=window,
            places=places,
        ),
        "template": "mart_price",
        "countries": iso3s,
        "places": [f"{p.level}:{p.name}" for p in places],
        "product_names": sorted({s.product_name for s in scoped})[:12],
        "price_source": source,
        "price_type": price_type,
        "widened": " ".join(notes),
        "table_id": "fct_prices",
    }


def _places_for(query: str, iso3s: list[str], series: list[PriceSeries]) -> tuple[list[PricePlace], str]:
    """Place tiers for the named market or region (narrowest first), and a note when it lacks the product."""
    if len(iso3s) != 1:
        return [], ""
    asked = resolve_price_place(query, iso3s[0])
    if asked is None:
        return [], ""
    places = [asked]
    if asked.level == "market" and (region := region_place(iso3s[0], asked.admin_1)) is not None:
        places.append(region)
    keys = {s.product_key for s in series}
    if any(p.product_keys & keys for p in places):
        return places, ""
    country = iso3_to_canonical_name(iso3s[0]) or iso3s[0]
    return places, f"No {asked.name} price rows for this product; rest of {country} shown."


__all__ = [
    "PriceWindow",
    "build_price_sql",
    "is_price_question",
    "plan_price_template",
    "price_window",
    "to_iso3",
]
