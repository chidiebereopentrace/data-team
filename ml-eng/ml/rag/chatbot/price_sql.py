"""One price template over fct_prices: product keys x country x time, labelled from the mart.

Every filter literal comes from the price vocabulary built off live BigQuery
(``price_vocab.py``): product_key, price_source, price_type, metric. Output rows
always carry product_name, price_type, price_source, unit, currency, the market
count and the observation dates, grouped so sources, units and currencies never blend.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from ml.rag.chatbot.agri_entities import CROP_COMMODITY_TERMS
from ml.rag.chatbot.bundle_metrics import AGRI_ACTIVITIES_STAPLES, food_price_basket
from ml.rag.chatbot.geo_iso3 import iso3_to_canonical_name, name_to_iso3
from ml.rag.chatbot.sql_ast_validate import string_literal as _lit
from ml.rag.chatbot.price_vocab import (
    PricePlace,
    PriceSeries,
    choose_price_type,
    held_price_sources,
    load_price_vocab,
    region_place,
    resolve_price_place,
    resolve_price_products,
)
from ml.rag.chatbot.time_resolver import DAILY_RECENCY_DAYS, parse_iso_date, resolve_relative_time

_ISO3_RE = re.compile(r"^[A-Z]{3}$")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PRICE_RE = re.compile(
    r"\b(price|prices|priced|cost|costs|how much|expensive|cheap|retail|wholesale|farm[- ]?gate|volatile|volatility)\b",
    re.IGNORECASE,
)
_ROW_LIMIT = 60
PRICE_LIST_ROW_LIMIT = 500
_PRICE_CROP_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, sorted(CROP_COMMODITY_TERMS, key=len, reverse=True))) + r")(?:e?s)?\b",
    re.I,
)
_PRICE_LIST_RE = re.compile(r"\b(?:price\s+list(?:s)?|list\b[^.?!]{0,60}\bprices?)\b", re.I)
_PROCESSED_RE = re.compile(r"\b(?:flour|bran|meal|oil|starch|flaked|feed)\b", re.I)


def requested_price_list_products(query: str) -> list[str]:
    """Identify explicit lists of separately named crops."""
    if not _PRICE_LIST_RE.search(query) or _PROCESSED_RE.search(query):
        return []
    products = list(dict.fromkeys(
        "maize" if match[1].lower() == "corn" else match[1].lower()
        for match in _PRICE_CROP_RE.finditer(query)
    ))
    return products if len(products) > 1 else []


@dataclass(frozen=True)
class PriceWindow:
    """Requested bounds and whether to return the latest month per series."""

    year_start: int | None
    year_end: int | None
    date_start: date | None = None
    date_end: date | None = None
    latest_in_window: bool = False

    @property
    def latest(self) -> bool:
        return self.latest_in_window or (self.year_start is None and self.year_end is None)

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
    start = parse_iso_date(time_start or "")
    end = parse_iso_date(time_end or "")
    low = _YEAR_RE.search(time_start or "")
    high = _YEAR_RE.search(time_end or "")
    if not low and not high:
        years = [int(m.group(0)) for m in _YEAR_RE.finditer(query or "")]
        return PriceWindow(min(years), max(years)) if years else PriceWindow(None, None)
    low_year = int(low.group(0)) if low else None
    high_year = int(high.group(0)) if high else None
    named = resolve_relative_time(query, end) if start and end and (end - start).days == DAILY_RECENCY_DAYS else []
    latest = (
        len(named) == 1 and not named[0].strict and named[0].grain == "range"
        and named[0].start == start and named[0].end == end
    )
    return PriceWindow(low_year or high_year, high_year or low_year, start, end, latest)


def _in(values: list[str]) -> str:
    return ", ".join(_lit(v) for v in sorted(set(values)))


def build_price_sql(
    *,
    project_id: str,
    dataset: str,
    iso3s: list[str],
    series: list[PriceSeries],
    price_source: str | list[str],
    price_type: str | None,
    window: PriceWindow,
    places: list[PricePlace] | None = None,
    product_groups: dict[str, list[PriceSeries]] | None = None,
    summarize: bool = False,
) -> str:
    """Code-owned price SQL. ``series`` are vocabulary rows already scoped to ``iso3s``.

    ``places`` are the named market and/or its admin-1 region, narrowest first. Rows
    are then split by place tier (market, region, rest of the country) so the
    answer gets the local figure when it exists and the national context always.
    """
    sources = [price_source] if isinstance(price_source, str) else list(price_source)
    sources = [s for s in sources if str(s).strip()]
    keys = sorted({s.product_key for s in series})
    metrics = sorted(
        {m for s in series if any(src in s.price_sources for src in sources) for m in s.metrics}
    )
    if len(sources) == 1:
        source_pred = f"price_source = {_lit(sources[0])}"
    else:
        source_pred = f"price_source IN ({_in(sources)})"
    filters = [
        f"country_iso3 IN ({_in(iso3s)})",
        f"product_key IN ({_in(keys)})",
        source_pred,
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
    else:
        # An unspecified type includes every held type, grouped separately below.
        held_types = [kind for row in series for kind in row.price_types if kind]
        if held_types:
            filters.append(f"price_type IN ({_in(held_types)})")
    if metrics:
        filters.append(f"metric IN ({_in(metrics)})")
    if window.date_start:
        filters.append(f"as_of_date >= DATE '{window.date_start}'")
    if window.date_end:
        filters.append(f"as_of_date <= DATE '{window.date_end}'")
    if window.year_start is None:
        floor = min(int(s.last_date[:4]) for s in series if s.last_date[:4].isdigit()) - 1
        filters.append(f"year >= {floor}")
    elif window.multi_year:
        filters.append(f"year BETWEEN {window.year_start} AND {window.year_end}")
    else:
        filters.append(f"year = {window.year_start}")
    qualify = ""
    if window.latest:
        qualify = (
            "QUALIFY DATE_TRUNC(as_of_date, MONTH) = MAX(DATE_TRUNC(as_of_date, MONTH)) OVER "
            f"(PARTITION BY country_iso3, product_key, price_source, price_type, unit, currency{scope_partition}) "
        )
    annual = window.multi_year and not window.latest
    year_col = "b.year, " if annual else ""
    fqn = f"`{project_id}.{dataset}"
    requested_inner = requested_outer = requested_group = ""
    if product_groups:
        cases = " ".join(
            f"WHEN product_key IN ({_in([s.product_key for s in members])}) THEN {_lit(product)}"
            for product, members in product_groups.items() if members
        )
        requested_inner = f"CASE {cases} END AS requested_product, "
        requested_outer = "b.requested_product, "
        requested_group = ", b.requested_product"
    sql = (
        f"SELECT b.country_iso3, {scope_outer}{requested_outer}d.product_name, b.price_type, b.price_source, b.unit, b.currency, {year_col}"
        "AVG(b.value) AS avg_price, MIN(b.value) AS min_price, "
        "MAX(b.value) AS max_price, COUNT(DISTINCT b.market_key) AS market_count, "
        "MIN(b.as_of_date) AS period_start, MAX(b.as_of_date) AS latest_date "
        "FROM ("
        f"SELECT country_iso3, {scope_inner}{requested_inner}product_key, price_type, price_source, unit, currency, "
        "year, as_of_date, market_key, value "
        f"FROM {fqn}.fct_prices` "
        f"WHERE {' AND '.join(filters)} "
        f"{qualify}"
        f") b JOIN {fqn}.dim_product` d ON d.product_key = b.product_key "
        f"GROUP BY b.country_iso3{scope_group}{requested_group}, d.product_name, b.price_type, b.price_source, b.unit, b.currency"
        f"{', b.year' if annual else ''} "
    )
    if product_groups:
        # Keep one series per crop throughout the period. Its type, currency and unit
        # remain separate and a missing year stays empty.
        series_keys = "country_iso3, requested_product, product_name, price_type, price_source, unit, currency"
        selected = (
            f"WITH prices AS ({sql}), coverage AS ("
            f"SELECT *, COUNT(*) OVER (PARTITION BY {series_keys}) AS covered_years, "
            f"MAX(latest_date) OVER (PARTITION BY {series_keys}) AS last_seen FROM prices) "
            "SELECT * EXCEPT(covered_years, last_seen) FROM coverage "
            "QUALIFY DENSE_RANK() OVER (PARTITION BY country_iso3, requested_product "
            "ORDER BY CASE price_type WHEN 'Retail' THEN 0 WHEN 'Wholesale' THEN 1 ELSE 2 END, "
            "covered_years DESC, last_seen DESC, product_name, price_source, unit, currency) = 1 "
        )
        if summarize:
            return _summarize_price_history(selected, series_keys)
        return selected + f"ORDER BY {'year DESC, ' if annual else ''}requested_product LIMIT {PRICE_LIST_ROW_LIMIT}"
    time_order = "latest_date DESC, " if window.latest else "b.year, " if annual else ""
    return sql + f"ORDER BY {scope_order}{time_order}market_count DESC LIMIT {_ROW_LIMIT}"


def _summarize_price_history(sql: str, series_keys: str) -> str:
    """One comparable series per country and basket item, retaining every annual figure."""
    first = "ARRAY_AGG(avg_price ORDER BY year LIMIT 1)[OFFSET(0)]"
    last = "ARRAY_AGG(avg_price ORDER BY year DESC LIMIT 1)[OFFSET(0)]"
    return (
        f"WITH selected AS ({sql}) SELECT {series_keys}, "
        f"{first} AS first_price, MIN(year) AS first_year, {last} AS avg_price, MAX(year) AS year, "
        f"IF(unit = '%', NULL, 100 * SAFE_DIVIDE({last} - {first}, {first})) AS change_pct, "
        f"IF(unit = '%', {last} - {first}, NULL) AS change_percentage_points, "
        "STRING_AGG(FORMAT('%d: %.2f', year, avg_price), '; ' ORDER BY year) AS annual_prices, "
        "COUNT(*) AS covered_years, MAX(market_count) AS market_count, "
        "MIN(period_start) AS period_start, MAX(latest_date) AS latest_date "
        f"FROM selected GROUP BY {series_keys} ORDER BY country_iso3, requested_product LIMIT {PRICE_LIST_ROW_LIMIT}"
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
    named = food_price_basket(query)
    window = price_window(time_start, time_end, query)
    report = named and window.multi_year
    products = requested_price_list_products(query) if len(iso3s) == 1 and not named else []
    groups: dict[str, list[PriceSeries]] = {}
    if products and not resolve_price_place(query, iso3s[0]):
        groups = {
            product: [
                row for row in resolve_price_products(product, iso3s[0]).series
                if not _PROCESSED_RE.search(row.product_name)
            ] for product in products
        }
        series = [row for members in groups.values() for row in members]
        named = True
    elif named:
        vocab = load_price_vocab()
        groups = {
            product: [row for code in iso3s for row in resolve_price_products(product, code).series]
            for product in AGRI_ACTIVITIES_STAPLES
        }
        if report:
            for product in ("Consumer Prices, Food Indices (2015 = 100)", "Food price inflation"):
                groups[product] = [row for code in iso3s for row in vocab.for_country(code) if row.product_name == product]
        series = [row for members in groups.values() for row in members]
        if not report:
            groups = {}
    else:
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
    report = report and not places
    if named and not products and not report:
        groups = {}
    if report:
        notes.append(
            "Commodity coverage is the available maize, rice, cassava, sorghum, millet and yam basket; "
            "it is not all foods or an official CPI basket. National food indices and food-price inflation "
            "are separate series. annual_prices contains every retrieved year. Changes use first_year "
            "and year for the same product, source, price type, unit and currency; missing years stay missing. "
            "Current-year averages cover only observations through latest_date and are provisional."
        )
    if window.year_start is not None:
        in_window = [s for s in series if _covers(s, window)]
        if in_window:
            series = in_window
    sources = held_price_sources(blob, series)
    if not sources:
        return None
    scoped = [s for s in series if any(src in s.price_sources for src in sources)]
    named_type = bool(re.search(r"\b(retail|wholesale|producer|farm[- ]?gate)\b", blob or "", re.I))
    price_type: str | None = None
    if named_type:
        price_type = choose_price_type(blob, scoped, sources[0] if len(sources) == 1 else None)
        if price_type is None and groups:
            return {"sql": "", "template": "price_gap", "gap_message": _gap_message(blob, iso3s, []), "table_id": "fct_prices"}
        if price_type is None:
            price_type = choose_price_type("", scoped, sources[0] if len(sources) == 1 else None)
            if price_type:
                notes.append(
                    f"The requested price type is not held for this product; {price_type} prices shown."
                )
    if window.year_start is None and not groups:
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
            price_source=sources if len(sources) > 1 else sources[0],
            price_type=price_type,
            window=window,
            places=places,
            product_groups=groups or None,
            summarize=report,
        ),
        "template": "mart_price",
        "countries": iso3s,
        "places": [f"{p.level}:{p.name}" for p in places],
        "product_names": sorted({s.product_name for s in scoped}),
        "price_list_products": products if groups else [],
        "price_report": report,
        "price_source": sources[0] if len(sources) == 1 else sources,
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
