"""Render a price history directly from the selected warehouse price series."""
from __future__ import annotations

from math import isfinite
import re
from typing import Any

from ml.rag.chatbot.price_sql import PriceWindow
from ml.rag.chatbot.geo_iso3 import iso3_to_canonical_name, name_to_iso3
from ml.rag.chatbot.time_resolver import parse_iso_date


def render_verified_price_report(
    sources: list[tuple[int, dict[str, Any]]], *, start: str, end: str,
) -> str:
    """Render SQL-computed annual histories independently of narrative generation."""
    first, last = parse_iso_date(start), parse_iso_date(end)
    if not first or not last:
        return ""
    countries: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for source_id, item in sources:
        meta = item.get("metadata") or {}
        row = meta.get("raw_row") or {}
        if not meta.get("price_report") or not row.get("annual_prices"):
            continue
        name = iso3_to_canonical_name(str(row.get("country_iso3") or ""))
        if name:
            countries.setdefault(name, []).append((source_id, row))
    if not countries:
        return ""
    years = list(range(first.year, last.year + 1))
    lines = [
        f"Food-price changes, {first.year}–{last.year}", "",
        "National food indicators and a sample of staple prices are shown separately. "
        "Figures are annual averages of available observations; a dash marks a missing year.",
    ]
    for country, rows in countries.items():
        lines.extend(("", f"**{country}**", ""))
        for source_id, row in rows:
            label = str(row.get("requested_product") or "")
            if "Food Indices" in label and row.get("change_pct") is not None:
                lines.extend((f"The national food-price index changed by {float(row['change_pct']):+,.2f}% "
                              f"between {row.get('first_year')} and {row.get('year')} [{source_id}].", ""))
            if label == "Food price inflation":
                observed = [(int(year), float(value)) for year, value in re.findall(
                    r"\b((?:19|20)\d{2}): ([+-]?\d+(?:\.\d+)?)", str(row["annual_prices"]))]
                if observed:
                    peak_year, peak = max(observed, key=lambda observation: observation[1])
                    latest_year, latest = max(observed)
                    lines.extend((f"Food inflation's highest reported annual average was {peak:,.2f}% in {peak_year}; "
                                  f"the latest is {latest:,.2f}% in {latest_year} [{source_id}].", ""))
        lines.extend((
                      "| Indicator or product | Unit | " + " | ".join(map(str, years)) + " | Change over available years |",
                      "|---|---|" + "---:|" * (len(years) + 1)))
        dates = []
        for source_id, row in rows:
            values = dict(re.findall(r"\b((?:19|20)\d{2}): ([+-]?\d+(?:\.\d+)?)", str(row["annual_prices"])))
            cells = [f"{float(values[str(year)]):,.2f}" if str(year) in values else "—" for year in years]
            label = str(row.get("product_name") or row.get("requested_product") or "")
            unit = " ".join(str(row.get(key) or "") for key in ("currency", "unit")).strip()
            if not unit and "(2015 = 100)" in label:
                unit = "index (2015 = 100)"
            change = "—"
            for column, suffix in (("change_pct", "%"), ("change_percentage_points", " pp")):
                value = row.get(column)
                if value is not None and isfinite(float(value)):
                    change = f"{float(value):+,.2f}{suffix} ({row.get('first_year')}–{row.get('year')})"
            lines.append(f"| {label} [{source_id}] | {unit or 'not recorded'} | " + " | ".join(cells) + f" | {change} |")
            dates.append(f"{row.get('requested_product') or label}: {row.get('latest_date')} [{source_id}]")
        lines.extend(("", "Latest observations — " + "; ".join(dates) + "."))
    lines.extend(("", "The staple sample covers the available maize, rice, cassava, sorghum, millet and yam series; "
                  "it is not all foods or an official CPI basket. Market averages can change with the markets observed. "
                  "Current-year figures cover only the reported dates and remain partial. NGN and XAF levels are not "
                  "directly comparable; relative changes describe each series separately. pp means percentage points."))
    return "\n".join(lines)


def render_verified_price(
    rows: list[dict[str, Any]], *, country: str, product: str,
    start: str, end: str, language: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    """Render one recent, directly comparable retail series with its real observation date."""
    if language not in ("en", "ig", "pcm"):
        return None
    first, last = parse_iso_date(start), parse_iso_date(end)
    iso = name_to_iso3(country)
    if not first or not last or not iso:
        return None
    offers = []
    for item in rows:
        meta = item.get("metadata") or {}
        raw = meta.get("raw_row") or {}
        label = str(raw.get("product_name") or "")
        if (
            meta.get("template") != "mart_price" or raw.get("price_type") != "Retail"
            or raw.get("country_iso3") != iso or label.casefold() != product.casefold()
            or raw.get("market_name") or raw.get("admin_1")
        ):
            continue
        latest = parse_iso_date(str(raw.get("latest_date") or ""))
        period_start = parse_iso_date(str(raw.get("period_start") or ""))
        if not latest or not period_start or not first <= period_start <= latest <= last:
            continue
        unit = str(raw.get("unit") or "").lower()
        currency = str(raw.get("currency") or "")
        source = {"wfp": "WFP", "fews": "FEWS NET", "faostat": "FAOSTAT"}.get(raw.get("price_source"))
        try:
            value = float(raw.get("avg_price"))
        except (TypeError, ValueError, OverflowError):
            continue
        if not unit or not currency or not source or not isfinite(value) or value <= 0:
            continue
        offers.append((latest, period_start, label, currency, unit, value, source, item))

    if not offers:
        return None
    latest = max(offer[0] for offer in offers)
    current = [offer for offer in offers if offer[0] == latest]
    if len(current) != 1:
        return None
    _, period_start, label, currency, unit, value, source, item = current[0]
    period = latest.isoformat() if period_start == latest else f"{period_start}–{latest}"
    if language == "ig":
        label = "osikapa" if product.casefold() == "rice" else label
        answer = f"Na {country}, nkezi ọnụ ahịa {label} n'ahịa bụ **{currency} {value:,.2f}/{unit}** ({period}, {source})."
    elif language == "pcm":
        answer = f"For {country}, di average retail price of {label.lower()} na **{currency} {value:,.2f}/{unit}** ({period}, {source})."
    else:
        answer = f"The average retail price of {label.lower()} in {country} was **{currency} {value:,.2f}/{unit}** ({period}, {source})."
    return answer, [item]


def render_price_list(
    rows: list[dict[str, Any]],
    products: list[str],
    *,
    country: str,
    country_iso3: str,
    window: PriceWindow,
    pidgin: bool,
) -> tuple[str, list[dict[str, Any]]]:
    values: dict[tuple[int, str], str] = {}
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for item in rows:
        meta = item.get("metadata") or {}
        raw = meta.get("raw_row") or {}
        crop = raw.get("requested_product")
        if crop not in products or raw.get("country_iso3") != country_iso3:
            continue
        date = str(raw.get("latest_date") or "")
        year = str(raw.get("year") or date)[:4]
        if not year.isdigit() or not raw.get("unit") or not raw.get("currency"):
            continue
        try:
            value = float(raw.get("avg_price"))
        except (TypeError, ValueError, OverflowError):
            continue
        if not isfinite(value):
            continue
        if window.year_start is not None and window.year_end is not None and not window.year_start <= int(year) <= window.year_end:
            continue
        values[int(year), crop] = f"{value:,.2f}"
        if crop not in latest or date > latest[crop][0]:
            latest[crop] = (date, item)

    heading = f"**{country}: {', '.join(products[:-1])} and {products[-1]} prices**"
    if not values:
        if pidgin:
            return f"{heading}\n\nI no find any price for these crops wey I fit confirm for the time wey you ask.", []
        return f"{heading}\n\nI couldn't verify any prices for these crops for the period you asked about.", []

    observed_years = sorted({year for year, _ in values})
    if window.year_start is not None and window.year_end is not None:
        years = range(window.year_start, window.year_end + 1)
        description = (
            "Na the yearly average prices wey I find for the markets inside the data be this. "
            "Each crop use one price series. **—** mean say I no find price for that year."
            if pidgin else
            "These are annual averages from the markets covered by the data. Each crop uses one price series. "
            "**—** means no price was verified for that year."
        )
    else:
        years = observed_years
        description = (
            "Na the latest prices wey I find for each crop be this. Check the date under the table."
            if pidgin else "These are the latest prices I could verify for each crop. See the dates below."
        )
    lines = [heading, "", description, "", "| Year | " + " | ".join(p.title() for p in products) + " |"]
    lines.append("| " + " | ".join(["---"] + ["---:"] * len(products)) + " |")
    for year in years:
        lines.append(f"| {year} | " + " | ".join(values.get((year, crop), "—") for crop in products) + " |")

    citations: list[dict[str, Any]] = []
    lines.extend(("", "**Unit and source for each crop**"))
    for crop in products:
        if crop not in latest:
            gap = "I no find price wey I fit confirm." if pidgin else "No price verified for the requested period."
            lines.append(f"- **{crop.title()}** — {gap}")
            continue
        date, item = latest[crop]
        raw = item["metadata"]["raw_row"]
        source = {"wfp": "WFP", "fews": "FEWS NET", "faostat": "FAOSTAT"}.get(raw.get("price_source"), raw.get("price_source"))
        unit = str(raw["unit"]).replace("_", " ").lower()
        detail = f"{raw['product_name']} · {raw['price_type']} · {raw['currency']} per {unit} · {source}"
        lines.append(f"- **{crop.title()}** — {detail} (latest observation {date[:7]}).")
        citations.append(item)
    return "\n".join(lines), citations
