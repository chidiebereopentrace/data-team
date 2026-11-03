"""Fill country, observation date, measure, and class columns. Reject rows that cannot."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from ingestion_engine.countries import Countries
from ingestion_engine.manifest import Source


def _blank(value: str) -> bool:
    return not value or not value.strip()


def _number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _observation(value: str, fmt: str) -> date | None:
    text = value.strip()
    if not text:
        return None
    try:
        if fmt == "excel_serial":
            return (date(1899, 12, 30) + timedelta(days=float(text)))
        if fmt == "%Y":
            return date(int(text), 1, 1)
        if fmt == "%Y-%m":
            year, month = text.split("-", 1)
            return date(int(year), int(month), 1)
        if fmt == "%YM%m":
            match = re.fullmatch(r"(\d{4})M(\d{2})", text)
            if match is None:
                return None
            return date(int(match.group(1)), int(match.group(2)), 1)
        return datetime.strptime(text, fmt).date()
    except (TypeError, ValueError):
        return None


def subject_iso3(source: Source, record: dict[str, str], countries: Countries) -> str | None:
    """Resolve the country the row is about. Other country columns are not consulted."""
    country_value = record.get(source.geo_country_column, "") if source.geo_country_column else ""
    if _blank(country_value):
        country_value = source.geo_country_constant
    if _blank(country_value):
        return None
    return countries.resolve(country_value, source.geo_code_standard)


def prepare_row(
    source: Source,
    record: dict[str, str],
    *,
    countries: Countries,
    today: date,
) -> tuple[dict[str, object] | None, str | None]:
    """Return a prepared landing row, or a quarantine reason."""
    if record.get("_width_error"):
        return None, str(record["_width_error"])
    observed = _observation(record.get(source.temporal_column, ""), source.temporal_format)
    if observed is None:
        return None, "unparsed observation date"
    if source.freshness_days is not None and observed < today - timedelta(days=source.freshness_days):
        return None, "stale observation"
    iso3 = subject_iso3(source, record, countries)
    if iso3 is None:
        return None, "unresolved country"
    admin1 = record.get(source.geo_admin1_column, "") if source.geo_admin1_column else ""
    admin2 = record.get(source.geo_admin2_column, "") if source.geo_admin2_column else ""
    market = record.get(source.geo_market_column, "") if source.geo_market_column else ""
    if source.geo_grain == "admin1" and _blank(admin1):
        return None, "missing admin1"
    if source.geo_grain == "admin2" and (_blank(admin1) or _blank(admin2)):
        return None, "missing admin2"
    if source.geo_grain == "market" and _blank(market):
        return None, "missing market"
    latitude = longitude = None
    if source.geo_grain == "point":
        latitude = _number(record.get(source.geo_lat_column, ""))
        longitude = _number(record.get(source.geo_lon_column, ""))
        if latitude is None or longitude is None:
            return None, "unparsed coordinates"
    value = _number(record.get(source.value_column, ""))
    if value is None:
        return None, "value is not numeric"
    for column in source.class_columns:
        if _blank(record.get(column, "")):
            return None, f"empty {column}"
    unit = source.unit_constant
    if source.unit_column:
        unit = record.get(source.unit_column, "")
    if _blank(unit):
        return None, "empty unit"
    prepared: dict[str, object] = {name: record.get(name, "") for name in source.column_names}
    prepared.update(
        country_iso3=iso3,
        country_name=countries.country_name(iso3),
        admin1_name=admin1,
        admin2_name=admin2,
        market_name=market,
        latitude=latitude,
        longitude=longitude,
        observation_date=observed.isoformat(),
        year=observed.year,
        temporal_grain=source.temporal_grain,
        indicator_class=source.indicator_class,
        value_num=value,
        unit=unit,
        measure_kind=source.measure_kind if source.indicator_class == "PROD" else "",
    )
    return prepared, None
