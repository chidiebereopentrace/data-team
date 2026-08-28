"""Deterministic SQL templates as NL2SQL fallback (metrics-layer lite)."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import columns_for_tables, match_product_samples
from ml.rag.chatbot.geo_regions import is_zone_label
from ml.rag.chatbot.query_decomposer import _extract_countries

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_RANK_RE = re.compile(
    r"\b(highest|lowest|top|rank|ranking|most|least|which country)\b",
    re.IGNORECASE,
)
_PRODUCTION_RE = re.compile(
    r"\b(production|agricultural\s+activity|agricultural\s+production|"
    r"ag(?:ricultural)?\s+output|crop\s+production|farming\s+activity|"
    r"best\s+agricultural|export|csv|chart|trend|time\s*series)\b",
    re.IGNORECASE,
)
_YIELD_RE = re.compile(r"\b(yield|productivity|hg/?ha|t/?ha)\b", re.IGNORECASE)
_PRICE_RE = re.compile(
    r"\b(price|prices|producer price|cost of|how much)\b",
    re.IGNORECASE,
)
_FOOD_SEC_RE = re.compile(
    r"\b(food security|ipc|phase\s*[3-5]|insecure|crisis|emergency|hunger)\b",
    re.IGNORECASE,
)
_CONTINENT_RE = re.compile(
    r"\b(africa|african|sub[- ]?saharan|west africa|east africa|southern africa|"
    r"north africa|central africa|sahel|horn of africa|maghreb|ecowas|sadc|eac|igad)\b",
    re.IGNORECASE,
)
_SERIES_RE = re.compile(
    r"\b(export|csv|chart|graph|plot|trend|time\s*series|over\s+time|by\s+year|"
    r"from\s+19|from\s+20|till\s+date|until\s+now)\b",
    re.IGNORECASE,
)

# Speech synonyms for crop detection. Warehouse labels come from YAML samples.
_CROP_ALIASES: tuple[tuple[str, str], ...] = (
    ("maize", "Maize"),
    ("corn", "Maize"),
    ("rice", "Rice"),
    ("wheat", "Wheat"),
    ("millet", "Millet"),
    ("sorghum", "Sorghum"),
    ("cassava", "Cassava, fresh"),
    ("yam", "Yams"),
    ("soy", "Soybeans"),
    ("groundnut", "Groundnuts"),
    ("peanut", "Groundnuts"),
)


def _year_from_context(
    *,
    time_start: str | None,
    time_end: str | None,
    query: str,
) -> int | None:
    for raw in (time_start, time_end):
        if not raw:
            continue
        m = _YEAR_RE.search(str(raw))
        if m:
            return int(m.group(0))
    m = _YEAR_RE.search(query or "")
    if m:
        return int(m.group(0))
    return None


def _tables_set(selected_tables: set[str] | list[str] | None) -> set[str]:
    return {str(t).strip().split(".")[-1].lower() for t in (selected_tables or []) if str(t).strip()}


def _blob(query: str, entities: list[str] | None) -> str:
    q = query or ""
    if entities:
        return f"{q} {' '.join(str(e) for e in entities)}"
    return q


def _extract_crop(blob: str) -> str | None:
    low = (blob or "").lower()
    for alias, product in _CROP_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", low):
            return product
    return None


def _sql_literal(value: str) -> str:
    return "'" + (value or "").replace("'", "''") + "'"


def _resolve_country(
    *,
    query: str,
    entities: list[str] | None,
    geo_country: str | None,
    geo_countries: list[str] | None,
) -> str | None:
    if geo_country and str(geo_country).strip():
        return str(geo_country).strip()
    for c in geo_countries or []:
        if str(c).strip():
            return str(c).strip()
    blob = _blob(query, entities)
    found = _extract_countries(blob)
    return found[0] if found else None


def _geo_col(table_id: str) -> str:
    if table_id.startswith("stg_fews") or table_id in {
        "stg_wfp_vampire_prices",
        "stg_yield_raw_data",
    }:
        return "country"
    return "country_name"


def _year_col(table_id: str) -> str:
    if table_id == "stg_yield_raw_data":
        return "harvest_year"
    return "year"


def _product_col(table_id: str) -> str:
    if table_id == "stg_yield_raw_data":
        return "product"
    return "product_name"


def _product_name_for_table(table_id: str, product_name: str | None) -> str | None:
    if not product_name:
        return None
    hits = match_product_samples(table_id, product_name)
    return hits[0] if hits else product_name


def _default_element(table_id: str, *, want_yield: bool) -> str | None:
    if table_id == "stg_faostat_prices":
        return "Producer Price (USD/tonne)"
    if table_id.startswith("stg_faostat"):
        return "Yield" if want_yield else "Production"
    return None


def _default_price_type(table_id: str) -> str | None:
    if table_id == "stg_fews_market_prices":
        return "Retail"
    return None


def match_faostat_crop_rank(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if "stg_faostat_production" not in tables:
        return False
    blob = _blob(query, entities)
    if _extract_crop(blob) is None:
        return False
    if not (_PRODUCTION_RE.search(blob) or _RANK_RE.search(blob)):
        return False
    if not _CONTINENT_RE.search(blob) and not _RANK_RE.search(blob):
        return False
    return _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None


def match_faostat_country_rank(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> bool:
    """True when the Africa production country-ranking template should apply."""
    tables = _tables_set(selected_tables)
    if "stg_faostat_production" not in tables:
        return False
    blob = _blob(query, entities)
    # Prefer crop-specific template when a crop is named.
    if _extract_crop(blob) is not None:
        return False
    if not _PRODUCTION_RE.search(blob):
        return False
    if not _RANK_RE.search(blob):
        return False
    if not _CONTINENT_RE.search(blob):
        return False
    return _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None


def match_faostat_price_rank(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if "stg_faostat_prices" not in tables:
        return False
    blob = _blob(query, entities)
    if not _PRICE_RE.search(blob):
        return False
    if not (_RANK_RE.search(blob) or _CONTINENT_RE.search(blob)):
        return False
    return _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None


def match_fews_food_security(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if "stg_fews_food_security" not in tables:
        return False
    blob = _blob(query, entities)
    if not _FOOD_SEC_RE.search(blob):
        return False
    # Year optional: latest-snapshot SQL when omitted (live IPC / assessments).
    return True


def _fews_countries(
    *,
    geo_country: str | None,
    geo_countries: list[str] | None,
) -> list[str]:
    raw: list[str] = []
    if geo_countries:
        raw.extend(str(c).strip() for c in geo_countries if str(c).strip())
    if geo_country and str(geo_country).strip():
        raw.append(str(geo_country).strip())
    out: list[str] = []
    seen: set[str] = set()
    for c in raw:
        if is_zone_label(c):
            continue
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            out.append(c)
    return out


def build_fews_food_security_sql(
    *,
    project_id: str,
    dataset: str,
    year: int | None = None,
    limit: int = 20,
    countries: list[str] | None = None,
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    fqn = f"`{project_id}.{dataset}.stg_fews_food_security`"
    country_clause = ""
    real_countries = _fews_countries(geo_country=None, geo_countries=countries)
    if real_countries:
        lits = ", ".join(_sql_literal(c) for c in real_countries)
        country_clause = f"AND country IN ({lits}) "
    if year is not None:
        year_clause = f"AND year = {int(year)} "
    else:
        year_clause = (
            f"AND year = (SELECT MAX(year) FROM {fqn} "
            f"WHERE measure_type = {_sql_literal('population')}) "
        )
    return (
        f"SELECT country, admin_1, AVG(SAFE_CAST(phase_code AS FLOAT64)) AS avg_phase "
        f"FROM {fqn} "
        f"WHERE measure_type = {_sql_literal('population')} "
        f"AND scenario_name = {_sql_literal('Current Situation')} "
        f"AND phase_code IS NOT NULL "
        f"{year_clause}"
        f"{country_clause}"
        f"GROUP BY country, admin_1 "
        f"ORDER BY avg_phase DESC "
        f"LIMIT {lim}"
    )


def match_country_crop_series(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    geo_country: str | None = None,
    geo_countries: list[str] | None = None,
) -> bool:
    """Country (+ optional crop) production/price/yield time series — chart/CSV/export style."""
    # Multi-country / regional expands must not collapse to a single-country series.
    multi = [str(c).strip() for c in (geo_countries or []) if str(c).strip()]
    if len(multi) > 1:
        return False
    tables = _tables_set(selected_tables)
    fact = None
    for tid in (
        "stg_faostat_production",
        "stg_faostat_prices",
        "stg_fews_market_prices",
    ):
        if tid in tables:
            fact = tid
            break
    if fact is None:
        return False
    blob = _blob(query, entities)
    country = _resolve_country(
        query=query,
        entities=entities,
        geo_country=geo_country,
        geo_countries=geo_countries,
    )
    if not country:
        return False
    if fact == "stg_faostat_prices" or fact == "stg_fews_market_prices":
        if not (_PRICE_RE.search(blob) or _SERIES_RE.search(blob)):
            return False
    elif not (
        _PRODUCTION_RE.search(blob)
        or _YIELD_RE.search(blob)
        or _SERIES_RE.search(blob)
    ):
        return False
    # Prefer series/export over continental ranking when a country is named.
    if _CONTINENT_RE.search(blob) and _RANK_RE.search(blob) and not _SERIES_RE.search(blob):
        return False
    return True


def build_faostat_country_rank_sql(
    *,
    project_id: str,
    dataset: str,
    year: int,
    limit: int = 20,
    element: str = "Production",
    product_name: str | None = None,
) -> str:
    """Single-table FAOSTAT country production ranking SQL."""
    lim = max(1, min(int(limit or 20), 100))
    fqn = f"`{project_id}.{dataset}.stg_faostat_production`"
    product_clause = ""
    if product_name:
        product_clause = f"AND product_name = {_sql_literal(product_name)} "
    return (
        f"SELECT country_name, SUM(value) AS total "
        f"FROM {fqn} "
        f"WHERE year = {int(year)} "
        f"AND element = {_sql_literal(element)} "
        f"{product_clause}"
        f"GROUP BY country_name "
        f"ORDER BY total DESC "
        f"LIMIT {lim}"
    )


def build_faostat_price_rank_sql(
    *,
    project_id: str,
    dataset: str,
    year: int,
    limit: int = 20,
    product_name: str | None = None,
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    fqn = f"`{project_id}.{dataset}.stg_faostat_prices`"
    product_clause = ""
    if product_name:
        product_clause = f"AND product_name = {_sql_literal(product_name)} "
    return (
        f"SELECT country_name, AVG(value) AS avg_price "
        f"FROM {fqn} "
        f"WHERE year = {int(year)} "
        f"AND element = {_sql_literal('Producer Price (USD/tonne)')} "
        f"{product_clause}"
        f"GROUP BY country_name "
        f"ORDER BY avg_price DESC "
        f"LIMIT {lim}"
    )


def build_country_crop_series_sql(
    *,
    project_id: str,
    dataset: str,
    table_id: str,
    country: str,
    product_name: str | None = None,
    element: str | None = None,
    price_type: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    limit: int = 100,
) -> str:
    """Country (+ crop) time series with exact metric-discriminator filters."""
    lim = max(1, min(int(limit or 100), 200))
    tid = table_id.strip().split(".")[-1].lower()
    fqn = f"`{project_id}.{dataset}.{tid}`"
    geo = _geo_col(tid)
    ycol = _year_col(tid)
    want_yield = bool(element and element.lower() == "yield")
    elem = element or _default_element(tid, want_yield=want_yield)
    ptype = price_type or _default_price_type(tid)

    clauses = [f"{geo} = {_sql_literal(country)}"]
    if elem:
        clauses.append(f"element = {_sql_literal(elem)}")
    if ptype:
        clauses.append(f"price_type = {_sql_literal(ptype)}")
    if product_name:
        clauses.append(
            f"{_product_col(tid)} = {_sql_literal(_product_name_for_table(tid, product_name) or product_name)}"
        )
    if year_start is not None:
        clauses.append(f"{ycol} >= {int(year_start)}")
    if year_end is not None:
        clauses.append(f"{ycol} <= {int(year_end)}")

    select_cols = [f"{ycol} AS year", "value"]
    cols = columns_for_tables({tid}).get(tid) or set()
    if "unit" in cols:
        select_cols.append("unit")
    select_cols.extend([_product_col(tid), geo])
    if elem and "element" in cols:
        select_cols.append("element")
    if ptype and "price_type" in cols:
        select_cols.append("price_type")

    where = " AND ".join(clauses)
    return (
        f"SELECT {', '.join(select_cols)} "
        f"FROM {fqn} "
        f"WHERE {where} "
        f"ORDER BY {ycol} "
        f"LIMIT {lim}"
    )


def try_sql_template(
    *,
    query: str,
    project_id: str,
    dataset: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    limit: int = 20,
    geo_country: str | None = None,
    geo_countries: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    Return ``{\"sql\": ..., \"template\": ...}`` when a template matches.

    NL2SQL remains primary; call this only after NL2SQL yields nothing or all prepares fail.
    Match order: country crop series → crop rank → country rank → prices → FEWS food security.
    """
    if not project_id or not dataset:
        return None
    year = _year_from_context(time_start=time_start, time_end=time_end, query=query or "")
    blob = _blob(query, entities)
    crop = _extract_crop(blob)
    country = _resolve_country(
        query=query,
        entities=entities,
        geo_country=geo_country,
        geo_countries=geo_countries,
    )

    if match_country_crop_series(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        geo_country=geo_country,
        geo_countries=geo_countries,
    ):
        tables = _tables_set(selected_tables)
        table_id = "stg_faostat_production"
        for tid in (
            "stg_faostat_production",
            "stg_faostat_prices",
            "stg_fews_market_prices",
        ):
            if tid in tables:
                table_id = tid
                break
        assert country is not None
        want_yield = bool(_YIELD_RE.search(blob)) and table_id == "stg_faostat_production"
        y_start = None
        y_end = None
        if time_start:
            m = _YEAR_RE.search(str(time_start))
            if m:
                y_start = int(m.group(0))
        if time_end:
            m = _YEAR_RE.search(str(time_end))
            if m:
                y_end = int(m.group(0))
        if y_start is None and y_end is None and year is not None:
            y_end = year
            y_start = year - 30
        return {
            "sql": build_country_crop_series_sql(
                project_id=project_id,
                dataset=dataset,
                table_id=table_id,
                country=country,
                product_name=crop,
                element=("Yield" if want_yield else None),
                year_start=y_start,
                year_end=y_end,
                limit=max(limit, 100),
            ),
            "template": "country_crop_series",
            "country": country,
            "product_name": crop,
            "table_id": table_id,
        }

    if match_faostat_crop_rank(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
    ):
        assert year is not None and crop is not None
        return {
            "sql": build_faostat_country_rank_sql(
                project_id=project_id,
                dataset=dataset,
                year=year,
                limit=limit,
                product_name=crop,
            ),
            "template": "faostat_crop_rank",
            "year": year,
            "product_name": crop,
        }

    if match_faostat_country_rank(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
    ):
        assert year is not None
        return {
            "sql": build_faostat_country_rank_sql(
                project_id=project_id,
                dataset=dataset,
                year=year,
                limit=limit,
            ),
            "template": "faostat_country_rank",
            "year": year,
        }

    if match_faostat_price_rank(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
    ):
        assert year is not None
        return {
            "sql": build_faostat_price_rank_sql(
                project_id=project_id,
                dataset=dataset,
                year=year,
                limit=limit,
                product_name=crop,
            ),
            "template": "faostat_price_rank",
            "year": year,
        }

    if match_fews_food_security(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
    ):
        fews_countries = _fews_countries(
            geo_country=geo_country,
            geo_countries=geo_countries,
        )
        return {
            "sql": build_fews_food_security_sql(
                project_id=project_id,
                dataset=dataset,
                year=year,
                limit=limit,
                countries=fews_countries or None,
            ),
            "template": "fews_food_security",
            "year": year,
            "countries": fews_countries,
        }

    return None
