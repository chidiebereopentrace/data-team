"""Deterministic mart SQL templates (fct_* / agg_* only) as NL2SQL fallback."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import (
    discriminator_equality_filters,
    geo_column,
    match_product_samples,
    measure_columns_mart,
    product_column,
    resolve_geo_filter_values,
    resolve_measure_column,
    year_column,
)
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

_MART_PRODUCTION_TABLES = (
    "agg_production_annual",
    "fct_production",
    "fct_yield",
)
_MART_PRICE_TABLES = ("fct_prices", "agg_prices_country_month")
_MART_FOOD_SECURITY_TABLES = ("fct_food_security", "agg_food_security_monthly")
_MART_RANK_TABLES = ("agg_production_annual", "fct_production", "fct_trade")
_MART_FACT_TABLES = frozenset(
    {
        *_MART_PRODUCTION_TABLES,
        *_MART_PRICE_TABLES,
        "fct_yield",
    }
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


def _resolve_countries(
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


def _pick_table(tables: set[str], candidates: tuple[str, ...]) -> str | None:
    for tid in candidates:
        if tid in tables:
            return tid
    return None


def _mart_geo_col(table_id: str) -> str:
    return geo_column(table_id) or "country_name"


def _mart_year_col(table_id: str) -> str:
    return year_column(table_id) or "year"


def _mart_product_col(table_id: str) -> str:
    return product_column(table_id) or "product_name"


def _product_filter(table_id: str, blob: str, product_name: str | None) -> tuple[str, str | None]:
    hits = match_product_samples(table_id, blob)
    pname = hits[0] if hits else product_name
    if not pname:
        return "", None
    col = _mart_product_col(table_id)
    return f"AND {col} = {_sql_literal(pname)} ", pname


def match_mart_point_fact(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    geo_country: str | None = None,
    geo_countries: list[str] | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if not tables.intersection(_MART_FACT_TABLES):
        return False
    multi = [str(c).strip() for c in (geo_countries or []) if str(c).strip()]
    if len(multi) > 1:
        return False
    if not _resolve_country(
        query=query,
        entities=entities,
        geo_country=geo_country,
        geo_countries=geo_countries,
    ):
        return False
    if _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is None:
        return False
    blob = _blob(query, entities)
    if _RANK_RE.search(blob) and _CONTINENT_RE.search(blob) and not _SERIES_RE.search(blob):
        return False
    if _SERIES_RE.search(blob):
        return False
    return bool(
        _PRODUCTION_RE.search(blob)
        or _YIELD_RE.search(blob)
        or _PRICE_RE.search(blob)
    )


def match_mart_country_series(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    geo_country: str | None = None,
    geo_countries: list[str] | None = None,
) -> bool:
    multi = [str(c).strip() for c in (geo_countries or []) if str(c).strip()]
    if len(multi) > 1:
        return False
    tables = _tables_set(selected_tables)
    table_id = _pick_table(tables, _MART_PRODUCTION_TABLES + _MART_PRICE_TABLES)
    if table_id is None:
        return False
    blob = _blob(query, entities)
    if not _resolve_country(
        query=query,
        entities=entities,
        geo_country=geo_country,
        geo_countries=geo_countries,
    ):
        return False
    if table_id in _MART_PRICE_TABLES:
        if not (_PRICE_RE.search(blob) or _SERIES_RE.search(blob)):
            return False
    elif not (_PRODUCTION_RE.search(blob) or _YIELD_RE.search(blob) or _SERIES_RE.search(blob)):
        return False
    if _CONTINENT_RE.search(blob) and _RANK_RE.search(blob) and not _SERIES_RE.search(blob):
        return False
    return True


def match_mart_country_rank(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if not _pick_table(tables, _MART_RANK_TABLES):
        return False
    blob = _blob(query, entities)
    if _extract_crop(blob) and _RANK_RE.search(blob):
        return _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None
    if not _PRODUCTION_RE.search(blob) and not _PRICE_RE.search(blob):
        return False
    if not _RANK_RE.search(blob):
        return False
    if not _CONTINENT_RE.search(blob):
        return False
    return _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None


def match_mart_food_security_snapshot(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if not _pick_table(tables, _MART_FOOD_SECURITY_TABLES):
        return False
    blob = _blob(query, entities)
    return bool(_FOOD_SEC_RE.search(blob))


def build_mart_point_fact_sql(
    *,
    project_id: str,
    dataset: str,
    table_id: str,
    country_labels: list[str],
    product_name: str | None,
    year: int,
    blob: str,
    limit: int = 5,
) -> str:
    lim = max(1, min(int(limit or 5), 20))
    geo_col = _mart_geo_col(table_id)
    ycol = _mart_year_col(table_id)
    metric = resolve_measure_column(table_id, "value") or (
        measure_columns_mart(table_id)[0] if measure_columns_mart(table_id) else "value"
    )
    geo_vals = resolve_geo_filter_values(table_id, country_labels)
    clauses = [f"{ycol} = {int(year)}"]
    if len(geo_vals) == 1:
        clauses.append(f"{geo_col} = {_sql_literal(geo_vals[0])}")
    elif geo_vals:
        lits = ", ".join(_sql_literal(v) for v in geo_vals)
        clauses.append(f"{geo_col} IN ({lits})")
    for col, val in discriminator_equality_filters(table_id, blob):
        clauses.append(f"{col} = {_sql_literal(val)}")
    prod_clause, _ = _product_filter(table_id, blob, product_name)
    where = " AND ".join(clauses)
    prod_col = _mart_product_col(table_id)
    select_cols = [geo_col, ycol, metric]
    if prod_col:
        select_cols.insert(1, prod_col)
    fqn = f"`{project_id}.{dataset}.{table_id}`"
    return (
        f"SELECT {', '.join(select_cols)} "
        f"FROM {fqn} "
        f"WHERE {where} "
        f"{prod_clause}"
        f"LIMIT {lim}"
    )


def build_mart_food_security_sql(
    *,
    project_id: str,
    dataset: str,
    table_id: str,
    year: int | None = None,
    limit: int = 20,
    countries: list[str] | None = None,
    blob: str = "",
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    fqn = f"`{project_id}.{dataset}.{table_id}`"
    geo_col = _mart_geo_col(table_id)
    ycol = _mart_year_col(table_id)
    metric = resolve_measure_column(table_id, "value") or "value"
    clauses: list[str] = []
    for col, val in discriminator_equality_filters(table_id, blob or "food security population"):
        clauses.append(f"{col} = {_sql_literal(val)}")
    real_countries = _resolve_countries(geo_country=None, geo_countries=countries)
    if real_countries:
        geo_vals = resolve_geo_filter_values(table_id, real_countries)
        if len(geo_vals) == 1:
            clauses.append(f"{geo_col} = {_sql_literal(geo_vals[0])}")
        elif geo_vals:
            lits = ", ".join(_sql_literal(c) for c in geo_vals)
            clauses.append(f"{geo_col} IN ({lits})")
    if year is not None:
        clauses.append(f"{ycol} = {int(year)}")
    else:
        clauses.append(
            f"{ycol} = (SELECT MAX({ycol}) FROM {fqn} "
            f"WHERE measure_type = {_sql_literal('population')})"
        )
    where = " AND ".join(clauses) if clauses else "1=1"
    return (
        f"SELECT {geo_col}, {ycol}, {metric} "
        f"FROM {fqn} "
        f"WHERE {where} "
        f"ORDER BY {metric} DESC "
        f"LIMIT {lim}"
    )


def _geo_clause_for_template(
    table_id: str,
    *,
    geo_country: str | None,
    geo_countries: list[str] | None,
) -> str:
    countries = _resolve_countries(geo_country=geo_country, geo_countries=geo_countries)
    if not countries:
        return ""
    resolved = resolve_geo_filter_values(table_id, countries)
    if not resolved:
        return ""
    col = _mart_geo_col(table_id)
    if len(resolved) == 1:
        return f"AND {col} = {_sql_literal(resolved[0])} "
    lits = ", ".join(_sql_literal(c) for c in resolved[:32])
    return f"AND {col} IN ({lits}) "


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
    Return ``{\"sql\": ..., \"template\": ...}`` when a mart template matches.

    Match order: point fact → country series → country rank → food security snapshot.
    """
    from ml.rag.chatbot.bq_sql_patterns import build_rank_by_sum_sql, build_time_series_sql

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
    tables = _tables_set(selected_tables)
    countries = _resolve_countries(geo_country=geo_country, geo_countries=geo_countries)

    if match_mart_point_fact(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
        geo_country=geo_country,
        geo_countries=geo_countries,
    ):
        assert year is not None and country is not None
        if _YIELD_RE.search(blob) and "fct_yield" in tables:
            table_id = "fct_yield"
        elif _PRICE_RE.search(blob):
            table_id = _pick_table(tables, _MART_PRICE_TABLES) or "fct_prices"
        else:
            table_id = _pick_table(tables, _MART_PRODUCTION_TABLES) or "fct_production"
        return {
            "sql": build_mart_point_fact_sql(
                project_id=project_id,
                dataset=dataset,
                table_id=table_id,
                country_labels=[country],
                product_name=crop,
                year=year,
                blob=blob,
                limit=limit,
            ),
            "template": "mart_point_fact",
            "country": country,
            "product_name": crop,
            "table_id": table_id,
            "year": year,
        }

    if match_mart_country_series(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        geo_country=geo_country,
        geo_countries=geo_countries,
    ):
        assert country is not None
        if _PRICE_RE.search(blob):
            table_id = _pick_table(tables, _MART_PRICE_TABLES) or "fct_prices"
        elif _YIELD_RE.search(blob) and "fct_yield" in tables:
            table_id = "fct_yield"
        else:
            table_id = _pick_table(tables, _MART_PRODUCTION_TABLES) or "fct_production"
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
        geo_sql = _geo_clause_for_template(
            table_id,
            geo_country=geo_country,
            geo_countries=geo_countries,
        )
        products = match_product_samples(table_id, blob)
        metric = resolve_measure_column(table_id, "value") or "value"
        return {
            "sql": build_time_series_sql(
                project_id=project_id,
                dataset=dataset,
                table_id=table_id,
                metric=metric,
                product_name=products[0] if products else crop,
                products=products or None,
                year=year,
                limit=max(limit, 100),
                blob=blob,
                geo_clause=geo_sql,
                time_start=time_start,
                time_end=time_end,
            ),
            "template": "mart_country_series",
            "country": country,
            "product_name": crop,
            "table_id": table_id,
        }

    if match_mart_country_rank(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
    ):
        assert year is not None
        table_id = _pick_table(tables, _MART_RANK_TABLES) or "fct_production"
        geo = geo_column(table_id) or "country_iso3"
        grain = [geo]
        products = match_product_samples(table_id, blob)
        return {
            "sql": build_rank_by_sum_sql(
                project_id=project_id,
                dataset=dataset,
                table_id=table_id,
                year=year,
                limit=limit,
                product_name=products[0] if products else crop,
                products=products or None,
                grain=grain,
                blob=blob,
                time_start=time_start,
                time_end=time_end,
            ),
            "template": "mart_country_rank",
            "year": year,
            "product_name": crop,
            "table_id": table_id,
        }

    if match_mart_food_security_snapshot(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
    ):
        table_id = _pick_table(tables, _MART_FOOD_SECURITY_TABLES) or "fct_food_security"
        return {
            "sql": build_mart_food_security_sql(
                project_id=project_id,
                dataset=dataset,
                table_id=table_id,
                year=year,
                limit=limit,
                countries=countries or None,
                blob=blob,
            ),
            "template": "mart_food_security_snapshot",
            "year": year,
            "countries": countries,
            "table_id": table_id,
        }

    return None


__all__ = [
    "try_sql_template",
    "match_mart_point_fact",
    "match_mart_country_series",
    "match_mart_country_rank",
    "match_mart_food_security_snapshot",
    "build_mart_point_fact_sql",
    "build_mart_food_security_sql",
    "_CROP_ALIASES",
    "_sql_literal",
    "_year_from_context",
]
