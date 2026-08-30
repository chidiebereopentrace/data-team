"""Deterministic mart SQL templates (fct_* / agg_* only) as NL2SQL fallback."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import (
    columns_for_mart_tables,
    compile_product_filter_sql,
    compile_measure_filters,
    compile_time_filter_sql,
    default_discriminator_value,
    discriminator_equality_filters,
    geo_column,
    load_mart_table_schema,
    match_product_samples,
    measure_blob,
    measure_columns_mart,
    product_blob,
    product_column,
    resolve_dictionary_label,
    resolve_geo_filter_values,
    resolve_measure_column,
    value_samples_for_mart_tables,
    year_column,
)
from ml.rag.chatbot.retrieval_contract import choose_agg_vs_fact
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
_RECENCY_RE = re.compile(
    r"\b(current|latest|now|right now|today)\b",
    re.IGNORECASE,
)
_PHASE3_POP_RE = re.compile(
    r"\b(phase\s*3\+|phase\s*3\s*or\s*higher|population_3\+|ipc\s*3\+)\b",
    re.IGNORECASE,
)
_SEASON_CLIMATE_RE = re.compile(
    r"\b(rainy season|rainy seasons|planting|planting window|onset|when does.*start)\b",
    re.IGNORECASE,
)
_TRADE_RE = re.compile(r"\b(trade|import|export)\b", re.IGNORECASE)

_MART_PRODUCTION_TABLES = (
    "agg_production_annual",
    "fct_production",
    "fct_yield",
)
# Point facts prefer fct_* (ACF contract + production_grain) over agg rollups.
_MART_POINT_FACT_PRODUCTION = (
    "fct_production",
    "fct_yield",
    "agg_production_annual",
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

_POINT_FACT_LINEAGE_COLS = (
    "source_key",
    "source_name",
    "tier",
    "data_level",
    "place_scope",
    "metric",
    "as_of_date",
    "as_of_date_basis",
    "unit",
    "production_unit",
    "production_grain",
    "geo_scope",
    "record_count",
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


def _has_recency_anchor(
    *,
    query: str,
    time_start: str | None = None,
    time_end: str | None = None,
    entities: list[str] | None = None,
) -> bool:
    blob = _blob(query, entities)
    if _RECENCY_RE.search(blob):
        return True
    for raw in (time_start, time_end):
        if raw and _RECENCY_RE.search(str(raw)):
            return True
    return False


def _tables_set(selected_tables: set[str] | list[str] | None) -> set[str]:
    return {str(t).strip().split(".")[-1].lower() for t in (selected_tables or []) if str(t).strip()}


def _blob(query: str, entities: list[str] | None) -> str:
    q = query or ""
    if entities:
        return f"{q} {' '.join(str(e) for e in entities)}"
    return q


def _resolve_dictionary_product(blob: str) -> str | None:
    return resolve_dictionary_label(column="product_name", blob=blob)


def _extract_crop(blob: str) -> str | None:
    """Facet-bridge helper; SQL filters must use compile_product_filter_sql instead."""
    return _resolve_dictionary_product(blob)


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


def _pick_point_fact_production_table(tables: set[str]) -> str:
    return _pick_table(tables, _MART_POINT_FACT_PRODUCTION) or "fct_production"


def _point_fact_select_cols(
    table_id: str,
    *,
    geo_col: str,
    ycol: str,
    prod_col: str | None,
    metric: str,
) -> list[str]:
    available = columns_for_mart_tables({table_id}).get(table_id) or set()
    out: list[str] = []
    for col in (geo_col, prod_col, ycol, metric):
        if col and col in available and col not in out:
            out.append(col)
    for col in _POINT_FACT_LINEAGE_COLS:
        if col in available and col not in out:
            out.append(col)
    return out or [geo_col, ycol, metric]


def _point_fact_order_clause(table_id: str) -> str:
    available = columns_for_mart_tables({table_id}).get(table_id) or set()
    if "record_count" in available:
        return "ORDER BY record_count DESC "
    if "tier" in available:
        return "ORDER BY tier ASC "
    return ""


def _mart_product_col(table_id: str) -> str:
    return product_column(table_id) or "product_name"


def _product_filter(
    table_id: str,
    blob: str,
    *,
    project_id: str,
    dataset: str,
) -> tuple[str, str | None]:
    sql, labels = compile_product_filter_sql(
        table_id,
        project_id=project_id,
        dataset=dataset,
        blob=blob,
    )
    return sql, (labels[0] if labels else None)


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
        blob = _blob(query, entities)
        if _PRICE_RE.search(blob) and _has_recency_anchor(
            query=query or "",
            time_start=time_start,
            time_end=time_end,
            entities=entities,
        ):
            return False
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
    if _resolve_dictionary_product(blob) and _RANK_RE.search(blob):
        return _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None
    if not _PRODUCTION_RE.search(blob) and not _PRICE_RE.search(blob) and not _TRADE_RE.search(blob):
        return False
    if not _RANK_RE.search(blob) and not (
        _SERIES_RE.search(blob) or re.search(r"\bacross\b", blob, re.IGNORECASE)
    ):
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


def match_mart_latest_price(
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
    if not _pick_table(tables, _MART_PRICE_TABLES):
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
    blob = _blob(query, entities)
    if not _PRICE_RE.search(blob):
        return False
    if _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None:
        return False
    return _has_recency_anchor(
        query=query or "",
        time_start=time_start,
        time_end=time_end,
        entities=entities,
    )


def build_mart_latest_price_sql(
    *,
    project_id: str,
    dataset: str,
    table_id: str,
    country_labels: list[str],
    blob: str,
    limit: int = 1,
    primary_measures: list[str] | None = None,
    query: str = "",
) -> str:
    lim = max(1, min(int(limit or 1), 5))
    mb = measure_blob(query or blob, primary_measures=primary_measures)
    geo_col = _mart_geo_col(table_id)
    metric = resolve_measure_column(table_id, "value") or "value"
    geo_vals = resolve_geo_filter_values(table_id, country_labels)
    clauses: list[str] = []
    if len(geo_vals) == 1:
        clauses.append(f"{geo_col} = {_sql_literal(geo_vals[0])}")
    elif geo_vals:
        lits = ", ".join(_sql_literal(v) for v in geo_vals)
        clauses.append(f"{geo_col} IN ({lits})")
    if re.search(r"\bretail\b", mb, re.IGNORECASE):
        clauses.append("price_type = 'retail'")
    for col, val in compile_measure_filters(
        table_id,
        measure_blob_text=mb,
        primary_measures=primary_measures,
    ):
        clauses.append(f"{col} = {_sql_literal(val)}")
    prod_clause, _ = _product_filter(
        table_id,
        blob,
        project_id=project_id,
        dataset=dataset,
    )
    where = " AND ".join(clauses) if clauses else "1=1"
    prod_col = _mart_product_col(table_id)
    ycol = _mart_year_col(table_id)
    select_cols = _point_fact_select_cols(
        table_id,
        geo_col=geo_col,
        ycol=ycol,
        prod_col=prod_col,
        metric=metric,
    )
    fqn = f"`{project_id}.{dataset}.{table_id}`"
    return (
        f"SELECT {', '.join(select_cols)} "
        f"FROM {fqn} "
        f"WHERE {where} "
        f"{prod_clause}"
        f"ORDER BY as_of_date DESC, tier ASC "
        f"LIMIT {lim}"
    )


def match_mart_regional_panel(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    geo_countries: list[str] | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if not _pick_table(tables, _MART_RANK_TABLES):
        return False
    blob = _blob(query, entities)
    if not _CONTINENT_RE.search(blob):
        return False
    if not (_PRODUCTION_RE.search(blob) or _TRADE_RE.search(blob)):
        return False
    if _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is None:
        return False
    multi = [str(c).strip() for c in (geo_countries or []) if str(c).strip()]
    if len(multi) >= 2:
        return True
    return bool(_SERIES_RE.search(blob) or re.search(r"\bacross\b", blob, re.IGNORECASE))


def match_mart_season_climate(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    geo_country: str | None = None,
    geo_countries: list[str] | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if "fct_yield" not in tables and "fct_climate" not in tables:
        return False
    blob = _blob(query, entities)
    if not _SEASON_CLIMATE_RE.search(blob):
        return False
    return bool(
        _resolve_country(
            query=query,
            entities=entities,
            geo_country=geo_country,
            geo_countries=geo_countries,
        )
    )


_EMPLOYMENT_SHARE_RE = re.compile(
    r"\b(employment\s+share|share\s+of\s+employment|employment\s+in\s+agricultur)",
    re.IGNORECASE,
)


def match_mart_employment_share(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    primary_measures: list[str] | None = None,
    template_key: str = "",
) -> bool:
    tables = _tables_set(selected_tables)
    if "fct_employment" not in tables and template_key not in (
        "employment_share_by_sex",
        "employment_share_total",
    ):
        return False
    pm = [str(m).strip().lower() for m in (primary_measures or []) if str(m).strip()]
    if "employment_share" in pm or template_key.startswith("employment_share"):
        return True
    return bool(_EMPLOYMENT_SHARE_RE.search(query or ""))


def build_mart_employment_share_sql(
    *,
    project_id: str,
    dataset: str,
    country_labels: list[str],
    by_sex: bool = False,
    time_start: str | None = None,
    time_end: str | None = None,
    latest_only: bool = False,
    limit: int = 10,
) -> str:
    table_id = "fct_employment"
    fq = f"`{project_id}.{dataset}.{table_id}`"
    geo_col = _mart_geo_col(table_id)
    resolved = resolve_geo_filter_values(table_id, country_labels)
    if not resolved:
        return ""
    geo_lit = _sql_literal(resolved[0]) if len(resolved) == 1 else ", ".join(
        _sql_literal(c) for c in resolved[:8]
    )
    geo_clause = (
        f"{geo_col} = {geo_lit} "
        if len(resolved) == 1
        else f"{geo_col} IN ({geo_lit}) "
    )
    indicator_filter = (
        "AND (LOWER(indicator) LIKE '%share%agricultur%' "
        "OR LOWER(indicator) LIKE '%employment%agricultur%') "
    )
    unit_filter = "AND unit = '%' "
    time_filter = compile_time_filter_sql(
        table_id,
        time_start=time_start,
        time_end=time_end,
    )
    sex_filter = ""
    select_cols = "year, indicator, sex, unit, value"
    if by_sex:
        sex_filter = "AND LOWER(sex) IN ('male', 'female', 'm', 'f') "
    else:
        sex_filter = "AND (sex IS NULL OR LOWER(sex) IN ('total', 'both sexes', 'both', 'all')) "
    latest_clause = ""
    if latest_only:
        latest_clause = "QUALIFY year = MAX(year) OVER (PARTITION BY sex) "
    lim = max(1, min(int(limit or 10), 20))
    return (
        f"SELECT {select_cols} "
        f"FROM {fq} "
        f"WHERE {geo_clause}"
        f"{indicator_filter}"
        f"{unit_filter}"
        f"{sex_filter}"
        f"{time_filter}"
        f"{latest_clause}"
        f"ORDER BY year DESC, sex "
        f"LIMIT {lim}"
    )


def build_mart_season_climate_sql(
    *,
    project_id: str,
    dataset: str,
    table_id: str,
    country_labels: list[str],
    blob: str,
    limit: int = 5,
    primary_measures: list[str] | None = None,
    query: str = "",
) -> str:
    lim = max(1, min(int(limit or 5), 20))
    y = _year_from_context(time_start=None, time_end=None, query=query or blob)
    if y is None:
        y = 2023
    return build_mart_point_fact_sql(
        project_id=project_id,
        dataset=dataset,
        table_id="fct_yield",
        country_labels=country_labels,
        year=y,
        blob=blob,
        limit=lim,
        primary_measures=primary_measures or ["yield"],
        query=query,
    )


def build_mart_point_fact_sql(
    *,
    project_id: str,
    dataset: str,
    table_id: str,
    country_labels: list[str],
    year: int,
    blob: str,
    limit: int = 1,
    primary_measures: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    query: str = "",
) -> str:
    lim = max(1, min(int(limit or 1), 5))
    mb = measure_blob(query or blob, primary_measures=primary_measures)
    geo_col = _mart_geo_col(table_id)
    ycol = _mart_year_col(table_id)
    metric = resolve_measure_column(table_id, "value") or (
        measure_columns_mart(table_id)[0] if measure_columns_mart(table_id) else "value"
    )
    geo_vals = resolve_geo_filter_values(table_id, country_labels)
    clauses: list[str] = []
    time_sql = compile_time_filter_sql(
        table_id,
        year=year,
        time_start=time_start,
        time_end=time_end,
    ).strip()
    if time_sql.startswith("AND "):
        clauses.append(time_sql[4:])
    else:
        clauses.append(f"{ycol} = {int(year)}")
    if len(geo_vals) == 1:
        clauses.append(f"{geo_col} = {_sql_literal(geo_vals[0])}")
    elif geo_vals:
        lits = ", ".join(_sql_literal(v) for v in geo_vals)
        clauses.append(f"{geo_col} IN ({lits})")
    for col, val in compile_measure_filters(
        table_id,
        measure_blob_text=mb,
        primary_measures=primary_measures,
    ):
        clauses.append(f"{col} = {_sql_literal(val)}")
    prod_clause, _ = _product_filter(
        table_id,
        blob,
        project_id=project_id,
        dataset=dataset,
    )
    where = " AND ".join(clauses)
    prod_col = _mart_product_col(table_id)
    select_cols = _point_fact_select_cols(
        table_id,
        geo_col=geo_col,
        ycol=ycol,
        prod_col=prod_col,
        metric=metric,
    )
    order_clause = _point_fact_order_clause(table_id)
    fqn = f"`{project_id}.{dataset}.{table_id}`"
    return (
        f"SELECT {', '.join(select_cols)} "
        f"FROM {fqn} "
        f"WHERE {where} "
        f"{prod_clause}"
        f"{order_clause}"
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
    blob_l = blob or "food security population"
    if _PHASE3_POP_RE.search(blob_l):
        clauses.append(f"measure_type = {_sql_literal('population')}")
        clauses.append(f"metric = {_sql_literal('population_3+')}")
    else:
        for col, val in discriminator_equality_filters(table_id, blob_l):
            clauses.append(f"{col} = {_sql_literal(val)}")
    if _RECENCY_RE.search(blob_l) or re.search(r"\bright now\b", blob_l, re.IGNORECASE):
        clauses.append(f"scenario_name = {_sql_literal('Current Situation')}")
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
    elif _RECENCY_RE.search(blob_l):
        clauses.append(
            f"as_of_date = (SELECT MAX(as_of_date) FROM {fqn} "
            f"WHERE measure_type = {_sql_literal('population')})"
        )
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
    primary_measures: list[str] | None = None,
    task_mode: str = "",
    template_key: str = "",
) -> dict[str, Any] | None:
    """
    Return ``{\"sql\": ..., \"template\": ...}`` when a mart template matches.

    Match order: point fact → country series → country rank → food security snapshot.
    """
    from ml.rag.chatbot.bq_sql_patterns import build_rank_by_sum_sql, build_time_series_sql

    if not project_id or not dataset:
        return None
    year = _year_from_context(time_start=time_start, time_end=time_end, query=query or "")
    blob = product_blob(query, entities)
    product_label = _resolve_dictionary_product(blob)
    country = _resolve_country(
        query=query,
        entities=entities,
        geo_country=geo_country,
        geo_countries=geo_countries,
    )
    tables = _tables_set(selected_tables)
    countries = _resolve_countries(geo_country=geo_country, geo_countries=geo_countries)
    pm = [str(m).strip().lower() for m in (primary_measures or []) if str(m).strip()]
    tkey = (template_key or "").strip().lower()

    if tkey in ("employment_share_by_sex", "employment_share_total") or match_mart_employment_share(
        query=query,
        selected_tables=selected_tables,
        primary_measures=primary_measures,
        template_key=tkey,
    ):
        countries = _resolve_countries(geo_country=geo_country, geo_countries=geo_countries)
        if not countries:
            country = _resolve_country(
                query=query,
                entities=entities,
                geo_country=geo_country,
                geo_countries=geo_countries,
            )
            countries = [country] if country else []
        if countries:
            by_sex = tkey == "employment_share_by_sex" or "sex" in (query or "").lower()
            latest_only = tkey == "employment_share_by_sex" or bool(_RECENCY_RE.search(query or ""))
            sql = build_mart_employment_share_sql(
                project_id=project_id,
                dataset=dataset,
                country_labels=countries,
                by_sex=by_sex,
                time_start=time_start,
                time_end=time_end,
                latest_only=latest_only,
                limit=1 if task_mode in ("fact_lookup", "data_export_only") else limit,
            )
            if sql:
                return {
                    "sql": sql,
                    "template": "employment_share_by_sex" if by_sex else "employment_share_total",
                    "countries": countries,
                    "table_id": "fct_employment",
                }

    if (
        "food_security_ipc" in pm
        or "food_security" in pm
    ) and match_mart_food_security_snapshot(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
    ):
        table_id = _pick_table(tables, _MART_FOOD_SECURITY_TABLES) or "fct_food_security"
        if _has_recency_anchor(
            query=query or "",
            time_start=time_start,
            time_end=time_end,
            entities=entities,
        ):
            table_id = _pick_table(tables, ("agg_food_security_monthly", "fct_food_security")) or table_id
        return {
            "sql": build_mart_food_security_sql(
                project_id=project_id,
                dataset=dataset,
                table_id=table_id,
                year=year,
                limit=1 if task_mode in ("fact_lookup", "data_export_only") else limit,
                countries=countries or None,
                blob=blob,
            ),
            "template": "mart_food_security_snapshot",
            "year": year,
            "countries": countries,
            "table_id": table_id,
        }

    if match_mart_regional_panel(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
        geo_countries=geo_countries,
    ):
        assert year is not None
        table_id = _pick_table(tables, _MART_RANK_TABLES) or "fct_production"
        geo = geo_column(table_id) or "country_iso3"
        products = match_product_samples(table_id, blob)
        geo_sql = _geo_clause_for_template(
            table_id,
            geo_country=geo_country,
            geo_countries=geo_countries,
        )
        return {
            "sql": build_rank_by_sum_sql(
                project_id=project_id,
                dataset=dataset,
                table_id=table_id,
                year=year,
                limit=max(limit, 16),
                product_name=products[0] if products else product_label,
                products=products or None,
                grain=[geo],
                blob=blob,
                time_start=time_start,
                time_end=time_end,
                geo_clause=geo_sql,
            ),
            "template": "mart_regional_panel",
            "year": year,
            "product_name": product_label,
            "table_id": table_id,
        }

    if match_mart_latest_price(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
        geo_country=geo_country,
        geo_countries=geo_countries,
    ):
        assert country is not None
        table_id = _pick_table(tables, _MART_PRICE_TABLES) or "fct_prices"
        return {
            "sql": build_mart_latest_price_sql(
                project_id=project_id,
                dataset=dataset,
                table_id=table_id,
                country_labels=[country],
                blob=blob,
                limit=1,
                primary_measures=primary_measures,
                query=query,
            ),
            "template": "mart_latest_price",
            "country": country,
            "table_id": table_id,
        }

    if match_mart_season_climate(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        geo_country=geo_country,
        geo_countries=geo_countries,
    ):
        assert country is not None
        table_id = "fct_yield" if "fct_yield" in tables else "fct_climate"
        return {
            "sql": build_mart_season_climate_sql(
                project_id=project_id,
                dataset=dataset,
                table_id=table_id,
                country_labels=[country],
                blob=blob,
                limit=5,
                primary_measures=primary_measures or ["yield"],
                query=query,
            ),
            "template": "mart_season_climate",
            "country": country,
            "table_id": table_id,
        }

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
            table_id = _pick_point_fact_production_table(tables)
        routed = choose_agg_vs_fact(
            table_id,
            query=query,
            multi_country=len(countries) > 1,
            year_hint=str(year or ""),
            single_country=bool(country),
        )
        if routed in tables:
            table_id = routed
        point_limit = 1
        return {
            "sql": build_mart_point_fact_sql(
                project_id=project_id,
                dataset=dataset,
                table_id=table_id,
                country_labels=[country],
                year=year,
                blob=blob,
                limit=point_limit,
                primary_measures=primary_measures,
                time_start=time_start,
                time_end=time_end,
                query=query,
            ),
            "template": "mart_point_fact",
            "country": country,
            "product_name": product_label,
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
                product_name=products[0] if products else product_label,
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
            "product_name": product_label,
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
                product_name=products[0] if products else product_label,
                products=products or None,
                grain=grain,
                blob=blob,
                time_start=time_start,
                time_end=time_end,
            ),
            "template": "mart_country_rank",
            "year": year,
            "product_name": product_label,
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
