"""Deterministic SQL pattern builders for structured reasoner intents."""
from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.bq_sql_templates import _year_from_context
from ml.rag.chatbot.bq_table_schema_yaml import (
    discriminator_equality_filters,
    geo_column,
    match_product_samples,
    measure_sql_aggregation,
    product_column,
    resolve_measure_column,
    table_supports_sql_pattern,
    year_column,
)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_PATTERNS = frozenset(
    {"rank_by_sum", "yoy_delta", "share_of_total", "time_series", "custom"}
)
_GEO_IN_CAP = 32


def normalize_pattern_name(raw: Any) -> str:
    name = str(raw or "custom").strip().lower()
    return name if name in _ALLOWED_PATTERNS else "custom"


def _safe_ident(name: str, *, default: str) -> str:
    cand = str(name or "").strip()
    if _IDENT_RE.match(cand):
        return cand
    return default


def _safe_idents(names: list[str] | None, *, default: list[str]) -> list[str]:
    out: list[str] = []
    for raw in names or []:
        cand = str(raw).strip()
        if _IDENT_RE.match(cand) and cand not in out:
            out.append(cand)
    return out or list(default)


def _sql_literal(value: str) -> str:
    return "'" + (value or "").replace("'", "''") + "'"


def _table_from_intent(
    intent: dict[str, Any],
    selected_tables: set[str] | list[str] | None,
) -> str | None:
    for raw in intent.get("tables") or []:
        tid = str(raw).strip().split(".")[-1].lower()
        if table_supports_sql_pattern(tid):
            return tid
    for raw in selected_tables or []:
        tid = str(raw).strip().split(".")[-1].lower()
        if table_supports_sql_pattern(tid):
            return tid
    return None


def _default_grain(table_id: str) -> list[str]:
    geo = geo_column(table_id)
    return [geo] if geo else ["country_name"]


def _normalize_grain(table_id: str, grain: list[str] | None) -> list[str] | None:
    """Map conceptual country labels onto the YAML geo column."""
    if not grain:
        return grain
    geo = geo_column(table_id) or "country_name"
    out: list[str] = []
    for raw in grain:
        g = str(raw).strip()
        if not g:
            continue
        if g.lower() in {"country", "country_name"}:
            g = geo
        if g not in out:
            out.append(g)
    return out or None


def _fqn(project_id: str, dataset: str, table_id: str) -> str:
    return f"`{project_id}.{dataset}.{table_id}`"


def _agg_expr(aggregation: str, metric_col: str) -> str:
    return f"AVG({metric_col})" if aggregation == "avg" else f"SUM({metric_col})"


def _product_clause(table_id: str, products: list[str] | str | None) -> str:
    if not products:
        return ""
    names = [products] if isinstance(products, str) else [p for p in products if str(p).strip()]
    if not names:
        return ""
    col = product_column(table_id) or (
        "product" if table_id == "stg_yield_raw_data" else "product_name"
    )
    if len(names) == 1:
        return f"AND {col} = {_sql_literal(names[0])} "
    literals = ", ".join(_sql_literal(n) for n in names[:16])
    return f"AND {col} IN ({literals}) "


def _discriminator_clause(table_id: str, blob: str) -> str:
    parts: list[str] = []
    for col, value in discriminator_equality_filters(table_id, blob):
        if not _IDENT_RE.match(col):
            continue
        parts.append(f"AND {col} = {_sql_literal(value)} ")
    return "".join(parts)


def _geo_clause(
    table_id: str,
    *,
    geo_country: str | None,
    geo_countries: list[str] | None,
) -> str:
    col = geo_column(table_id)
    if not col:
        return ""
    countries: list[str] = []
    if geo_countries:
        countries = [str(c).strip() for c in geo_countries if str(c).strip()]
    elif geo_country and str(geo_country).strip():
        countries = [str(geo_country).strip()]
    if not countries:
        return ""
    if len(countries) == 1:
        return f"AND {col} = {_sql_literal(countries[0])} "
    literals = ", ".join(_sql_literal(c) for c in countries[:_GEO_IN_CAP])
    return f"AND {col} IN ({literals}) "


def _year_col(table_id: str) -> str:
    return year_column(table_id) or "year"


def build_rank_by_sum_sql(
    *,
    project_id: str,
    dataset: str,
    table_id: str,
    year: int,
    metric: str = "value",
    grain: list[str] | None = None,
    order_by: str = "total DESC",
    product_name: str | None = None,
    products: list[str] | None = None,
    limit: int = 20,
    element: str | None = None,
    blob: str = "",
    geo_clause: str = "",
    aggregation: str = "sum",
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    metric_col = _safe_ident(metric, default="value")
    group_cols = _safe_idents(grain, default=_default_grain(table_id))
    order = str(order_by or "total DESC").strip() or "total DESC"
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s+(ASC|DESC)$", order, re.IGNORECASE):
        order = "total DESC"
    ycol = _year_col(table_id)
    select_cols = ", ".join(group_cols)
    disc_blob = blob or (element or "")
    product_list = products or ([product_name] if product_name else None)
    return (
        f"SELECT {select_cols}, {_agg_expr(aggregation, metric_col)} AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE {ycol} = {int(year)} "
        f"{geo_clause}"
        f"{_discriminator_clause(table_id, disc_blob)}"
        f"{_product_clause(table_id, product_list)}"
        f"GROUP BY {select_cols} "
        f"ORDER BY {order} "
        f"LIMIT {lim}"
    )


def build_time_series_sql(
    *,
    project_id: str,
    dataset: str,
    table_id: str,
    metric: str = "value",
    product_name: str | None = None,
    products: list[str] | None = None,
    year: int | None = None,
    limit: int = 50,
    element: str | None = None,
    blob: str = "",
    geo_clause: str = "",
    aggregation: str = "sum",
) -> str:
    lim = max(1, min(int(limit or 50), 100))
    metric_col = _safe_ident(metric, default="value")
    ycol = _year_col(table_id)
    year_filter = f"AND {ycol} >= {int(year) - 10} AND {ycol} <= {int(year)} " if year else ""
    disc_blob = blob or (element or "")
    product_list = products or ([product_name] if product_name else None)
    return (
        f"SELECT {ycol} AS year, {_agg_expr(aggregation, metric_col)} AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE 1=1 "
        f"{year_filter}"
        f"{geo_clause}"
        f"{_discriminator_clause(table_id, disc_blob)}"
        f"{_product_clause(table_id, product_list)}"
        f"GROUP BY {ycol} "
        f"ORDER BY {ycol} "
        f"LIMIT {lim}"
    )


def build_yoy_delta_sql(
    *,
    project_id: str,
    dataset: str,
    table_id: str,
    year: int,
    metric: str = "value",
    grain: list[str] | None = None,
    product_name: str | None = None,
    products: list[str] | None = None,
    limit: int = 20,
    element: str | None = None,
    blob: str = "",
    geo_clause: str = "",
    aggregation: str = "sum",
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    metric_col = _safe_ident(metric, default="value")
    group_cols = _safe_idents(grain, default=_default_grain(table_id))
    ycol = _year_col(table_id)
    select_cols = ", ".join(group_cols)
    join_on = " AND ".join(f"curr.{c} = prev.{c}" for c in group_cols)
    disc_blob = blob or (element or "")
    product_list = products or ([product_name] if product_name else None)
    return (
        f"WITH yearly AS ("
        f"SELECT {select_cols}, {ycol} AS year, {_agg_expr(aggregation, metric_col)} AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE {ycol} IN ({int(year)}, {int(year) - 1}) "
        f"{geo_clause}"
        f"{_discriminator_clause(table_id, disc_blob)}"
        f"{_product_clause(table_id, product_list)}"
        f"GROUP BY {select_cols}, {ycol}"
        f") "
        f"SELECT {', '.join(f'curr.{c}' for c in group_cols)}, "
        f"curr.total AS total_curr, prev.total AS total_prev, "
        f"(curr.total - prev.total) AS yoy_delta "
        f"FROM yearly curr "
        f"JOIN yearly prev ON {join_on} AND prev.year = curr.year - 1 "
        f"WHERE curr.year = {int(year)} "
        f"ORDER BY yoy_delta DESC "
        f"LIMIT {lim}"
    )


def build_share_of_total_sql(
    *,
    project_id: str,
    dataset: str,
    table_id: str,
    year: int,
    metric: str = "value",
    grain: list[str] | None = None,
    product_name: str | None = None,
    products: list[str] | None = None,
    limit: int = 20,
    element: str | None = None,
    blob: str = "",
    geo_clause: str = "",
    aggregation: str = "sum",
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    metric_col = _safe_ident(metric, default="value")
    group_cols = _safe_idents(grain, default=_default_grain(table_id))
    ycol = _year_col(table_id)
    select_cols = ", ".join(group_cols)
    disc_blob = blob or (element or "")
    product_list = products or ([product_name] if product_name else None)
    return (
        f"WITH base AS ("
        f"SELECT {select_cols}, {_agg_expr(aggregation, metric_col)} AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE {ycol} = {int(year)} "
        f"{geo_clause}"
        f"{_discriminator_clause(table_id, disc_blob)}"
        f"{_product_clause(table_id, product_list)}"
        f"GROUP BY {select_cols}"
        f"), tot AS (SELECT SUM(total) AS continent_total FROM base) "
        f"SELECT {', '.join(f'b.{c}' for c in group_cols)}, b.total, "
        f"SAFE_DIVIDE(b.total, t.continent_total) AS share "
        f"FROM base b CROSS JOIN tot t "
        f"ORDER BY share DESC "
        f"LIMIT {lim}"
    )


def try_sql_pattern(
    intent: dict[str, Any],
    *,
    project_id: str,
    dataset: str,
    query: str = "",
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    selected_tables: set[str] | list[str] | None = None,
    limit: int = 20,
    geo_country: str | None = None,
    geo_countries: list[str] | None = None,
) -> dict[str, Any] | None:
    """Compile SQL when intent.pattern is a known builder and required fields resolve."""
    if not project_id or not dataset or not isinstance(intent, dict):
        return None
    pattern = normalize_pattern_name(intent.get("pattern"))
    if pattern == "custom":
        return None
    table_id = _table_from_intent(intent, selected_tables)
    if not table_id:
        return None
    year = _year_from_context(time_start=time_start, time_end=time_end, query=query or "")
    blob_parts = [query or "", str(intent.get("filters") or ""), str(intent.get("goal") or "")]
    if entities:
        blob_parts.extend(str(e) for e in entities)
    blob = " ".join(blob_parts)
    products = match_product_samples(table_id, blob)
    crop = products[0] if products else None
    metric = resolve_measure_column(table_id, str(intent.get("metric") or "").strip() or None)
    if not metric:
        return None
    disc = discriminator_equality_filters(table_id, blob)
    element = next((v for c, v in disc if c.lower() == "element"), None)
    aggregation = measure_sql_aggregation(table_id, metric, element=element)
    geo_sql = _geo_clause(table_id, geo_country=geo_country, geo_countries=geo_countries)
    raw_grain = intent.get("grain")
    grain: list[str] | None
    if isinstance(raw_grain, list):
        grain = [str(g).strip() for g in raw_grain if str(g).strip()]
    elif isinstance(raw_grain, str) and raw_grain.strip():
        grain = [g.strip() for g in raw_grain.split(",") if g.strip()]
    else:
        grain = None
    grain = _normalize_grain(table_id, grain)
    order_by = str(intent.get("order_by") or "total DESC").strip() or "total DESC"

    if pattern in {"rank_by_sum", "yoy_delta", "share_of_total"} and year is None:
        return None

    common: dict[str, Any] = {
        "project_id": project_id,
        "dataset": dataset,
        "table_id": table_id,
        "metric": metric,
        "products": products or None,
        "product_name": crop,
        "element": element,
        "blob": blob,
        "geo_clause": geo_sql,
        "aggregation": aggregation,
    }

    if pattern == "rank_by_sum":
        assert year is not None
        sql = build_rank_by_sum_sql(
            **common,
            year=year,
            grain=grain,
            order_by=order_by,
            limit=limit,
        )
    elif pattern == "time_series":
        sql = build_time_series_sql(
            **common,
            year=year,
            limit=max(limit, 50),
        )
    elif pattern == "yoy_delta":
        assert year is not None
        sql = build_yoy_delta_sql(
            **common,
            year=year,
            grain=grain,
            limit=limit,
        )
    elif pattern == "share_of_total":
        assert year is not None
        sql = build_share_of_total_sql(
            **common,
            year=year,
            grain=grain,
            limit=limit,
        )
    else:
        return None

    if not sql.upper().lstrip().startswith(("SELECT", "WITH")):
        return None
    return {
        "sql": sql,
        "pattern": pattern,
        "table_id": table_id,
        "year": year,
        "product_name": crop,
        "products": products,
        "element": element,
        "metric": metric,
        "aggregation": aggregation,
    }


def try_sql_patterns(
    intents: list[Any] | None,
    *,
    project_id: str,
    dataset: str,
    query: str = "",
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    selected_tables: set[str] | list[str] | None = None,
    limit: int = 20,
    geo_country: str | None = None,
    geo_countries: list[str] | None = None,
    max_queries: int | None = None,
) -> list[dict[str, Any]]:
    """Compile every structured intent that a YAML-capable pattern can bind."""
    if not isinstance(intents, list):
        return []
    cap = max_queries
    if cap is None:
        try:
            cap = int(os.environ.get("RAG_BQ_MAX_SQL_QUERIES", "10") or 10)
        except ValueError:
            cap = 10
    cap = max(1, min(int(cap), 20))
    hits: list[dict[str, Any]] = []
    for idx, intent in enumerate(intents):
        if len(hits) >= cap:
            break
        if not isinstance(intent, dict):
            continue
        hit = try_sql_pattern(
            intent,
            project_id=project_id,
            dataset=dataset,
            query=query,
            entities=entities,
            time_start=time_start,
            time_end=time_end,
            selected_tables=selected_tables,
            limit=limit,
            geo_country=geo_country,
            geo_countries=geo_countries,
        )
        if hit:
            hit["intent_index"] = idx
            hits.append(hit)
    return hits
