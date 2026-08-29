"""Deterministic SQL pattern builders for structured reasoner intents."""
from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.bq_sql_templates import _year_from_context
from ml.rag.chatbot.bq_table_schema_yaml import (
    compile_product_filter_sql,
    discriminator_equality_filters,
    geo_column,
    load_mart_table_schema,
    match_product_samples,
    measure_blob,
    measure_sql_aggregation,
    product_blob,
    resolve_geo_filter_values,
    resolve_measure_column,
    table_supports_sql_pattern,
    year_column,
)
from ml.rag.chatbot.retrieval_contract import choose_agg_vs_fact

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


def _product_clause(
    table_id: str,
    products: list[str] | str | None,
    *,
    project_id: str,
    dataset: str,
    blob: str = "",
) -> str:
    labels: list[str] | None
    if isinstance(products, str):
        labels = [products] if str(products).strip() else None
    elif products:
        labels = [str(p).strip() for p in products if str(p).strip()]
    else:
        labels = None
    sql, _ = compile_product_filter_sql(
        table_id,
        project_id=project_id,
        dataset=dataset,
        blob=blob,
        labels=labels,
    )
    return sql


def _discriminator_clause(
    table_id: str,
    blob: str,
    *,
    primary_measures: list[str] | None = None,
    query: str = "",
) -> str:
    mb = measure_blob(query or blob, primary_measures=primary_measures)
    parts: list[str] = []
    for col, value in discriminator_equality_filters(
        table_id,
        mb,
        primary_measures=primary_measures,
    ):
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
    resolved = resolve_geo_filter_values(table_id, countries)
    if not resolved:
        return ""
    if len(resolved) == 1:
        return f"AND {col} = {_sql_literal(resolved[0])} "
    literals = ", ".join(_sql_literal(c) for c in resolved[:_GEO_IN_CAP])
    return f"AND {col} IN ({literals}) "


def _year_col(table_id: str) -> str:
    return year_column(table_id) or "year"


def _partition_date_col(table_id: str) -> str | None:
    schema = load_mart_table_schema(table_id)
    if not schema:
        return None
    for col in schema.get("columns") or []:
        name = str(col.get("name") or "").strip()
        if name.lower() == "as_of_date":
            return name
    return None


def _time_window_clause(
    table_id: str,
    *,
    year: int | None,
    time_start: str | None = None,
    time_end: str | None = None,
    point_year: bool = False,
) -> str:
    ts = (time_start or "")[:10]
    te = (time_end or "")[:10]
    pcol = _partition_date_col(table_id)
    if pcol and ts and te:
        return f"AND {pcol} BETWEEN DATE '{ts}' AND DATE '{te}' "
    ycol = _year_col(table_id)
    if ts and te and ts[:4].isdigit() and te[:4].isdigit():
        return f"AND {ycol} BETWEEN {int(ts[:4])} AND {int(te[:4])} "
    if point_year and year is not None:
        return f"AND {ycol} = {int(year)} "
    if year is not None:
        return f"AND {ycol} >= {int(year) - 10} AND {ycol} <= {int(year)} "
    return ""


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
    time_start: str | None = None,
    time_end: str | None = None,
    primary_measures: list[str] | None = None,
    query: str = "",
    measure_blob_text: str = "",
    product_blob_text: str = "",
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    metric_col = _safe_ident(metric, default="value")
    group_cols = _safe_idents(grain, default=_default_grain(table_id))
    order = str(order_by or "total DESC").strip() or "total DESC"
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s+(ASC|DESC)$", order, re.IGNORECASE):
        order = "total DESC"
    ycol = _year_col(table_id)
    time_clause = _time_window_clause(
        table_id,
        year=year,
        time_start=time_start,
        time_end=time_end,
        point_year=True,
    )
    year_clause = "WHERE 1=1 " if time_clause.strip() else f"WHERE {ycol} = {int(year)} "
    select_cols = ", ".join(group_cols)
    disc_blob = measure_blob_text or blob or (element or "")
    product_blob_use = product_blob_text or blob or (element or "")
    product_list = products or ([product_name] if product_name else None)
    return (
        f"SELECT {select_cols}, {_agg_expr(aggregation, metric_col)} AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"{year_clause}"
        f"{time_clause}"
        f"{geo_clause}"
        f"{_discriminator_clause(table_id, disc_blob, primary_measures=primary_measures, query=query)}"
        f"{_product_clause(table_id, product_list, project_id=project_id, dataset=dataset, blob=product_blob_use)}"
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
    grain: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    primary_measures: list[str] | None = None,
    query: str = "",
    measure_blob_text: str = "",
    product_blob_text: str = "",
) -> str:
    ycol = _year_col(table_id)
    extra = _safe_idents(
        [g for g in (grain or []) if str(g).strip().lower() not in {ycol.lower(), "year"}],
        default=[],
    )
    lim = max(1, min(int(limit or 50), 200 if extra else 100))
    metric_col = _safe_ident(metric, default="value")
    year_filter = _time_window_clause(
        table_id,
        year=year,
        time_start=time_start,
        time_end=time_end,
    )
    disc_blob = measure_blob_text or blob or (element or "")
    product_blob_use = product_blob_text or blob or (element or "")
    product_list = products or ([product_name] if product_name else None)
    if extra:
        select_cols = ", ".join([*extra, f"{ycol} AS year"])
        group_cols = ", ".join([*extra, ycol])
        order_cols = f"{ycol}, {extra[0]}"
    else:
        select_cols = f"{ycol} AS year"
        group_cols = ycol
        order_cols = ycol
    return (
        f"SELECT {select_cols}, {_agg_expr(aggregation, metric_col)} AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE 1=1 "
        f"{year_filter}"
        f"{geo_clause}"
        f"{_discriminator_clause(table_id, disc_blob, primary_measures=primary_measures, query=query)}"
        f"{_product_clause(table_id, product_list, project_id=project_id, dataset=dataset, blob=product_blob_use)}"
        f"GROUP BY {group_cols} "
        f"ORDER BY {order_cols} "
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
    primary_measures: list[str] | None = None,
    query: str = "",
    measure_blob_text: str = "",
    product_blob_text: str = "",
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    metric_col = _safe_ident(metric, default="value")
    group_cols = _safe_idents(grain, default=_default_grain(table_id))
    ycol = _year_col(table_id)
    select_cols = ", ".join(group_cols)
    join_on = " AND ".join(f"curr.{c} = prev.{c}" for c in group_cols)
    disc_blob = measure_blob_text or blob or (element or "")
    product_blob_use = product_blob_text or blob or (element or "")
    product_list = products or ([product_name] if product_name else None)
    return (
        f"WITH yearly AS ("
        f"SELECT {select_cols}, {ycol} AS year, {_agg_expr(aggregation, metric_col)} AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE {ycol} IN ({int(year)}, {int(year) - 1}) "
        f"{geo_clause}"
        f"{_discriminator_clause(table_id, disc_blob, primary_measures=primary_measures, query=query)}"
        f"{_product_clause(table_id, product_list, project_id=project_id, dataset=dataset, blob=product_blob_use)}"
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
    primary_measures: list[str] | None = None,
    query: str = "",
    measure_blob_text: str = "",
    product_blob_text: str = "",
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    metric_col = _safe_ident(metric, default="value")
    group_cols = _safe_idents(grain, default=_default_grain(table_id))
    ycol = _year_col(table_id)
    select_cols = ", ".join(group_cols)
    disc_blob = measure_blob_text or blob or (element or "")
    product_blob_use = product_blob_text or blob or (element or "")
    product_list = products or ([product_name] if product_name else None)
    return (
        f"WITH base AS ("
        f"SELECT {select_cols}, {_agg_expr(aggregation, metric_col)} AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE {ycol} = {int(year)} "
        f"{geo_clause}"
        f"{_discriminator_clause(table_id, disc_blob, primary_measures=primary_measures, query=query)}"
        f"{_product_clause(table_id, product_list, project_id=project_id, dataset=dataset, blob=product_blob_use)}"
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
    primary_measures: list[str] | None = None,
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
    tables = {
        str(t).strip().split(".")[-1].lower()
        for t in (selected_tables or [])
        if str(t).strip()
    }
    year = _year_from_context(time_start=time_start, time_end=time_end, query=query or "")
    blob_parts = [query or "", str(intent.get("filters") or ""), str(intent.get("goal") or "")]
    product_text = product_blob(query, entities)
    measure_text = measure_blob(query, primary_measures=primary_measures)
    blob = " ".join(blob_parts)
    routed = choose_agg_vs_fact(
        table_id,
        query=query,
        multi_country=bool(geo_countries and len(geo_countries) > 1),
        year_hint=str(year or ""),
        single_country=bool(geo_country or (geo_countries and len(geo_countries) == 1)),
    )
    if routed in tables:
        table_id = routed
    products = match_product_samples(table_id, product_text or blob)
    crop = products[0] if products else None
    metric = resolve_measure_column(table_id, str(intent.get("metric") or "").strip() or None)
    if not metric:
        return None
    disc = discriminator_equality_filters(
        table_id,
        measure_text,
        primary_measures=primary_measures,
    )
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
        "measure_blob_text": measure_text,
        "product_blob_text": product_text,
        "primary_measures": primary_measures,
        "query": query,
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
            time_start=time_start,
            time_end=time_end,
        )
    elif pattern == "time_series":
        sql = build_time_series_sql(
            **common,
            year=year,
            grain=grain,
            limit=max(limit, 50),
            time_start=time_start,
            time_end=time_end,
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
    primary_measures: list[str] | None = None,
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
            primary_measures=primary_measures,
        )
        if hit:
            hit["intent_index"] = idx
            hits.append(hit)
    return hits
