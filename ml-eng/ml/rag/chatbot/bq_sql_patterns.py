"""Deterministic SQL pattern builders for structured reasoner intents."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.bq_sql_templates import _extract_crop, _year_from_context

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_PATTERNS = frozenset(
    {"rank_by_sum", "yoy_delta", "share_of_total", "time_series", "custom"}
)

# Staging facts that fit the SUM(value) / year grain patterns.
_FACT_TABLES = frozenset(
    {
        "stg_faostat_production",
        "stg_faostat_prices",
        "stg_faostat_trade",
        "stg_faostat_food_balances",
        "stg_faostat_emissions",
        "stg_faostat_land_inputs",
        "stg_faostat_population_employment",
        "stg_faostat_investment_asti",
        "stg_faostat_sdg_hdi",
        "stg_fews_market_prices",
        "stg_wfp_vampire_prices",
        "stg_fews_food_security",
        "stg_yield_raw_data",
    }
)


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


def _table_from_intent(
    intent: dict[str, Any],
    selected_tables: set[str] | list[str] | None,
) -> str | None:
    for raw in intent.get("tables") or []:
        tid = str(raw).strip().split(".")[-1].lower()
        if tid in _FACT_TABLES:
            return tid
    for raw in selected_tables or []:
        tid = str(raw).strip().split(".")[-1].lower()
        if tid in _FACT_TABLES:
            return tid
    return None


def _default_grain(table_id: str) -> list[str]:
    if table_id.startswith("stg_fews") or table_id in {
        "stg_wfp_vampire_prices",
        "stg_yield_raw_data",
    }:
        return ["country"]
    return ["country_name"]


def _normalize_grain(table_id: str, grain: list[str] | None) -> list[str] | None:
    """Map conceptual ``country`` → real column for tables that use country_name."""
    if not grain:
        return grain
    # FEWS / WFP / yield keep bare country when that is the real column.
    if table_id.startswith("stg_fews") or table_id in {
        "stg_wfp_vampire_prices",
        "stg_yield_raw_data",
    }:
        return grain
    out: list[str] = []
    for raw in grain:
        g = str(raw).strip()
        if not g:
            continue
        if g.lower() == "country":
            g = "country_name"
        if g not in out:
            out.append(g)
    return out or None


def _fqn(project_id: str, dataset: str, table_id: str) -> str:
    return f"`{project_id}.{dataset}.{table_id}`"


def _product_clause(table_id: str, crop: str | None) -> str:
    if not crop:
        return ""
    safe = crop.replace("'", "''")
    col = "product" if table_id == "stg_yield_raw_data" else "product_name"
    return f"AND {col} = '{safe}' "


def _discriminator_clause(table_id: str, *, element: str | None = None) -> str:
    """Exact metric-discriminator filters from YAML sample defaults."""
    if table_id == "stg_faostat_prices":
        return "AND element = 'Producer Price (USD/tonne)' "
    if table_id == "stg_fews_market_prices":
        return "AND price_type = 'Retail' "
    if table_id == "stg_fews_food_security":
        return (
            "AND measure_type = 'population' "
            "AND scenario_name = 'Current Situation' "
        )
    if table_id.startswith("stg_faostat"):
        el = (element or "Production").strip() or "Production"
        if el not in ("Production", "Yield", "Area harvested", "Import Quantity", "Export Quantity"):
            # Allow common FAOSTAT elements; fall back safely.
            if el.lower() == "yield":
                el = "Yield"
            else:
                el = "Production"
        return f"AND element = '{el}' "
    return ""


def _element_clause(table_id: str, *, element: str | None = None) -> str:
    """Backward-compatible alias for pattern builders."""
    return _discriminator_clause(table_id, element=element)


def _infer_element_from_blob(blob: str) -> str | None:
    b = (blob or "").lower()
    if re.search(r"\byields?\b", b) or "element='yield'" in b or 'element="yield"' in b:
        return "Yield"
    if "element='production'" in b or "element=\"production\"" in b:
        return "Production"
    m = re.search(r"element\s*=\s*'([^']+)'", blob or "", re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _year_col(table_id: str) -> str:
    if table_id == "stg_yield_raw_data":
        return "harvest_year"
    return "year"


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
    limit: int = 20,
    element: str | None = None,
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    metric_col = _safe_ident(metric, default="value")
    group_cols = _safe_idents(grain, default=_default_grain(table_id))
    order = str(order_by or "total DESC").strip() or "total DESC"
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s+(ASC|DESC)$", order, re.IGNORECASE):
        order = "total DESC"
    ycol = _year_col(table_id)
    select_cols = ", ".join(group_cols)
    return (
        f"SELECT {select_cols}, SUM({metric_col}) AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE {ycol} = {int(year)} "
        f"{_element_clause(table_id, element=element)}"
        f"{_product_clause(table_id, product_name)}"
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
    year: int | None = None,
    limit: int = 50,
    element: str | None = None,
) -> str:
    lim = max(1, min(int(limit or 50), 100))
    metric_col = _safe_ident(metric, default="value")
    ycol = _year_col(table_id)
    year_filter = f"AND {ycol} >= {int(year) - 10} AND {ycol} <= {int(year)} " if year else ""
    return (
        f"SELECT {ycol} AS year, SUM({metric_col}) AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE 1=1 "
        f"{year_filter}"
        f"{_element_clause(table_id, element=element)}"
        f"{_product_clause(table_id, product_name)}"
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
    limit: int = 20,
    element: str | None = None,
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    metric_col = _safe_ident(metric, default="value")
    group_cols = _safe_idents(grain, default=_default_grain(table_id))
    ycol = _year_col(table_id)
    select_cols = ", ".join(group_cols)
    join_on = " AND ".join(f"curr.{c} = prev.{c}" for c in group_cols)
    return (
        f"WITH yearly AS ("
        f"SELECT {select_cols}, {ycol} AS year, SUM({metric_col}) AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE {ycol} IN ({int(year)}, {int(year) - 1}) "
        f"{_element_clause(table_id, element=element)}"
        f"{_product_clause(table_id, product_name)}"
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
    limit: int = 20,
    element: str | None = None,
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    metric_col = _safe_ident(metric, default="value")
    group_cols = _safe_idents(grain, default=_default_grain(table_id))
    ycol = _year_col(table_id)
    select_cols = ", ".join(group_cols)
    return (
        f"WITH base AS ("
        f"SELECT {select_cols}, SUM({metric_col}) AS total "
        f"FROM {_fqn(project_id, dataset, table_id)} "
        f"WHERE {ycol} = {int(year)} "
        f"{_element_clause(table_id, element=element)}"
        f"{_product_clause(table_id, product_name)}"
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
    crop = _extract_crop(blob)
    element = _infer_element_from_blob(blob)
    metric = str(intent.get("metric") or "value").strip() or "value"
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

    if pattern == "rank_by_sum":
        assert year is not None
        sql = build_rank_by_sum_sql(
            project_id=project_id,
            dataset=dataset,
            table_id=table_id,
            year=year,
            metric=metric,
            grain=grain,
            order_by=order_by,
            product_name=crop,
            limit=limit,
            element=element,
        )
    elif pattern == "time_series":
        sql = build_time_series_sql(
            project_id=project_id,
            dataset=dataset,
            table_id=table_id,
            metric=metric,
            product_name=crop,
            year=year,
            limit=max(limit, 50),
            element=element,
        )
    elif pattern == "yoy_delta":
        assert year is not None
        sql = build_yoy_delta_sql(
            project_id=project_id,
            dataset=dataset,
            table_id=table_id,
            year=year,
            metric=metric,
            grain=grain,
            product_name=crop,
            limit=limit,
            element=element,
        )
    elif pattern == "share_of_total":
        assert year is not None
        sql = build_share_of_total_sql(
            project_id=project_id,
            dataset=dataset,
            table_id=table_id,
            year=year,
            metric=metric,
            grain=grain,
            product_name=crop,
            limit=limit,
            element=element,
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
        "element": element,
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
) -> dict[str, Any] | None:
    """First matching structured intent wins."""
    if not isinstance(intents, list):
        return None
    for intent in intents:
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
        )
        if hit:
            return hit
    return None
