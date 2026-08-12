"""Lightweight FAOSTAT trend lookup for ranked-table ACF direction (Y-1 vs Y+1)."""
from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.bq_sql_templates import _sql_literal

_TREND_STABLE_THRESHOLD = 0.02


def build_faostat_trend_companion_sql(
    *,
    project_id: str,
    dataset: str,
    country_name: str,
    focal_year: int,
    product_name: str | None = None,
) -> str:
    """Production + yield totals for bracketing years around a focal year."""
    year_prior = int(focal_year) - 1
    year_after = int(focal_year) + 1
    fqn = f"`{project_id}.{dataset}.stg_faostat_production`"
    product_clause = ""
    if product_name:
        product_clause = f"AND product_name = {_sql_literal(product_name)} "
    return (
        f"SELECT element, year, SUM(value) AS total "
        f"FROM {fqn} "
        f"WHERE country_name = {_sql_literal(country_name)} "
        f"AND year IN ({year_prior}, {year_after}) "
        f"AND element IN ('Production', 'Yield') "
        f"{product_clause}"
        f"GROUP BY element, year "
        f"ORDER BY element, year"
    )


def _metric_direction(prior: float | None, after: float | None) -> tuple[str, float | None]:
    if prior is None or after is None:
        return "unknown", None
    if prior == 0:
        return "unknown", None
    pct = (after - prior) / abs(prior)
    magnitude = pct * 100.0
    if abs(pct) <= _TREND_STABLE_THRESHOLD:
        return "stable", magnitude
    if after > prior:
        return "increasing", magnitude
    return "decreasing", magnitude


def parse_trend_companion_rows(rows: list[dict[str, Any]], *, focal_year: int) -> dict[str, Any]:
    """Normalize BQ rows into production/yield bracketing-year values."""
    year_prior = focal_year - 1
    year_after = focal_year + 1
    by_element: dict[str, dict[int, float]] = {}
    for row in rows:
        element = str(row.get("element") or "").strip()
        if not element:
            continue
        try:
            year = int(row.get("year"))
            total = float(row.get("total"))
        except (TypeError, ValueError):
            continue
        by_element.setdefault(element, {})[year] = total

    out: dict[str, Any] = {}
    for element in ("Production", "Yield"):
        vals = by_element.get(element, {})
        prior = vals.get(year_prior)
        after = vals.get(year_after)
        direction, magnitude = _metric_direction(prior, after)
        out[element.lower()] = {
            "prior_value": prior,
            "value": after,
            "direction": direction,
            "magnitude": magnitude,
            "year_prior": year_prior,
            "year_after": year_after,
        }
    return out


def align_trend_directions(trend: dict[str, Any]) -> dict[str, Any]:
    """Align production + yield directions; stamp mixed-signal metadata."""
    prod = trend.get("production") or {}
    yld = trend.get("yield") or {}
    prod_dir = str(prod.get("direction") or "unknown")
    yield_dir = str(yld.get("direction") or "unknown")

    mixed = False
    direction = "unknown"
    if prod_dir != "unknown" and yield_dir != "unknown":
        if prod_dir == yield_dir:
            direction = prod_dir
        else:
            mixed = True
            direction = "unknown"
    elif prod_dir != "unknown":
        direction = prod_dir
    elif yield_dir != "unknown":
        direction = yield_dir

    magnitude = prod.get("magnitude")
    if magnitude is None:
        magnitude = yld.get("magnitude")

    return {
        "direction": direction,
        "value": prod.get("value"),
        "prior_value": prod.get("prior_value"),
        "magnitude": magnitude,
        "trend_mixed": mixed,
        "trend_companion": {"production": prod, "yield": yld},
    }


def fetch_faostat_trend_companion(
    *,
    country_name: str,
    focal_year: int,
    product_name: str | None = None,
    project_id: str | None = None,
    dataset: str | None = None,
) -> dict[str, Any] | None:
    """Run companion BQ query; return None when BQ is unavailable or query fails."""
    pid = (project_id or os.environ.get("BQ_PROJECT") or "").strip()
    if not pid:
        return None
    ds = (dataset or os.environ.get("BQ_DATASET_SILVER") or "staging_dev").strip()
    sql = build_faostat_trend_companion_sql(
        project_id=pid,
        dataset=ds,
        country_name=country_name,
        focal_year=focal_year,
        product_name=product_name,
    )
    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=pid)
        rows = [dict(r) for r in client.query(sql).result()]
    except Exception:
        return None
    if not rows:
        return None
    parsed = parse_trend_companion_rows(rows, focal_year=focal_year)
    aligned = align_trend_directions(parsed)
    aligned["sql"] = sql
    return aligned


def _product_from_sql(sql: str) -> str | None:
    m = re.search(r"product_name\s*=\s*['\"]([^'\"]+)['\"]", sql or "", re.IGNORECASE)
    return m.group(1).strip() if m else None


def maybe_attach_ranking_trend(
    meta: dict[str, Any],
    *,
    sql: str,
    template: str,
) -> dict[str, Any]:
    """Attach trend direction to ranked_table metadata when FAOSTAT rank template applies."""
    if meta.get("bq_enrichment") != "ranked_table":
        return meta
    if str(template or "") not in ("faostat_country_rank", "faostat_production_rank"):
        if "stg_faostat_production" not in (sql or "").lower():
            return meta
    ranked_rows = meta.get("ranked_rows")
    if not isinstance(ranked_rows, list) or not ranked_rows:
        return meta
    focal_year = meta.get("year")
    if focal_year is None:
        return meta
    try:
        focal_year = int(focal_year)
    except (TypeError, ValueError):
        return meta
    top_country = str(ranked_rows[0].get("label") or "").strip()
    if not top_country:
        return meta

    trend = fetch_faostat_trend_companion(
        country_name=top_country,
        focal_year=focal_year,
        product_name=_product_from_sql(sql),
    )
    if not trend:
        return meta

    if trend.get("direction"):
        meta["direction"] = trend["direction"]
    if trend.get("value") is not None:
        meta["value"] = trend["value"]
    if trend.get("prior_value") is not None:
        meta["prior_value"] = trend["prior_value"]
    if trend.get("magnitude") is not None:
        meta["magnitude"] = trend["magnitude"]
    if trend.get("trend_mixed"):
        meta["trend_mixed"] = True
    if trend.get("trend_companion"):
        meta["trend_companion"] = trend["trend_companion"]
    if trend.get("sql"):
        meta["trend_companion_sql"] = trend["sql"]
    return meta


__all__ = [
    "align_trend_directions",
    "build_faostat_trend_companion_sql",
    "fetch_faostat_trend_companion",
    "maybe_attach_ranking_trend",
    "parse_trend_companion_rows",
]
