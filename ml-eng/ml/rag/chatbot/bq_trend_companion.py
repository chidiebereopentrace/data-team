"""Lightweight mart production trend lookup for ranked-table ACF direction (Y-1 vs Y+1)."""
from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.acf_metadata import metric_direction
from ml.rag.chatbot.bq_sql_templates import _sql_literal

_MART_PRODUCTION_RE = re.compile(r"\b(?:fct_production|agg_production_annual)\b", re.IGNORECASE)
_RANK_TEMPLATES = frozenset({"faostat_country_rank", "faostat_production_rank", "mart_production_rank"})


def build_mart_production_trend_companion_sql(
    *,
    project_id: str,
    dataset: str,
    country_iso3: str,
    focal_year: int,
    product_key: str | None = None,
) -> str:
    """Production + national yield totals for bracketing years around a focal year."""
    year_prior = int(focal_year) - 1
    year_after = int(focal_year) + 1
    fqn = f"`{project_id}.{dataset}.fct_production`"
    product_clause = ""
    if product_key:
        product_clause = f"AND product_key = {_sql_literal(product_key)} "
    return (
        f"SELECT element, year, SUM(value) AS total "
        f"FROM {fqn} "
        f"WHERE country_iso3 = {_sql_literal(country_iso3)} "
        f"AND production_grain = 'physical' "
        f"AND year IN ({year_prior}, {year_after}) "
        f"AND element IN ('Production', 'Yield') "
        f"{product_clause}"
        f"GROUP BY element, year "
        f"ORDER BY element, year"
    )


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
        direction, magnitude = metric_direction(prior, after)
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


def fetch_mart_production_trend_companion(
    *,
    country_iso3: str,
    focal_year: int,
    product_key: str | None = None,
    project_id: str | None = None,
    dataset: str | None = None,
) -> dict[str, Any] | None:
    """Run companion BQ query; return None when BQ is unavailable or query fails."""
    pid = (project_id or os.environ.get("BQ_PROJECT") or "").strip()
    if not pid:
        return None
    ds = (dataset or os.environ.get("BQ_DATASET_GOLD") or "mart_dev").strip()
    sql = build_mart_production_trend_companion_sql(
        project_id=pid,
        dataset=ds,
        country_iso3=country_iso3,
        focal_year=focal_year,
        product_key=product_key,
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
    text = sql or ""
    for pattern in (
        r"product_key\s*=\s*['\"]([^'\"]+)['\"]",
        r"product_name\s*=\s*['\"]([^'\"]+)['\"]",
        r"metric\s*=\s*['\"]([^'\"]+)['\"]",
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _country_iso3_from_ranked(meta: dict[str, Any]) -> str:
    ranked_rows = meta.get("ranked_rows")
    if not isinstance(ranked_rows, list) or not ranked_rows:
        return ""
    top = ranked_rows[0]
    if not isinstance(top, dict):
        return ""
    raw = top.get("raw_row")
    if isinstance(raw, dict):
        iso = str(raw.get("country_iso3") or "").strip()
        if iso:
            return iso
    label = str(top.get("label") or "").strip()
    if len(label) == 3 and label.isalpha():
        return label.upper()
    return label


def _is_production_rank_context(*, sql: str, template: str) -> bool:
    if str(template or "") in _RANK_TEMPLATES:
        return True
    return bool(_MART_PRODUCTION_RE.search(sql or ""))


def maybe_attach_ranking_trend(
    meta: dict[str, Any],
    *,
    sql: str,
    template: str,
) -> dict[str, Any]:
    """Attach trend direction to ranked_table metadata when mart production rank applies."""
    if meta.get("bq_enrichment") != "ranked_table":
        return meta
    if not _is_production_rank_context(sql=sql, template=template):
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
    country_iso3 = _country_iso3_from_ranked(meta)
    if not country_iso3:
        return meta

    trend = fetch_mart_production_trend_companion(
        country_iso3=country_iso3,
        focal_year=focal_year,
        product_key=_product_from_sql(sql),
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
    "build_mart_production_trend_companion_sql",
    "fetch_mart_production_trend_companion",
    "maybe_attach_ranking_trend",
    "parse_trend_companion_rows",
]
