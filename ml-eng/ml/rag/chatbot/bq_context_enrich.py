"""Deterministic BigQuery row enrichment: value semantics + readable context prose."""
from __future__ import annotations

import ast
import re
from typing import Any

from ml.rag.chatbot.acf_metadata import project_bq_row_acf, stamp_temporal_direction
from ml.rag.chatbot.bq_table_schema_yaml import (
    column_description,
    discriminator_columns,
    load_mart_table_schema,
    measure_columns,
    table_source_meta,
)
from ml.rag.chatbot.bq_trend_companion import maybe_attach_ranking_trend
from ml.rag.chatbot.generator import (
    _BQ_PUBLIC_LABEL,
    _public_source_label,
    is_ranking_numeric_query,
    is_usable_context_item,
)

_YEAR_SQL_RE = re.compile(r"\byear\s*=\s*(\d{4})\b", re.IGNORECASE)
_ELEMENT_SQL_RE = re.compile(r"\belement\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)

_STG_TABLE_RE = re.compile(r"\bstg_[a-z0-9_]+\b", re.IGNORECASE)
_MART_TABLE_RE = re.compile(r"\b(?:fct|agg|dim)_[a-z0-9_]+\b", re.IGNORECASE)
_SELECT_ALIAS_RE = re.compile(
    r"\b(?:SUM|AVG|MIN|MAX|COUNT)\s*\([^)]+\)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_BQ_PROVENANCE_KEYS = frozenset(
    {
        "sql",
        "sql_index",
        "sql_count",
        "sql_source",
        "nl2sql_model",
        "template",
        "pattern",
        "status",
        "validation_failed",
        "execution_error",
        "prep_error",
        "nl2sql_raw",
        "tier",
        "data_level",
        "source_id",
        "geo_scope",
        "geo_country_primary",
        "geo_countries",
        "as_of_date",
        "metric",
        "unit",
        "direction",
        "prior_value",
        "value_semantics",
        "raw_row",
        "ranked_rows",
        "bq_enrichment",
        "year",
        "trend_companion",
        "trend_mixed",
        "trend_companion_sql",
        "coverage_strength",
        "bq_context_truncated",
        "table_id",
        "table_description",
        "source_domain",
        "source_layer",
    }
)

_GEO_KEYS = (
    "country_iso3",
    "country_name",
    "country",
    "geographic_unit_name",
    "market_name",
    "admin_1",
    "admin_2",
    "fnid",
    "region",
)
_TIME_KEYS = (
    "year",
    "month",
    "harvest_year",
    "planting_year",
    "observation_year",
    "mp_year",
    "mp_month",
)
_LABEL_KEYS = (
    "country_iso3",
    "country_name",
    "country",
    "geographic_unit_name",
    "market_name",
    "product_name",
    "product",
)
_RANK_VALUE_KEYS = (
    "total",
    "total_production_qty",
    "production_qty",
    "sum_value",
    "value",
    "gdp_per_capita_ppp",
    "hdi_value",
    "production",
    "yield",
)


def _s(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _table_from_sql(sql: str) -> str:
    text = sql or ""
    m = _MART_TABLE_RE.search(text)
    if m:
        return m.group(0).lower()
    m = _STG_TABLE_RE.search(text)
    return m.group(0).lower() if m else ""


def _parse_row_content(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if not text or text.startswith("[BQ"):
        return None
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return val if isinstance(val, dict) else None


def _raw_row_from_item(item: dict[str, Any]) -> dict[str, Any]:
    meta_raw = item.get("metadata")
    if not isinstance(meta_raw, dict):
        meta_raw = {}
    meta: dict[str, Any] = meta_raw
    existing = meta.get("raw_row")
    if isinstance(existing, dict):
        return dict(existing)
    content_raw = item.get("content")
    if not isinstance(content_raw, str):
        content_raw = ""
    parsed = _parse_row_content(str(content_raw))
    if parsed:
        return dict(parsed)
    return {k: v for k, v in (meta or {}).items() if k not in _BQ_PROVENANCE_KEYS and v is not None}


def _row_coherent_with_decomposition(
    raw: dict[str, Any],
    decomposition: dict[str, Any] | None,
) -> bool:
    """Drop rows whose measure/geo/year contradict decomposition intent."""
    if not raw or not decomposition:
        return True

    pm = [
        str(m).strip().lower()
        for m in (decomposition.get("primary_measures") or [])
        if str(m).strip()
    ]
    element = str(raw.get("element") or "").strip().lower()
    metric = str(raw.get("metric") or "").strip().lower()
    production_intent = any(m in ("production", "prod") for m in pm)
    yield_intent = any("yield" in m for m in pm)
    if production_intent and not yield_intent:
        if element == "yield":
            return False
        if metric and "yield" in metric and "production_production" not in metric:
            return False
    if yield_intent and not production_intent and element == "production":
        return False

    geo = [str(g).strip() for g in (decomposition.get("geography") or []) if str(g).strip()]
    if geo:
        row_geo_parts = [
            str(raw.get(k) or "").strip().lower()
            for k in _GEO_KEYS
            if raw.get(k) is not None and str(raw.get(k)).strip()
        ]
        row_geo = " ".join(row_geo_parts)
        if row_geo and not any(
            g.lower() in row_geo or row_geo in g.lower() for g in geo
        ):
            return False

    ts = str(decomposition.get("time_start") or "")[:4]
    te = str(decomposition.get("time_end") or "")[:4]
    if ts.isdigit() and te.isdigit():
        row_year = raw.get("year") or raw.get("time_key") or raw.get("observation_year")
        if row_year is not None:
            try:
                y = int(row_year)
            except (TypeError, ValueError):
                return True
            if y < int(ts) or y > int(te):
                return False

    return True


def _reject_incoherent_item(
    item: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    out = dict(item)
    meta = dict(out.get("metadata") or {})
    meta["semantic_row_rejected"] = True
    meta["semantic_row_reason"] = reason
    out["metadata"] = meta
    return out

def _sql_aliases(sql: str) -> list[str]:
    return [m.group(1).lower() for m in _SELECT_ALIAS_RE.finditer(sql or "")]


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        val = _s(row.get(key))
        if val:
            return val
    return ""


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return _s(value)
    if abs(num - round(num)) < 1e-6:
        return f"{int(round(num)):,}"
    return f"{num:,.2f}"


def _measure_kind(table_id: str, row: dict[str, Any]) -> str:
    bare = table_id.lower()
    if bare in ("fct_economics",) or "economics" in bare:
        return "macro_gdp"
    if bare in ("fct_hdi", "agg_hdi_latest") or bare.endswith("_hdi") or "hdi" in bare:
        return "macro_hdi"
    if bare in ("fct_food_security", "agg_food_security_monthly") or "fews_food_security" in bare:
        return "fews_food_security"
    if bare in ("fct_prices", "agg_prices_country_month") or "market_prices" in bare or "vampire_prices" in bare:
        return "fews_market_price"
    if bare == "fct_yield" or "yield_raw_data" in bare:
        return "subnational_yield"
    if bare in ("fct_production", "agg_production_annual"):
        grain = _s(row.get("production_grain"))
        if grain:
            return f"mart_production_{grain}"
        return "mart_production"
    if bare == "fct_climate":
        return "mart_climate"
    if bare.startswith("stg_faostat_"):
        return f"faostat_{bare.replace('stg_faostat_', '')}"
    if bare.startswith(("fct_", "agg_", "dim_")):
        return bare
    return bare.replace("stg_", "")


def _element_measure_label(element: str) -> str:
    el = element.strip()
    low = el.lower()
    if low == "production":
        return "Physical crop/livestock production output"
    if low == "yield":
        return "Crop/livestock yield (productivity per land area)"
    if "area harvested" in low:
        return "Harvested land area"
    if "index" in low:
        return f"FAOSTAT index measure ({el})"
    if low in {"stocks", "milk animals", "laying", "producing animals/slaughtered"}:
        return f"Livestock inventory or animal count ({el})"
    if "carcass" in low or "weight" in low:
        return f"Livestock productivity measure ({el})"
    return f"FAOSTAT measure ({el})"


def _fews_measure_label(row: dict[str, Any]) -> str:
    mt = _s(row.get("measure_type")).lower()
    if mt == "population":
        phase = _s(row.get("phase_name")) or _s(row.get("phase_code"))
        if phase:
            return f"Food-insecure population estimate ({phase})"
        return "Food-insecure population estimate"
    if mt == "classification":
        scale = _s(row.get("classification_scale"))
        if scale:
            return f"IPC area food-security classification ({scale})"
        return "IPC area food-security classification"
    return "FEWS NET food-security measure"


def _price_measure_label(row: dict[str, Any]) -> str:
    pt = _s(row.get("price_type")) or "market price"
    product = _s(row.get("product_name")) or _s(row.get("product"))
    market = _s(row.get("market_name"))
    bits = [f"{pt} market price"]
    if product:
        bits.append(f"for {product}")
    if market:
        bits.append(f"at {market}")
    return " ".join(bits)


def _resolve_unit(table_id: str, row: dict[str, Any], measure_col: str) -> str:
    bare = table_id.lower()
    if "gdp" in bare and measure_col == "gdp_per_capita_ppp":
        return "international dollars (PPP per capita)"
    if "hdi" in bare and measure_col == "hdi_value":
        return "index score (0–1)"
    mt = _s(row.get("measure_type")).lower()
    if mt == "population":
        return "people"
    if mt == "classification":
        return "IPC classification"
    prod_unit = _s(row.get("production_unit"))
    if prod_unit:
        return prod_unit
    unit = _s(row.get("unit"))
    currency = _s(row.get("currency"))
    if currency and unit:
        return f"{currency}/{unit}"
    if unit:
        return unit
    if bare in ("fct_production", "agg_production_annual") or "production" in bare:
        if measure_col in (
            "total_production_qty",
            "production_qty",
            "total",
            "production",
            "value",
        ):
            return "tonnes"
    if measure_col == "yield":
        return "yield units (see source table)"
    if measure_col == "area":
        return "area units (see source table)"
    if measure_col == "production":
        return "production units (see source table)"
    if measure_col.startswith("pct_"):
        return "share (%)"
    return unit or "units"


def _resolve_measure_label(
    table_id: str,
    row: dict[str, Any],
    measure_col: str,
) -> str:
    bare = table_id.lower()
    element = _s(row.get("element"))
    if element:
        return _element_measure_label(element)
    if bare in ("fct_food_security", "agg_food_security_monthly") or "fews_food_security" in bare:
        return _fews_measure_label(row)
    if bare in ("fct_prices", "agg_prices_country_month") or "market_prices" in bare or "vampire_prices" in bare:
        return _price_measure_label(row)
    if bare in ("fct_economics",) or "economics" in bare:
        if measure_col == "gdp_per_capita_ppp":
            return "GDP per capita at purchasing power parity"
    if measure_col == "gdp_per_capita_ppp":
        return "GDP per capita at purchasing power parity"
    if bare in ("fct_hdi", "agg_hdi_latest") or measure_col == "hdi_value":
        return "Human Development Index score"
    if measure_col == "hdi_value":
        return "Human Development Index score"
    if measure_col == "yield":
        return "Crop yield (productivity per land area)"
    if measure_col in ("production", "total_production_qty", "production_qty"):
        return "Crop production volume"
    if measure_col == "area":
        return "Harvested or cultivated area"
    desc = column_description(table_id, measure_col)
    if desc:
        return desc[:160]
    return measure_col.replace("_", " ")


def _not_this_list(table_id: str, row: dict[str, Any], measure_col: str) -> list[str]:
    bare = table_id.lower()
    element = _s(row.get("element")).lower()
    mt = _s(row.get("measure_type")).lower()
    out: list[str] = []

    if "gdp" in bare or bare == "fct_economics" or measure_col == "gdp_per_capita_ppp":
        out.extend(["agricultural production volume", "crop yield", "food security IPC phase", "market retail price"])
    elif "hdi" in bare or bare in ("fct_hdi", "agg_hdi_latest") or measure_col == "hdi_value":
        out.extend(["agricultural production", "crop yield", "GDP", "market price"])
    elif bare in ("fct_food_security", "agg_food_security_monthly") or "fews_food_security" in bare:
        if mt == "population":
            out.extend(["crop production tonnes", "GDP", "retail market price", "IPC area classification map"])
        elif mt == "classification":
            out.extend(["population headcount", "crop production tonnes", "GDP", "market price"])
        else:
            out.extend(["crop production tonnes", "GDP"])
    elif bare in ("fct_prices", "agg_prices_country_month") or "market_prices" in bare or "vampire_prices" in bare:
        out.extend(["production volume", "crop yield", "GDP", "food security phase population"])
    elif bare == "fct_yield" or "yield_raw_data" in bare:
        if measure_col == "yield":
            out.extend(["total production volume", "GDP", "food security phase"])
        elif measure_col == "production":
            out.extend(["yield per hectare", "GDP", "food security phase"])
        else:
            out.extend(["GDP", "food security phase"])
    elif element == "production":
        out.extend(["crop yield per hectare", "GDP", "food security IPC phase"])
    elif element == "yield":
        out.extend(["total production volume", "GDP", "food security IPC phase"])
    elif "price" in bare or _s(row.get("price_type")):
        out.extend(["production volume", "GDP", "food security phase"])
    else:
        out.extend(["GDP", "food security IPC phase"])

    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:4]


def _pick_measure_column(
    row: dict[str, Any],
    *,
    table_id: str,
    sql: str,
) -> tuple[str, Any]:
    aliases = _sql_aliases(sql)
    for alias in aliases:
        if alias in row and row[alias] is not None:
            return alias, row[alias]
    for key in _RANK_VALUE_KEYS:
        if key in row and row[key] is not None:
            return key, row[key]
    for col in measure_columns(table_id):
        if col in row and row[col] is not None:
            return col, row[col]
    if row.get("value") is not None:
        return "value", row["value"]
    for key, val in row.items():
        if key in _BQ_PROVENANCE_KEYS or key in _GEO_KEYS or key in _TIME_KEYS:
            continue
        if isinstance(val, (int, float)):
            return key, val
    return "", None


def resolve_row_semantics(
    row: dict[str, Any],
    *,
    table_id: str,
    schema: dict[str, Any] | None = None,
    sql: str = "",
) -> dict[str, Any]:
    """Return structured value semantics for prose and metadata stamping."""
    bare = table_id.lower()
    if not schema:
        schema = load_mart_table_schema(bare) or {}
    src = table_source_meta(bare)
    measure_col, measure_val = _pick_measure_column(row, table_id=bare, sql=sql)
    measure_label = _resolve_measure_label(bare, row, measure_col)
    unit = _resolve_unit(bare, row, measure_col)
    geo = _first_present(row, _GEO_KEYS)
    time_bits = [_first_present(row, (k,)) for k in _TIME_KEYS]
    time_s = " ".join(t for t in time_bits if t)

    discriminators: dict[str, str] = {}
    for col in discriminator_columns(bare):
        val = _s(row.get(col))
        if val:
            discriminators[col] = val

    return {
        "measure_kind": _measure_kind(bare, row),
        "measure_column": measure_col,
        "measure_label": measure_label,
        "measure_value": measure_val,
        "discriminators": discriminators,
        "unit": unit,
        "geo": geo,
        "time": time_s,
        "table_id": bare,
        "table_description": src.get("description") or "",
        "source_domain": src.get("source_domain") or "",
        "source_layer": src.get("source_layer") or "mart_dev",
        "grain": src.get("grain") or "",
        "source_name": _s(row.get("source_name")),
        "source_natural_key": _s(row.get("source_natural_key")),
        "organisation_name": _s(row.get("organisation_name")),
        "price_source": _s(row.get("price_source")),
        "not_this": _not_this_list(bare, row, measure_col),
    }


def _format_discriminator_lines(semantics: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    table_id = str(semantics.get("table_id") or "")
    skip_cols = {"source_name", "source_natural_key", "source_key"}
    for col, val in (semantics.get("discriminators") or {}).items():
        if col in skip_cols:
            continue
        desc = column_description(table_id, str(col))
        snippet = desc[:120] if desc else ""
        if snippet:
            lines.append(f"{col}={val} — {snippet}")
        else:
            lines.append(f"{col}={val}")
    return lines


def _format_trend_line(direction: Any, magnitude: Any) -> str | None:
    d = _s(direction)
    if not d or d.lower() == "unknown":
        return None
    mag_part = ""
    if magnitude is not None:
        try:
            mag_part = f" ({float(magnitude):+.0f}% change)"
        except (TypeError, ValueError):
            pass
    return f"Trend: {d}{mag_part}"


def format_row_prose(
    semantics: dict[str, Any],
    *,
    sql_source: str = "",
    template: str = "",
    direction: Any = None,
    magnitude: Any = None,
) -> str:
    """Build human-readable context prose from resolved semantics."""
    table_id = _s(semantics.get("table_id"))
    domain = _s(semantics.get("source_domain"))
    label_meta = {
        "source_domain": domain,
        "source_name": semantics.get("source_name"),
        "source_natural_key": semantics.get("source_natural_key"),
        "organisation_name": semantics.get("organisation_name"),
        "price_source": semantics.get("price_source"),
    }
    institution = _public_source_label(table_id, label_meta)
    if institution:
        title = institution
    else:
        title = _BQ_PUBLIC_LABEL
        if domain and "stg_" not in domain.lower():
            title += f" [{domain}]"

    lines = [title]
    desc = _s(semantics.get("table_description"))
    grain = _s(semantics.get("grain"))
    if desc:
        lines.append(f"Source: {desc}")
    if grain:
        lines.append(f"Grain: {grain}")

    measure_label = _s(semantics.get("measure_label"))
    disc_lines = _format_discriminator_lines(semantics)
    if disc_lines:
        lines.append("Filters: " + "; ".join(disc_lines[:5]))
    lines.append(f"What this value is: {measure_label}.")

    unit = _s(semantics.get("unit"))
    time_s = _s(semantics.get("time"))
    geo = _s(semantics.get("geo"))
    meta_bits = []
    if unit:
        meta_bits.append(f"Unit: {unit}")
    if time_s:
        meta_bits.append(f"Time: {time_s}")
    if geo:
        meta_bits.append(f"Place: {geo}")
    if meta_bits:
        lines.append(" | ".join(meta_bits))

    val = semantics.get("measure_value")
    if val is not None:
        formatted = _format_number(val)
        suffix = f" {unit}" if unit and unit not in formatted else ""
        lines.append(f"Value: {formatted}{suffix}")

    not_this = semantics.get("not_this") or []
    if not_this:
        lines.append(f"What this is NOT: {', '.join(str(x) for x in not_this)}.")

    trend = _format_trend_line(
        direction if direction is not None else semantics.get("direction"),
        magnitude if magnitude is not None else semantics.get("magnitude"),
    )
    if trend:
        lines.append(trend)

    # template / sql_source stay in metadata only — do not surface internal
    # pipeline vocabulary into the LLM context.
    _ = template, sql_source

    return "\n".join(lines)


def _rank_label(row: dict[str, Any]) -> str:
    return _first_present(row, _LABEL_KEYS) or "Unknown"


def _rank_value(row: dict[str, Any], *, table_id: str, sql: str) -> tuple[str, Any]:
    col, val = _pick_measure_column(row, table_id=table_id, sql=sql)
    return col, val


def _year_from_sql(sql: str) -> int | None:
    m = _YEAR_SQL_RE.search(sql or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _year_from_decomposition(decomposition: dict[str, Any] | None) -> int | None:
    if not isinstance(decomposition, dict):
        return None
    for key in ("time_end", "time_start"):
        raw = _s(decomposition.get(key))
        if not raw:
            continue
        m = re.match(r"(\d{4})", raw)
        if m:
            try:
                return int(m.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _resolve_observation_date(
    row: dict[str, Any],
    *,
    sql: str,
    decomposition: dict[str, Any] | None,
) -> tuple[int | None, str | None]:
    """Return (year, as_of_date ISO) from row keys, SQL year filter, or decomposition."""
    for key in _TIME_KEYS:
        raw = row.get(key)
        if raw is None:
            continue
        text = _s(raw)
        if not text:
            continue
        ym = re.match(r"(\d{4})", text)
        if ym:
            year = int(ym.group(1))
            return year, f"{year}-01-01"

    year = _year_from_sql(sql)
    if year is not None:
        return year, f"{year}-01-01"

    dec_year = _year_from_decomposition(decomposition)
    if dec_year is not None:
        as_of = f"{dec_year}-01-01"
        if isinstance(decomposition, dict):
            te = _s(decomposition.get("time_end"))
            if te and len(te) >= 10:
                as_of = te[:10]
        return dec_year, as_of

    return None, None


def _element_from_sql(sql: str) -> str:
    m = _ELEMENT_SQL_RE.search(sql or "")
    return m.group(1).strip() if m else ""


def _stamp_acf_metadata(
    meta: dict[str, Any],
    semantics: dict[str, Any] | None,
    *,
    sql: str,
    decomposition: dict[str, Any] | None,
    row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp year, as_of_date, metric, unit, and value fields for ACF scoring."""
    out = dict(meta)
    raw_existing = out.get("raw_row")
    raw: dict[str, Any] = row if isinstance(row, dict) else (
        raw_existing if isinstance(raw_existing, dict) else {}
    )
    year, as_of = _resolve_observation_date(raw, sql=sql, decomposition=decomposition)
    if year is not None:
        out["year"] = year
    if as_of:
        out["as_of_date"] = as_of

    sem_raw = semantics if isinstance(semantics, dict) else out.get("value_semantics")
    warehouse_metric = _s(raw.get("metric"))
    if isinstance(sem_raw, dict):
        measure_label = _s(sem_raw.get("measure_label"))
        if measure_label and not warehouse_metric:
            out["metric"] = measure_label
        unit = _s(sem_raw.get("unit"))
        if unit:
            out["unit"] = unit
        if out.get("value") is None and sem_raw.get("measure_value") is not None:
            out["value"] = sem_raw.get("measure_value")

    element = _element_from_sql(sql) or _s(raw.get("element"))
    if element and not _s(out.get("metric")) and not warehouse_metric:
        out["metric"] = element

    ranked_rows = out.get("ranked_rows")
    if isinstance(ranked_rows, list) and ranked_rows:
        top = ranked_rows[0]
        if isinstance(top, dict):
            label = _s(top.get("label"))
            if label:
                out.setdefault("geo_country_primary", label)
                out.setdefault("geo_countries", label)
            if out.get("value") is None and top.get("value") is not None:
                out["value"] = top["value"]
            if not _s(out.get("unit")):
                out["unit"] = _s(top.get("unit"))
        out["coverage_strength"] = min(1.0, len(ranked_rows) / 10.0)

    merged = stamp_temporal_direction({**raw, **out})
    out.update({k: v for k, v in merged.items() if v is not None})

    projected = project_bq_row_acf({**raw, **out}, table_hint=_s(out.get("table_id")))
    for key in (
        "tier",
        "data_level",
        "as_of_date",
        "as_of_date_basis",
        "region",
        "source_id",
        "source_key",
        "metric",
        "unit",
        "geo_scope",
        "place_scope",
        "geo_country_primary",
        "geo_countries",
        "value",
        "prior_value",
        "direction",
        "magnitude",
    ):
        if projected.get(key) is not None:
            out[key] = projected[key]
    if warehouse_metric:
        out["metric"] = warehouse_metric
    for key in ("place_scope", "source_key", "as_of_date_basis"):
        if raw.get(key) is not None and out.get(key) is None:
            out[key] = raw[key]
    return out


def _point_fact_canonical_score(raw: dict[str, Any], table_id: str) -> tuple[int, int, int, float]:
    """Sort key for picking one point-fact row (higher is better)."""
    bare = table_id.lower()
    if bare.startswith("fct_"):
        kind = 3
    elif bare.startswith("agg_"):
        kind = 1
    else:
        kind = 0
    tier_raw = raw.get("tier")
    tier_boost = 0
    if tier_raw is not None:
        try:
            tier_int = int(tier_raw)
            tier_boost = 2 if tier_int == 2 else (1 if tier_int == 1 else 0)
        except (TypeError, ValueError):
            tier_boost = 0
    faostat = 1 if "faostat" in _s(raw.get("source_key")).lower() else 0
    try:
        record_count = float(raw.get("record_count") or 0)
    except (TypeError, ValueError):
        record_count = 0.0
    return (kind, tier_boost, faostat, record_count)


def _consolidate_point_fact_batch(
    items: list[dict[str, Any]],
    *,
    table_id: str,
    decomposition: dict[str, Any] | None = None,
    task_mode: str = "",
) -> dict[str, Any] | None:
    if len(items) < 2:
        return None
    meta0 = dict(items[0].get("metadata") or {})
    template = _s(meta0.get("template"))
    if template != "mart_point_fact" and task_mode != "fact_lookup":
        return None
    if template != "mart_point_fact":
        return None
    sql = _s(meta0.get("sql"))
    if not sql:
        return None
    for item in items[1:]:
        if _s((item.get("metadata") or {}).get("sql")) != sql:
            return None

    candidates: list[dict[str, Any]] = []
    for item in items:
        raw = _raw_row_from_item(item)
        semantics = resolve_row_semantics(raw, table_id=table_id, sql=sql)
        val = semantics.get("measure_value")
        if val is None:
            continue
        try:
            float(val)
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "item": item,
                "raw": raw,
                "semantics": semantics,
                "score": _point_fact_canonical_score(raw, table_id),
            }
        )
    if len(candidates) < 1:
        return None

    candidates.sort(key=lambda c: c["score"], reverse=True)
    winner = candidates[0]
    head_sem = winner["semantics"]

    numeric_values: list[float] = []
    alternate_values: list[dict[str, Any]] = []
    for entry in candidates:
        try:
            num = float(entry["semantics"]["measure_value"])
        except (TypeError, ValueError):
            continue
        numeric_values.append(num)
        alternate_values.append(
            {
                "value": num,
                "source_key": _s(entry["raw"].get("source_key")),
                "source_name": _s(entry["raw"].get("source_name")),
            }
        )

    value_conflict = False
    if len(numeric_values) >= 2:
        spread = max(numeric_values) - min(numeric_values)
        denom = max(abs(max(numeric_values)), 1e-9)
        if spread / denom > 0.01:
            value_conflict = True

    out_meta = dict(meta0)
    out_meta["raw_row"] = winner["raw"]
    out_meta["value_semantics"] = head_sem
    out_meta["bq_enrichment"] = "point_fact"
    for key in (
        "source_key",
        "source_name",
        "tier",
        "data_level",
        "place_scope",
        "metric",
        "unit",
        "production_unit",
        "production_grain",
        "record_count",
    ):
        if winner["raw"].get(key) is not None:
            out_meta[key] = winner["raw"][key]
    if table_id:
        out_meta["table_id"] = table_id
    if value_conflict:
        out_meta["value_conflict"] = True
        out_meta["alternate_values"] = alternate_values
    out_meta = _stamp_acf_metadata(
        out_meta,
        head_sem,
        sql=sql,
        decomposition=decomposition,
        row=winner["raw"],
    )

    content = format_row_prose(
        head_sem,
        sql_source=_s(out_meta.get("sql_source")),
        template=template,
        direction=out_meta.get("direction"),
        magnitude=out_meta.get("magnitude"),
    )
    if value_conflict:
        content += "\nNote: multiple warehouse rows matched this point fact; canonical row selected."

    return {
        "content": content,
        "source": "bigquery",
        "metadata": out_meta,
    }


def _consolidate_ranking_batch(
    items: list[dict[str, Any]],
    *,
    query: str,
    table_id: str,
    decomposition: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not is_ranking_numeric_query(query) or len(items) < 2:
        return None
    sql = _s((items[0].get("metadata") or {}).get("sql"))
    if not sql:
        return None
    for item in items[1:]:
        if _s((item.get("metadata") or {}).get("sql")) != sql:
            return None

    ranked: list[dict[str, Any]] = []
    for item in items:
        raw = _raw_row_from_item(item)
        semantics = resolve_row_semantics(raw, table_id=table_id, sql=sql)
        label = _rank_label(raw)
        val = semantics.get("measure_value")
        if val is None:
            continue
        try:
            sort_val = float(val)
        except (TypeError, ValueError):
            continue
        ranked.append(
            {
                "label": label,
                "value": val,
                "sort_value": sort_val,
                "unit": semantics.get("unit") or "",
                "measure_label": semantics.get("measure_label") or "",
                "time": semantics.get("time") or "",
                "raw_row": raw,
                "semantics": semantics,
            }
        )
    if len(ranked) < 2:
        return None

    ranked.sort(key=lambda r: r["sort_value"], reverse=True)
    head_sem = ranked[0]["semantics"]
    meta0 = dict(items[0].get("metadata") or {})
    template = _s(meta0.get("template"))
    sql_source = _s(meta0.get("sql_source"))

    rank_lines = []
    ranked_rows = []
    for idx, entry in enumerate(ranked, start=1):
        unit = _s(entry.get("unit"))
        val_s = _format_number(entry.get("value"))
        suffix = f" {unit}" if unit else ""
        ml = _s(entry.get("measure_label"))
        time_s = _s(entry.get("time"))
        detail = f" ({ml}, {time_s})" if ml and time_s else (f" ({ml})" if ml else "")
        rank_lines.append(f"{idx}. {entry['label']} — {val_s}{suffix}{detail}")
        ranked_rows.append(
            {
                "rank": idx,
                "label": entry["label"],
                "value": entry["value"],
                "unit": unit,
                "measure_label": ml,
                "time": time_s,
                "raw_row": entry["raw_row"],
            }
        )

    not_this = head_sem.get("not_this") or []
    header = format_row_prose(head_sem, sql_source=sql_source, template=template)
    body = "\n".join(
        [
            "",
            "Ranked results (highest first):",
            *rank_lines,
        ]
    )
    content = header + body

    out_meta = dict(meta0)
    out_meta["raw_row"] = ranked[0]["raw_row"]
    out_meta["value_semantics"] = head_sem
    out_meta["ranked_rows"] = ranked_rows
    out_meta["bq_enrichment"] = "ranked_table"
    if table_id:
        out_meta["table_id"] = table_id
    out_meta = _stamp_acf_metadata(
        out_meta,
        head_sem,
        sql=sql,
        decomposition=decomposition,
        row=ranked[0]["raw_row"],
    )
    out_meta = maybe_attach_ranking_trend(
        out_meta,
        sql=sql,
        template=template,
    )

    trend = _format_trend_line(out_meta.get("direction"), out_meta.get("magnitude"))
    if trend:
        content = content + "\n" + trend

    return {
        "content": content,
        "source": "bigquery",
        "metadata": out_meta,
    }


def _enrich_single_item(
    item: dict[str, Any],
    *,
    table_id: str,
    decomposition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(item.get("metadata") or {})
    raw = _raw_row_from_item(item)
    sql = _s(meta.get("sql"))
    if not table_id:
        table_id = _table_from_sql(sql)
    if not table_id:
        table_id = _s(meta.get("table_id"))

    semantics = resolve_row_semantics(raw, table_id=table_id or "unknown", sql=sql)
    meta["raw_row"] = raw
    meta["value_semantics"] = semantics
    if table_id:
        meta["table_id"] = table_id
        src = table_source_meta(table_id)
        meta["table_description"] = src.get("description")
        meta["source_domain"] = src.get("source_domain")
        meta["source_layer"] = src.get("source_layer")
    meta = _stamp_acf_metadata(meta, semantics, sql=sql, decomposition=decomposition, row=raw)

    content = format_row_prose(
        semantics,
        sql_source=_s(meta.get("sql_source")),
        template=_s(meta.get("template")),
        direction=meta.get("direction"),
        magnitude=meta.get("magnitude"),
    )
    return {**item, "content": content, "metadata": meta}


def enrich_bq_results(
    items: list[dict[str, Any]],
    *,
    query: str,
    plan: dict[str, Any] | None = None,
    decomposition: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Enrich usable BQ context items with value semantics prose before merge/rerank."""
    plan = plan if isinstance(plan, dict) else {}
    dec = decomposition if isinstance(decomposition, dict) else None
    selected = [str(t).strip().split(".")[-1].lower() for t in (plan.get("selected_tables") or []) if str(t).strip()]

    usable: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for item in items or []:
        if is_usable_context_item(item):
            usable.append(item)
        else:
            diagnostics.append(item)

    if dec:
        kept: list[dict[str, Any]] = []
        for item in usable:
            raw = _raw_row_from_item(item)
            if _row_coherent_with_decomposition(raw, dec):
                kept.append(item)
            else:
                diagnostics.append(
                    _reject_incoherent_item(
                        item,
                        reason="row outside decomposition geo/measure/year",
                    )
                )
        usable = kept

    # Group usable rows by SQL for optional ranking consolidation.
    by_sql: dict[str, list[dict[str, Any]]] = {}
    for item in usable:
        sql = _s((item.get("metadata") or {}).get("sql")) or "__no_sql__"
        by_sql.setdefault(sql, []).append(item)

    enriched: list[dict[str, Any]] = []
    for sql, group in by_sql.items():
        table_id = _table_from_sql(sql if sql != "__no_sql__" else "")
        if not table_id and selected:
            table_id = selected[0]
        consolidated = _consolidate_point_fact_batch(
            group,
            table_id=table_id,
            decomposition=dec,
            task_mode=str(plan.get("task_mode") or "").strip().lower(),
        )
        if consolidated:
            enriched.append(consolidated)
            continue
        consolidated = _consolidate_ranking_batch(
            group,
            query=query,
            table_id=table_id,
            decomposition=dec,
        )
        if consolidated:
            enriched.append(consolidated)
            continue
        for item in group:
            enriched.append(_enrich_single_item(item, table_id=table_id, decomposition=dec))

    return enriched


__all__ = [
    "enrich_bq_results",
    "format_row_prose",
    "resolve_row_semantics",
]
