"""
Deterministic ACF provenance fields for OpenTrace chunks and BQ rows.

OpenTrace Qdrant ``geo_scope`` is a coverage class
(``country|multi_country|regional|global|unknown``). ACF ``geo_scope`` is a
set of place names — never pass the keyword enum into ``from_payload``.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from ml.rag.text_processors.normalize_dates import normalize_to_iso_date

# Coverage-class keyword used in Qdrant payloads (not ACF place-name geo_scope).
_COVERAGE_GLOBAL = frozenset({"global"})
_COVERAGE_MULTI = frozenset({"multi_country", "regional"})
_COVERAGE_COUNTRY = frozenset({"country"})

_ADMIN_KEYS = (
    "region",
    "admin_1",
    "admin1",
    "adm1_name",
    "geographic_unit_name",
    "admin_region",
    "fewsnet_region",
    "admin_2",
    "admin2",
    "adm2_name",
)

_COUNTRY_KEYS = (
    "geo_country_primary",
    "country",
    "country_name",
    "area",
    "adm0_name",
    "admin_0",
    "reporting_country",
)

_LOCAL_MARKERS = re.compile(
    r"\b(ward|village|kebele|parish|settlement|community)\b",
    re.IGNORECASE,
)

_DATE_KEYS = (
    "as_of_date",
    "published_at",
    "reporting_date",
    "period_date",
    "projection_start",
    "first_period_date",
)

_YEAR_KEYS = ("publication_year", "year", "harvest_year", "planting_year", "mp_year", "TIME_PERIOD")

_METRIC_KEYS = (
    "metric",
    "Measure",
    "measure",
    "indicator",
    "element",
    "item",
    "product",
    "Commodity",
    "cm_name",
)

_UNIT_KEYS = ("unit", "Unit of measure", "UNIT_Short", "unit_of_measure")

_VALUE_KEYS = ("value", "OBS_VALUE", "yield", "production", "mp_price", "ipc_phase_value")


def _s(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _split_countries(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set, frozenset)):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = _s(raw)
    if not text:
        return []
    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip()]
    if "," in text and len(text) > 3:
        return [p.strip() for p in text.split(",") if p.strip()]
    return [text]


def coverage_class(meta: dict[str, Any]) -> str:
    """Return OpenTrace coverage keyword (not ACF place geo_scope)."""
    raw = _s(meta.get("geo_scope")).lower()
    if raw in _COVERAGE_GLOBAL | _COVERAGE_MULTI | _COVERAGE_COUNTRY | {"unknown"}:
        return raw
    countries = _split_countries(meta.get("geo_countries") or meta.get("country"))
    if len(countries) >= 5:
        return "global"
    if len(countries) > 1:
        return "multi_country"
    if len(countries) == 1:
        return "country"
    return "unknown"


def place_names(meta: dict[str, Any]) -> list[str]:
    """Place names for ACF pattern agreement (adapter ``geo_scope``)."""
    places: list[str] = []
    for key in _ADMIN_KEYS:
        val = _s(meta.get(key))
        if val and val not in places:
            places.append(val)
    for c in _split_countries(
        meta.get("geo_countries")
        or meta.get("geo_country_primary")
        or meta.get("country")
        or meta.get("country_name")
        or meta.get("area")
        or meta.get("adm0_name")
    ):
        if c not in places:
            places.append(c)
    return places


def derive_tier_and_data_level(meta: dict[str, Any]) -> tuple[int, str]:
    """Geo-scale ACF tier (1–3) and data_level from coverage + region."""
    if meta.get("tier") is not None and meta.get("data_level"):
        try:
            t = int(meta["tier"])
            dl = _s(meta.get("data_level")).lower()
            if t in (1, 2, 3) and dl in ("global", "national", "sub_national", "community"):
                return t, dl
        except (TypeError, ValueError):
            pass

    region = ""
    for key in _ADMIN_KEYS:
        region = _s(meta.get(key))
        if region:
            break

    cov = coverage_class(meta)
    countries = _split_countries(meta.get("geo_countries") or meta.get("country"))

    if region and _LOCAL_MARKERS.search(region):
        return 3, "community"
    if region:
        return 3, "sub_national"

    if cov in _COVERAGE_GLOBAL or (cov == "multi_country" and len(countries) >= 5):
        return 1, "global"
    if cov in _COVERAGE_MULTI:
        return 1 if len(countries) >= 3 else 2, "global" if len(countries) >= 3 else "national"
    if cov in _COVERAGE_COUNTRY or len(countries) == 1:
        return 2, "national"
    return 2, "national"


def _parse_iso_date(raw: Any) -> str | None:
    """Parse date-like values to ISO ``YYYY-MM-DD`` (RFC-822 OK; truncated → None)."""
    return normalize_to_iso_date(raw)


def derive_as_of_date(meta: dict[str, Any]) -> str | None:
    if meta.get("as_of_date") is not None:
        parsed = _parse_iso_date(meta.get("as_of_date"))
        if parsed:
            return parsed
    for key in _DATE_KEYS:
        parsed = _parse_iso_date(meta.get(key))
        if parsed:
            return parsed
    year = meta.get("publication_year")
    for key in _YEAR_KEYS:
        if meta.get(key) is not None:
            year = meta.get(key)
            break
    if year is not None:
        y = _s(year)
        # TIME_PERIOD sometimes "2024-Q1" or "2024"
        ym = re.match(r"(\d{4})", y)
        if ym:
            month = 1
            mp_month = meta.get("mp_month")
            if mp_month is not None:
                try:
                    month = max(1, min(12, int(mp_month)))
                except (TypeError, ValueError):
                    month = 1
            return f"{ym.group(1)}-{month:02d}-01"
    return None


def derive_source_id(meta: dict[str, Any], *, fallback_prefix: str = "chunk") -> str:
    for key in ("source_id", "document_id", "doi", "ota_record_id", "bq_table_id", "url"):
        val = _s(meta.get(key))
        if val:
            return val[:256]
    # Stable hash of remaining identity-ish fields
    grain = "|".join(
        _s(meta.get(k))
        for k in ("title", "article_title", "table_name", "sql_index", "chunk_index")
        if _s(meta.get(k))
    )
    if not grain:
        grain = str(sorted((k, _s(v)) for k, v in meta.items() if k not in ("content", "sql")))[:500]
    digest = hashlib.sha256(grain.encode("utf-8")).hexdigest()[:24]
    return f"{fallback_prefix}:{digest}"


def derive_metric(meta: dict[str, Any]) -> str:
    for key in _METRIC_KEYS:
        val = _s(meta.get(key))
        if val:
            return val[:128]
    domains = meta.get("domains")
    if isinstance(domains, str) and domains.strip():
        first = domains.split(";")[0].strip() or domains.split(",")[0].strip()
        if first:
            return first[:128]
    if isinstance(domains, (list, tuple)) and domains:
        return _s(domains[0])[:128] or "general"
    return "general"


def derive_unit(meta: dict[str, Any]) -> str | None:
    for key in _UNIT_KEYS:
        val = _s(meta.get(key))
        if val:
            return val[:64]
    return None


def _first_numeric(meta: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in meta or meta[key] is None:
            continue
        try:
            return float(meta[key])
        except (TypeError, ValueError):
            continue
    return None


def enrich_acf_payload_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """Stamp tier / data_level / as_of_date / region / source_id onto metadata (ingest)."""
    out = dict(meta or {})
    # Repair or clear garbage published_at before as_of derivation.
    if "published_at" in out:
        pub_raw = out.get("published_at")
        if pub_raw is not None and _s(pub_raw):
            pub_norm = normalize_to_iso_date(pub_raw)
            if pub_norm:
                out["published_at"] = pub_norm
            else:
                out.pop("published_at", None)
        else:
            out.pop("published_at", None)
    # Clear invalid as_of_date so derive can fall through to other keys.
    if "as_of_date" in out and out.get("as_of_date") is not None:
        if not normalize_to_iso_date(out.get("as_of_date")):
            out.pop("as_of_date", None)
    tier, data_level = derive_tier_and_data_level(out)
    out["tier"] = tier
    out["data_level"] = data_level
    as_of = derive_as_of_date(out)
    if as_of:
        out["as_of_date"] = as_of
    elif "as_of_date" in out:
        out.pop("as_of_date", None)
    # region: keep existing or promote first admin place
    if not _s(out.get("region")):
        for key in _ADMIN_KEYS:
            if key == "region":
                continue
            val = _s(out.get(key))
            if val:
                out["region"] = val
                break
    out["source_id"] = derive_source_id(out)
    return out


def context_item_to_acf_record(item: dict[str, Any]) -> dict[str, Any] | None:
    """
    Build a dict suitable for ``acf.from_payload`` / ``from_row``.

    Skips records missing ``as_of_date`` (cannot score without freshness).
    Prefers ingested ``metric`` / ``direction`` / ``magnitude`` / ``unit`` from
    claim extraction; prose without D/M gets ``direction=unknown``.
    Remaps OpenTrace coverage-class ``geo_scope`` → ACF place-name list.
    ``finding`` stays on metadata for audit — not passed to ``from_payload``.
    """
    meta = dict(item.get("metadata") or {})
    # Merge top-level ACF fields if present (BQ projection may set them).
    for key in (
        "tier",
        "data_level",
        "as_of_date",
        "region",
        "source_id",
        "metric",
        "direction",
        "magnitude",
        "unit",
        "value",
        "prior_value",
        "coverage_strength",
        "finding",
        "bq_enrichment",
        "ranked_rows",
        "trend_mixed",
    ):
        if item.get(key) is not None and meta.get(key) is None:
            meta[key] = item[key]

    # Ranked-table BQ items: use rank-1 country and stamped trend fields.
    if meta.get("bq_enrichment") == "ranked_table":
        ranked_rows = meta.get("ranked_rows")
        if isinstance(ranked_rows, list) and ranked_rows:
            top = ranked_rows[0]
            if isinstance(top, dict):
                label = _s(top.get("label"))
                if label:
                    meta.setdefault("geo_country_primary", label)
                    meta.setdefault("geo_countries", label)
                if meta.get("value") is None and top.get("value") is not None:
                    meta["value"] = top["value"]
                if not _s(meta.get("unit")):
                    unit_top = _s(top.get("unit"))
                    if unit_top:
                        meta["unit"] = unit_top
            if meta.get("coverage_strength") is None:
                meta["coverage_strength"] = min(1.0, len(ranked_rows) / 10.0)
        sem = meta.get("value_semantics")
        if isinstance(sem, dict):
            ml = _s(sem.get("measure_label"))
            if ml and (not _s(meta.get("metric")) or meta.get("metric") == "general"):
                meta["metric"] = ml

    as_of = derive_as_of_date(meta)
    if not as_of:
        return None

    tier, data_level = derive_tier_and_data_level(meta)
    places = place_names(meta)

    # Prefer ingested metric; fall back to domain heuristics
    ingested_metric = _s(meta.get("metric"))
    metric = ingested_metric if ingested_metric else derive_metric(meta)

    record: dict[str, Any] = {
        "tier": tier,
        "data_level": data_level,
        "as_of_date": as_of,
        "metric": metric,
        "source_id": derive_source_id(meta),
        # ACF place-name geo_scope (list), NOT coverage keyword
        "geo_scope": places,
    }
    region = _s(meta.get("region"))
    if region:
        record["region"] = region

    unit = _s(meta.get("unit")) or derive_unit(meta)
    if unit:
        record["unit"] = unit

    if meta.get("coverage_strength") is not None:
        try:
            record["coverage_strength"] = float(meta["coverage_strength"])
        except (TypeError, ValueError):
            pass

    # D/M: prefer ingested structured fields (do not overwrite with unknown)
    ingested_direction = _s(meta.get("direction") or meta.get("trend")).lower()
    if ingested_direction in ("increasing", "decreasing", "stable", "unknown"):
        record["direction"] = ingested_direction
        if meta.get("magnitude") is not None:
            try:
                record["magnitude"] = float(meta["magnitude"])
            except (TypeError, ValueError):
                pass
    elif meta.get("value") is not None:
        record["value"] = meta["value"]
        if meta.get("prior_value") is not None:
            record["prior_value"] = meta["prior_value"]
    elif meta.get("magnitude") is not None:
        try:
            record["magnitude"] = float(meta["magnitude"])
        except (TypeError, ValueError):
            record["direction"] = "unknown"
    else:
        val = _first_numeric(meta, _VALUE_KEYS)
        if val is not None and meta.get("prior_value") is not None:
            record["value"] = val
            record["prior_value"] = meta["prior_value"]
        else:
            record["direction"] = "unknown"

    return record


def project_bq_row_acf(row: dict[str, Any], *, table_hint: str | None = None) -> dict[str, Any]:
    """Project a BigQuery row dict into ACF-oriented metadata fields (merged into item metadata)."""
    meta = dict(row)
    # Infer table from SQL if present
    table = table_hint or ""
    sql = _s(meta.get("sql"))
    if not table and sql:
        m = re.search(r"FROM\s+`?(?:[\w-]+\.)*([\w-]+)`?", sql, re.IGNORECASE)
        if m:
            table = m.group(1)

    admin = ""
    for key in _ADMIN_KEYS:
        admin = _s(meta.get(key))
        if admin:
            meta.setdefault("region", admin)
            break

    country = ""
    for key in _COUNTRY_KEYS:
        country = _s(meta.get(key))
        if country:
            break
    if country:
        meta.setdefault("geo_country_primary", country)
        meta.setdefault("geo_countries", country)
        meta.setdefault("geo_scope", "country" if not admin else "country")

    if admin:
        meta["tier"] = 3
        meta["data_level"] = "community" if _LOCAL_MARKERS.search(admin) else "sub_national"
    elif country:
        meta["tier"] = 2
        meta["data_level"] = "national"
    else:
        meta["tier"] = 1
        meta["data_level"] = "global"
        meta.setdefault("geo_scope", "global")

    as_of = derive_as_of_date(meta)
    if as_of:
        meta["as_of_date"] = as_of

    metric = derive_metric(meta)
    meta["metric"] = metric
    unit = derive_unit(meta)
    if unit:
        meta["unit"] = unit

    # Grain key for durable source_id
    grain_parts = [table or "bq"]
    for key in ("country", "country_name", "area", "region", "geographic_unit_name", "product", "item", "year", "harvest_year", "TIME_PERIOD", "mp_year", "mp_month"):
        val = _s(meta.get(key))
        if val:
            grain_parts.append(f"{key}={val}")
    meta["source_id"] = ":".join(grain_parts)[:256]

    # value / prior_value when both present under common names
    val = _first_numeric(meta, _VALUE_KEYS)
    if val is not None:
        meta.setdefault("value", val)
    if meta.get("prior_value") is None:
        for key in ("prior_value", "previous_value", "prior_year_value"):
            if meta.get(key) is not None:
                try:
                    meta["prior_value"] = float(meta[key])
                except (TypeError, ValueError):
                    pass
                break

    if meta.get("direction") is None and meta.get("value") is not None and meta.get("prior_value") is None:
        # Single snapshot — unknown direction (prose-equivalent)
        meta["direction"] = "unknown"

    return meta
