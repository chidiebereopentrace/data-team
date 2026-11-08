"""Per-table YAML loader for BigQuery semantic schemas.

Mart_dev ``fct_*`` / ``agg_*`` YAMLs under ``bq_mart_tables_yaml_files/`` are the sole
table catalog for Ask ADZA NL-to-SQL. Staging ``stg_*`` YAMLs remain for reference/tests.

Public API:
- ``load_mart_table_schema(name)``       -> mart_dev YAML dict.
- ``format_table_schema(name, loader=...)`` -> compact SQL-prompt block.
- ``list_mart_table_index()``            -> compact index rows for the SQL reasoner.
- ``format_mart_reasoner_index(...)``      -> byte-capped mart index for LLM prompt.
- ``pack_mart_table_hints(...)``         -> byte-capped full YAML packs for NL2SQL.
- ``columns_for_mart_tables(...)``       -> YAML column names per table (SQL allowlist).
- ``value_samples_for_mart_tables(...)`` -> enum/sample labels per column.
- Staging helpers (``load_table_schema``, ``list_staging_table_index``, …) retained for tests.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ml.rag.chatbot.agri_measure_ontology import entity_is_measure_noise
from ml.rag.chatbot.bq_byte_budget import hint_max_bytes, pack_lines, reasoner_index_max_bytes, truncate_utf8, utf8_len
from ml.rag.chatbot.mart_indicator_classes import (
    families_for_fact,
    indicator_classes_for_table,
)
from ml.rag.helpers.mart_semantic_relationships import SEMANTIC_RELATIONSHIPS
from ml.rag.helpers.mart_semantic_relationships import compact_rels_summary as mart_compact_rels_summary
from ml.rag.helpers.mart_semantic_relationships import format_join_fragments_for_nl2sql as mart_join_fragments
from ml.rag.helpers.staging_semantic_relationships import compact_rels_summary
from ml.rag.mart_yaml_contract import dim_expose_columns

# bq_table_schema_yaml.py lives at ml/rag/chatbot/, YAMLs live at ml/rag/bq_tables_yaml_files/.
_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "bq_tables_yaml_files"
_DEFAULT_MART_DIR = Path(__file__).resolve().parents[1] / "bq_mart_tables_yaml_files"

_INDEX_RECHECK_S = 2.0
# directory -> (file signature, last check, index)
_index_cache: dict[str, tuple[tuple[tuple[str, int], ...], float, dict[str, dict[str, Any]]]] = {}


def _yaml_dir() -> Path:
    raw = os.environ.get("RAG_BQ_TABLES_YAML_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_DIR.resolve()


def _mart_yaml_dir() -> Path:
    raw = os.environ.get("RAG_BQ_MART_TABLES_YAML_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_MART_DIR.resolve()


def _strip_fqn(table_name: str) -> str:
    """Return the bare table name (last dotted segment) from a possibly fully-qualified id."""
    text = (table_name or "").strip().strip("`")
    if not text:
        return ""
    return text.split(".")[-1]


_MART_TABLE_PREFIXES = ("fct_", "agg_", "dim_", "bridge_")


def _is_mart_table_id(table_id: str) -> bool:
    bare = _strip_fqn(table_id).lower()
    return bare.startswith(_MART_TABLE_PREFIXES)


def _schema_loader_for(table_id: str):
    if _is_mart_table_id(table_id):
        return load_mart_table_schema
    return load_table_schema


def _index_yaml_files(directory: Path) -> dict[str, dict[str, Any]]:
    """Build name -> table_schema_dict index, keyed by both bare and fully-qualified names."""
    out: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return out
    for path in directory.glob("*.yml"):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        bare = path.stem
        # File-name key is authoritative; the explicit table_name field (often a FQN) is an alias.
        out[bare] = data
        declared = str(data.get("table_name") or "").strip().strip("`")
        if declared:
            out[declared] = data
            short = _strip_fqn(declared)
            if short and short != bare:
                out[short] = data
    return out


def _file_signature(directory: Path) -> tuple[tuple[str, int], ...]:
    try:
        return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in directory.glob("*.yml")))
    except OSError:
        return tuple()


def _cached_index(directory: Path) -> dict[str, dict[str, Any]]:
    """YAML index for a directory. File mtimes are re-checked at most every few seconds,
    so edits are picked up without stat-ing every file on each schema lookup."""
    key = str(directory)
    now = time.monotonic()
    cached = _index_cache.get(key)
    if cached is not None and now - cached[1] < _INDEX_RECHECK_S:
        return cached[2]
    signature = _file_signature(directory) if directory.is_dir() else tuple()
    if cached is not None and cached[0] == signature:
        _index_cache[key] = (signature, now, cached[2])
        return cached[2]
    index = _index_yaml_files(directory)
    _index_cache[key] = (signature, now, index)
    return index


def _build_index() -> dict[str, dict[str, Any]]:
    return _cached_index(_yaml_dir())


def known_table_names() -> set[str]:
    """All names (bare + FQN aliases) for which a YAML schema is available."""
    return set(_build_index().keys())


def load_table_schema(table_name: str) -> dict[str, Any] | None:
    """Resolve a table name to its raw YAML dict (trying FQN, bare, file-stem)."""
    name = (table_name or "").strip()
    if not name:
        return None
    index = _build_index()
    if name in index:
        return index[name]
    bare = _strip_fqn(name)
    if bare and bare in index:
        return index[bare]
    return None


def _build_mart_index() -> dict[str, dict[str, Any]]:
    return _cached_index(_mart_yaml_dir())


def known_mart_table_names() -> set[str]:
    """All names (bare + FQN aliases) for which a mart YAML schema is available."""
    return set(_build_mart_index().keys())


def load_mart_table_schema(table_name: str) -> dict[str, Any] | None:
    """Resolve a mart table name to its raw YAML dict."""
    name = (table_name or "").strip()
    if not name:
        return None
    index = _build_mart_index()
    if name in index:
        return index[name]
    bare = _strip_fqn(name)
    if bare and bare in index:
        return index[bare]
    return None


def bind_spine_map(table_id: str) -> dict[str, str]:
    """FK column → dim_table.column join targets from mart YAML bind_spine block."""
    schema = load_mart_table_schema(table_id) or {}
    raw = schema.get("bind_spine")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for fk_col, dim_ref in raw.items():
        fk = str(fk_col or "").strip()
        ref = str(dim_ref or "").strip()
        if fk and ref and "." in ref:
            out[fk] = ref
    return out


_GRAIN_TOKEN_COLUMNS: dict[str, tuple[str, ...]] = {
    "country": ("country_iso3", "country_name", "geography_key", "geo_key", "area"),
    "area": ("country_iso3", "country_name", "geography_key", "geo_key", "area"),
    "geography": ("geography_key", "geo_key", "country_iso3"),
    "product": ("product_key", "product_name", "item"),
    "commodity": ("product_key", "product_name", "commodity", "item"),
    "crop": ("product_key", "product_name", "crop", "item"),
    "item": ("item", "item_key", "item_name", "product_key"),
    "year": ("year", "time_key", "harvest_year", "observation_year", "mp_year"),
    "month": ("month", "as_of_date"),
    "date": ("as_of_date", "date"),
    "source": ("source_key", "price_source", "source"),
    "market": ("market_key", "market_name", "market"),
    "element": ("element", "element_key"),
    "indicator": ("indicator", "indicator_key", "indicator_name"),
    "sex": ("sex",),
    "measure": ("measure_type",),
    "season": ("season", "season_key"),
    "admin": ("admin_1", "admin_1_name"),
    "region": ("admin_1", "admin_1_name", "region"),
}
_GRAIN_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]*")


def grain_columns(table_id: str) -> list[str]:
    """Columns that together identify one row: YAML ``grain_columns`` or the parsed ``grain`` line."""
    schema = load_mart_table_schema(table_id) or load_table_schema(table_id) or {}
    by_low = {
        str(c.get("name") or "").strip().lower(): str(c.get("name") or "").strip()
        for c in _schema_columns(schema)
        if str(c.get("name") or "").strip()
    }
    explicit = schema.get("grain_columns")
    if isinstance(explicit, list):
        return [by_low[str(c).strip().lower()] for c in explicit if str(c).strip().lower() in by_low]
    out: list[str] = []
    for token in _GRAIN_TOKEN_RE.findall(str(schema.get("grain") or "").lower()):
        for cand in (token, *_GRAIN_TOKEN_COLUMNS.get(token, ())):
            if cand in by_low:
                if by_low[cand] not in out:
                    out.append(by_low[cand])
                break
    return out


def column_units(table_id: str) -> dict[str, str]:
    """Measure column → unit for tables without a unit column (YAML ``column_units``)."""
    schema = load_mart_table_schema(table_id) or load_table_schema(table_id) or {}
    raw = schema.get("column_units")
    if not isinstance(raw, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip() and str(v).strip()}


def with_column_units(row: dict[str, Any], table_ids: set[str]) -> dict[str, Any]:
    """Row plus ``<measure>_unit`` for each declared unit the row does not already carry."""
    out = dict(row)
    for table_id in sorted(table_ids):
        for col, unit in column_units(table_id).items():
            key = f"{col}_unit"
            if col in out and key not in out:
                out[key] = unit
    return out


_DIM_LABEL_COL = {
    "dim_product": "product_name",
    "dim_source": "organisation_name",
    "dim_market": "market_name",
    "dim_geography": "country_name",
    "dim_element": "element_name",
    "dim_unit": "unit_code",
    "dim_season": "season_name",
    "dim_indicator": "indicator_name",
    "dim_item": "item_name",
}


def dim_label_joins(table_id: str) -> dict[str, tuple[str, str, str]]:
    """FK column → (dim table, dim key column, display label) from ``bind_spine``."""
    out: dict[str, tuple[str, str, str]] = {}
    for fk, ref in bind_spine_map(table_id).items():
        dim, filter_col = ref.split(".", 1)
        key = _FK_DIM_SELECT_COL.get((fk.lower(), dim.lower()), fk)
        label = _DIM_LABEL_COL.get(dim.lower(), filter_col)
        out[fk] = (dim, key, label)
    return out


# Legacy facts use geo_key as FK to dim_geography.geography_key (same hash).
_FK_DIM_SELECT_COL: dict[tuple[str, str], str] = {
    ("geo_key", "dim_geography"): "geography_key",
}

_GEO_SUBNATIONAL_SAMPLE_COLS: tuple[str, ...] = (
    "admin_1_name",
    "admin_2_name",
    "city_name",
)


def _compile_fk_spine_filter_sql(
    table_id: str,
    *,
    fk_col: str,
    dim_ref: str,
    literals: list[str],
    project_id: str,
    dataset: str,
    extra_predicates: list[tuple[str, str]] | None = None,
) -> str:
    dim_table, dim_col = dim_ref.split(".", 1)
    dim_table = dim_table.strip()
    dim_fqn = f"`{project_id}.{dataset}.{dim_table}`"
    fk = fk_col.strip()
    dcol = dim_col.strip()
    select_col = _FK_DIM_SELECT_COL.get((fk.lower(), dim_table.lower()), fk)
    where_parts: list[str] = []
    if literals:
        if len(literals) == 1:
            where_parts.append(f"{dcol} = {_sql_literal(literals[0])}")
        else:
            lits = ", ".join(_sql_literal(v) for v in literals[:32])
            where_parts.append(f"{dcol} IN ({lits})")
    for ecol, eval_ in extra_predicates or []:
        col = str(ecol or "").strip()
        val = str(eval_ or "").strip()
        if col and val:
            where_parts.append(f"{col} = {_sql_literal(val)}")
    if not where_parts:
        return ""
    where_sql = " AND ".join(where_parts)
    return f"AND {fk} IN (SELECT {select_col} FROM {dim_fqn} WHERE {where_sql}) "


def match_subnational_geo_labels(blob: str) -> list[tuple[str, str]]:
    """Speech-match admin1/admin2/city samples on dim_geography (never invent)."""
    text = (blob or "").strip()
    if not text:
        return []
    out: list[tuple[str, str]] = []
    seen_cols: set[str] = set()
    for col in _GEO_SUBNATIONAL_SAMPLE_COLS:
        samples = column_samples_for_table("dim_geography", col)
        if not samples:
            continue
        matched = match_value_samples(text, samples)
        for cand in matched:
            cand_s = str(cand).strip()
            if not cand_s:
                continue
            # Skip ultra-short single-token place names (Northern/Eastern noise).
            toks = _alnum_tokens(cand_s)
            if len(toks) < 2 and len(cand_s) <= 8:
                blob_tokens = set(_alnum_tokens(expand_speech_text(text)))
                if not toks or toks[0] not in blob_tokens:
                    continue
                if len(toks[0]) < 4:
                    continue
            out.append((col, cand_s))
            seen_cols.add(col)
            break
        if col in seen_cols:
            continue
    return out


def match_geo_level_label(blob: str) -> str | None:
    """Match geo_level sample when speech names a place grain."""
    samples = column_samples_for_table("dim_geography", "geo_level")
    if not samples:
        return None
    matched = match_value_samples(blob or "", samples)
    for cand in matched:
        cand_s = str(cand).strip()
        if cand_s:
            return cand_s
    # Synonym grains not identical to sample tokens.
    low = expand_speech_text(blob or "")
    synonym_map = (
        ("admin 1", "admin1"),
        ("admin1", "admin1"),
        ("province", "admin1"),
        ("region", "admin1"),
        ("admin 2", "admin2"),
        ("admin2", "admin2"),
        ("district", "admin2"),
        ("city", "city"),
        ("town", "city"),
        ("fnid", "fnid"),
    )
    sample_by_low = {str(s).strip().lower(): str(s).strip() for s in samples if str(s).strip()}
    for trigger, level in synonym_map:
        if re.search(rf"\b{re.escape(trigger)}\b", low):
            hit = sample_by_low.get(level)
            if hit:
                return hit
    return None


# FKs already bound by geo / product / lineage paths — not entity spine matching.
_SPINE_ENTITY_SKIP_FKS = frozenset({"geography_key", "geo_key", "source_key", "product_key"})

# Only these FKs auto-bind from speech (blocks noisy unit/sex/household/org/person keys).
_SPINE_ENTITY_ALLOW_FKS = frozenset(
    {
        "land_use_key",
        "disease_key",
        "pest_key",
        "item_key",
        "element_key",
        "season_key",
        "classification_key",
        "scenario_key",
        "livestock_key",
        "market_key",
        "soil_property_key",
    }
)

# Typed IR entity roles → bind_spine FK (Phase 5).
_ENTITY_ROLE_TO_FK: dict[str, str] = {
    "land_use": "land_use_key",
    "hazard": "disease_key",
    "item": "item_key",
    "pest": "pest_key",
    "season": "season_key",
    "livestock": "livestock_key",
}


def _short_token_sample_ok(sample: str, blob: str) -> bool:
    """Short single-token samples (≤3 chars) need exact token equality in blob."""
    cand_s = (sample or "").strip()
    toks = _alnum_tokens(cand_s)
    if len(toks) >= 2 or len(cand_s) > 3:
        return True
    blob_tokens = set(_alnum_tokens(expand_speech_text(blob)))
    return bool(toks) and toks[0] in blob_tokens and cand_s.lower() in blob_tokens


def _preferred_labels_from_entity_roles(
    entity_roles: dict[str, list[str]] | list[tuple[str, list[str]]] | None,
) -> dict[str, list[str]]:
    """Map role → labels into FK → labels for spine prefer path."""
    out: dict[str, list[str]] = {}
    if not entity_roles:
        return out
    items: list[tuple[str, list[str]]]
    if isinstance(entity_roles, dict):
        items = [(str(k), list(v) if isinstance(v, (list, tuple)) else []) for k, v in entity_roles.items()]
    else:
        items = [(str(k), list(v) if isinstance(v, (list, tuple)) else []) for k, v in entity_roles]
    for role, labels in items:
        fk = _ENTITY_ROLE_TO_FK.get(str(role).strip().lower())
        if not fk:
            continue
        cleaned = [str(x).strip() for x in labels if str(x).strip()]
        if cleaned:
            out[fk] = cleaned
    return out


def compile_spine_entity_filters(
    table_id: str,
    *,
    blob: str,
    project_id: str,
    dataset: str,
    skip_fks: set[str] | frozenset[str] | None = None,
    consumed_labels: list[str] | None = None,
    entity_roles: dict[str, list[str]] | list[tuple[str, list[str]]] | None = None,
) -> list[tuple[str, str, str]]:
    """Match speech (or typed entity_roles) to bind_spine dim samples; emit FK subquery filters.

    Returns ``(fk_col, matched_label, sql_fragment)`` — at most one match per FK.
    Never invents literals: only YAML dim ``*_value_samples``.
    """
    bare = _strip_fqn(table_id).lower()
    spine = bind_spine_map(bare)
    preferred_by_fk = _preferred_labels_from_entity_roles(entity_roles)
    if not spine:
        return []
    if not (blob or "").strip() and not preferred_by_fk:
        return []

    skip = set(_SPINE_ENTITY_SKIP_FKS)
    if skip_fks:
        skip |= {str(x).strip().lower() for x in skip_fks if str(x).strip()}

    consumed_low = {
        str(x).strip().lower() for x in (consumed_labels or []) if str(x).strip()
    }
    consumed_tokens: set[str] = set()
    for lab in consumed_low:
        consumed_tokens.update(_alnum_tokens(lab))

    fact_cols = {name.lower() for name in _column_names(bare)}
    out: list[tuple[str, str, str]] = []
    for fk_col, dim_ref in sorted(spine.items()):
        fk = str(fk_col or "").strip()
        ref = str(dim_ref or "").strip()
        fk_l = fk.lower()
        if not fk or fk_l in skip or "." not in ref:
            continue
        if fk_l not in _SPINE_ENTITY_ALLOW_FKS or fk_l not in fact_cols:
            continue
        dim_table, dim_col = ref.split(".", 1)
        dim_table = dim_table.strip()
        dim_col = dim_col.strip()
        if not dim_table or not dim_col:
            continue
        if fk_l.removesuffix("_key") in fact_cols or dim_col.lower() in fact_cols:
            continue
        samples = column_samples_for_table(dim_table, dim_col)
        if not samples:
            continue
        sample_by_low = {str(s).strip().lower(): str(s).strip() for s in samples if str(s).strip()}

        label: str | None = None
        preferred = preferred_by_fk.get(fk) or preferred_by_fk.get(fk_l)
        if preferred:
            for raw in preferred:
                hit = sample_by_low.get(str(raw).strip().lower())
                if hit and hit.lower() not in consumed_low:
                    label = hit
                    break
            if not label:
                matched_pref = match_value_samples(" ".join(preferred), samples)
                for cand in matched_pref:
                    cand_s = str(cand).strip()
                    if cand_s and cand_s.lower() not in consumed_low:
                        label = cand_s
                        break

        if not label and (blob or "").strip():
            matched = match_value_samples(blob, samples)
            for cand in matched:
                cand_s = str(cand).strip()
                if not cand_s:
                    continue
                if cand_s.lower() in consumed_low:
                    continue
                cand_toks = set(_alnum_tokens(cand_s))
                if cand_toks and cand_toks <= consumed_tokens:
                    continue
                if not _short_token_sample_ok(cand_s, blob):
                    continue
                label = cand_s
                break
        if not label:
            continue
        sql = _compile_fk_spine_filter_sql(
            bare,
            fk_col=fk,
            dim_ref=ref,
            literals=[label],
            project_id=project_id,
            dataset=dataset,
        )
        if not sql.strip():
            continue
        out.append((fk, label, sql.strip()))
        consumed_low.add(label.lower())
        consumed_tokens.update(_alnum_tokens(label))
    return out


CORE_METRIC_DISCRIMINATORS = frozenset(
    {
        "element",
        "indicator",
        "price_type",
        "measure_type",
        "treatment",
        "metric",
        "production_grain",
        "price_source",
        "trade_grain",
        "climate_grain",
        "input_grain",
        "vegetation_grain",
    }
)
GRAIN_METRIC_DISCRIMINATORS = frozenset(
    {
        "classification_scale",
        "scenario_name",
    }
)
_CORE_METRIC_DISCRIMINATORS = CORE_METRIC_DISCRIMINATORS
_GRAIN_METRIC_DISCRIMINATORS = GRAIN_METRIC_DISCRIMINATORS

_NUMERIC_COLUMN_TYPES = frozenset({"INT64", "FLOAT64", "NUMERIC", "BIGNUMERIC", "INTEGER", "FLOAT"})
_MEASURE_SKIP_COLUMNS = frozenset(
    {
        "year",
        "month",
        "planting_year",
        "harvest_year",
        "observation_year",
        "planting_month",
        "harvest_month",
        "mp_year",
        "mp_month",
        "qc_flag",
        "hh_size",
        "individual_count",
        "latitude",
        "longitude",
        "fnid",
        "country_code",
        "area_code",
        "item_code",
        "objectid",
        "record_count",
    }
)


def _schema_columns(schema: dict[str, Any]) -> list[dict[str, Any]]:
    cols_raw = schema.get("columns")
    if not isinstance(cols_raw, list):
        return []
    return [c for c in cols_raw if isinstance(c, dict)]


def column_description(table_id: str, column: str) -> str:
    """Return trimmed YAML ``columns[].description`` for a physical column name."""
    schema = load_table_schema(table_id)
    if not schema:
        return ""
    col_name = (column or "").strip()
    if not col_name:
        return ""
    for col in _schema_columns(schema):
        if str(col.get("name") or "").strip() == col_name:
            desc = col.get("description")
            if desc is None:
                return ""
            text = str(desc).strip()
            if not text:
                return ""
            return " ".join(text.split())[:400]
    return ""


# Explicit overrides when sample key stem ≠ physical column name.
_SAMPLE_KEY_OVERRIDES: dict[str, str] = {
    "product_value_samples": "product_name",
    "market_value_samples": "market_name",
    "item_value_samples": "item",
}

_SAMPLE_KEY_SUFFIX = "_value_samples"


def column_for_sample_key(sample_key: str) -> str:
    """Map YAML ``*_value_samples`` key → physical column name."""
    key = (sample_key or "").strip()
    if key in _SAMPLE_KEY_OVERRIDES:
        return _SAMPLE_KEY_OVERRIDES[key]
    if key.endswith(_SAMPLE_KEY_SUFFIX):
        return key[: -len(_SAMPLE_KEY_SUFFIX)]
    return key


def discriminator_columns(table_id: str) -> list[str]:
    """Physical columns with YAML ``*_value_samples`` (metric grain discriminators)."""
    schema = _schema_loader_for(table_id)(table_id)
    if not schema:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for sample_key in schema:
        if not isinstance(sample_key, str) or not sample_key.endswith(_SAMPLE_KEY_SUFFIX):
            continue
        col = column_for_sample_key(sample_key)
        if col and col not in seen:
            seen.add(col)
            out.append(col)
    return out


def measure_columns(table_id: str) -> list[str]:
    """Numeric measure columns excluding geo/time keys and metric discriminators."""
    if _is_mart_table_id(table_id):
        return measure_columns_mart(table_id)
    schema = load_table_schema(table_id)
    if not schema:
        return []
    disc = {c.lower() for c in discriminator_columns(table_id)}
    out: list[str] = []
    for col in _schema_columns(schema):
        name = str(col.get("name") or "").strip()
        if not name:
            continue
        low = name.lower()
        typ = str(col.get("type") or "").upper()
        if typ not in _NUMERIC_COLUMN_TYPES:
            continue
        if low in _MEASURE_SKIP_COLUMNS:
            continue
        if low in disc or low in _CORE_METRIC_DISCRIMINATORS or low in _GRAIN_METRIC_DISCRIMINATORS:
            continue
        out.append(name)
    return out


def measure_columns_mart(table_id: str) -> list[str]:
    schema = load_mart_table_schema(table_id)
    if not schema:
        return []
    disc = {c.lower() for c in discriminator_columns(table_id)}
    out: list[str] = []
    col_names = {str(c.get("name") or "").strip().lower() for c in _schema_columns(schema)}
    if "value" in col_names:
        return ["value"]
    for col in _schema_columns(schema):
        name = str(col.get("name") or "").strip()
        if not name:
            continue
        low = name.lower()
        typ = str(col.get("type") or "").upper()
        if typ not in _NUMERIC_COLUMN_TYPES:
            continue
        if low in _MEASURE_SKIP_COLUMNS or low.endswith("_key"):
            continue
        if low in disc or low in _CORE_METRIC_DISCRIMINATORS or low in _GRAIN_METRIC_DISCRIMINATORS:
            continue
        out.append(name)
    return out[:6]


_GEO_COLUMN_CANDIDATES = ("country_name", "country")
_MART_GEO_COLUMN_CANDIDATES = ("country_iso3", "country_name", "country")
_YEAR_COLUMN_CANDIDATES = ("year", "time_key", "harvest_year", "observation_year", "mp_year")

# Canonical African country names (aligned with query_decomposer) → ISO3 for country_iso3 filters.
_AFRICA_COUNTRY_ISO3: dict[str, str] = {
    "Algeria": "DZA",
    "Angola": "AGO",
    "Benin": "BEN",
    "Botswana": "BWA",
    "Burkina Faso": "BFA",
    "Burundi": "BDI",
    "Cabo Verde": "CPV",
    "Cameroon": "CMR",
    "Central African Republic": "CAF",
    "Chad": "TCD",
    "Comoros": "COM",
    "Republic of the Congo": "COG",
    "Democratic Republic of the Congo": "COD",
    "Djibouti": "DJI",
    "Egypt": "EGY",
    "Equatorial Guinea": "GNQ",
    "Eritrea": "ERI",
    "Eswatini": "SWZ",
    "Ethiopia": "ETH",
    "Gabon": "GAB",
    "Gambia": "GMB",
    "Ghana": "GHA",
    "Guinea": "GIN",
    "Guinea-Bissau": "GNB",
    "Côte d'Ivoire": "CIV",
    "Kenya": "KEN",
    "Lesotho": "LSO",
    "Liberia": "LBR",
    "Libya": "LBY",
    "Madagascar": "MDG",
    "Malawi": "MWI",
    "Mali": "MLI",
    "Mauritania": "MRT",
    "Mauritius": "MUS",
    "Morocco": "MAR",
    "Mozambique": "MOZ",
    "Namibia": "NAM",
    "Niger": "NER",
    "Nigeria": "NGA",
    "Rwanda": "RWA",
    "Sao Tome and Principe": "STP",
    "Senegal": "SEN",
    "Seychelles": "SYC",
    "Sierra Leone": "SLE",
    "Somalia": "SOM",
    "South Africa": "ZAF",
    "South Sudan": "SSD",
    "Sudan": "SDN",
    "Tanzania": "TZA",
    "Togo": "TGO",
    "Tunisia": "TUN",
    "Uganda": "UGA",
    "Zambia": "ZMB",
    "Zimbabwe": "ZWE",
}
_ISO3_RE = re.compile(r"^[A-Z]{3}$")

_SERIES_QUERY_RE = re.compile(
    r"\b(export|csv|chart|graph|plot|trend|time\s*series|over\s+time|by\s+year|"
    r"from\s+19|from\s+20|till\s+date|until\s+now)\b",
    re.IGNORECASE,
)
_RANK_QUERY_RE = re.compile(
    r"\b(highest|lowest|top|rank|ranking|most|least|which country)\b",
    re.IGNORECASE,
)
_YOY_QUERY_RE = re.compile(r"\b(yoy|year[\s-]over[\s-]year|year on year)\b", re.IGNORECASE)
_SHARE_QUERY_RE = re.compile(r"\b(share of|percent of|percentage of|proportion of)\b", re.IGNORECASE)
_PRODUCT_COLUMN_CANDIDATES = ("product_name", "product", "item")
_PATTERN_DENY_TABLES = frozenset(
    {
        "stg_fews_cross_border_trade",
        "stg_ilri_household_food_security",
    }
)
_PATTERN_DENY_GRAIN_RE = re.compile(
    r"household|border_point|lat/lon|\blat\b|plot_id|\bplot\b|germplasm|"
    r"grid_id|\bgrid\b|sensor|occurrence|farm/cow|\bfarm\b|respondent|"
    r"entity row|protected_area|study\s*×",
    re.IGNORECASE,
)
_AVG_SEMANTIC_RE = re.compile(
    r"per[_\s-]?capita|\brate\b|\bindex\b|\bshare\b|\byield\b|\bprice",
    re.IGNORECASE,
)
_SPEECH_SYNONYMS = {
    "corn": "maize",
    "peanut": "groundnut",
    "peanuts": "groundnuts",
    "soya": "soy",
    "soyabean": "soybean",
    "soyabeans": "soybeans",
}
_GENERIC_SAMPLE_TOKENS = frozenset(
    {
        "of",
        "and",
        "or",
        "the",
        "with",
        "from",
        "in",
        "on",
        "at",
        "to",
        "for",
        "by",
        "per",
        "a",
        "an",
        "is",
        "n",
        "e",
        "c",
        "other",
        "products",
        "nes",
        "nec",
    }
)
_AUTO_DEFAULT_DISCRIMINATOR_COLS = frozenset({"element", "price_type", "measure_type"})
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_YIELD_QUERY_RE = re.compile(r"\byields?\b", re.IGNORECASE)
_PRODUCTION_QUERY_RE = re.compile(
    r"\b(production|produced|output|tonnes?|tons?)\b",
    re.IGNORECASE,
)
_PRODUCER_PRICE_QUERY_RE = re.compile(r"\b(producer|farm\s*gate)\b", re.IGNORECASE)
_WHOLESALE_QUERY_RE = re.compile(r"\bwholesale\b", re.IGNORECASE)
_RETAIL_QUERY_RE = re.compile(r"\bretail\b", re.IGNORECASE)
_POPULATION_QUERY_RE = re.compile(r"\b(population|people|ipc)\b", re.IGNORECASE)
_CLASSIFICATION_QUERY_RE = re.compile(r"\b(classification|phase)\b", re.IGNORECASE)
_MEASURE_DISCRIMINATOR_COLS = frozenset(
    {
        "element",
        "metric",
        "production_grain",
        "price_type",
        "measure_type",
        "indicator",
        "trade_grain",
        "price_source",
    }
)


def _column_names(table_id: str) -> set[str]:
    loader = _schema_loader_for(table_id)
    schema = loader(table_id)
    if not schema:
        return set()
    return {str(c.get("name") or "").strip() for c in _schema_columns(schema) if str(c.get("name") or "").strip()}


def geo_column(table_id: str) -> str | None:
    """Country label column from YAML, preferring mart ``country_iso3`` or staging ``country_name``."""
    if _is_mart_table_id(table_id):
        return geo_column_mart(table_id)
    names = _column_names(table_id)
    by_low = {n.lower(): n for n in names}
    for cand in _GEO_COLUMN_CANDIDATES:
        if cand in by_low:
            return by_low[cand]
    return None


def geo_column_mart(table_id: str) -> str | None:
    schema = load_mart_table_schema(table_id)
    if not schema:
        return None
    names = {str(c.get("name") or "").strip() for c in _schema_columns(schema) if str(c.get("name") or "").strip()}
    by_low = {n.lower(): n for n in names}
    for cand in _MART_GEO_COLUMN_CANDIDATES:
        if cand in by_low:
            return by_low[cand]
    # Legacy facts: only geo_key (FK to dim_geography.geography_key).
    if "geography_key" in by_low:
        return by_low["geography_key"]
    if "geo_key" in by_low:
        return by_low["geo_key"]
    return None


def year_column(table_id: str) -> str | None:
    names = _column_names(table_id)
    by_low = {n.lower(): n for n in names}
    for cand in _YEAR_COLUMN_CANDIDATES:
        if cand in by_low:
            return by_low[cand]
    return None


_ISO3_TO_NAME: dict[str, str] = {iso: name for name, iso in _AFRICA_COUNTRY_ISO3.items()}


def _pick_sample_label(label: str, sample_by_low: dict[str, str]) -> str | None:
    if not label:
        return None
    low = label.lower()
    if low in sample_by_low:
        return sample_by_low[low]
    for s_low, canonical in sample_by_low.items():
        if low in s_low or s_low.startswith(low):
            return canonical
    return None


def _labels_to_iso3_spine(labels: list[str]) -> list[str]:
    """Normalize geography labels to ISO3 where possible (continental spine input)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        label = str(raw).strip()
        if not label:
            continue
        if _ISO3_RE.match(label.upper()):
            val = label.upper()
        elif label in _AFRICA_COUNTRY_ISO3:
            val = _AFRICA_COUNTRY_ISO3[label]
        else:
            val = label
            for name, iso in _AFRICA_COUNTRY_ISO3.items():
                if name.lower() == label.lower():
                    val = iso
                    break
        key = str(val).lower()
        if key not in seen:
            seen.add(key)
            out.append(str(val))
    return out


def resolve_geo_literals_for_table(table_id: str, labels: list[str] | None) -> list[str]:
    """Map ISO3 spine (or country names) to warehouse literals for the table geo column."""
    if not labels:
        return []
    col = geo_column(table_id)
    spine = _labels_to_iso3_spine([str(g).strip() for g in labels if str(g).strip()])
    if not col:
        return spine
    # geo_key / geography_key bind via dim_geography.country_iso3 — never hash literals.
    if col in ("geo_key", "geography_key"):
        out_iso: list[str] = []
        seen_iso: set[str] = set()
        for item in spine:
            if _ISO3_RE.match(str(item).upper()):
                val = str(item).upper()
            else:
                val = _AFRICA_COUNTRY_ISO3.get(str(item), str(item).upper())
            key = val.lower()
            if key not in seen_iso:
                seen_iso.add(key)
                out_iso.append(val)
        return out_iso
    samples = column_samples_for_table(table_id, col)
    sample_by_low = {s.lower(): s for s in samples}
    out: list[str] = []
    seen: set[str] = set()
    for item in spine:
        if col == "country_iso3":
            if _ISO3_RE.match(str(item).upper()):
                val = str(item).upper()
            else:
                val = _AFRICA_COUNTRY_ISO3.get(str(item), str(item).upper())
        elif col in ("country_name", "country"):
            if _ISO3_RE.match(str(item).upper()):
                iso = str(item).upper()
                canonical = _ISO3_TO_NAME.get(iso, "")
                val = _pick_sample_label(canonical, sample_by_low) or canonical or iso
            else:
                val = _pick_sample_label(str(item), sample_by_low) or str(item)
        else:
            val = str(item)
        key = val.lower()
        if key not in seen:
            seen.add(key)
            out.append(val)
    return out


def resolve_geo_filter_values(table_id: str, labels: list[str] | None) -> list[str]:
    """Map decomposition geography labels to warehouse filter values for the table geo column."""
    if not labels:
        return []
    return resolve_geo_literals_for_table(table_id, labels)


def geo_filter_string(
    table_id: str,
    geo_labels: list[str],
    *,
    multi_country: bool,
) -> str:
    """Human-readable filter clause using schema-correct geo column and values."""
    if not geo_labels:
        return "geography from question (country_iso3 or join dim_geography)"
    resolved = resolve_geo_filter_values(table_id, geo_labels)
    col = geo_column(table_id) or "country_iso3"
    if not resolved:
        return "geography from question (country_iso3 or join dim_geography)"
    if len(resolved) == 1:
        return f"{col} = '{resolved[0]}'"
    lits = ", ".join(f"'{v}'" for v in resolved[:16])
    return f"{col} in ({lits})"


def time_bounds_filter_string(
    table_id: str,
    *,
    time_start: str,
    time_end: str,
    year_hint: str,
) -> str:
    ts = (time_start or "")[:10]
    te = (time_end or "")[:10]
    if ts and te:
        schema = load_mart_table_schema(table_id)
        names = {
            str(c.get("name") or "").strip().lower()
            for c in (schema or {}).get("columns") or []
            if str(c.get("name") or "").strip()
        }
        if "as_of_date" in names:
            return f"as_of_date BETWEEN '{ts}' AND '{te}'"
        ycol = year_column(table_id) or "year"
        if ts[:4].isdigit() and te[:4].isdigit():
            return f"{ycol} BETWEEN {ts[:4]} AND {te[:4]}"
    return f"year≈{year_hint}"


def intent_pattern_for_query(
    query: str,
    table_id: str,
    *,
    multi_country: bool,
    africa_panel: bool = False,
    time_start: str = "",
    time_end: str = "",
) -> str:
    """Pick a compilable SQL pattern from question shape and mart table capability."""
    if _is_mart_table_id(table_id):
        if not table_supports_sql_pattern_mart(table_id):
            return "custom"
    elif not table_supports_sql_pattern(table_id):
        return "custom"

    q = query or ""
    if _YOY_QUERY_RE.search(q):
        return "yoy_delta"
    if _SHARE_QUERY_RE.search(q):
        return "share_of_total"
    if _SERIES_QUERY_RE.search(q):
        return "time_series"
    if multi_country or africa_panel or _RANK_QUERY_RE.search(q):
        return "rank_by_sum"
    ts = (time_start or "")[:10]
    te = (time_end or "")[:10]
    if (
        not multi_country
        and not africa_panel
        and not _SERIES_QUERY_RE.search(q)
        and ts
        and te
        and ts[:4].isdigit()
        and te[:4].isdigit()
        and ts[:4] == te[:4]
    ):
        return "custom"
    if ts and te and ts[:4].isdigit():
        return "rank_by_sum"
    if re.search(r"\b(19|20)\d{2}\b", q):
        return "rank_by_sum"
    return "rank_by_sum"


def intent_grain_for_table(
    table_id: str,
    *,
    multi_country: bool,
    pattern: str,
) -> list[str]:
    geo = geo_column(table_id) or "country_name"
    prod = product_column(table_id)
    if pattern == "time_series":
        grain = [geo]
        if prod:
            grain.append(prod)
        return grain
    if multi_country or pattern in {"share_of_total", "yoy_delta"}:
        return [geo]
    grain = [geo]
    if prod:
        grain.append(prod)
    return grain


def _table_filter_prefix(table_id: str, measure_id: str = "") -> str:
    if table_id == "fct_production" or "agg_production" in table_id:
        return "production_grain='physical'"
    if table_id == "fct_prices":
        return "price_source from question"
    if table_id == "fct_trade":
        return "trade_grain from question"
    if table_id == "fct_food_security":
        return "measure_type from question (population vs classification)"
    if table_id == "fct_yield":
        return "season_key/harvest_year"
    if table_id == "fct_employment":
        return "JOIN dim_indicator; unit=% vs headcount"
    return ""


def compile_intent_for_table(
    table_id: str,
    *,
    measure_id: str,
    query: str = "",
    geo_labels: list[str] | None = None,
    year_hint: str = "",
    multi_country: bool = False,
    africa_panel: bool = False,
    time_start: str = "",
    time_end: str = "",
    extra_filters: str = "",
) -> dict[str, Any]:
    """Schema-driven BQ intent shared by contract, ontology fallback, and fact plans."""
    geo_filter = geo_filter_string(table_id, geo_labels or [], multi_country=multi_country)
    time_filter = time_bounds_filter_string(
        table_id,
        time_start=time_start,
        time_end=time_end,
        year_hint=year_hint,
    )
    prefix = extra_filters or _table_filter_prefix(table_id, measure_id)
    filters = f"{prefix}; {geo_filter}; {time_filter}" if prefix else f"{geo_filter}; {time_filter}"
    pattern = intent_pattern_for_query(
        query,
        table_id,
        multi_country=multi_country,
        africa_panel=africa_panel,
        time_start=time_start,
        time_end=time_end,
    )
    grain = intent_grain_for_table(table_id, multi_country=multi_country, pattern=pattern)
    rankish = multi_country or africa_panel or pattern in {"rank_by_sum", "share_of_total", "yoy_delta"}
    order_by = "total DESC" if rankish else "value DESC"
    return {
        "goal": f"{measure_id} signal from {table_id}",
        "tables": [table_id],
        "filters": filters,
        "notes": f"contract_{measure_id}_{table_id}",
        "pattern": pattern,
        "metric": "value",
        "grain": grain,
        "order_by": order_by,
    }


def product_column(table_id: str) -> str | None:
    if _is_mart_table_id(table_id):
        return product_column_mart(table_id)
    names = _column_names(table_id)
    by_low = {n.lower(): n for n in names}
    samples = value_samples_for_tables({table_id}).get(_strip_fqn(table_id).lower()) or {}
    for cand in _PRODUCT_COLUMN_CANDIDATES:
        if cand in samples and cand in by_low:
            return by_low[cand]
    for cand in _PRODUCT_COLUMN_CANDIDATES:
        if cand in by_low:
            return by_low[cand]
    return None


def product_column_mart(table_id: str) -> str | None:
    """Product filter column on mart facts (product_key only — join dim_product for names)."""
    schema = load_mart_table_schema(table_id)
    if not schema:
        return None
    names = {str(c.get("name") or "").strip() for c in _schema_columns(schema) if str(c.get("name") or "").strip()}
    by_low = {n.lower(): n for n in names}
    for cand in ("product_name", "product_key", "item_name", "item_key"):
        if cand in by_low:
            return by_low[cand]
    return None


def table_supports_sql_pattern(table_id: str) -> bool:
    """True when YAML grain is country×year facts that SUM/AVG patterns can compile."""
    if _is_mart_table_id(table_id):
        return table_supports_sql_pattern_mart(table_id)
    bare = _strip_fqn(table_id).lower()
    if not bare or bare in _PATTERN_DENY_TABLES or bare.startswith("stg_ilri_"):
        return False
    schema = load_table_schema(bare)
    if not schema:
        return False
    grain = str(schema.get("grain") or "")
    if _PATTERN_DENY_GRAIN_RE.search(grain):
        return False
    if geo_column(bare) is None:
        return False
    if year_column(bare) is None:
        return False
    return bool(measure_columns(bare))


def table_supports_sql_pattern_mart(table_id: str) -> bool:
    bare = _strip_fqn(table_id).lower()
    if not bare or bare.startswith("dim_") or bare.startswith("bridge_"):
        return False
    schema = load_mart_table_schema(bare)
    if not schema:
        return False
    grain = str(schema.get("grain") or "")
    if _PATTERN_DENY_GRAIN_RE.search(grain):
        return False
    if geo_column_mart(bare) is None:
        return False
    if year_column(bare) is None and "as_of_date" not in _column_names(bare):
        return False
    return bool(measure_columns_mart(bare))


def join_fragments_for_tables(selected_tables: list[str] | set[str] | None) -> str:
    """Standard LEFT JOIN blocks for selected mart tables."""
    return mart_join_fragments(selected_tables)


def measure_sql_aggregation(table_id: str, column: str, *, element: str | None = None) -> str:
    """``sum`` or ``avg`` from YAML ``sql_aggregation``, else measure semantics."""
    el = str(element or "").strip()
    if el and _AVG_SEMANTIC_RE.search(el):
        return "avg"
    schema = load_table_schema(table_id)
    want = str(column or "").strip()
    if schema and want:
        for col in _schema_columns(schema):
            if str(col.get("name") or "").strip() != want:
                continue
            explicit = str(col.get("sql_aggregation") or "").strip().lower()
            if explicit in {"sum", "avg"}:
                return explicit
            blob = f"{want} {col.get('description') or ''}"
            return "avg" if _AVG_SEMANTIC_RE.search(blob) else "sum"
    if want and _AVG_SEMANTIC_RE.search(want):
        return "avg"
    return "sum"


def resolve_measure_column(table_id: str, requested: str | None) -> str | None:
    measures = measure_columns(table_id)
    if not measures:
        return None
    want = str(requested or "").strip()
    if want:
        for name in measures:
            if name.lower() == want.lower():
                return name
    return measures[0]


def fold_accents(text: str) -> str:
    """Lowercase and strip diacritics so 'Ségou' tokenizes as one word, 'segou'."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _alnum_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _WORD_RE.finditer(fold_accents(text))]


def _singular(token: str) -> str:
    if len(token) <= 3 or token.endswith("ss"):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("oes"):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def _canon_token(token: str) -> str:
    return _singular(_SPEECH_SYNONYMS.get(token, token))


def expand_speech_text(text: str) -> str:
    out = fold_accents(text)
    for src, dst in _SPEECH_SYNONYMS.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out)
    return out


def match_value_samples(blob: str, samples: set[str] | list[str]) -> list[str]:
    """Pick warehouse labels whose head token appears in the query (speech synonyms expanded).

    Tokens are accent-folded and singularized on both sides; a head token must be at
    least two characters so possessive 's' or initials never select a label.
    """
    expanded = expand_speech_text(blob)
    raw_blob_tokens = {_singular(t) for t in _alnum_tokens(expanded)}
    blob_tokens = {_canon_token(t) for t in _alnum_tokens(expanded)}
    if not blob_tokens:
        return []
    grouped: dict[str, list[str]] = {}
    sample_order = {str(s).strip(): i for i, s in enumerate(samples or [])}
    for raw in samples or []:
        sample = str(raw).strip()
        if not sample:
            continue
        toks = [t for t in _alnum_tokens(sample) if t not in _GENERIC_SAMPLE_TOKENS]
        if not toks:
            continue
        head = _canon_token(toks[0])
        if len(head) < 2 or head not in blob_tokens:
            continue
        grouped.setdefault(head, []).append(sample)
    chosen: list[str] = []
    for head, group in grouped.items():
        scored: list[tuple[Any, ...]] = []
        for sample in group:
            sample_tokens = [t for t in _alnum_tokens(sample) if t not in _GENERIC_SAMPLE_TOKENS]

            def _said(token: str) -> bool:
                if _singular(token) in raw_blob_tokens:
                    return True
                canonical = _canon_token(token)
                return canonical != head and canonical in blob_tokens

            extra = []
            for token in sample_tokens:
                canonical = _canon_token(token)
                if canonical == head and (token == sample_tokens[0] or _singular(token) in raw_blob_tokens):
                    continue
                extra.append(token)
            extra_hits = sum(1 for token in extra if _said(token))
            unmatched = {token for token in extra if not _said(token)}
            order = sample_order.get(sample, 10**9)
            if extra and extra_hits == 0:
                scored.append((1, len(unmatched), 0, order, sample))
            else:
                scored.append((0, len(unmatched), -extra_hits, order, sample))
        scored.sort()
        chosen.append(scored[0][-1])

    def _head_pos(sample: str) -> int:
        toks = [t for t in _alnum_tokens(sample) if t not in _GENERIC_SAMPLE_TOKENS]
        if not toks:
            return 10**9
        head = _SPEECH_SYNONYMS.get(toks[0], toks[0])
        found = re.search(rf"\b{re.escape(head)}\b", expanded)
        return found.start() if found else 10**9

    chosen.sort(key=_head_pos)
    return chosen


def _match_metric_slug(blob: str, samples: set[str] | list[str]) -> str | None:
    """Match metric slug samples using crop/entity tokens (e.g. production_maize_yield)."""
    expanded = expand_speech_text(blob or "")
    blob_tokens = set(_alnum_tokens(expanded))
    if not blob_tokens:
        return None
    skip_toks = _GENERIC_SAMPLE_TOKENS | frozenset({"production", "yield", "physical"})
    best: str | None = None
    best_hits = 0
    for raw in samples or []:
        sample = str(raw).strip()
        if not sample:
            continue
        toks = [t for t in _alnum_tokens(sample) if t not in skip_toks]
        hits = sum(
            1 for t in toks if t in blob_tokens or _SPEECH_SYNONYMS.get(t, t) in blob_tokens
        )
        if hits > best_hits:
            best_hits = hits
            best = sample
    return best if best_hits > 0 else None


_PRODUCT_FK_COL = "product_key"
_PRODUCT_LABEL_COL = "product_name"
_PRODUCT_JOIN_DIM = "dim_product"
_PRODUCT_PREFER_TABLES = ("dim_product", "agg_production_annual")
_SHARED_DICTIONARY_COLUMNS = frozenset({_PRODUCT_LABEL_COL})


def _sql_literal(value: str) -> str:
    return "'" + (value or "").replace("'", "''") + "'"


@dataclass(frozen=True)
class SemanticFilterClause:
    sql: str
    label: str | None = None
    labels: tuple[str, ...] = ()


def _mart_bare_table_ids() -> list[str]:
    directory = _mart_yaml_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yml"))


def column_samples_for_table(table_id: str, column: str) -> list[str]:
    """Profiled ``{column}_value_samples`` for one table (mart or staging YAML)."""
    bare = _strip_fqn(table_id).lower()
    col = (column or "").strip().lower()
    if not bare or not col:
        return []
    for loader in (value_samples_for_mart_tables, value_samples_for_tables):
        samples_map = loader({bare}).get(bare) or {}
        for sample_col, vals in samples_map.items():
            if sample_col.lower() == col:
                return [str(v).strip() for v in vals if str(v).strip()]
    return []


def dictionary_samples_for_column(
    column: str,
    *,
    prefer_tables: list[str] | None = None,
) -> list[str]:
    """Union of profiled ``{column}_value_samples`` across mart and staging YAML tables."""
    col = (column or "").strip().lower()
    if not col:
        return []
    ordered: list[str] = []
    seen: set[str] = set()

    def _merge(table_id: str) -> None:
        for text in column_samples_for_table(table_id, col):
            if text in seen:
                continue
            seen.add(text)
            ordered.append(text)

    for tid in prefer_tables or []:
        _merge(tid)
    for tid in _mart_bare_table_ids():
        _merge(tid)
    return ordered


def resolve_dictionary_label(
    *,
    column: str,
    blob: str,
    prefer_tables: list[str] | None = None,
) -> str | None:
    """Match speech to one dictionary label for ``column``; never invent literals."""
    samples = dictionary_samples_for_column(column, prefer_tables=prefer_tables)
    if not samples:
        return None
    matched = match_value_samples(blob, samples)
    return matched[0] if matched else None


def resolve_dictionary_labels(
    *,
    column: str,
    blob: str,
    prefer_tables: list[str] | None = None,
    limit: int = 6,
) -> list[str]:
    samples = dictionary_samples_for_column(column, prefer_tables=prefer_tables)
    if not samples:
        return []
    matched = match_value_samples(blob, samples)
    return matched[: max(1, int(limit or 6))]


def _product_join_dim(table_id: str) -> str | None:
    bare = _strip_fqn(table_id).lower()
    rels = SEMANTIC_RELATIONSHIPS.get(bare) or {}
    for join in rels.get("joins_with") or []:
        if not isinstance(join, dict):
            continue
        jt = str(join.get("table") or "").strip().lower()
        on_keys = [str(x).split("≈")[0].strip().lower() for x in (join.get("on") or [])]
        if jt == _PRODUCT_JOIN_DIM and _PRODUCT_FK_COL in on_keys:
            return jt
    schema = load_mart_table_schema(bare) or {}
    col_names = {
        str(c.get("name") or "").strip().lower()
        for c in (schema.get("columns") or [])
        if str(c.get("name") or "").strip()
    }
    if _PRODUCT_FK_COL in col_names and _PRODUCT_LABEL_COL not in col_names:
        return _PRODUCT_JOIN_DIM
    return None


def _fact_has_denormalized_product_name(table_id: str) -> bool:
    return bool(column_samples_for_table(table_id, _PRODUCT_LABEL_COL))


def crop_entities_for_bind(
    entities: list[str] | None,
    *,
    primary_measures: list[str] | None = None,
) -> list[str]:
    """Decomposition entities with measure/domain tokens removed for product binding."""
    out: list[str] = []
    for raw in entities or []:
        text = str(raw).strip()
        if not text or entity_is_measure_noise(text, primary_measures=primary_measures):
            continue
        out.append(text)
    return out


def _filter_measure_tokens_from_labels(
    labels: list[str],
    *,
    primary_measures: list[str] | None = None,
) -> list[str]:
    return [
        label
        for label in labels
        if not entity_is_measure_noise(label, primary_measures=primary_measures)
    ]


def _resolve_product_labels_for_table(
    table_id: str,
    *,
    blob: str,
    labels: list[str] | None = None,
    primary_measures: list[str] | None = None,
) -> list[str]:
    """Resolve speech or hints to dictionary labels for one table's filter column."""
    bare = _strip_fqn(table_id).lower()
    hinted = _filter_measure_tokens_from_labels(
        [str(x).strip() for x in (labels or []) if str(x).strip()],
        primary_measures=primary_measures,
    )
    table_samples = column_samples_for_table(bare, _PRODUCT_LABEL_COL)
    dictionary = dictionary_samples_for_column(
        _PRODUCT_LABEL_COL, prefer_tables=[bare, *_PRODUCT_PREFER_TABLES]
    )
    pool = table_samples or dictionary
    if hinted:
        resolved = _resolve_hints_against(hinted, pool, pool)
        if resolved:
            return resolved[:6]
    return match_value_samples(blob, pool)[:6]


def _resolve_hints_against(hints: list[str], exact_pool: list[str], fuzzy_pool: list[str]) -> list[str]:
    """Map each hint to a warehouse label (exact first, then speech match); unmatched hints are dropped."""
    exact = {s.casefold(): s for s in reversed(exact_pool)}
    out: list[str] = []
    for hint in hints:
        label = exact.get(hint.casefold())
        if label is None:
            matched = match_value_samples(hint, fuzzy_pool)
            label = matched[0] if matched else None
        if label and label not in out:
            out.append(label)
    return out


def _resolve_bind_product_labels(
    table_id: str,
    *,
    query: str,
    facets: dict[str, Any],
    entities: list[str],
    primary_measures: list[str] | None,
    product_labels: list[str] | None,
    product_blob_text: str,
) -> list[str]:
    """Product literals for bind contract: engine hits, staples, then crop entities."""
    bare = _strip_fqn(table_id).lower()
    explicit = [str(x).strip() for x in (product_labels or []) if str(x).strip()]
    if explicit:
        return explicit[:6]
    from ml.rag.chatbot.bundle_metrics import resolve_staple_products

    staples = resolve_staple_products(query, facets)
    if staples:
        return staples[:6]
    crop_ents = crop_entities_for_bind(entities, primary_measures=primary_measures)
    if crop_ents:
        resolved = _resolve_product_labels_for_table(
            bare,
            blob=product_blob_text,
            labels=crop_ents,
            primary_measures=primary_measures,
        )
        if resolved:
            return resolved[:6]
    resolved = _resolve_product_labels_for_table(
        bare,
        blob=product_blob_text,
        labels=None,
        primary_measures=primary_measures,
    )
    return resolved[:6] if resolved else []


def compile_product_filter_sql(
    table_id: str,
    *,
    project_id: str,
    dataset: str,
    blob: str = "",
    labels: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Compile a product filter SQL fragment using dictionary labels + join graph.

    Returns ``(sql_fragment, resolved_labels)``. Fragment is empty when no dictionary match.
    """
    bare = _strip_fqn(table_id).lower()
    if not product_column(bare):
        return "", []
    resolved = _resolve_product_labels_for_table(bare, blob=blob, labels=labels)
    if not resolved:
        return "", []

    spine = bind_spine_map(bare)
    product_spine = spine.get(_PRODUCT_FK_COL) or spine.get("product_key")

    if _fact_has_denormalized_product_name(bare):
        col = _PRODUCT_LABEL_COL
        if len(resolved) == 1:
            return f"AND {col} = {_sql_literal(resolved[0])} ", resolved
        lits = ", ".join(_sql_literal(v) for v in resolved[:16])
        return f"AND {col} IN ({lits}) ", resolved

    if product_spine:
        fk_col = _PRODUCT_FK_COL if _PRODUCT_FK_COL in {
            str(c.get("name") or "").strip().lower()
            for c in (load_mart_table_schema(bare) or {}).get("columns") or []
        } else (product_column_mart(bare) or _PRODUCT_FK_COL)
        sql = _compile_fk_spine_filter_sql(
            bare,
            fk_col=fk_col,
            dim_ref=product_spine,
            literals=resolved,
            project_id=project_id,
            dataset=dataset,
        )
        if sql:
            return sql, resolved

    dim = _product_join_dim(bare) or _PRODUCT_JOIN_DIM
    schema = load_mart_table_schema(bare) or {}
    col_names = {
        str(c.get("name") or "").strip().lower()
        for c in (schema.get("columns") or [])
        if str(c.get("name") or "").strip()
    }
    fk_col = _PRODUCT_FK_COL if _PRODUCT_FK_COL in col_names else (
        product_column_mart(bare) or _PRODUCT_FK_COL
    )
    dim_fqn = f"`{project_id}.{dataset}.{dim}`"
    if len(resolved) == 1:
        return (
            f"AND {fk_col} IN (SELECT {_PRODUCT_FK_COL} FROM {dim_fqn} "
            f"WHERE {_PRODUCT_LABEL_COL} = {_sql_literal(resolved[0])}) ",
            resolved,
        )
    lits = ", ".join(_sql_literal(v) for v in resolved[:16])
    return (
        f"AND {fk_col} IN (SELECT {_PRODUCT_FK_COL} FROM {dim_fqn} "
        f"WHERE {_PRODUCT_LABEL_COL} IN ({lits})) ",
        resolved,
    )


def product_blob(query: str, entities: list[str] | None = None) -> str:
    """Speech blob for product/geo matching — query plus crop-like entities only."""
    q = (query or "").strip()
    parts = [q] if q else []
    for raw in entities or []:
        text = str(raw).strip()
        if not text:
            continue
        if entity_is_measure_noise(text):
            continue
        if resolve_dictionary_label(column=_PRODUCT_LABEL_COL, blob=text):
            parts.append(text)
    return " ".join(parts)


def measure_blob(
    query: str,
    *,
    primary_measures: list[str] | None = None,
    task_mode: str = "",
) -> str:
    """Speech blob for measure discriminators — query + contract measures, not domain tags."""
    parts = [str(query or "").strip()]
    parts.extend(str(m).strip() for m in (primary_measures or []) if str(m).strip())
    mode = str(task_mode or "").strip()
    if mode:
        parts.append(mode)
    return " ".join(p for p in parts if p)


def value_samples_for_table(table_id: str) -> dict[str, list[str]]:
    """Column samples from mart or staging YAML for one table."""
    bare = _strip_fqn(table_id).lower()
    if _is_mart_table_id(bare):
        return value_samples_for_mart_tables({bare}).get(bare) or {}
    return value_samples_for_tables({bare}).get(bare) or {}


def _primary_measure_ids(primary_measures: list[str] | None) -> list[str]:
    return [str(m).strip().lower() for m in (primary_measures or []) if str(m).strip()]


def _wants_production_volume(primary_measures: list[str] | None, measure_blob_text: str) -> bool:
    pm = _primary_measure_ids(primary_measures)
    if any(m in ("production", "prod") for m in pm):
        return True
    blob = measure_blob_text or ""
    if not _PRODUCTION_QUERY_RE.search(blob):
        return False
    if any("yield" in m for m in pm):
        return False
    return True


def _wants_yield(primary_measures: list[str] | None, measure_blob_text: str) -> bool:
    pm = _primary_measure_ids(primary_measures)
    if any("yield" in m for m in pm):
        return True
    return bool(_YIELD_QUERY_RE.search(measure_blob_text or "")) and not _wants_production_volume(
        primary_measures, measure_blob_text
    )


def _every_token_said(sample: str, blob: str) -> bool:
    """Slug samples fuse measure and product; one said token does not name the slug."""
    blob_tokens = set(_alnum_tokens(expand_speech_text(blob)))
    tokens = [t for t in _alnum_tokens(sample) if t not in _GENERIC_SAMPLE_TOKENS]
    return bool(tokens) and all(_SPEECH_SYNONYMS.get(t, t) in blob_tokens or t in blob_tokens for t in tokens)


def compile_measure_filters(
    table_id: str,
    *,
    measure_blob_text: str = "",
    primary_measures: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Compile equality filters for measure discriminators bound to primary_measures."""
    bare = _strip_fqn(table_id).lower()
    samples_map = value_samples_for_table(bare)
    if not samples_map:
        return []

    schema = (_schema_loader_for(bare))(bare) or {}
    col_names = {
        str(c.get("name") or "").strip().lower()
        for c in (schema.get("columns") or [])
        if str(c.get("name") or "").strip()
    }
    pm = _primary_measure_ids(primary_measures)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(col: str, val: str | None) -> None:
        if not val or col.lower() in seen:
            return
        seen.add(col.lower())
        out.append((col, val))

    production_vol = _wants_production_volume(pm, measure_blob_text)
    yield_q = _wants_yield(pm, measure_blob_text)

    if production_vol and bare in ("fct_production", "agg_production_annual"):
        if "production_grain" in col_names:
            grain_samples = samples_map.get("production_grain") or []
            grain = default_discriminator_value(
                "production_grain",
                grain_samples,
                query=measure_blob_text,
                primary_measures=pm,
            )
            if grain:
                _add("production_grain", grain)
        element_samples = samples_map.get("element") or []
        _add(
            "element",
            default_discriminator_value(
                "element",
                element_samples,
                query=measure_blob_text,
                primary_measures=pm,
            ),
        )
        if "metric" in col_names:
            _add(
                "metric",
                default_discriminator_value(
                    "metric",
                    samples_map.get("metric") or [],
                    query=measure_blob_text,
                    primary_measures=pm,
                ),
            )
        return out

    if yield_q:
        if "element" in samples_map:
            _add(
                "element",
                default_discriminator_value(
                    "element",
                    samples_map.get("element") or [],
                    query=measure_blob_text,
                    primary_measures=pm,
                ),
            )
        if "metric" in col_names:
            metric_samples = samples_map.get("metric") or []
            metric_val = _match_metric_slug(measure_blob_text, metric_samples)
            if not metric_val:
                matched = [
                    m
                    for m in match_value_samples(measure_blob_text, metric_samples)
                    if _every_token_said(m, measure_blob_text)
                ]
                metric_val = matched[0] if matched else default_discriminator_value(
                    "metric",
                    metric_samples,
                    query=measure_blob_text,
                    primary_measures=pm,
                )
            _add("metric", metric_val)
        return out

    product_col = (product_column(bare) or "").lower()
    skip = {product_col, "unit", "country", "country_name", "market_name"} - {""}
    for col, samples in samples_map.items():
        col_l = col.lower()
        if col_l in skip or col_l not in _MEASURE_DISCRIMINATOR_COLS:
            continue
        matched = match_value_samples(measure_blob_text, samples)
        if col_l == "metric":
            matched = [m for m in matched if _every_token_said(m, measure_blob_text)]
        if matched:
            _add(col, matched[0])
            continue
        if col_l not in _AUTO_DEFAULT_DISCRIMINATOR_COLS:
            continue
        default = default_discriminator_value(
            col,
            samples,
            query=measure_blob_text,
            primary_measures=pm,
        )
        if default:
            _add(col, default)
    return out


def compile_measure_filter_sql(
    table_id: str,
    *,
    measure_blob_text: str = "",
    primary_measures: list[str] | None = None,
) -> str:
    """AND-prefixed SQL fragments for measure discriminators."""
    parts: list[str] = []
    for col, val in compile_measure_filters(
        table_id,
        measure_blob_text=measure_blob_text,
        primary_measures=primary_measures,
    ):
        if _IDENT_RE.match(col):
            parts.append(f"AND {col} = {_sql_literal(val)} ")
    return "".join(parts)


def compile_time_filter_sql(
    table_id: str,
    *,
    year: int | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> str:
    """Time predicate for one table. A passed year is an equality. A start and end in different years is BETWEEN on as_of_date when that column exists, otherwise on the table year column."""
    bare = _strip_fqn(table_id).lower()
    schema = (_schema_loader_for(bare))(bare) or {}
    col_names = {
        str(c.get("name") or "").strip().lower()
        for c in (schema.get("columns") or [])
        if str(c.get("name") or "").strip()
    }
    ts = (time_start or "")[:10]
    te = (time_end or "")[:10]
    if "as_of_date" in col_names and ts and te:
        return f"AND as_of_date BETWEEN DATE '{ts}' AND DATE '{te}' "
    ycol = year_column(bare)
    if not ycol:
        return ""
    if year is not None:
        return f"AND {ycol} = {int(year)} "
    if ts[:4].isdigit() and te[:4].isdigit():
        return f"AND {ycol} BETWEEN {int(ts[:4])} AND {int(te[:4])} "
    return ""


def compile_semantic_filter(
    table_id: str,
    facet: str,
    *,
    blob: str,
    project_id: str,
    dataset: str,
    labels: list[str] | None = None,
    primary_measures: list[str] | None = None,
    measure_blob_text: str = "",
    year: int | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    country_labels: list[str] | None = None,
    ontology_context: Any | None = None,
) -> SemanticFilterClause | None:
    """Compile one semantic facet filter (dictionary label + join routing)."""
    pm = list(primary_measures) if primary_measures else []
    mb = measure_blob_text or blob
    if ontology_context is not None:
        if not pm:
            pm = list(getattr(ontology_context, "primary_measures", None) or [])
        if not mb and getattr(ontology_context, "query", ""):
            mb = measure_blob(
                str(ontology_context.query),
                primary_measures=pm or None,
            )

    facet_l = str(facet or "").strip().lower()
    if facet_l == "product":
        sql, resolved = compile_product_filter_sql(
            table_id,
            project_id=project_id,
            dataset=dataset,
            blob=blob,
            labels=labels,
        )
        if not sql:
            return None
        return SemanticFilterClause(
            sql=sql,
            label=resolved[0] if resolved else None,
            labels=tuple(resolved),
        )
    if facet_l == "measure":
        filters = compile_measure_filters(
            table_id,
            measure_blob_text=mb,
            primary_measures=pm or None,
        )
        if not filters:
            return None
        sql = compile_measure_filter_sql(
            table_id,
            measure_blob_text=mb,
            primary_measures=pm or None,
        )
        return SemanticFilterClause(
            sql=sql,
            label=filters[0][1] if filters else None,
            labels=tuple(v for _, v in filters),
        )
    if facet_l == "geo":
        resolved = resolve_geo_filter_values(table_id, country_labels) if country_labels else []
        place_blob = blob or ""
        for name in [*(country_labels or []), *resolved]:
            if str(name).strip():
                place_blob = re.sub(re.escape(str(name).strip()), " ", place_blob, flags=re.IGNORECASE)
        subnational = match_subnational_geo_labels(place_blob)
        geo_level = match_geo_level_label(place_blob)
        extra: list[tuple[str, str]] = list(subnational)
        if geo_level:
            extra.append(("geo_level", geo_level))
        if not resolved and not extra:
            return None

        col = geo_column(table_id) or "country_name"
        spine = bind_spine_map(table_id)
        schema = load_mart_table_schema(table_id) or {}
        col_names = {
            str(c.get("name") or "").strip().lower()
            for c in (schema.get("columns") or [])
            if str(c.get("name") or "").strip()
        }

        fk_for_dim: str | None = None
        spine_ref: str | None = None
        if col in ("geo_key", "geography_key"):
            fk_for_dim = col
            spine_ref = (
                spine.get(col)
                or spine.get("geography_key")
                or spine.get("geo_key")
                or "dim_geography.country_iso3"
            )
        elif extra:
            if "geography_key" in col_names:
                fk_for_dim = "geography_key"
                spine_ref = spine.get("geography_key") or "dim_geography.country_iso3"
            elif "geo_key" in col_names:
                fk_for_dim = "geo_key"
                spine_ref = spine.get("geo_key") or "dim_geography.country_iso3"

        sql_parts: list[str] = []
        # Denormalized country column on the fact (never hash-filter geo keys).
        if col not in ("geo_key", "geography_key") and resolved:
            if len(resolved) == 1:
                sql_parts.append(f"AND {col} = {_sql_literal(resolved[0])}")
            else:
                lits = ", ".join(_sql_literal(v) for v in resolved[:32])
                sql_parts.append(f"AND {col} IN ({lits})")

        need_dim_subq = bool(fk_for_dim and spine_ref) and (
            col in ("geo_key", "geography_key") or bool(extra)
        )
        if need_dim_subq and fk_for_dim and spine_ref:
            sub_sql = _compile_fk_spine_filter_sql(
                table_id,
                fk_col=fk_for_dim,
                dim_ref=spine_ref,
                literals=resolved if col in ("geo_key", "geography_key") else [],
                project_id=project_id,
                dataset=dataset,
                extra_predicates=extra or None,
            )
            if sub_sql.strip():
                sql_parts.append(sub_sql.strip())

        if not sql_parts:
            return None
        label = resolved[0] if resolved else (extra[0][1] if extra else None)
        clause_labels = tuple(resolved) if resolved else tuple(v for _, v in extra)
        return SemanticFilterClause(
            sql=" ".join(sql_parts) + " ",
            label=label,
            labels=clause_labels,
        )
    if facet_l == "time":
        sql = compile_time_filter_sql(
            table_id,
            year=year,
            time_start=time_start,
            time_end=time_end,
        )
        if not sql:
            return None
        return SemanticFilterClause(sql=sql)
    return None


@dataclass
class TableBindContract:
    """Table-driven facet binding for NL2SQL / template / validation (all facets, not geo-only)."""

    table_id: str
    geo_column: str | None = None
    geo_literals: list[str] = field(default_factory=list)
    time_column: str | None = None
    time_sql: str | None = None
    product_column: str | None = None
    product_literals: list[str] = field(default_factory=list)
    measure_columns: list[str] = field(default_factory=list)
    measure_filters: list[tuple[str, str]] = field(default_factory=list)
    spine_entity_filters: list[tuple[str, str]] = field(default_factory=list)
    required_filters_sql: str = ""
    nomenclature: str = ""
    anti_patterns: list[str] = field(default_factory=list)
    bind_spine: dict[str, str] = field(default_factory=dict)
    filter_provenance: dict[str, str] = field(default_factory=dict)
    time_policy: dict[str, Any] = field(default_factory=dict)

    def unproven_filters(self) -> list[str]:
        """Bound facets with no recorded origin (Question, resolved request or documented policy)."""
        bound = {
            "geo": bool(self.geo_literals),
            "product": bool(self.product_literals),
            "time": bool(str(self.time_sql or "").strip()),
            "spine_entities": bool(self.spine_entity_filters),
            "subnational": bool(re.search(r"\b(admin_\d\w*|city_name)\b", self.required_filters_sql or "")),
        }
        return [facet for facet, is_bound in bound.items() if is_bound and facet not in self.filter_provenance]

    def to_dict(self) -> dict[str, Any]:
        return {
            "filter_provenance": dict(self.filter_provenance),
            "time_policy": dict(self.time_policy),
            "table_id": self.table_id,
            "geo_column": self.geo_column,
            "geo_literals": list(self.geo_literals),
            "time_column": self.time_column,
            "time_sql": self.time_sql,
            "product_column": self.product_column,
            "product_literals": list(self.product_literals),
            "measure_columns": list(self.measure_columns),
            "measure_filters": [[c, v] for c, v in self.measure_filters],
            "spine_entity_filters": [[fk, lab] for fk, lab in self.spine_entity_filters],
            "required_filters_sql": self.required_filters_sql,
            "nomenclature": self.nomenclature,
            "anti_patterns": list(self.anti_patterns),
            "bind_spine": dict(self.bind_spine),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> TableBindContract | None:
        if not isinstance(raw, dict) or not str(raw.get("table_id") or "").strip():
            return None
        mf_raw = raw.get("measure_filters") or []
        measure_filters: list[tuple[str, str]] = []
        for item in mf_raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                measure_filters.append((str(item[0]), str(item[1])))
        se_raw = raw.get("spine_entity_filters") or []
        spine_entity_filters: list[tuple[str, str]] = []
        for item in se_raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                spine_entity_filters.append((str(item[0]), str(item[1])))
        return cls(
            table_id=str(raw.get("table_id") or ""),
            geo_column=str(raw.get("geo_column") or "") or None,
            geo_literals=[str(x) for x in (raw.get("geo_literals") or []) if str(x).strip()],
            time_column=str(raw.get("time_column") or "") or None,
            time_sql=str(raw.get("time_sql") or "") or None,
            product_column=str(raw.get("product_column") or "") or None,
            product_literals=[str(x) for x in (raw.get("product_literals") or []) if str(x).strip()],
            measure_columns=[str(x) for x in (raw.get("measure_columns") or []) if str(x).strip()],
            measure_filters=measure_filters,
            spine_entity_filters=spine_entity_filters,
            required_filters_sql=str(raw.get("required_filters_sql") or ""),
            nomenclature=str(raw.get("nomenclature") or ""),
            anti_patterns=[str(x) for x in (raw.get("anti_patterns") or []) if str(x).strip()],
            bind_spine={
                str(k): str(v)
                for k, v in (raw.get("bind_spine") or {}).items()
                if str(k).strip() and str(v).strip()
            },
            filter_provenance={
                str(k): str(v) for k, v in (raw.get("filter_provenance") or {}).items()
            },
            time_policy=dict(raw.get("time_policy") or {}),
        )


def format_bind_nomenclature(contract: TableBindContract) -> str:
    """Model-facing mandatory bind block for NL2SQL / reasoner prompts."""
    lines = [f"TABLE: {contract.table_id}"]
    if contract.geo_column and contract.geo_literals:
        if len(contract.geo_literals) == 1:
            lines.append(f"GEO: {contract.geo_column} = {_sql_literal(contract.geo_literals[0])}")
        else:
            lits = ", ".join(_sql_literal(v) for v in contract.geo_literals[:16])
            lines.append(f"GEO: {contract.geo_column} IN ({lits})")
    if contract.time_column and contract.time_sql:
        lines.append(f"TIME: use column {contract.time_column} ({contract.time_sql.strip()})")
    elif contract.time_column:
        lines.append(f"TIME_COLUMN: {contract.time_column}")
    if contract.product_column and contract.product_literals:
        prod_col = contract.product_column
        if prod_col.lower() in {"product_key", "product_id"}:
            prod_col = "product_name"
        if len(contract.product_literals) == 1:
            lines.append(
                f"PRODUCT: {prod_col} = {_sql_literal(contract.product_literals[0])}"
            )
        else:
            lits = ", ".join(_sql_literal(v) for v in contract.product_literals[:8])
            lines.append(f"PRODUCT: {prod_col} IN ({lits})")
    if contract.measure_columns:
        lines.append(f"MEASURE_COLUMNS: {', '.join(contract.measure_columns[:4])}")
    for col, val in contract.measure_filters:
        lines.append(f"DISCRIMINATOR: {col} = {_sql_literal(val)}")
    for fk_col, label in contract.spine_entity_filters:
        lines.append(f"SPINE_ENTITY: {fk_col} → {_sql_literal(label)}")
    for anti in contract.anti_patterns:
        lines.append(f"ANTI: {anti}")
    dim_seen: set[str] = set()
    for fk_col, dim_ref in sorted(contract.bind_spine.items()):
        lines.append(f"JOIN_SPINE: {fk_col} → {dim_ref}")
        dim_table = str(dim_ref).split(".", 1)[0].strip().lower()
        if not dim_table or dim_table in dim_seen:
            continue
        dim_seen.add(dim_table)
        expose = dim_expose_columns(dim_table)
        if expose:
            lines.append(f"DIM_CONTEXT: {dim_table} SELECT {', '.join(expose)}")
        if dim_table == "dim_geography":
            lines.append(
                "DIM_GEO_HINT: filter geo_level + admin/city via samples; "
                "SELECT population/coords when useful; never filter geo_key with hashes"
            )
    if contract.required_filters_sql.strip():
        lines.append(f"REQUIRED_SQL_FRAGMENTS: {contract.required_filters_sql.strip()}")
    return "\n".join(lines)


def _bind_anti_patterns(table_id: str, *, geo_column: str | None, time_column: str | None) -> list[str]:
    bare = _strip_fqn(table_id).lower()
    schema = load_mart_table_schema(bare) or {}
    col_names = {
        str(c.get("name") or "").strip().lower()
        for c in (schema.get("columns") or [])
        if str(c.get("name") or "").strip()
    }
    anti: list[str] = []
    if geo_column == "country_name" and "country_code" in col_names:
        anti.append("DO NOT filter country_code with 3-letter ISO3 codes")
    if time_column == "time_key" and "year" not in col_names:
        anti.append("DO NOT use column year — this table uses time_key")
    if time_column and time_column != "year" and "year" in col_names and time_column != "year":
        anti.append(f"Prefer {time_column} over year for time filters on this table")
    return anti


def compile_table_bind_contract(
    table_id: str,
    *,
    facets: dict[str, Any] | None = None,
    card: dict[str, Any] | None = None,
    query: str = "",
    country_labels: list[str] | None = None,
    product_labels: list[str] | None = None,
    project_id: str | None = None,
    dataset: str | None = None,
) -> TableBindContract:
    """Compile all facet bindings for one table from YAML dictionary + decomposition spine."""
    from ml.rag.chatbot.bq_mart_sql import mart_dataset

    bare = _strip_fqn(table_id).lower()
    dec = facets if isinstance(facets, dict) else {}
    proj = (project_id or os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4")).strip()
    ds = (dataset or mart_dataset()).strip()

    geo_labels = list(country_labels or [])
    if not geo_labels:
        geo_raw = dec.get("geography")
        if isinstance(geo_raw, list):
            geo_labels = [str(g).strip() for g in geo_raw if str(g).strip()]
    geo_literals = resolve_geo_literals_for_table(bare, geo_labels)
    gcol = geo_column(bare)

    entities_raw = dec.get("entities")
    entities = entities_raw if isinstance(entities_raw, list) else []
    pm_raw = dec.get("primary_measures")
    primary_measures = pm_raw if isinstance(pm_raw, list) else None
    mb = measure_blob(query, primary_measures=primary_measures, task_mode=str(dec.get("task_mode") or ""))
    pb = product_blob(query, entities)

    ts = str(dec.get("time_start") or "")[:10]
    te = str(dec.get("time_end") or "")[:10]
    start_year = int(ts[:4]) if ts[:4].isdigit() else None
    end_year = int(te[:4]) if te[:4].isdigit() else None
    # A year equality is only a point. A span stays a span so the time filter
    # can emit BETWEEN on whatever time column the table has.
    year_hint: int | None = None
    if start_year is not None and end_year is not None and start_year == end_year:
        year_hint = start_year
    elif start_year is not None and end_year is None:
        year_hint = start_year
    elif end_year is not None and start_year is None:
        year_hint = end_year

    parts: list[str] = []
    provenance: dict[str, str] = {}
    # Places come only from the Question or the resolved geography, never from entities.
    geo_blob = " ".join([query or "", *geo_labels]).strip()
    geo_clause = compile_semantic_filter(
        bare,
        "geo",
        blob=geo_blob,
        project_id=proj,
        dataset=ds,
        country_labels=geo_labels or geo_literals,
    )
    if geo_clause and geo_clause.sql.strip():
        parts.append(geo_clause.sql.strip())
        folded_q = fold_accents(query)
        named = [g for g in geo_labels if fold_accents(g) in folded_q]
        provenance["geo"] = "question" if named or not geo_labels else "resolved_request"
        if "admin_" in geo_clause.sql or "city_name" in geo_clause.sql:
            provenance["subnational"] = "question"

    time_clause = compile_semantic_filter(
        bare,
        "time",
        blob=query,
        project_id=proj,
        dataset=ds,
        year=year_hint,
        time_start=ts or None,
        time_end=te or None,
    )
    tcol = year_column(bare)
    time_sql = time_clause.sql.strip() if time_clause else None
    if time_clause and time_clause.sql.strip():
        parts.append(time_clause.sql.strip())
        provenance["time"] = str(dec.get("time_policy_source") or dec.get("time_source") or "question")

    bind_product_labels = _resolve_bind_product_labels(
        bare,
        query=query,
        facets=dec,
        entities=entities,
        primary_measures=primary_measures,
        product_labels=product_labels,
        product_blob_text=pb,
    )
    prod_clause = compile_semantic_filter(
        bare,
        "product",
        blob=pb,
        project_id=proj,
        dataset=ds,
        labels=bind_product_labels or None,
    )
    pcol = product_column(bare)
    product_literals = list(prod_clause.labels) if prod_clause and prod_clause.labels else []
    if prod_clause and prod_clause.sql.strip():
        parts.append(prod_clause.sql.strip())
        provenance["product"] = "question" if match_value_samples(query, product_literals) else "resolved_request"

    roles_raw = dec.get("entity_roles")
    entity_roles: dict[str, list[str]] | None = None
    if isinstance(roles_raw, dict):
        entity_roles = {
            str(k): [str(x).strip() for x in (v or []) if str(x).strip()]
            for k, v in roles_raw.items()
            if str(k).strip()
        }
    elif isinstance(roles_raw, (list, tuple)):
        entity_roles = {
            str(k): [str(x).strip() for x in (v or []) if str(x).strip()]
            for k, v in roles_raw
            if str(k).strip()
        }

    spine_hits = compile_spine_entity_filters(
        bare,
        blob=pb or query,
        project_id=proj,
        dataset=ds,
        consumed_labels=product_literals,
        entity_roles=entity_roles,
    )
    spine_entity_filters = [(fk, lab) for fk, lab, _sql in spine_hits]
    for _fk, _lab, sql in spine_hits:
        if sql.strip():
            parts.append(sql.strip())

    meas_clause = compile_semantic_filter(
        bare,
        "measure",
        blob=mb,
        project_id=proj,
        dataset=ds,
        primary_measures=primary_measures,
        measure_blob_text=mb,
    )
    measure_filters = compile_measure_filters(
        bare,
        measure_blob_text=mb,
        primary_measures=primary_measures,
    )
    if meas_clause and meas_clause.sql.strip():
        parts.append(meas_clause.sql.strip())
        provenance["measure"] = "question" if primary_measures is None else "resolved_request"
    if spine_entity_filters:
        provenance["spine_entities"] = "question"

    required_sql = " ".join(parts)
    measure_cols = measure_columns_mart(bare) or measure_columns(bare)
    anti = _bind_anti_patterns(bare, geo_column=gcol, time_column=tcol)
    if card:
        for rule in card.get("hard_rules") or []:
            anti.append(f"CARD_RULE: {rule}")

    contract = TableBindContract(
        table_id=bare,
        geo_column=gcol,
        geo_literals=geo_literals,
        time_column=tcol,
        time_sql=time_sql,
        product_column=pcol,
        product_literals=product_literals,
        measure_columns=measure_cols,
        measure_filters=measure_filters,
        spine_entity_filters=spine_entity_filters,
        required_filters_sql=required_sql,
        anti_patterns=anti,
        nomenclature="",
        bind_spine=bind_spine_map(bare),
        filter_provenance=provenance,
    )
    contract.nomenclature = format_bind_nomenclature(contract)
    return contract


def match_product_samples(table_id: str, blob: str) -> list[str]:
    """Dictionary-resolved product labels for SQL filters (not fact-column hash samples)."""
    return _resolve_product_labels_for_table(table_id, blob=blob)


def default_discriminator_value(
    col: str,
    samples: set[str] | list[str] | None,
    *,
    query: str = "",
    primary_measures: list[str] | None = None,
) -> str | None:
    """Pick a YAML sample for a discriminator column. Never invent labels."""
    sample_set = {str(s).strip() for s in (samples or []) if str(s).strip()}
    if not sample_set:
        return None
    by_low = {s.lower(): s for s in sample_set}
    col_l = (col or "").strip().lower()
    q = query or ""
    pm = _primary_measure_ids(primary_measures)
    production_vol = _wants_production_volume(pm, q)
    yield_q = _wants_yield(pm, q)

    def _from_cands(*cands: str) -> str | None:
        for cand in cands:
            hit = by_low.get(cand.lower())
            if hit:
                return hit
        return None

    if col_l == "element":
        if production_vol:
            hit = _from_cands("Production")
            if hit:
                return hit
        if yield_q:
            hit = _from_cands("Yield")
            if hit:
                return hit
        if _PRODUCTION_QUERY_RE.search(q):
            hit = _from_cands("Production")
            if hit:
                return hit
        if _YIELD_QUERY_RE.search(q):
            hit = _from_cands("Yield")
            if hit:
                return hit

    if col_l == "metric":
        if production_vol:
            hit = _from_cands("production_production_physical")
            if hit:
                return hit
        if yield_q:
            hit = _from_cands("production_yield_physical")
            if hit:
                return hit
        if _PRODUCTION_QUERY_RE.search(q):
            hit = _from_cands("production_production_physical")
            if hit:
                return hit
        if _YIELD_QUERY_RE.search(q):
            hit = _from_cands("production_yield_physical")
            if hit:
                return hit

    if col_l == "production_grain" and production_vol:
        hit = _from_cands("physical")
        if hit:
            return hit

    if col_l == "price_type":
        if _PRODUCER_PRICE_QUERY_RE.search(q):
            hit = _from_cands("Producer")
            if hit:
                return hit
        if _WHOLESALE_QUERY_RE.search(q):
            hit = _from_cands("Wholesale")
            if hit:
                return hit
        if _RETAIL_QUERY_RE.search(q):
            hit = _from_cands("Retail")
            if hit:
                return hit

    if col_l == "measure_type":
        if _POPULATION_QUERY_RE.search(q):
            hit = _from_cands("population")
            if hit:
                return hit
        if _CLASSIFICATION_QUERY_RE.search(q):
            hit = _from_cands("classification")
            if hit:
                return hit

    matched = match_value_samples(q, sample_set)
    if matched:
        return matched[0]

    return None


def discriminator_equality_filters(
    table_id: str,
    blob: str,
    *,
    primary_measures: list[str] | None = None,
) -> list[tuple[str, str]]:
    """``(column, sample)`` filters from YAML samples; prefer query matches then known defaults."""
    mb = measure_blob(blob, primary_measures=primary_measures) if primary_measures else blob
    return compile_measure_filters(
        table_id,
        measure_blob_text=mb,
        primary_measures=primary_measures,
    )


def table_source_meta(table_id: str) -> dict[str, Any]:
    """Compact table-level metadata for BQ context enrichment."""
    loader = _schema_loader_for(table_id)
    schema = loader(table_id) or {}
    bare = _strip_fqn(table_id).lower()
    source_obj = schema.get("source")
    source: dict[str, Any] = source_obj if isinstance(source_obj, dict) else {}
    semantic_obj = schema.get("semantic_role")
    semantic: dict[str, Any] = semantic_obj if isinstance(semantic_obj, dict) else {}
    supports = semantic.get("supports")
    layer = str(source.get("layer") or ("mart_dev" if _is_mart_table_id(bare) else "staging_dev")).strip()
    return {
        "table_id": bare,
        "table_name": str(schema.get("table_name") or bare).strip(),
        "description": " ".join(str(schema.get("description") or "").split())[:500],
        "grain": str(schema.get("grain") or "").strip(),
        "entity_type": str(schema.get("entity_type") or "").strip(),
        "source_layer": layer,
        "source_domain": str(source.get("domain") or semantic.get("primary_domain") or "").strip(),
        "supports": list(supports) if isinstance(supports, list) else [],
    }


# Legacy alias kept for imports / tests that referenced the old fixed map.
_SAMPLE_KEY_TO_COLUMN: dict[str, str] = {
    "element_value_samples": "element",
    "product_value_samples": "product_name",
    "item_value_samples": "item",
    "unit_value_samples": "unit",
    "donor_value_samples": "donor",
    "purpose_value_samples": "purpose",
    "indicator_value_samples": "indicator",
    "institution_value_samples": "institution",
    "degree_value_samples": "degree",
    "source_value_samples": "source",
    "currency_value_samples": "currency",
    "price_type_value_samples": "price_type",
    "market_value_samples": "market_name",
    "phase_code_value_samples": "phase_code",
    "phase_name_value_samples": "phase_name",
    "classification_scale_value_samples": "classification_scale",
    "scenario_name_value_samples": "scenario_name",
    "measure_type_value_samples": "measure_type",
    "treatment_value_samples": "treatment",
    "food_value_value_samples": "food_value",
    "industry_value_samples": "industry",
    "factor_value_samples": "factor",
    "release_value_samples": "release",
}


def columns_for_tables(table_ids: set[str] | list[str]) -> dict[str, set[str]]:
    """Return ``{bare_table_id: {column_name, ...}}`` from YAML for each known table."""
    out: dict[str, set[str]] = {}
    for raw in table_ids or []:
        bare = _strip_fqn(str(raw)).lower()
        if not bare:
            continue
        schema = load_table_schema(bare)
        if not schema:
            continue
        cols_raw = schema.get("columns")
        names: set[str] = set()
        if isinstance(cols_raw, list):
            for col in cols_raw:
                if not isinstance(col, dict):
                    continue
                name = str(col.get("name") or "").strip()
                if name:
                    names.add(name)
        if names:
            out[bare] = names
    return out


def columns_for_mart_tables(table_ids: set[str] | list[str]) -> dict[str, set[str]]:
    """Return column allowlists from mart YAMLs (includes dim join targets)."""
    return _columns_for_schemas(table_ids, load_mart_table_schema)


def _columns_for_schemas(table_ids: set[str] | list[str], loader) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for raw in table_ids or []:
        bare = _strip_fqn(str(raw)).lower()
        if not bare:
            continue
        schema = loader(bare)
        if not schema:
            continue
        cols_raw = schema.get("columns")
        names: set[str] = set()
        if isinstance(cols_raw, list):
            for col in cols_raw:
                if not isinstance(col, dict):
                    continue
                name = str(col.get("name") or "").strip()
                if name:
                    names.add(name)
        if names:
            out[bare] = names
    return out


def value_samples_for_tables(
    table_ids: set[str] | list[str],
) -> dict[str, dict[str, list[str]]]:
    """Return ``{bare_table: {column: [sample_values]}}`` from YAML ``*_value_samples``.

    Sample order follows the YAML list (first listed label is the catalog default).
    """
    return _value_samples_for_schemas(table_ids, load_table_schema)


def value_samples_for_mart_tables(
    table_ids: set[str] | list[str],
) -> dict[str, dict[str, list[str]]]:
    """Return mart table column samples from ``bq_mart_tables_yaml_files``."""
    return _value_samples_for_schemas(table_ids, load_mart_table_schema)


def _value_samples_for_schemas(
    table_ids: set[str] | list[str],
    loader,
) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for raw in table_ids or []:
        bare = _strip_fqn(str(raw)).lower()
        if not bare:
            continue
        schema = loader(bare)
        if not schema:
            continue
        yaml_cols: set[str] = set()
        cols_raw = schema.get("columns")
        if isinstance(cols_raw, list):
            for col in cols_raw:
                if isinstance(col, dict):
                    n = str(col.get("name") or "").strip()
                    if n:
                        yaml_cols.add(n)
        by_col: dict[str, list[str]] = {}
        for sample_key, samples in schema.items():
            if not isinstance(sample_key, str) or not sample_key.endswith(_SAMPLE_KEY_SUFFIX):
                continue
            if not isinstance(samples, list) or not samples:
                continue
            vals: list[str] = []
            seen: set[str] = set()
            for item in samples:
                text = str(item).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                vals.append(text)
            if not vals:
                continue
            col_name = column_for_sample_key(sample_key)
            target = col_name
            if col_name not in yaml_cols:
                if sample_key in ("item_value_samples", "product_value_samples") and "product_name" in yaml_cols:
                    target = "product_name"
                elif col_name == "product" and "product_name" in yaml_cols:
                    target = "product_name"
            existing = by_col.setdefault(target, [])
            existing_set = set(existing)
            for text in vals:
                if text not in existing_set:
                    existing.append(text)
                    existing_set.add(text)
        if by_col:
            out[bare] = by_col
    return out


# --- formatting -------------------------------------------------------------

_MAX_LINE = 140
_MAX_COL_DESC = 600
_MAX_COLUMNS = 30
_MAX_VALUE_SAMPLES = 400
_VALUE_SAMPLE_MATCH_CAP = 80
_VALUE_SAMPLE_HEAD_KEEP = 12
_VALUE_SAMPLE_KEYS = frozenset(_SAMPLE_KEY_TO_COLUMN.keys())
_GUIDANCE_LIST_KEYS = frozenset(
    {
        "filtering_guidance",
        "sql_generation_hints",
        "business_questions_supported",
        "aggregation_rules",
    }
)
_MAX_GUIDANCE_ITEMS = 24


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _normalize_query_terms(query_terms: list[str] | None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in query_terms or []:
        t = str(raw).strip().lower()
        if len(t) < 2 or t in seen:
            continue
        seen.add(t)
        terms.append(t)
    return terms


def _prefer_matching_samples(
    items: list[str],
    query_terms: list[str] | None,
    *,
    max_items: int,
    head_keep: int = _VALUE_SAMPLE_HEAD_KEEP,
) -> list[str]:
    """Prefer enum values that match query terms; keep a short discovery head."""
    if not items:
        return []
    cap = max(1, min(max_items, _VALUE_SAMPLE_MATCH_CAP if query_terms else max_items))
    terms = _normalize_query_terms(query_terms)
    if not terms:
        return items[:cap]
    matched: list[str] = []
    seen: set[str] = set()
    for item in items:
        low = item.lower()
        if any(term in low for term in terms):
            if item not in seen:
                matched.append(item)
                seen.add(item)
            if len(matched) >= cap:
                return matched
    head = max(0, min(head_keep, cap - len(matched)))
    for item in items[:head]:
        if item not in seen:
            matched.append(item)
            seen.add(item)
        if len(matched) >= cap:
            break
    if len(matched) < cap:
        for item in items:
            if item in seen:
                continue
            matched.append(item)
            seen.add(item)
            if len(matched) >= cap:
                break
    return matched


def _format_value_samples(
    label: str,
    value: Any,
    *,
    max_items: int = _MAX_VALUE_SAMPLES,
    query_terms: list[str] | None = None,
) -> str | None:
    """Render element/product sample lists as multi-line bullets for NL2SQL packs."""
    if not isinstance(value, list) or not value:
        return None
    items = [
        str(x).strip()
        for x in value
        if isinstance(x, (str, int, float, bool)) and str(x).strip()
    ]
    if not items:
        return None
    shown = _prefer_matching_samples(items, query_terms, max_items=max_items)
    lines = [f"{label}:"]
    for item in shown:
        lines.append(f"  - {item}")
    remaining = len(items) - len(shown)
    if remaining > 0:
        lines.append(f"  - … +{remaining} more")
    return "\n".join(lines)


def _format_guidance_list(label: str, value: Any, *, max_items: int = _MAX_GUIDANCE_ITEMS) -> str | None:
    """Render filtering/SQL hint lists as multi-line bullets (not one truncated line)."""
    if not isinstance(value, list) or not value:
        return None
    items = [
        str(x).strip()
        for x in value
        if isinstance(x, (str, int, float, bool)) and str(x).strip()
    ]
    if not items:
        return None
    shown = items[:max_items]
    lines = [f"{label}:"]
    for item in shown:
        lines.append(f"  - {_truncate(item, 220)}")
    remaining = len(items) - len(shown)
    if remaining > 0:
        lines.append(f"  - … +{remaining} more")
    return "\n".join(lines)


def _format_list_field(label: str, value: Any) -> str | None:
    """Render scalar/list/dict YAML node as a single compact line."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        if not text:
            return None
        return f"{label}: {_truncate(text, _MAX_LINE)}"
    if isinstance(value, list):
        flat: list[str] = []
        for item in value:
            if isinstance(item, (str, int, float, bool)):
                flat.append(str(item).strip())
            elif isinstance(item, dict):
                pair = next(iter(item.items()), None)
                if pair is not None:
                    k, v = pair
                    flat.append(f"{k}={_truncate(str(v), 60)}")
            if not flat:
                continue
        if not flat:
            return None
        return f"{label}: " + _truncate(", ".join(filter(None, flat)), _MAX_LINE)
    if isinstance(value, dict):
        bits: list[str] = []
        for k, v in value.items():
            if isinstance(v, (str, int, float, bool)):
                bits.append(f"{k}={_truncate(str(v), 60)}")
            elif isinstance(v, list):
                inner = ", ".join(str(x) for x in v if isinstance(x, (str, int, float, bool)))
                if inner:
                    bits.append(f"{k}=[{_truncate(inner, 80)}]")
        if not bits:
            return None
        return f"{label}: " + _truncate("; ".join(bits), _MAX_LINE)
    return None


def _format_columns(
    columns: Any,
    *,
    max_columns: int = _MAX_COLUMNS,
    column_allowlist: set[str] | frozenset[str] | None = None,
) -> str:
    """Render a YAML columns list as `name (type, role): description` lines."""
    if not isinstance(columns, list):
        return ""
    allow = {c.lower() for c in column_allowlist} if column_allowlist else None
    filtered = columns
    if allow is not None:
        filtered = [
            col
            for col in columns
            if isinstance(col, dict) and str(col.get("name") or "").strip().lower() in allow
        ]
    lines: list[str] = []
    for col in filtered[:max_columns]:
        if not isinstance(col, dict):
            continue
        name = str(col.get("name") or "").strip()
        if not name:
            continue
        typ = str(col.get("type") or "").strip()
        role = str(col.get("semantic_role") or "").strip()
        desc = " ".join(str(col.get("description") or "").split())
        example = col.get("example")
        head = name
        meta_bits = []
        if typ:
            meta_bits.append(typ)
        if role:
            meta_bits.append(role)
        if meta_bits:
            head = f"{name} ({', '.join(meta_bits)})"
        tail = desc
        if example not in (None, ""):
            ex = _truncate(str(example), 40)
            tail = f"{tail} [ex: {ex}]" if tail else f"[ex: {ex}]"
        # Long metric/product glossaries need more than the compact line budget.
        desc_limit = _MAX_COL_DESC if name in ("product_name", "item", "element", "unit", "value") else max(
            40, _MAX_LINE - len(head) - 4
        )
        line = f"  - {head}" + (f": {_truncate(tail, desc_limit)}" if tail else "")
        lines.append(line)
    if isinstance(filtered, list) and len(filtered) > max_columns:
        lines.append(f"  - … {len(filtered) - max_columns} more columns")
    return "\n".join(lines)


# Section ordering optimized for NL-to-SQL prompt usefulness.
_SECTION_ORDER: list[tuple[str, str]] = [
    ("description", "Description"),
    ("grain", "Grain"),
    ("primary_keys", "Primary keys"),
    ("relationships", "Relationships"),
    ("semantic_relationships", "Semantic relationships"),
    ("join_logic", "Join logic"),
    ("time_dimensions", "Time dimensions"),
    ("geography", "Geography columns"),
    ("metrics", "Metric columns"),
    ("scenario_context", "Scenario context"),
    ("semantic_role", "Semantic role"),
    ("indicator_classes", "Indicator classes"),
    ("indicator_families", "Indicator families"),
    ("business_questions_supported", "Business questions supported"),
    ("aggregation_rules", "Aggregation rules"),
    ("filtering_guidance", "Filtering guidance"),
    ("sql_generation_hints", "SQL generation hints"),
    ("element_value_samples", "Element value samples"),
    ("product_value_samples", "Product value samples"),
    ("item_value_samples", "Item value samples"),
    ("unit_value_samples", "Unit value samples"),
    ("donor_value_samples", "Donor value samples"),
    ("purpose_value_samples", "Purpose value samples"),
    ("indicator_value_samples", "Indicator value samples"),
    ("institution_value_samples", "Institution value samples"),
    ("degree_value_samples", "Degree value samples"),
    ("source_value_samples", "Source value samples"),
    ("currency_value_samples", "Currency value samples"),
    ("price_type_value_samples", "Price type value samples"),
    ("market_value_samples", "Market value samples"),
    ("phase_code_value_samples", "Phase code value samples"),
    ("phase_name_value_samples", "Phase name value samples"),
    ("classification_scale_value_samples", "Classification scale value samples"),
    ("scenario_name_value_samples", "Scenario name value samples"),
    ("measure_type_value_samples", "Measure type value samples"),
    ("treatment_value_samples", "Treatment value samples"),
    ("food_value_value_samples", "Food value value samples"),
    ("industry_value_samples", "Industry value samples"),
    ("factor_value_samples", "Factor value samples"),
    ("release_value_samples", "Release value samples"),
    ("data_quality", "Data quality"),
    ("temporal_model", "Temporal model"),
]


def _in_selected_set(table: str, selected_tables: set[str] | None) -> bool:
    if not selected_tables:
        return True
    bare = table.strip().split(".")[-1].lower()
    return bare in {t.lower() for t in selected_tables}


def _format_semantic_relationships(
    value: Any,
    *,
    selected_tables: set[str] | None = None,
) -> str | None:
    """Compact multi-table relationship block for NL2SQL / reasoner packs."""
    if not isinstance(value, dict):
        return None
    lines: list[str] = ["Semantic relationships:"]
    joins = value.get("joins_with")
    if isinstance(joins, list) and joins:
        lines.append("  joins_with:")
        for item in joins[:8]:
            if not isinstance(item, dict):
                continue
            table = str(item.get("table") or "").strip()
            if not table:
                continue
            if selected_tables and not _in_selected_set(table, selected_tables):
                continue
            on = item.get("on")
            on_s = ",".join(str(x) for x in on) if isinstance(on, list) else str(on or "")
            how = str(item.get("how") or "").strip()
            note = str(item.get("note") or "").strip()
            lines.append(
                f"    - {table} on=[{on_s}] how={how}" + (f" ({note})" if note else "")
            )
    comps = value.get("companions")
    if isinstance(comps, list) and comps:
        lines.append("  companions:")
        for item in comps[:6]:
            if not isinstance(item, dict):
                continue
            table = str(item.get("table") or "").strip()
            if selected_tables and not _in_selected_set(table, selected_tables):
                continue
            when = str(item.get("when") or "").strip()
            role = str(item.get("role") or "").strip()
            if table:
                lines.append(f"    - {table} when={when}" + (f" role={role}" if role else ""))
    avoid = value.get("do_not_join")
    if isinstance(avoid, list) and avoid:
        lines.append("  do_not_join:")
        for item in avoid[:6]:
            if not isinstance(item, dict):
                continue
            table = str(item.get("table") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if table:
                lines.append(f"    - {table}: {reason}" if reason else f"    - {table}")
    return "\n".join(lines) if len(lines) > 1 else None


def format_table_schema(
    table_name: str,
    *,
    max_chars: int = 2400,
    max_bytes: int | None = None,
    include_columns: bool = True,
    selected_tables: set[str] | None = None,
    query_terms: list[str] | None = None,
    column_allowlist: set[str] | frozenset[str] | tuple[str, ...] | None = None,
    loader=None,
) -> str:
    """Compact, SQL-prompt-friendly rendering of a per-table YAML schema.

    Returns "" when no YAML is known for the table. Output is bounded by
    ``max_bytes`` (preferred) or ``max_chars``. Value-sample lists prefer
    entries matching ``query_terms`` so large FAOSTAT enums fit the hint budget.
    When ``column_allowlist`` is set, only those columns and their ``*_value_samples``
    are packed (used for spine dim context snippets).
    """
    load_fn = loader or _schema_loader_for(table_name)
    schema = load_fn(table_name)
    if not schema:
        return ""

    allow = {str(c).strip().lower() for c in column_allowlist} if column_allowlist else None

    fqn = str(schema.get("table_name") or table_name).strip().strip("`")
    header = f"Table: {fqn or table_name}"
    parts: list[str] = [header]
    deferred_samples: list[str] = []

    def _sample_col_allowed(sample_key: str) -> bool:
        if allow is None:
            return True
        col = sample_key[: -len(_SAMPLE_KEY_SUFFIX)] if sample_key.endswith(_SAMPLE_KEY_SUFFIX) else sample_key
        return col.lower() in allow

    for key, label in _SECTION_ORDER:
        if key not in schema:
            continue
        if key == "semantic_relationships":
            if allow is not None:
                continue
            block = _format_semantic_relationships(
                schema[key],
                selected_tables=selected_tables,
            )
            if block:
                parts.append(block)
            continue
        if key.endswith(_SAMPLE_KEY_SUFFIX) or key in _VALUE_SAMPLE_KEYS:
            if not _sample_col_allowed(key):
                continue
            block = _format_value_samples(
                label,
                schema[key],
                query_terms=query_terms,
            )
            if block:
                deferred_samples.append(block)
            continue
        if key in _GUIDANCE_LIST_KEYS:
            block = _format_guidance_list(label, schema[key])
            if block:
                parts.append(block)
            continue
        if allow is not None:
            continue
        line = _format_list_field(label, schema[key])
        if line:
            parts.append(line)

    # Pack any remaining *_value_samples not listed in _SECTION_ORDER.
    seen_sample_keys = {k for k, _ in _SECTION_ORDER if k.endswith(_SAMPLE_KEY_SUFFIX)}
    for key, value in schema.items():
        if not isinstance(key, str) or not key.endswith(_SAMPLE_KEY_SUFFIX):
            continue
        if key in seen_sample_keys:
            continue
        if not _sample_col_allowed(key):
            continue
        label = key.replace("_", " ").strip().title()
        block = _format_value_samples(label, value, query_terms=query_terms)
        if block:
            deferred_samples.append(block)

    # Columns before enum samples so byte truncation keeps schema usable.
    if include_columns and isinstance(schema.get("columns"), list):
        col_block = _format_columns(
            schema["columns"],
            column_allowlist=allow,
        )
        if col_block:
            parts.append("Columns:")
            parts.append(col_block)

    parts.extend(deferred_samples)

    text = "\n".join(parts)
    budget = max_bytes if max_bytes is not None else max_chars
    if budget <= 0:
        return ""
    if max_bytes is not None:
        out, _ = truncate_utf8(text, budget)
        return out
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 1)].rstrip() + "…"


def list_staging_table_index() -> list[dict[str, Any]]:
    """Compact catalog for the SQL reasoner (one row per unique ``stg_*`` YAML)."""
    index = _build_index()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for key, schema in index.items():
        if not isinstance(schema, dict):
            continue
        fqn = str(schema.get("table_name") or "").strip().strip("`")
        bare = _strip_fqn(fqn) or (key if key.startswith("stg_") else "")
        if not bare.startswith("stg_") or bare in seen:
            continue
        seen.add(bare)
        raw_role = schema.get("semantic_role")
        role: dict[str, Any] = raw_role if isinstance(raw_role, dict) else {}
        raw_tags = role.get("supports")
        tags: list[Any] = raw_tags if isinstance(raw_tags, list) else []
        raw_source = schema.get("source")
        source: dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
        domain = str(
            role.get("primary_domain")
            or source.get("domain")
            or schema.get("entity_type")
            or ""
        ).strip()
        rows.append(
            {
                "table_id": bare,
                "fqn": fqn or bare,
                "description": str(schema.get("description") or "").strip(),
                "grain": str(schema.get("grain") or "").strip(),
                "domain": domain,
                "tags": [str(t).strip() for t in tags if str(t).strip()],
                "rels": compact_rels_summary(bare),
            }
        )
    rows.sort(key=lambda r: r["table_id"])
    return rows


def format_reasoner_index(
    *,
    max_bytes: int | None = None,
    table_ids: list[str] | None = None,
    domains: list[str] | None = None,
) -> tuple[str, bool]:
    """Byte-capped one-line-per-table index for the SQL reasoner prompt.

    When ``table_ids`` or ``domains`` are provided, prefer matching rows first
    (ontology scope). If the filter yields nothing, fall back to the full index.
    """
    budget = reasoner_index_max_bytes() if max_bytes is None else max(0, max_bytes)
    prefer = {str(t).strip().split(".")[-1].lower() for t in (table_ids or []) if str(t).strip()}
    prefer_domains = {str(d).strip().lower() for d in (domains or []) if str(d).strip()}
    rows = list_staging_table_index()
    if prefer or prefer_domains:
        scoped = [
            r
            for r in rows
            if (prefer and str(r.get("table_id") or "").lower() in prefer)
            or (
                prefer_domains
                and str(r.get("domain") or "").lower() in prefer_domains
            )
        ]
        # Always include explicit candidate table ids even if domain mismatch.
        if prefer:
            have = {str(r.get("table_id") or "").lower() for r in scoped}
            for r in rows:
                tid = str(r.get("table_id") or "").lower()
                if tid in prefer and tid not in have:
                    scoped.append(r)
                    have.add(tid)
        if scoped:
            rows = scoped
    lines: list[str] = []
    for row in rows:
        tags = ", ".join(row.get("tags") or [])
        desc = str(row.get("description") or "")
        if len(desc) > 120:
            desc = desc[:119].rstrip() + "…"
        rels = str(row.get("rels") or compact_rels_summary(str(row["table_id"])))
        if len(rels) > 140:
            rels = rels[:139].rstrip() + "…"
        lines.append(
            f"- {row['table_id']} | domain={row.get('domain') or '-'} | "
            f"grain={row.get('grain') or '-'} | tags={tags or '-'} | "
            f"rels={rels} | {desc}"
        )
    return pack_lines(lines, budget)


def pack_selected_table_hints(
    table_ids: list[str],
    *,
    max_bytes: int | None = None,
    query_terms: list[str] | None = None,
) -> tuple[list[str], bool]:
    """Full YAML packs for selected tables, truncated to the NL2SQL hint byte budget."""
    budget = hint_max_bytes() if max_bytes is None else max(0, max_bytes)
    if budget <= 0 or not table_ids:
        return [], bool(table_ids)
    selected = {str(t).strip().split(".")[-1].lower() for t in table_ids if str(t).strip()}
    per = max(400, budget // max(1, len(table_ids)))
    hints: list[str] = []
    used = 0
    truncated = False
    known_count = 0
    for tid in table_ids:
        if not load_table_schema(tid):
            continue
        known_count += 1
        if used > 0:
            remain_total = budget - used - 1  # newline separator when joined
        else:
            remain_total = budget - used
        if remain_total <= 0:
            truncated = True
            break
        block = format_table_schema(
            tid,
            max_bytes=min(per, remain_total),
            include_columns=True,
            selected_tables=selected,
            query_terms=query_terms,
        )
        if not block:
            continue
        cost = utf8_len(block) + (1 if hints else 0)
        if used + cost > budget:
            frag, _ = truncate_utf8(block, remain_total)
            if frag:
                hints.append(frag)
            truncated = True
            break
        hints.append(block)
        used += cost
    return hints, truncated or len(hints) < known_count


_MART_INDEX_PREFIXES = ("fct_", "agg_", "dim_")


def list_mart_table_index() -> list[dict[str, Any]]:
    """Compact catalog for the mart SQL reasoner (fct/agg/dim YAMLs)."""
    index = _build_mart_index()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for key, schema in index.items():
        if not isinstance(schema, dict):
            continue
        fqn = str(schema.get("table_name") or "").strip().strip("`")
        bare = _strip_fqn(fqn) or (key if key.startswith(_MART_INDEX_PREFIXES) else "")
        if not bare.startswith(_MART_INDEX_PREFIXES) or bare in seen:
            continue
        seen.add(bare)
        raw_role = schema.get("semantic_role")
        role: dict[str, Any] = raw_role if isinstance(raw_role, dict) else {}
        raw_tags = role.get("supports")
        tags: list[Any] = raw_tags if isinstance(raw_tags, list) else []
        raw_ic = schema.get("indicator_classes")
        iclasses: list[str] = [str(c) for c in raw_ic] if isinstance(raw_ic, list) else indicator_classes_for_table(bare)
        raw_fams = schema.get("indicator_families")
        fam_ids: list[str] = []
        if isinstance(raw_fams, list):
            fam_ids = [str(f.get("id") or "") for f in raw_fams if isinstance(f, dict) and f.get("id")]
        if not fam_ids:
            fam_ids = [str(f.get("id") or "") for f in families_for_fact(bare) if f.get("id")]
        domain = str(
            role.get("primary_domain")
            or schema.get("entity_type")
            or ""
        ).strip()
        rows.append(
            {
                "table_id": bare,
                "fqn": fqn or bare,
                "description": str(schema.get("description") or "").strip(),
                "grain": str(schema.get("grain") or "").strip(),
                "domain": domain,
                "tags": [str(t).strip() for t in tags if str(t).strip()],
                "indicator_classes": iclasses,
                "families": fam_ids,
                "rels": mart_compact_rels_summary(bare),
            }
        )
    rows.sort(key=lambda r: r["table_id"])
    return rows


def format_mart_reasoner_index(
    *,
    max_bytes: int | None = None,
    table_ids: list[str] | None = None,
    domains: list[str] | None = None,
    indicator_classes: list[str] | None = None,
) -> tuple[str, bool]:
    """Byte-capped mart index for the SQL reasoner (class-scoped when possible)."""
    budget = reasoner_index_max_bytes() if max_bytes is None else max(0, max_bytes)
    prefer = {str(t).strip().split(".")[-1].lower() for t in (table_ids or []) if str(t).strip()}
    prefer_domains = {str(d).strip().lower() for d in (domains or []) if str(d).strip()}
    prefer_classes = {str(c).strip().upper() for c in (indicator_classes or []) if str(c).strip()}
    rows = list_mart_table_index()
    if prefer or prefer_domains or prefer_classes:
        scoped = [
            r
            for r in rows
            if (prefer and str(r.get("table_id") or "").lower() in prefer)
            or (
                prefer_domains
                and str(r.get("domain") or "").lower() in prefer_domains
            )
            or (
                prefer_classes
                and prefer_classes.intersection({str(c).upper() for c in (r.get("indicator_classes") or [])})
            )
        ]
        if prefer:
            have = {str(r.get("table_id") or "").lower() for r in scoped}
            for r in rows:
                tid = str(r.get("table_id") or "").lower()
                if tid in prefer and tid not in have:
                    scoped.append(r)
                    have.add(tid)
        if scoped:
            rows = scoped
    lines: list[str] = []
    for row in rows:
        tags = ", ".join(row.get("tags") or [])
        ic = ",".join(row.get("indicator_classes") or []) or "-"
        fams = ",".join(row.get("families") or []) or "-"
        desc = str(row.get("description") or "")
        if len(desc) > 120:
            desc = desc[:119].rstrip() + "…"
        rels = str(row.get("rels") or mart_compact_rels_summary(str(row["table_id"])))
        if len(rels) > 140:
            rels = rels[:139].rstrip() + "…"
        lines.append(
            f"- {row['table_id']} | classes={ic} | families={fams} | "
            f"domain={row.get('domain') or '-'} | grain={row.get('grain') or '-'} | "
            f"tags={tags or '-'} | rels={rels} | {desc}"
        )
    return pack_lines(lines, budget)


def pack_mart_table_hints(
    table_ids: list[str],
    *,
    max_bytes: int | None = None,
    query_terms: list[str] | None = None,
) -> tuple[list[str], bool]:
    """Full mart YAML packs for selected tables, truncated to the NL2SQL hint byte budget.

    Also packs spine-target dim YAMLs (expose columns + samples) so FK joins carry
    rich dim context without inventing filter literals.
    """
    budget = hint_max_bytes() if max_bytes is None else max(0, max_bytes)
    if budget <= 0 or not table_ids:
        return [], bool(table_ids)
    selected = {str(t).strip().split(".")[-1].lower() for t in table_ids if str(t).strip()}
    spine_dims: list[str] = []
    seen_dims: set[str] = set()
    for tid in table_ids:
        bare = str(tid).strip().split(".")[-1].lower()
        for _fk, dim_ref in bind_spine_map(bare).items():
            dim = str(dim_ref).split(".", 1)[0].strip().lower()
            if not dim or dim in selected or dim in seen_dims:
                continue
            if not dim_expose_columns(dim):
                continue
            seen_dims.add(dim)
            spine_dims.append(dim)

    pack_ids = list(table_ids) + spine_dims
    per = max(400, budget // max(1, len(pack_ids)))
    hints: list[str] = []
    used = 0
    truncated = False
    known_count = 0
    for tid in pack_ids:
        if not load_mart_table_schema(tid):
            continue
        known_count += 1
        remain_total = budget - used - (1 if hints else 0)
        if remain_total <= 0:
            truncated = True
            break
        bare = str(tid).strip().split(".")[-1].lower()
        allow = dim_expose_columns(bare) if bare in seen_dims else None
        # Dim packs get a smaller share so fact YAMLs stay primary.
        dim_budget = min(per, remain_total, 900) if allow else min(per, remain_total)
        block = format_table_schema(
            tid,
            max_bytes=dim_budget,
            include_columns=True,
            selected_tables=selected | seen_dims,
            query_terms=query_terms,
            column_allowlist=allow,
            loader=load_mart_table_schema,
        )
        if not block:
            continue
        cost = utf8_len(block) + (1 if hints else 0)
        if used + cost > budget:
            frag, _ = truncate_utf8(block, remain_total)
            if frag:
                hints.append(frag)
            truncated = True
            break
        hints.append(block)
        used += cost
    return hints, truncated or len(hints) < known_count
