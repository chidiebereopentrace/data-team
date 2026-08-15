"""Per-table YAML loader for BigQuery semantic schemas under ``ml/rag/bq_tables_yaml_files``.

Staging_dev ``stg_*`` YAMLs are the sole table catalog for Ask ADZA NL-to-SQL
(no Qdrant table-description matching). Each YAML carries grain, keys, hints,
and columns for the SQL reasoner and NL-to-SQL prompts.

Public API:
- ``load_table_schema(name)``       -> raw dict for the table, or None.
- ``format_table_schema(name, ...)`` -> compact SQL-prompt block string, or "".
- ``known_table_names()``           -> set[str] of all indexed names (bare + FQN).
- ``list_staging_table_index()``    -> compact index rows for the SQL reasoner.
- ``pack_selected_table_hints(...)`` -> byte-capped full YAML packs for NL2SQL.
- ``columns_for_tables(...)``       -> YAML column names per table (SQL allowlist).
- ``value_samples_for_tables(...)`` -> enum/sample labels per column for soft checks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ml.rag.chatbot.bq_byte_budget import hint_max_bytes, pack_lines, reasoner_index_max_bytes, truncate_utf8, utf8_len
from ml.rag.helpers.staging_semantic_relationships import compact_rels_summary

# bq_table_schema_yaml.py lives at ml/rag/chatbot/, YAMLs live at ml/rag/bq_tables_yaml_files/.
_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "bq_tables_yaml_files"

# Cache: (cache_key, index)
_cache: tuple[tuple[Any, ...], dict[str, dict[str, Any]]] | None = None


def _yaml_dir() -> Path:
    raw = os.environ.get("RAG_BQ_TABLES_YAML_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_DIR.resolve()


def _strip_fqn(table_name: str) -> str:
    """Return the bare table name (last dotted segment) from a possibly fully-qualified id."""
    text = (table_name or "").strip().strip("`")
    if not text:
        return ""
    return text.split(".")[-1]


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


def _build_index() -> dict[str, dict[str, Any]]:
    """Load all YAML files; cache by (dir_path, dir_mtime, sorted file mtimes)."""
    global _cache
    directory = _yaml_dir()
    try:
        dir_mtime = directory.stat().st_mtime_ns if directory.is_dir() else None
    except OSError:
        dir_mtime = None
    file_sig: tuple[tuple[str, int], ...] = tuple()
    if dir_mtime is not None and directory.is_dir():
        try:
            file_sig = tuple(
                sorted(
                    (p.name, p.stat().st_mtime_ns)
                    for p in directory.glob("*.yml")
                )
            )
        except OSError:
            file_sig = tuple()
    cache_key = ("v1", str(directory), dir_mtime, file_sig)
    if _cache is not None and _cache[0] == cache_key:
        return _cache[1]
    index = _index_yaml_files(directory)
    _cache = (cache_key, index)
    return index


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


# Aligned with bq_sql_validate metric-discriminator sets.
_CORE_METRIC_DISCRIMINATORS = frozenset(
    {
        "element",
        "indicator",
        "price_type",
        "measure_type",
        "treatment",
    }
)
_GRAIN_METRIC_DISCRIMINATORS = frozenset(
    {
        "classification_scale",
        "scenario_name",
    }
)

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
    schema = load_table_schema(table_id)
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


def table_source_meta(table_id: str) -> dict[str, Any]:
    """Compact table-level metadata for BQ context enrichment."""
    schema = load_table_schema(table_id) or {}
    bare = _strip_fqn(table_id).lower()
    source_obj = schema.get("source")
    source: dict[str, Any] = source_obj if isinstance(source_obj, dict) else {}
    semantic_obj = schema.get("semantic_role")
    semantic: dict[str, Any] = semantic_obj if isinstance(semantic_obj, dict) else {}
    supports = semantic.get("supports")
    return {
        "table_id": bare,
        "table_name": str(schema.get("table_name") or bare).strip(),
        "description": " ".join(str(schema.get("description") or "").split())[:500],
        "grain": str(schema.get("grain") or "").strip(),
        "entity_type": str(schema.get("entity_type") or "").strip(),
        "source_layer": str(source.get("layer") or "staging_dev").strip(),
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


def value_samples_for_tables(
    table_ids: set[str] | list[str],
) -> dict[str, dict[str, set[str]]]:
    """Return ``{bare_table: {column: {sample_values}}}`` from YAML ``*_value_samples``."""
    out: dict[str, dict[str, set[str]]] = {}
    for raw in table_ids or []:
        bare = _strip_fqn(str(raw)).lower()
        if not bare:
            continue
        schema = load_table_schema(bare)
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
        by_col: dict[str, set[str]] = {}
        for sample_key, samples in schema.items():
            if not isinstance(sample_key, str) or not sample_key.endswith(_SAMPLE_KEY_SUFFIX):
                continue
            if not isinstance(samples, list) or not samples:
                continue
            vals: set[str] = {str(item).strip() for item in samples if str(item).strip()}
            if not vals:
                continue
            col_name = column_for_sample_key(sample_key)
            target = col_name
            if col_name not in yaml_cols:
                if sample_key in ("item_value_samples", "product_value_samples") and "product_name" in yaml_cols:
                    target = "product_name"
                elif col_name == "product" and "product_name" in yaml_cols:
                    target = "product_name"
            by_col.setdefault(target, set()).update(vals)
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


def _format_columns(columns: Any, *, max_columns: int = _MAX_COLUMNS) -> str:
    """Render a YAML columns list as `name (type, role): description` lines."""
    if not isinstance(columns, list):
        return ""
    lines: list[str] = []
    for col in columns[:max_columns]:
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
    if isinstance(columns, list) and len(columns) > max_columns:
        lines.append(f"  - … {len(columns) - max_columns} more columns")
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
) -> str:
    """Compact, SQL-prompt-friendly rendering of a per-table YAML schema.

    Returns "" when no YAML is known for the table. Output is bounded by
    ``max_bytes`` (preferred) or ``max_chars``. Value-sample lists prefer
    entries matching ``query_terms`` so large FAOSTAT enums fit the hint budget.
    """
    schema = load_table_schema(table_name)
    if not schema:
        return ""

    fqn = str(schema.get("table_name") or table_name).strip().strip("`")
    header = f"Table: {fqn or table_name}"
    parts: list[str] = [header]
    deferred_samples: list[str] = []

    for key, label in _SECTION_ORDER:
        if key not in schema:
            continue
        if key == "semantic_relationships":
            block = _format_semantic_relationships(
                schema[key],
                selected_tables=selected_tables,
            )
            if block:
                parts.append(block)
            continue
        if key.endswith(_SAMPLE_KEY_SUFFIX) or key in _VALUE_SAMPLE_KEYS:
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
        label = key.replace("_", " ").strip().title()
        block = _format_value_samples(label, value, query_terms=query_terms)
        if block:
            deferred_samples.append(block)

    # Columns before enum samples so byte truncation keeps schema usable.
    if include_columns and isinstance(schema.get("columns"), list):
        col_block = _format_columns(schema["columns"])
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
