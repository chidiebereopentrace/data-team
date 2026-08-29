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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ml.rag.chatbot.bq_byte_budget import hint_max_bytes, pack_lines, reasoner_index_max_bytes, truncate_utf8, utf8_len
from ml.rag.chatbot.mart_indicator_classes import (
    families_for_fact,
    indicator_classes_for_table,
)
from ml.rag.helpers.mart_semantic_relationships import SEMANTIC_RELATIONSHIPS
from ml.rag.helpers.mart_semantic_relationships import compact_rels_summary as mart_compact_rels_summary
from ml.rag.helpers.mart_semantic_relationships import format_join_fragments_for_nl2sql as mart_join_fragments
from ml.rag.helpers.staging_semantic_relationships import compact_rels_summary

# bq_table_schema_yaml.py lives at ml/rag/chatbot/, YAMLs live at ml/rag/bq_tables_yaml_files/.
_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "bq_tables_yaml_files"
_DEFAULT_MART_DIR = Path(__file__).resolve().parents[1] / "bq_mart_tables_yaml_files"

# Cache: (cache_key, index)
_cache: tuple[tuple[Any, ...], dict[str, dict[str, Any]]] | None = None
_mart_cache: tuple[tuple[Any, ...], dict[str, dict[str, Any]]] | None = None


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


def _build_mart_index() -> dict[str, dict[str, Any]]:
    """Load all mart YAML files; cache by (dir_path, dir_mtime, sorted file mtimes)."""
    global _mart_cache
    directory = _mart_yaml_dir()
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
    if _mart_cache is not None and _mart_cache[0] == cache_key:
        return _mart_cache[1]
    index = _index_yaml_files(directory)
    _mart_cache = (cache_key, index)
    return index


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
        "n",
        "e",
        "c",
        "other",
        "products",
        "nes",
        "nec",
    }
)
_PREFERRED_DISCRIMINATOR_DEFAULTS = (
    "Production",
    "Export quantity",
    "Producer Price (USD/tonne)",
    "Retail",
    "population",
)
_AUTO_DEFAULT_DISCRIMINATOR_COLS = frozenset({"element", "price_type", "measure_type"})
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_YIELD_QUERY_RE = re.compile(r"\byields?\b", re.IGNORECASE)
_PRODUCTION_QUERY_RE = re.compile(
    r"\b(production|produced|output|tonnes?|tons?)\b",
    re.IGNORECASE,
)
_PRODUCER_PRICE_QUERY_RE = re.compile(r"\b(producer|farm\s*gate)\b", re.IGNORECASE)
_WHOLESALE_QUERY_RE = re.compile(r"\bwholesale\b", re.IGNORECASE)
_POPULATION_QUERY_RE = re.compile(r"\b(population|people|ipc)\b", re.IGNORECASE)
_CLASSIFICATION_QUERY_RE = re.compile(r"\b(classification|phase)\b", re.IGNORECASE)


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
    return None


def year_column(table_id: str) -> str | None:
    names = _column_names(table_id)
    by_low = {n.lower(): n for n in names}
    for cand in _YEAR_COLUMN_CANDIDATES:
        if cand in by_low:
            return by_low[cand]
    return None


def resolve_geo_filter_values(table_id: str, labels: list[str] | None) -> list[str]:
    """Map decomposition geography labels to warehouse filter values for the table geo column."""
    col = geo_column(table_id)
    if not col or not labels:
        return [str(g).strip() for g in (labels or []) if str(g).strip()]
    out: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        label = str(raw).strip()
        if not label:
            continue
        if col == "country_iso3":
            if _ISO3_RE.match(label.upper()):
                val = label.upper()
            else:
                val = _AFRICA_COUNTRY_ISO3.get(label, label)
        else:
            val = label
        key = val.lower()
        if key not in seen:
            seen.add(key)
            out.append(val)
    return out


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


def _alnum_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def expand_speech_text(text: str) -> str:
    out = (text or "").lower()
    for src, dst in _SPEECH_SYNONYMS.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out)
    return out


def match_value_samples(blob: str, samples: set[str] | list[str]) -> list[str]:
    """Pick warehouse labels whose head token appears in the query (speech synonyms expanded)."""
    expanded = expand_speech_text(blob)
    blob_tokens = set(_alnum_tokens(expanded))
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
        head = _SPEECH_SYNONYMS.get(toks[0], toks[0])
        if head not in blob_tokens:
            continue
        grouped.setdefault(head, []).append(sample)
    chosen: list[str] = []
    for head, group in grouped.items():
        scored: list[tuple[Any, ...]] = []
        for sample in group:
            extra = [
                t
                for t in _alnum_tokens(sample)
                if t not in _GENERIC_SAMPLE_TOKENS and _SPEECH_SYNONYMS.get(t, t) != head
            ]
            extra_hits = sum(
                1 for t in extra if t in blob_tokens or _SPEECH_SYNONYMS.get(t, t) in blob_tokens
            )
            if extra and extra_hits == 0:
                scored.append((1, len(extra), sample_order.get(sample, 10**9), sample))
            else:
                scored.append((0, -extra_hits, len(sample), sample))
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


def _resolve_product_labels_for_table(
    table_id: str,
    *,
    blob: str,
    labels: list[str] | None = None,
) -> list[str]:
    """Resolve speech or hints to dictionary labels for one table's filter column."""
    bare = _strip_fqn(table_id).lower()
    table_samples = column_samples_for_table(bare, _PRODUCT_LABEL_COL)
    if table_samples:
        hinted = [str(x).strip() for x in (labels or []) if str(x).strip()]
        if hinted:
            in_table = [h for h in hinted if h in table_samples]
            if in_table:
                return in_table[:6]
        matched = match_value_samples(blob, table_samples)
        if matched:
            return matched[:6]
        if hinted:
            return hinted[:6]
        return []

    prefer = [bare, *_PRODUCT_PREFER_TABLES]
    hinted = [str(x).strip() for x in (labels or []) if str(x).strip()]
    if hinted:
        return hinted[:6]
    return resolve_dictionary_labels(
        column=_PRODUCT_LABEL_COL,
        blob=blob,
        prefer_tables=prefer,
    )


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
    resolved = _resolve_product_labels_for_table(bare, blob=blob, labels=labels)
    if not resolved:
        return "", []

    if _fact_has_denormalized_product_name(bare):
        col = _PRODUCT_LABEL_COL
        if len(resolved) == 1:
            return f"AND {col} = {_sql_literal(resolved[0])} ", resolved
        lits = ", ".join(_sql_literal(v) for v in resolved[:16])
        return f"AND {col} IN ({lits}) ", resolved

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


def compile_semantic_filter(
    table_id: str,
    facet: str,
    *,
    blob: str,
    project_id: str,
    dataset: str,
    labels: list[str] | None = None,
) -> SemanticFilterClause | None:
    """Compile one semantic facet filter (dictionary label + join routing)."""
    facet_l = str(facet or "").strip().lower()
    if facet_l != "product":
        return None
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


def match_product_samples(table_id: str, blob: str) -> list[str]:
    """Dictionary-resolved product labels for SQL filters (not fact-column hash samples)."""
    return _resolve_product_labels_for_table(table_id, blob=blob)


def default_discriminator_value(
    col: str,
    samples: set[str] | list[str] | None,
    *,
    query: str = "",
) -> str | None:
    """Pick a YAML sample for a discriminator column. Never invent labels."""
    sample_set = {str(s).strip() for s in (samples or []) if str(s).strip()}
    if not sample_set:
        return None
    by_low = {s.lower(): s for s in sample_set}
    col_l = (col or "").strip().lower()
    q = query or ""

    def _from_cands(*cands: str) -> str | None:
        for cand in cands:
            hit = by_low.get(cand.lower())
            if hit:
                return hit
        return None

    if col_l == "element":
        if _YIELD_QUERY_RE.search(q):
            hit = _from_cands("Yield")
            if hit:
                return hit
        if _PRODUCTION_QUERY_RE.search(q):
            hit = _from_cands("Production")
            if hit:
                return hit

    if col_l == "metric":
        if _YIELD_QUERY_RE.search(q):
            hit = _from_cands("production_yield_physical")
            if hit:
                return hit
        if _PRODUCTION_QUERY_RE.search(q):
            hit = _from_cands("production_production_physical")
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

    for preferred in _PREFERRED_DISCRIMINATOR_DEFAULTS:
        hit = by_low.get(preferred.lower())
        if hit:
            return hit

    ordered = sorted(sample_set, key=lambda s: (len(s), s.lower()))
    return ordered[0]


def discriminator_equality_filters(table_id: str, blob: str) -> list[tuple[str, str]]:
    """``(column, sample)`` filters from YAML samples; prefer query matches then known defaults."""
    samples_map = value_samples_for_tables({table_id}).get(_strip_fqn(table_id).lower()) or {}
    product_col = (product_column(table_id) or "").lower()
    skip = {product_col, "unit", "country", "country_name", "market_name"} - {""}
    out: list[tuple[str, str]] = []
    for col, samples in samples_map.items():
        if col.lower() in skip:
            continue
        matched = match_value_samples(blob, samples)
        if matched:
            out.append((col, matched[0]))
            continue
        if col.lower() not in _AUTO_DEFAULT_DISCRIMINATOR_COLS:
            continue
        default = default_discriminator_value(col, samples, query=blob)
        if default:
            out.append((col, default))
    return out


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
    loader=None,
) -> str:
    """Compact, SQL-prompt-friendly rendering of a per-table YAML schema.

    Returns "" when no YAML is known for the table. Output is bounded by
    ``max_bytes`` (preferred) or ``max_chars``. Value-sample lists prefer
    entries matching ``query_terms`` so large FAOSTAT enums fit the hint budget.
    """
    load_fn = loader or _schema_loader_for(table_name)
    schema = load_fn(table_name)
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
    """Full mart YAML packs for selected tables, truncated to the NL2SQL hint byte budget."""
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
        if not load_mart_table_schema(tid):
            continue
        known_count += 1
        remain_total = budget - used - (1 if hints else 0)
        if remain_total <= 0:
            truncated = True
            break
        block = format_table_schema(
            tid,
            max_bytes=min(per, remain_total),
            include_columns=True,
            selected_tables=selected,
            query_terms=query_terms,
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
