"""BigQuery NL2SQL validation: table allowlist, YAML columns, value samples, dry-run."""
from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import (
    CORE_METRIC_DISCRIMINATORS as _CORE_METRIC_DISCRIMINATORS,
    GRAIN_METRIC_DISCRIMINATORS as _GRAIN_METRIC_DISCRIMINATORS,
    TableBindContract,
    columns_for_mart_tables,
    columns_for_tables,
    default_discriminator_value,
    dictionary_samples_for_column,
    geo_column,
    load_mart_table_schema,
    load_table_schema,
    value_samples_for_mart_tables,
    value_samples_for_tables,
    year_column,
)
from ml.rag.chatbot.price_vocab import price_vocab_values
from ml.rag.chatbot.sql_ast_validate import parse_select, physical_tables, unknown_columns
from ml.rag.chatbot.value_index import complete_enum
from ml.rag.helpers.mart_semantic_relationships import SEMANTIC_RELATIONSHIPS

_SHARED_DICTIONARY_COLUMNS = frozenset({"product_name"})

# Profiling stores at most this many samples per column; a shorter list is the whole domain.
PROFILE_SAMPLE_CAP = 50

_DIMENSION_LABEL_COLUMNS: dict[str, tuple[str, str]] = {
    "product_name": ("dim_product", "product_name"),
    "element": ("dim_element", "element_name"),
    "indicator": ("dim_indicator", "indicator_name"),
    "indicator_name": ("dim_indicator", "indicator_name"),
}

_LARGE_PARTITIONED_TABLES = frozenset({"fct_production", "fct_trade", "fct_prices", "fct_yield"})

_YIELD_ELEMENT_VALUES = frozenset({"yield"})
_YIELD_METRIC_TOKENS = ("yield",)
_PRODUCTION_ELEMENT_VALUES = frozenset({"production"})

_MART_TABLE_RE = re.compile(
    r"\b(?:fct|agg|dim|bridge)_[a-z0-9_]+\b",
    re.IGNORECASE,
)
_STG_TABLE_RE = re.compile(r"\bstg_[a-z0-9_]+\b", re.IGNORECASE)
_TABLE_HEADER_RE = re.compile(
    r"^Table:\s*[`\w.-]*\.?((?:fct|agg|dim|bridge|stg)_[a-z0-9_]+)",
    re.IGNORECASE | re.MULTILINE,
)
# FROM/JOIN target: backticked FQN or dotted/bare identifier (not a subquery paren).
_FROM_JOIN_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?![\(\s])(`[^`]+`|(?:[A-Za-z_][\w-]*)(?:\.[A-Za-z_][\w-]*)*)",
    re.IGNORECASE,
)
_STRING_LIT_RE = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
_BACKTICK_RE = re.compile(r"`[^`]+`")
_AS_ALIAS_RE = re.compile(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_CTE_NAME_RE = re.compile(r"(?:WITH|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", re.IGNORECASE)
_TABLE_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?:`[^`]+`|[A-Za-z_][\w.]*)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_EQ_FILTER_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'((?:''|[^'])*)'",
    re.IGNORECASE,
)
_IN_FILTER_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s+IN\s*\(([^)]*)\)",
    re.IGNORECASE,
)
_PREDICATE_RE = re.compile(
    r"^(?P<col>[A-Za-z_][A-Za-z0-9_]*)\s*(?:"
    r"(?:NOT\s+)?IN\s*\(|"
    r"(?:NOT\s+)?BETWEEN\b|"
    r"(?:NOT\s+)?LIKE\b|"
    r"IS\s+(?:NOT\s+)?|"
    r"=|<>|!=|>=|<=|>|<"
    r")",
    re.IGNORECASE,
)
_BETWEEN_OPEN_RE = re.compile(
    r"\bBETWEEN\s+(?:DATE\s+)?(?:'(?:''|[^'])*'|\d{4}(?:-\d{2}-\d{2})?)\s*$",
    re.IGNORECASE,
)
# Equality or IN on a column (string or non-string RHS) — used for required filters.
_COL_FILTERED_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*|IN\s*\()",
    re.IGNORECASE,
)
_WHERE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)
_TRAILING_CLAUSE_RE = re.compile(
    r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING|QUALIFY)\b",
    re.IGNORECASE,
)
_YEAR_EQ_RE = re.compile(
    r"\b(year|harvest_year|planting_year)\s*=\s*(\d{4})\b",
    re.IGNORECASE,
)
_PRODUCT_EQ_RE = re.compile(
    r"\b(product_name|product|item)\s*=\s*'((?:''|[^'])*)'",
    re.IGNORECASE,
)
_PRODUCT_IN_RE = re.compile(
    r"\b(product_name|product|item)\s+IN\s*\([^)]*\)",
    re.IGNORECASE,
)

# Metric discriminators live in bq_table_schema_yaml (one list).

_SQL_KEYWORDS = frozenset(
    {
        "select",
        "from",
        "where",
        "and",
        "or",
        "not",
        "in",
        "on",
        "as",
        "join",
        "left",
        "right",
        "inner",
        "outer",
        "full",
        "cross",
        "group",
        "by",
        "order",
        "asc",
        "desc",
        "limit",
        "offset",
        "having",
        "with",
        "union",
        "all",
        "distinct",
        "case",
        "when",
        "then",
        "else",
        "end",
        "null",
        "true",
        "false",
        "is",
        "between",
        "like",
        "ilike",
        "cast",
        "safe_cast",
        "safe_divide",
        "sum",
        "avg",
        "min",
        "max",
        "count",
        "coalesce",
        "ifnull",
        "if",
        "over",
        "partition",
        "row_number",
        "rank",
        "dense_rank",
        "date",
        "datetime",
        "timestamp",
        "extract",
        "interval",
        "unnest",
        "array",
        "struct",
        "exists",
        "except",
        "intersect",
        "qualify",
        "window",
        "using",
        "natural",
        "lateral",
        "values",
        "current_timestamp",
        "current_date",
        "lower",
        "upper",
        "trim",
        "substr",
        "substring",
        "length",
        "round",
        "abs",
        "floor",
        "ceil",
        "greatest",
        "least",
        "concat",
        "format",
        "string",
        "float64",
        "int64",
        "bool",
        "numeric",
        "bignumeric",
    }
)


def dry_run_enabled() -> bool:
    return os.environ.get("RAG_BQ_SQL_DRY_RUN", "on").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def sql_retry_enabled() -> bool:
    try:
        return int(os.environ.get("RAG_BQ_SQL_RETRY", "1") or 1) > 0
    except ValueError:
        return True


def bare_table_ids_from_hints(hints: list[str]) -> set[str]:
    """Extract mart/staging table ids from packed YAML hint blocks."""
    out: set[str] = set()
    for hint in hints:
        text = str(hint or "")
        if not text.strip():
            continue
        for match in _TABLE_HEADER_RE.finditer(text):
            out.add(match.group(1).lower())
        for match in _MART_TABLE_RE.finditer(text):
            name = match.group(0).lower()
            if name.startswith(("fct_", "agg_", "dim_", "bridge_")):
                out.add(name)
        for match in _STG_TABLE_RE.finditer(text):
            name = match.group(0).lower()
            if name.startswith("stg_"):
                out.add(name)
    return out


def _bare_table_name(raw: str) -> str:
    text = (raw or "").strip().strip("`").strip()
    if not text:
        return ""
    return text.split(".")[-1].lower()


def referenced_tables(sql: str) -> set[str]:
    """Extract bare table names from FROM/JOIN clauses (any id, not only stg_*)."""
    out: set[str] = set()
    for match in _FROM_JOIN_TABLE_RE.finditer(sql or ""):
        bare = _bare_table_name(match.group(1))
        if bare:
            out.add(bare)
    return out


def referenced_mart_tables(sql: str) -> set[str]:
    """Extract mart table names referenced in SQL."""
    refs = referenced_tables(sql)
    mart = {t for t in refs if t.startswith(("fct_", "agg_", "dim_", "bridge_"))}
    if mart:
        return mart
    return {m.lower() for m in _MART_TABLE_RE.findall(sql or "")}


def referenced_stg_tables(sql: str) -> set[str]:
    """Extract stg_* table names referenced in SQL (legacy tests)."""
    refs = referenced_tables(sql)
    stg = {t for t in refs if t.startswith("stg_")}
    if stg:
        return stg
    return {m.lower() for m in _STG_TABLE_RE.findall(sql or "")}


def _catalog_uses_mart(table_ids: set[str]) -> bool:
    return any(t.startswith(("fct_", "agg_", "dim_", "bridge_")) for t in table_ids)


def _columns_for_refs(refs: set[str]) -> dict[str, set[str]]:
    if _catalog_uses_mart(refs):
        return columns_for_mart_tables(refs)
    return columns_for_tables(refs)


def _samples_for_refs(refs: set[str]) -> dict[str, dict[str, list[str]]]:
    if _catalog_uses_mart(refs):
        return value_samples_for_mart_tables(refs)
    return value_samples_for_tables(refs)


def _load_schema_for_ref(bare: str):
    if bare.startswith(("fct_", "agg_", "dim_", "bridge_")):
        return load_mart_table_schema(bare)
    return load_table_schema(bare)


def _reasoner_allowed_tables(selected: set[str]) -> set[str]:
    """Reasoner-selected tables plus documented semantic join dims (not ad-hoc subqueries)."""
    allowed = {
        str(t).strip().split(".")[-1].lower()
        for t in (selected or [])
        if str(t).strip()
    }
    expanded = set(allowed)
    for bare in allowed:
        rel = SEMANTIC_RELATIONSHIPS.get(bare) or {}
        for join in rel.get("joins_with") or []:
            if not isinstance(join, dict):
                continue
            jt = str(join.get("table") or "").strip().lower()
            if jt:
                expanded.add(jt)
    return expanded


def validate_sql_table_allowlist(sql: str, allowed: set[str]) -> str | None:
    """
    Return an error message if SQL references tables outside ``allowed``.

    When ``allowed`` is non-empty, every FROM/JOIN table (including ``dim_*`` and
    ``bridge_*`` joins) must be in the allowed set (typically reasoner-selected
    mart ``fct_*`` / ``agg_*`` tables plus explicit join dims).
    Empty ``allowed`` skips the check.
    """
    if not allowed:
        return None
    allowed_lower = _reasoner_allowed_tables(allowed)
    if not allowed_lower:
        return None
    refs = referenced_tables(sql)
    if not refs:
        refs = referenced_mart_tables(sql) or referenced_stg_tables(sql)
    if not refs:
        return None
    extra = sorted(refs - allowed_lower)
    if extra:
        return (
            f"SQL references tables not in the reasoner selection: {', '.join(extra)}. "
            f"Allowed: {', '.join(sorted(allowed_lower))}. "
            "Do not invent warehouse tables; use only the allowed mart/staging tables."
        )
    return None


def _strip_string_literals(sql: str) -> str:
    return _STRING_LIT_RE.sub("''", sql or "")


def _filtered_columns(sql: str) -> set[str]:
    """Column names that appear in equality or IN filters."""
    scrubbed = _strip_string_literals(sql)
    return {m.group(1).lower() for m in _COL_FILTERED_RE.finditer(scrubbed)}


def metric_discriminator_names() -> frozenset[str]:
    """Columns whose values must not be blended across a numeric measure."""
    return _CORE_METRIC_DISCRIMINATORS | _GRAIN_METRIC_DISCRIMINATORS


def _non_aggregate_exposes_column(sql: str, column: str) -> bool:
    """True when a non-aggregated statement selects ``column``, so each row stays labeled."""
    scrubbed = _strip_string_literals(sql or "")
    if re.search(r"\bGROUP\s+BY\b", scrubbed, re.IGNORECASE):
        return False
    if re.search(r"\b(?:SUM|AVG|COUNT|MIN|MAX)\s*\(", scrubbed, re.IGNORECASE):
        return False
    select_m = re.search(r"\bSELECT\b(?P<body>.*?)\bFROM\b", scrubbed, re.IGNORECASE | re.DOTALL)
    if not select_m:
        return False
    return re.search(rf"\b{re.escape(column)}\b", select_m.group("body"), re.IGNORECASE) is not None


def _split_top_level_and(expr: str) -> list[str]:
    """Split a filter expression on AND outside quotes, parentheses, and BETWEEN pairs."""
    text = (expr or "").strip()
    if text.upper().startswith("AND "):
        text = text[4:].strip()
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_quote = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_quote:
            buf.append(ch)
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    buf.append(text[i + 1])
                    i += 2
                    continue
                in_quote = False
            i += 1
            continue
        if ch == "'":
            in_quote = True
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue
        and_at = (
            depth == 0
            and text[i : i + 3].upper() == "AND"
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
            and (i + 3 >= n or not (text[i + 3].isalnum() or text[i + 3] == "_"))
        )
        if and_at and _BETWEEN_OPEN_RE.search("".join(buf)):
            buf.append(text[i : i + 3])
            i += 3
            continue
        if and_at:
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
            i += 3
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _complete_predicate(piece: str) -> str | None:
    """Return a finished predicate, or None when the fragment is truncated or unbalanced."""
    text = piece.strip().rstrip(";")
    if not text or not _PREDICATE_RE.match(text):
        return None
    if text.count("(") != text.count(")"):
        return None
    quotes = 0
    i = 0
    while i < len(text):
        if text[i] == "'":
            if i + 1 < len(text) and text[i + 1] == "'":
                i += 2
                continue
            quotes += 1
        i += 1
    if quotes % 2:
        return None
    return text


def _required_metric_cols_for_table(bare: str, samples_by_col: dict[str, set[str]]) -> list[str]:
    """Return metric-discriminator columns that must be filtered for this table."""
    schema = _load_schema_for_ref(bare) or {}
    grain_raw = schema.get("grain")
    if not isinstance(grain_raw, str):
        grain_raw = ""
    grain = str(grain_raw).lower().replace("×", " ")
    grain_tokens = set(re.findall(r"[a-z][a-z0-9_]*", grain))
    sample_cols = {str(k).lower(): k for k in (samples_by_col or {})}
    required: list[str] = []
    for cl, _orig in sample_cols.items():
        stem = cl[:-5] if cl.endswith("_name") else cl
        in_grain = cl in grain_tokens or stem in grain_tokens
        if cl in _CORE_METRIC_DISCRIMINATORS:
            required.append(cl)
        elif cl in _GRAIN_METRIC_DISCRIMINATORS and in_grain:
            required.append(cl)
    seen: set[str] = set()
    out: list[str] = []
    for c in required:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _split_trailing_clauses(sql: str) -> tuple[str, str]:
    """Split SQL into (head, trailing GROUP/ORDER/LIMIT/HAVING/QUALIFY tail)."""
    # Search the original SQL. Do not map offsets from a string-stripped copy —
    # shorter replacements shift indices into the middle of identifiers.
    match = _TRAILING_CLAUSE_RE.search(sql or "")
    if not match:
        return (sql or "").rstrip().rstrip(";"), ""
    return (sql or "")[: match.start()].rstrip(), (sql or "")[match.start() :]


def _append_where_filters(sql: str, extra: str) -> str:
    """AND extra predicates into WHERE, or insert WHERE before trailing clauses."""
    extra = extra.strip()
    if not extra:
        return sql
    head, tail = _split_trailing_clauses(sql)
    if _WHERE_RE.search(_strip_string_literals(head)):
        joined = f"{head} AND {extra}"
    else:
        joined = f"{head} WHERE {extra}"
    if tail:
        return f"{joined} {tail}".rstrip()
    return joined.rstrip()


def inject_missing_metric_filters(
    sql: str,
    table_ids: set[str] | None = None,
    *,
    query: str = "",
    primary_measures: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Auto-inject equality filters for required metric discriminators that are missing.

    Returns (possibly_modified_sql, notes like ``stg_faostat_production.element='Production'``).
    Only injects values that exist in YAML ``*_value_samples``.
    """
    text = (sql or "").strip()
    if not text:
        return text, []

    refs = _refs_for_sql_checks(text, table_ids)
    if not refs:
        return text, []

    samples_map = _samples_for_refs(refs)
    if not samples_map:
        return text, []

    filtered = _filtered_columns(text)
    clauses: list[str] = []
    notes: list[str] = []

    for bare in sorted(refs):
        by_col = samples_map.get(bare) or {}
        by_col_set = {k: set(v) for k, v in by_col.items()}
        for col in _required_metric_cols_for_table(bare, by_col_set):
            if col in filtered:
                continue
            if _non_aggregate_exposes_column(text, col):
                continue
            sample_vals: set[str] = set()
            orig_col = col
            for key, vals in by_col.items():
                if str(key).lower() == col:
                    sample_vals = set(vals)
                    orig_col = str(key)
                    break
            default = default_discriminator_value(
                orig_col,
                sample_vals,
                query=query,
                primary_measures=primary_measures,
            )
            if not default:
                continue
            lit = default.replace("'", "''")
            clauses.append(f"{orig_col} = '{lit}'")
            notes.append(f"{bare}.{orig_col}='{default}'")

    if not clauses:
        return text, []

    extra = " AND ".join(clauses)
    return _append_where_filters(text, extra), notes


def inject_bind_contract_filters(
    sql: str,
    bind_contracts: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    """Inject missing facet filters from the bind contract of each table the statement reads."""
    text = (sql or "").strip()
    if not text or not isinstance(bind_contracts, dict) or not bind_contracts:
        return text, []

    refs = referenced_tables(text)
    filtered = _filtered_columns(text)
    clauses: list[str] = []
    notes: list[str] = []

    for tid in sorted(bind_contracts.keys()):
        raw = bind_contracts.get(tid)
        contract = TableBindContract.from_dict(raw if isinstance(raw, dict) else None)
        if contract is None:
            continue
        bare = str(contract.table_id or tid).strip().split(".")[-1].lower()
        if bare not in refs:
            continue
        if contract.geo_column and contract.geo_literals:
            gcol = contract.geo_column
            if gcol.lower() not in filtered:
                if len(contract.geo_literals) == 1:
                    lit = contract.geo_literals[0].replace("'", "''")
                    clauses.append(f"{gcol} = '{lit}'")
                    notes.append(f"{bare}.{gcol}='{contract.geo_literals[0]}'")
                else:
                    lits = ", ".join(
                        f"'{v.replace(chr(39), chr(39)+chr(39))}'"
                        for v in contract.geo_literals[:16]
                    )
                    clauses.append(f"{gcol} IN ({lits})")
                    notes.append(f"{bare}.{gcol} IN (...)")
                filtered.add(gcol.lower())
        for col, val in contract.measure_filters:
            if col.lower() in filtered:
                continue
            lit = val.replace("'", "''")
            clauses.append(f"{col} = '{lit}'")
            notes.append(f"{bare}.{col}='{val}'")
            filtered.add(col.lower())
        for piece in _split_top_level_and(str(contract.required_filters_sql or "")):
            predicate = _complete_predicate(piece)
            if predicate is None:
                continue
            col_m = _PREDICATE_RE.match(predicate)
            if col_m is None or col_m.group("col").lower() in filtered:
                continue
            clauses.append(predicate)
            notes.append(f"{bare}.{col_m.group('col')}")
            filtered.add(col_m.group("col").lower())

    if not clauses:
        return text, []

    extra = " AND ".join(clauses)
    return _append_where_filters(text, extra), notes


def _table_has_column(table_id: str, col_name: str) -> bool:
    schema = load_mart_table_schema(table_id) or load_table_schema(table_id)
    if not schema:
        return False
    target = col_name.lower()
    for col in schema.get("columns") or []:
        if str(col.get("name") or "").strip().lower() == target:
            return True
    return False


def _column_has_time_predicate(scrubbed: str, column: str) -> bool:
    col_l = column.lower()
    return bool(
        re.search(
            rf"\b{re.escape(col_l)}\b\s*(=|between|>=|<=|>|<|in\s*\()",
            scrubbed,
        )
    )


def _sql_has_time_filter(
    sql: str,
    table_ids: set[str] | None = None,
) -> bool:
    """True when SQL filters as_of_date or the table year column (year, time_key, …)."""
    stripped = _strip_string_literals(sql or "").lower()
    if re.search(
        r"\bas_of_date\b\s*(=|between|>=|<=|>|<|in\s*\()",
        stripped,
    ):
        return True

    refs = _refs_for_sql_checks(sql or "", table_ids)
    if refs:
        for bare in sorted(refs):
            ycol = year_column(bare) or "year"
            if _column_has_time_predicate(stripped, ycol):
                return True
        return False

    for cand in ("year", "time_key", "harvest_year", "observation_year", "mp_year"):
        if _column_has_time_predicate(stripped, cand):
            return True
    return False


def _sql_has_time_bounds(
    sql: str,
    table_ids: set[str] | None = None,
) -> bool:
    return _sql_has_time_filter(sql, table_ids)


def inject_time_bounds(
    sql: str,
    decomposition: dict[str, Any] | None,
    table_ids: set[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Inject partition-friendly time bounds when decomposition has dates and SQL lacks them.

    Prefers ``as_of_date BETWEEN`` on mart facts; falls back to ``year BETWEEN``.
    """
    text = (sql or "").strip()
    if not text:
        return text, []

    refs = _refs_for_sql_checks(text, table_ids)
    if _sql_has_time_filter(text, table_ids):
        return text, []

    decomp = decomposition or {}
    ts = str(decomp.get("time_start") or "")[:10]
    te = str(decomp.get("time_end") or "")[:10]
    if not ts or not te:
        return text, []

    refs = _refs_for_sql_checks(text, table_ids)
    if not refs:
        return text, []

    for bare in sorted(refs):
        if _table_has_column(bare, "as_of_date"):
            clause = f"as_of_date BETWEEN '{ts}' AND '{te}'"
            return _append_where_filters(text, clause), [f"{bare}.as_of_date BETWEEN {ts} AND {te}"]
        ycol = year_column(bare)
        if ycol and ts[:4].isdigit() and te[:4].isdigit():
            clause = f"{ycol} BETWEEN {ts[:4]} AND {te[:4]}"
            return _append_where_filters(text, clause), [f"{bare}.{ycol} BETWEEN {ts[:4]} AND {te[:4]}"]

    return text, []


def broaden_empty_sql_once(
    sql: str,
    *,
    crop_required: bool = True,
    geography_required: bool = True,
) -> str | None:
    """
    Soften a valid-but-empty SQL once: widen a single-year equality, optionally drop product.

    Never drops geography filters. Returns None when nothing can be softened.
    """
    text = (sql or "").strip()
    if not text:
        return None
    out = text
    changed = False
    year_match = _YEAR_EQ_RE.search(out)
    if year_match:
        year = int(year_match.group(2))
        col = year_match.group(1)
        out = _YEAR_EQ_RE.sub(f"{col} BETWEEN {year - 1} AND {year + 1}", out, count=1)
        changed = True
    if not crop_required:
        replaced = _PRODUCT_EQ_RE.sub("1=1", out)
        replaced = _PRODUCT_IN_RE.sub("1=1", replaced)
        if replaced != out:
            out = replaced
            changed = True
    _ = geography_required
    return out if changed else None


def _refs_for_sql_checks(sql: str, table_ids: set[str] | None = None) -> set[str]:
    """
    Tables to validate against for metric/column/sample checks.

    Prefer tables actually referenced in the SQL. Only fall back to ``table_ids``
    when the SQL has no parseable mart/staging refs — never union plan-selected
    companions into a single-table statement (that falsely demands ASTI
    ``indicator`` filters on production SQL, etc.).
    """
    refs = referenced_mart_tables(sql)
    if refs:
        return refs
    refs = referenced_stg_tables(sql)
    if refs:
        return refs
    if not table_ids:
        return set()
    return {str(t).strip().split(".")[-1].lower() for t in table_ids if str(t).strip()}


def validate_required_metric_filters(sql: str, table_ids: set[str] | None = None) -> str | None:
    """
    Require equality/IN filters on YAML metric-discriminator columns that have samples.

    Applies to every referenced mart/staging table (element, price_type, measure_type,
    production_grain, price_source, …), not only FAOSTAT.
    """
    refs = _refs_for_sql_checks(sql, table_ids)
    if not refs:
        return None
    samples_map = _samples_for_refs(refs)
    if not samples_map:
        return None
    filtered = _filtered_columns(sql)
    missing: list[str] = []
    previews: list[str] = []
    for bare in sorted(refs):
        by_col = samples_map.get(bare) or {}
        by_col_set: dict[str, set[str]] = {k: set(v) for k, v in by_col.items()}
        for col in _required_metric_cols_for_table(bare, by_col_set):
            if col in filtered:
                continue
            if _non_aggregate_exposes_column(sql, col):
                continue
            missing.append(f"{bare}.{col}")
            vals: list[str] = []
            for k, v in by_col.items():
                if str(k).lower() == col:
                    vals = sorted(v)
                    break
            preview = ", ".join(repr(v) for v in vals[:6])
            previews.append(f"{col} (e.g. {preview})" if preview else col)
    if not missing:
        return None
    return (
        f"SQL must equality-filter metric discriminator column(s): {', '.join(missing)}. "
        f"Use exact YAML value samples — {'; '.join(previews[:4])}. "
        "Do not SELECT * or SUM across mixed discriminator values."
    )


def _yaml_columns_for_table(bare: str) -> set[str] | None:
    names = _columns_for_refs({bare}).get(bare)
    return set(names) if names else None


def validate_sql_column_allowlist(sql: str, table_ids: set[str] | None = None) -> str | None:
    """
    Reject column references that no source in their scope exposes.

    Columns are resolved on the parsed statement: qualified names against their table
    alias, CTE or derived table; unqualified names against every source in scope.
    Sources without YAML columns are not checked.
    """
    root, err = parse_select(sql)
    if err:
        return err
    bad = unknown_columns(root, _yaml_columns_for_table)
    if not bad:
        return None
    refs = {t.name for t in physical_tables(root)} | {
        str(t).strip().split(".")[-1].lower() for t in (table_ids or set())
    }
    allowed: set[str] = set()
    for names in _columns_for_refs(refs).values():
        allowed |= {n.lower() for n in names}
    allowed_preview = ", ".join(sorted(allowed)[:40])
    return (
        f"SQL uses columns not in the YAML Columns blocks: {', '.join(bad)}. "
        f"Allowed columns include: {allowed_preview}. "
        "Use only exact column names from the table hints; do not invent bronze/raw names."
    )


def validate_sql_complete_index_literals(
    sql: str,
    table_ids: set[str] | None = None,
) -> str | None:
    """Reject string literals not present in complete value index (closed-world)."""
    refs = _refs_for_sql_checks(sql, table_ids)
    if not refs:
        return None

    def _ok(table: str, col: str, literal: str) -> bool:
        allowed = complete_enum(table, col)
        if not allowed:
            return True
        lit = literal.replace("''", "'").strip()
        return lit in allowed or lit.lower() in {a.lower() for a in allowed}

    bad: list[str] = []
    for match in _EQ_FILTER_RE.finditer(sql or ""):
        col, lit = match.group(1), match.group(2)
        for ref in refs:
            if _ok(ref, col, lit):
                break
        else:
            if complete_enum(next(iter(refs), ""), col):
                bad.append(f"{col}='{lit}'")
    for match in _IN_FILTER_RE.finditer(sql or ""):
        col = match.group(1)
        body = match.group(2)
        if "select" in body.lower():
            continue
        for lit_m in re.finditer(r"'((?:''|[^'])*)'", body):
            lit = lit_m.group(1)
            for ref in refs:
                if _ok(ref, col, lit):
                    break
            else:
                if complete_enum(next(iter(refs), ""), col):
                    bad.append(f"{col} IN … '{lit}'")
    if not bad:
        return None
    return f"SQL uses labels not in complete value index: {', '.join(bad[:8])}"


def _profiled_domain(table_id: str, column: str) -> list[str] | None:
    """All labels of ``table.column`` when the profile captured the full domain, else None."""
    schema = _load_schema_for_ref(table_id) or {}
    samples = schema.get(f"{column}_value_samples")
    if not isinstance(samples, list) or not samples:
        return None
    stats = schema.get(f"{column}_value_stats")
    if isinstance(stats, dict) and "is_truncated" in stats:
        complete = stats.get("is_truncated") is False
    else:
        complete = len(samples) < PROFILE_SAMPLE_CAP
    return [str(s).strip() for s in samples if str(s).strip()] if complete else None


def label_domain(table_ids: set[str], column: str) -> set[str] | None:
    """Complete lower-cased label domain for a filter column, from the fact profile or its dimension.

    None means only a capped sample is known, so an unseen label is not evidence of a bad label.
    """
    col = column.lower()
    domains = [
        _profiled_domain(t, col)
        for t in table_ids
        if (_load_schema_for_ref(t) or {}).get(f"{col}_value_samples")
    ]
    if domains and all(d is not None for d in domains):
        return {label.lower() for d in domains for label in d or []}
    dimension = _DIMENSION_LABEL_COLUMNS.get(col)
    if dimension:
        dim_domain = _profiled_domain(*dimension)
        if dim_domain is not None:
            return {label.lower() for label in dim_domain}
    return None


def validate_sql_value_samples(sql: str, table_ids: set[str] | None = None) -> str | None:
    """
    Check equality / IN string literals against the label domain of their column.

    A label outside the YAML samples is rejected only when the samples (or the column's
    dimension) cover the full domain; capped samples cannot prove a label absent.
    Skips columns with no samples and leaves LIKE filters alone.
    """
    refs = _refs_for_sql_checks(sql, table_ids)
    if not refs:
        return None
    samples_map = _samples_for_refs(refs)
    if not samples_map:
        return None
    by_col: dict[str, set[str]] = {}
    for per_table in samples_map.values():
        for col, vals in per_table.items():
            by_col.setdefault(col.lower(), set()).update(vals)
    for col in _SHARED_DICTIONARY_COLUMNS:
        by_col.setdefault(col, set()).update(dictionary_samples_for_column(col))
    if refs & {"fct_prices", "agg_prices_country_month"}:
        for col in ("price_type", "price_source", "metric"):
            if col in by_col:
                by_col[col].update(price_vocab_values(col))

    def _ok(col: str, literal: str) -> bool:
        col_l = col.lower()
        allowed = by_col.get(col_l)
        if col_l in _SHARED_DICTIONARY_COLUMNS:
            shared = dictionary_samples_for_column(col_l)
            if shared:
                allowed = set(shared)
        if not allowed:
            return True
        lit = literal.replace("''", "'").strip().lower()
        if lit in {a.lower() for a in allowed}:
            return True
        domain = label_domain(refs, col_l)
        return domain is None or lit in domain

    bad: list[str] = []
    for match in _EQ_FILTER_RE.finditer(sql or ""):
        col, lit = match.group(1), match.group(2)
        if not _ok(col, lit):
            bad.append(f"{col}='{lit}'")
    for match in _IN_FILTER_RE.finditer(sql or ""):
        col = match.group(1)
        if col.lower() not in by_col:
            continue
        body = match.group(2)
        if "select" in body.lower():
            continue
        for lit_m in re.finditer(r"'((?:''|[^'])*)'", body):
            lit = lit_m.group(1)
            if not _ok(col, lit):
                bad.append(f"{col} IN … '{lit}'")
    if not bad:
        return None
    return (
        f"SQL equality filters use labels not in YAML value samples: {', '.join(bad[:8])}. "
        "Use exact strings from element/product/item *_value_samples in the table hints."
    )


def dry_run_sql(client: Any, sql: str) -> str | None:
    """Run BigQuery dry-run; return error message or None on success."""
    if not sql or not dry_run_enabled():
        return None
    try:
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        client.query(sql, job_config=job_config)
        return None
    except Exception as exc:
        return str(exc)[:500]


def dry_run_bytes_processed(client: Any, sql: str) -> int | None:
    """Return dry-run totalBytesProcessed or None when unavailable."""
    if not sql or not dry_run_enabled():
        return None
    try:
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(sql, job_config=job_config)
        return int(getattr(job, "total_bytes_processed", 0) or 0)
    except Exception:
        return None


def _eq_literals_for_column(sql: str, column: str) -> list[str]:
    """Equality / IN string literals for one column."""
    col_l = column.lower()
    out: list[str] = []
    for match in _EQ_FILTER_RE.finditer(sql or ""):
        if match.group(1).lower() != col_l:
            continue
        out.append(match.group(2).replace("''", "'").strip())
    for match in _IN_FILTER_RE.finditer(sql or ""):
        if match.group(1).lower() != col_l:
            continue
        body = match.group(2)
        if "select" in body.lower():
            continue
        for lit_m in re.finditer(r"'((?:''|[^'])*)'", body):
            out.append(lit_m.group(1).replace("''", "'").strip())
    return out


def _sql_has_product_filter(sql: str) -> bool:
    scrubbed = _strip_string_literals(sql or "").lower()
    if re.search(r"\bproduct_name\s*(=|in\s*\()", scrubbed):
        return True
    if re.search(r"\bproduct_key\s+in\s*\(", scrubbed):
        return True
    return False


def _sql_has_geo_filter(sql: str, geo_col: str | None) -> bool:
    if not geo_col:
        return True
    scrubbed = _strip_string_literals(sql or "").lower()
    return bool(re.search(rf"\b{re.escape(geo_col.lower())}\s*(=|in\s*\()", scrubbed))


def _sql_has_partition_predicate(sql: str, table_id: str) -> bool:
    bare = str(table_id or "").strip().split(".")[-1].lower()
    if bare not in _LARGE_PARTITIONED_TABLES:
        return True
    schema = load_mart_table_schema(bare) or {}
    col_names = {
        str(c.get("name") or "").strip().lower()
        for c in (schema.get("columns") or [])
        if str(c.get("name") or "").strip()
    }
    scrubbed = _strip_string_literals(sql or "").lower()
    if "as_of_date" in col_names and re.search(
        r"\bas_of_date\b\s*(=|between|>=|<=|>|<|in\s*\()",
        scrubbed,
    ):
        return True
    ycol = (year_column(bare) or "year").lower()
    return bool(re.search(rf"\b{re.escape(ycol)}\b\s*(=|between|>=|<=|>|<|in\s*\()", scrubbed))


def validate_semantic_coherence(
    sql: str,
    *,
    query: str = "",
    primary_measures: list[str] | None = None,
    geography: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    crop_required: bool | None = None,
    geography_required: bool | None = None,
    table_ids: set[str] | None = None,
    decomposition: dict[str, Any] | None = None,
) -> str | None:
    """
    Reject warehouse-valid SQL that contradicts decomposition intent.

    Checks measure/geo/product/time coherence — not YAML literal presence alone.
    """
    from ml.rag.chatbot.ontology_context import build_ontology_context

    ontology = build_ontology_context(query, decomposition) if decomposition else None

    raw_pm = list(primary_measures) if primary_measures else []
    if not raw_pm and ontology:
        raw_pm = list(ontology.primary_measures)
    pm = [str(m).strip().lower() for m in raw_pm if str(m).strip()]

    geo_labels = [str(g).strip() for g in (geography or []) if str(g).strip()]
    if not geo_labels and ontology:
        geo_labels = list(ontology.geography)

    ts = str(time_start or (ontology.time_start if ontology else "") or "")[:10]
    te = str(time_end or (ontology.time_end if ontology else "") or "")[:10]

    crop_req = (
        crop_required
        if crop_required is not None
        else (ontology.crop_required if ontology else True)
    )
    geo_req = (
        geography_required
        if geography_required is not None
        else (ontology.geography_required if ontology else True)
    )

    text = (sql or "").strip()
    if not text:
        return None

    production_intent = any(m in ("production", "prod") for m in pm) or bool(
        re.search(r"\b(production|tonnes?|tons?)\b", (query or ""), re.IGNORECASE)
    )
    yield_intent = any("yield" in m for m in pm) or bool(
        re.search(r"\byields?\b", (query or ""), re.IGNORECASE)
    )

    if production_intent and not yield_intent:
        for lit in _eq_literals_for_column(text, "element"):
            if lit.lower() in _YIELD_ELEMENT_VALUES:
                return (
                    "SQL filters element='Yield' but question/primary_measures "
                    "request production volume. Use element='Production' and a "
                    "production volume metric."
                )
        for lit in _eq_literals_for_column(text, "metric"):
            low = lit.lower()
            if any(tok in low for tok in _YIELD_METRIC_TOKENS) and "production_production" not in low:
                return (
                    "SQL filters a yield metric but question/primary_measures "
                    "request production volume."
                )

    if yield_intent and not production_intent:
        for lit in _eq_literals_for_column(text, "element"):
            if lit.lower() in _PRODUCTION_ELEMENT_VALUES:
                return "SQL filters element='Production' but question requests yield."

    if geo_req and geo_labels:
        refs = _refs_for_sql_checks(text, table_ids)
        geo_ok = False
        for bare in refs:
            gcol = geo_column(bare)
            if _sql_has_geo_filter(text, gcol):
                geo_ok = True
                break
        if not geo_ok:
            return (
                "SQL must filter geography when decomposition specifies countries. "
                f"Expected filter on geo column for: {', '.join(geo_labels[:5])}."
            )

    if crop_req and not _sql_has_product_filter(text):
        q = (query or "").lower()
        if re.search(r"\b(maize|rice|wheat|sorghum|millet|cassava|crop|commodity)\b", q):
            return (
                "SQL must include a dictionary-backed product filter when the "
                "question names a crop/commodity."
            )

    if ts and te and not _sql_has_time_bounds(text, table_ids):
        return (
            "SQL must include a time bound (year or as_of_date) matching the "
            "decomposition time window."
        )

    refs = _refs_for_sql_checks(text, table_ids)
    for bare in sorted(refs):
        if bare in _LARGE_PARTITIONED_TABLES and not _sql_has_partition_predicate(text, bare):
            return (
                f"SQL querying `{bare}` must filter as_of_date or year for partition "
                "pruning on large mart facts."
            )

    return None


def validate_dry_run_bytes(
    client: Any,
    sql: str,
    *,
    max_bytes: int | None = None,
    sql_source: str = "",
) -> str | None:
    """Optional dry-run bytes ceiling (template vs nl2sql defaults)."""
    if not max_bytes or max_bytes <= 0:
        return None
    processed = dry_run_bytes_processed(client, sql)
    if processed is None:
        return None
    if processed > max_bytes:
        mb = processed / (1024 * 1024)
        cap = max_bytes / (1024 * 1024)
        return (
            f"SQL dry-run would scan {mb:.0f}MB (cap {cap:.0f}MB for {sql_source or 'query'}). "
            "Add partition/time bounds or route to an agg table."
        )
    return None


def max_bytes_billed_for_source(sql_source: str) -> int:
    """Bytes cap by SQL producer (NL2SQL stricter than template/pattern)."""
    src = (sql_source or "").strip().lower()
    if src in ("template", "pattern"):
        return int(os.environ.get("RAG_BQ_TEMPLATE_MAX_BYTES_BILLED", str(512 * 1024 * 1024)) or 0)
    return int(os.environ.get("RAG_BQ_MAX_BYTES_BILLED", str(250 * 1024 * 1024)) or 0)


_QUAL_EQ_FILTER_RE = re.compile(
    r"\b(?:([A-Za-z_][A-Za-z0-9_]*)\.)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'((?:''|[^'])*)'",
    re.IGNORECASE,
)
_QUAL_IN_FILTER_RE = re.compile(
    r"\b(?:([A-Za-z_][A-Za-z0-9_]*)\.)?([A-Za-z_][A-Za-z0-9_]*)\s+IN\s*\(([^)]*)\)",
    re.IGNORECASE,
)
_FROM_ALIAS_MAP_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?:`([^`]+)`|([A-Za-z_][\w.]*))"
    r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)


def _ontology_bq_strict() -> bool:
    return os.environ.get("ADZA_ONTOLOGY_BQ_STRICT", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def _bare_from_fqn(raw: str) -> str:
    text = (raw or "").strip().strip("`")
    if not text:
        return ""
    return text.split(".")[-1].lower()


def _alias_to_table(sql: str) -> dict[str, str]:
    """Map SQL alias (and bare table) → bare mart table id."""
    out: dict[str, str] = {}
    for match in _FROM_ALIAS_MAP_RE.finditer(sql or ""):
        target = match.group(1) or match.group(2) or ""
        bare = _bare_from_fqn(target)
        if not bare.startswith(("fct_", "agg_", "dim_", "bridge_")):
            continue
        out[bare] = bare
        alias = (match.group(3) or "").strip().lower()
        if alias and alias not in _SQL_KEYWORDS:
            out[alias] = bare
    return out


def _allowed_for_column(
    filters: dict[str, list[str]],
    *,
    column: str,
    table_hint: str | None,
    tables_in_sql: set[str],
) -> tuple[str | None, list[str]]:
    """Pick the best filter key for a column reference; return (key, values)."""
    col_l = column.lower()
    candidates: list[tuple[str, list[str]]] = []
    for key, vals in (filters or {}).items():
        if "." not in key:
            continue
        table, col = key.split(".", 1)
        if col.lower() != col_l:
            continue
        table_l = table.lower()
        if table_hint and table_l != table_hint.lower():
            continue
        if tables_in_sql and table_l not in tables_in_sql and not table_hint:
            continue
        candidates.append((key, list(vals)))
    if not candidates and not table_hint:
        for key, vals in (filters or {}).items():
            if "." not in key:
                continue
            _table, col = key.split(".", 1)
            if col.lower() == col_l:
                candidates.append((key, list(vals)))
    if not candidates:
        return None, []
    if table_hint:
        for key, vals in candidates:
            if key.split(".", 1)[0].lower() == table_hint.lower():
                return key, vals
    return candidates[0]


def _rewrite_via_bq_targets(col_key: str, literal: str, allowed: list[str]) -> str | None:
    """Map an alternate country/product form onto an allowed stored literal when possible."""
    allowed_l = {a.lower(): a for a in allowed}
    if literal.lower() in allowed_l:
        return allowed_l[literal.lower()]
    try:
        import opentrace_ontology as oo
    except Exception:
        return None
    try:
        ents = oo.resolve(literal)
    except Exception:
        return None
    for ent in ents or []:
        eid = getattr(ent, "entity_id", None) or (ent.get("entity_id") if isinstance(ent, dict) else None)
        if not eid:
            continue
        try:
            targets = oo.bq_targets(str(eid)) or {}
        except Exception:
            continue
        for val in targets.get(col_key) or []:
            if val in allowed:
                return val
            if str(val).lower() in allowed_l:
                return allowed_l[str(val).lower()]
    return None


def validate_literals_against_value_map(
    sql: str,
    filters: dict[str, list[str]] | None,
) -> tuple[str, list[str]]:
    """Rewrite / check equality and IN string literals against ontology value-map filters.

    Returns ``(sql, notes)``. When ``ADZA_ONTOLOGY_BQ_STRICT=1``, unrepaired
    mismatches produce a ``strict_reject:…`` note (caller should fail closed).
    """
    notes: list[str] = []
    text = sql or ""
    if not text or not filters:
        return text, notes

    alias_map = _alias_to_table(text)
    tables_in_sql = set(alias_map.values()) | referenced_tables(text)
    strict = _ontology_bq_strict()
    replacements: list[tuple[str, str]] = []

    def _handle(col: str, literal: str, table_hint: str | None) -> None:
        lit = literal.replace("''", "'").strip()
        key, allowed = _allowed_for_column(
            filters,
            column=col,
            table_hint=table_hint,
            tables_in_sql=tables_in_sql,
        )
        if not key or not allowed:
            return
        if lit in allowed:
            return
        rewritten = _rewrite_via_bq_targets(key, lit, allowed)
        if rewritten and rewritten != lit:
            replacements.append((lit, rewritten))
            notes.append(f"rewrote {key} '{lit}' -> '{rewritten}'")
            return
        if lit.lower() in {a.lower() for a in allowed}:
            for a in allowed:
                if a.lower() == lit.lower() and a != lit:
                    replacements.append((lit, a))
                    notes.append(f"rewrote {key} '{lit}' -> '{a}' (case)")
                    return
            return
        msg = f"literal not in value map: {key}='{lit}'"
        if strict:
            notes.append(f"strict_reject:{msg}")
        else:
            notes.append(msg)

    for match in _QUAL_EQ_FILTER_RE.finditer(text):
        prefix, col, lit = match.group(1), match.group(2), match.group(3)
        table_hint = alias_map.get((prefix or "").lower()) if prefix else None
        _handle(col, lit, table_hint)

    for match in _QUAL_IN_FILTER_RE.finditer(text):
        prefix, col, body = match.group(1), match.group(2), match.group(3)
        if "select" in (body or "").lower():
            continue
        table_hint = alias_map.get((prefix or "").lower()) if prefix else None
        for lit_m in re.finditer(r"'((?:''|[^'])*)'", body or ""):
            _handle(col, lit_m.group(1), table_hint)

    out = text
    # Longest-first so partial overlaps do not corrupt longer literals.
    for old, new in sorted(set(replacements), key=lambda p: -len(p[0])):
        if old == new:
            continue
        out = out.replace(f"'{old}'", f"'{new}'")
        out = out.replace(f"'{old.replace(chr(39), chr(39)+chr(39))}'", f"'{new}'")

    return out, notes
