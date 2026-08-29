"""BigQuery NL2SQL validation: table allowlist, YAML columns, value samples, dry-run."""
from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import (
    columns_for_mart_tables,
    columns_for_tables,
    default_discriminator_value,
    load_mart_table_schema,
    load_table_schema,
    value_samples_for_mart_tables,
    value_samples_for_tables,
    year_column,
)

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

# Metric discriminators: change the meaning of ``value`` / measures.
# Optional ones are required only when mentioned in the table grain.
_CORE_METRIC_DISCRIMINATORS = frozenset(
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
_GRAIN_METRIC_DISCRIMINATORS = frozenset(
    {
        "classification_scale",
        "scenario_name",
    }
)

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
    allowed_lower = {t.lower() for t in allowed if str(t).strip()}
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


def _aliases_in_sql(sql: str) -> set[str]:
    names = {m.group(1).lower() for m in _AS_ALIAS_RE.finditer(sql or "")}
    names |= {m.group(1).lower() for m in _CTE_NAME_RE.finditer(sql or "")}
    for match in _TABLE_ALIAS_RE.finditer(sql or ""):
        alias = match.group(1).lower()
        # Avoid treating ON/WHERE keywords after FROM as aliases when regex over-matches.
        if alias in _SQL_KEYWORDS:
            continue
        names.add(alias)
    return names


def _strip_table_refs(sql: str) -> str:
    """Remove FROM/JOIN table targets and backtick FQNs so project ids are not scanned as columns."""

    def _repl(match: re.Match[str]) -> str:
        full = match.group(0)
        target = match.group(1)
        return full[: len(full) - len(target)] + " "

    text = _FROM_JOIN_TABLE_RE.sub(_repl, sql or "")
    return _BACKTICK_RE.sub(" ", text)


def candidate_column_idents(sql: str) -> set[str]:
    """Heuristic column identifiers referenced outside string literals and table FQNs."""
    scrubbed = _strip_table_refs(_strip_string_literals(sql))
    aliases = _aliases_in_sql(sql)
    tables = referenced_tables(sql)
    out: set[str] = set()
    for match in _IDENT_RE.finditer(scrubbed):
        ident = match.group(1)
        low = ident.lower()
        if low in _SQL_KEYWORDS:
            continue
        if low in aliases:
            continue
        if low in tables:
            continue
        if low.startswith("stg_") or low.startswith(("fct_", "agg_", "dim_", "bridge_")):
            continue
        if low in {"proj", "project", "staging_dev", "mart_dev", "raw_dev", "opentrace"}:
            continue
        # Hyphen-split GCP project fragments (e.g. opentrace-prod-5ga4 → prod).
        if low in {"prod", "dev", "raw"} or low[:1].isdigit():
            continue
        out.add(ident)
    return out


def _filtered_columns(sql: str) -> set[str]:
    """Column names that appear in equality or IN filters."""
    scrubbed = _strip_string_literals(sql)
    return {m.group(1).lower() for m in _COL_FILTERED_RE.finditer(scrubbed)}


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
            sample_vals: set[str] = set()
            orig_col = col
            for key, vals in by_col.items():
                if str(key).lower() == col:
                    sample_vals = set(vals)
                    orig_col = str(key)
                    break
            default = default_discriminator_value(orig_col, sample_vals, query=query)
            if not default:
                continue
            lit = default.replace("'", "''")
            clauses.append(f"{orig_col} = '{lit}'")
            notes.append(f"{bare}.{orig_col}='{default}'")

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


def _sql_has_time_filter(sql: str) -> bool:
    stripped = _strip_string_literals(sql or "").lower()
    if re.search(r"\bas_of_date\b", stripped):
        return True
    if re.search(r"\byear\b", stripped) and re.search(
        r"\b(?:=|between|>=|<=|>|<|in\s*\()",
        stripped,
    ):
        return True
    return False


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
    if not text or _sql_has_time_filter(text):
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


def validate_sql_column_allowlist(sql: str, table_ids: set[str] | None = None) -> str | None:
    """
    Reject identifiers that are not YAML columns for referenced (or provided) tables.

    When YAML columns cannot be loaded for any referenced table, skip the check.
    """
    refs = _refs_for_sql_checks(sql, table_ids)
    if not refs:
        return None
    col_map = _columns_for_refs(refs)
    if not col_map:
        return None
    allowed: set[str] = set()
    for names in col_map.values():
        allowed |= {n.lower() for n in names}
    if not allowed:
        return None
    aliases = _aliases_in_sql(sql)
    bad: list[str] = []
    for ident in candidate_column_idents(sql):
        low = ident.lower()
        if low in aliases:
            continue
        if low in allowed:
            continue
        bad.append(ident)
    if not bad:
        return None
    seen: set[str] = set()
    uniq: list[str] = []
    for b in sorted(bad, key=str.lower):
        if b.lower() in seen:
            continue
        seen.add(b.lower())
        uniq.append(b)
    allowed_preview = ", ".join(sorted(allowed)[:40])
    return (
        f"SQL uses columns not in the YAML Columns blocks: {', '.join(uniq)}. "
        f"Allowed columns include: {allowed_preview}. "
        "Use only exact column names from the table hints; do not invent bronze/raw names."
    )


def validate_sql_value_samples(sql: str, table_ids: set[str] | None = None) -> str | None:
    """
    Soft-check equality / IN string literals against YAML ``*_value_samples``.

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

    def _ok(col: str, literal: str) -> bool:
        allowed = by_col.get(col.lower())
        if not allowed:
            return True
        lit = literal.replace("''", "'").strip()
        allowed_l = {a.lower() for a in allowed}
        return lit.lower() in allowed_l

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
