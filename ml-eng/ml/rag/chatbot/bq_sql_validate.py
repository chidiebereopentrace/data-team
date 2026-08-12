"""BigQuery NL2SQL validation: table allowlist, YAML columns, value samples, dry-run."""
from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import columns_for_tables, value_samples_for_tables

_STG_TABLE_RE = re.compile(r"\bstg_[a-z0-9_]+\b", re.IGNORECASE)
_TABLE_HEADER_RE = re.compile(r"^Table:\s*[`\w.-]*\.?(\bstg_[a-z0-9_]+)", re.IGNORECASE | re.MULTILINE)
# FROM/JOIN target: backticked FQN or dotted/bare identifier (not a subquery paren).
_FROM_JOIN_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?![\(\s])(`[^`]+`|(?:[A-Za-z_][\w-]*)(?:\.[A-Za-z_][\w-]*)*)",
    re.IGNORECASE,
)
_STRING_LIT_RE = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
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
    """Extract stg_* table ids from packed YAML hint blocks."""
    out: set[str] = set()
    for hint in hints:
        text = str(hint or "")
        if not text.strip():
            continue
        for match in _TABLE_HEADER_RE.finditer(text):
            out.add(match.group(1).lower())
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


def referenced_stg_tables(sql: str) -> set[str]:
    """Extract stg_* table names referenced in SQL (hint/compat helper)."""
    refs = referenced_tables(sql)
    stg = {t for t in refs if t.startswith("stg_")}
    if stg:
        return stg
    # Fallback for odd SQL shapes that omit FROM/JOIN capture but still mention stg_*.
    return {m.lower() for m in _STG_TABLE_RE.findall(sql or "")}


def validate_sql_table_allowlist(sql: str, allowed: set[str]) -> str | None:
    """
    Return an error message if SQL references tables outside ``allowed``.

    When ``allowed`` is non-empty, every FROM/JOIN table (including ``dim_*`` and
    non-stg names) must be in the allowed set (typically reasoner-selected ``stg_*``).
    Empty ``allowed`` skips the check.
    """
    if not allowed:
        return None
    allowed_lower = {t.lower() for t in allowed if str(t).strip()}
    if not allowed_lower:
        return None
    refs = referenced_tables(sql)
    if not refs:
        # No FROM/JOIN parse — still catch loose stg_* mentions outside the set.
        refs = referenced_stg_tables(sql)
    if not refs:
        return None
    extra = sorted(refs - allowed_lower)
    if extra:
        return (
            f"SQL references tables not in the reasoner selection: {', '.join(extra)}. "
            f"Allowed: {', '.join(sorted(allowed_lower))}. "
            "Do not invent dim_* or other warehouse tables; use only the allowed stg_* tables."
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


def candidate_column_idents(sql: str) -> set[str]:
    """Heuristic column identifiers referenced outside string literals."""
    scrubbed = _strip_string_literals(sql)
    aliases = _aliases_in_sql(scrubbed)
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
        if low.startswith("stg_"):
            continue
        if low in {"proj", "project", "staging_dev", "raw_dev", "opentrace"}:
            continue
        out.add(ident)
    return out


def validate_sql_column_allowlist(sql: str, table_ids: set[str] | None = None) -> str | None:
    """
    Reject identifiers that are not YAML columns for referenced (or provided) tables.

    When YAML columns cannot be loaded for any referenced table, skip the check.
    """
    refs = referenced_stg_tables(sql)
    if table_ids:
        refs = refs | {str(t).strip().split(".")[-1].lower() for t in table_ids if str(t).strip()}
    if not refs:
        return None
    col_map = columns_for_tables(refs)
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
    refs = referenced_stg_tables(sql)
    if table_ids:
        refs = refs | {str(t).strip().split(".")[-1].lower() for t in table_ids if str(t).strip()}
    if not refs:
        return None
    samples_map = value_samples_for_tables(refs)
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
