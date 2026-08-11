"""BigQuery NL2SQL validation: table allowlist and dry-run checks."""
from __future__ import annotations

import os
import re
from typing import Any

_STG_TABLE_RE = re.compile(r"\bstg_[a-z0-9_]+\b", re.IGNORECASE)
_TABLE_HEADER_RE = re.compile(r"^Table:\s*[`\w.-]*\.?(\bstg_[a-z0-9_]+)", re.IGNORECASE | re.MULTILINE)
# FROM/JOIN target: backticked FQN or dotted/bare identifier (not a subquery paren).
_FROM_JOIN_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?![\(\s])(`[^`]+`|(?:[A-Za-z_][\w-]*)(?:\.[A-Za-z_][\w-]*)*)",
    re.IGNORECASE,
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
