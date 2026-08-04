"""Extract tabular rows from BQ retrieval results for export builders."""
from __future__ import annotations

import ast
import re
from typing import Any

_BQ_SKIP_META = frozenset({
    "sql",
    "sql_index",
    "sql_count",
    "execution_error",
    "validation_failed",
    "tier",
    "data_level",
    "source_id",
})


def _parse_row_content(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if not text or text.startswith("[BQ"):
        return None
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return val if isinstance(val, dict) else None


def rows_from_bq_results(bq_results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Collect row dicts from BQ context items, stripping SQL/metadata fields."""
    out: list[dict[str, Any]] = []
    for item in bq_results or []:
        if str(item.get("source") or "") != "bigquery":
            continue
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        row = _parse_row_content(str(item.get("content") or ""))
        if row is None and meta:
            row = {k: v for k, v in meta.items() if k not in _BQ_SKIP_META}
        if not row:
            continue
        clean = {k: v for k, v in row.items() if k not in _BQ_SKIP_META and v is not None}
        if clean:
            out.append(clean)
    return out


def slugify_filename(text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^\w\s-]", "", (text or "").lower())
    slug = re.sub(r"[\s_]+", "_", slug).strip("_")
    return (slug[:max_len] or "export").strip("_")


__all__ = ["rows_from_bq_results", "slugify_filename"]
