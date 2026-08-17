"""Extract tabular rows from BQ retrieval results for export builders."""
from __future__ import annotations

import ast
import re
from typing import Any

from ml.rag.chatbot.generator import is_usable_context_item

_BQ_SKIP_META = frozenset({
    "sql",
    "sql_index",
    "sql_count",
    "execution_error",
    "validation_failed",
    "status",
    "prep_error",
    "nl2sql_raw",
    "nl2sql_model",
    "sql_source",
    "tier",
    "data_level",
    "source_id",
    "value_semantics",
    "raw_row",
    "ranked_rows",
    "bq_enrichment",
    "table_id",
    "table_description",
    "source_domain",
    "source_layer",
})


def _item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("metadata")
    return raw if isinstance(raw, dict) else {}


def _parse_row_content(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if not text or text.startswith("[BQ"):
        return None
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return val if isinstance(val, dict) else None


def _row_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    meta = _item_metadata(item)
    raw = meta.get("raw_row")
    if isinstance(raw, dict) and raw:
        return dict(raw)
    row = _parse_row_content(str(item.get("content") or ""))
    if row is None and meta:
        row = {k: v for k, v in meta.items() if k not in _BQ_SKIP_META}
    return row


def rows_from_bq_results(bq_results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Collect row dicts from BQ context items, stripping SQL/metadata fields."""
    out: list[dict[str, Any]] = []
    for item in bq_results or []:
        if str(item.get("source") or "") != "bigquery":
            continue
        if not is_usable_context_item(item):
            continue
        meta = _item_metadata(item)
        raw_ranked = meta.get("ranked_rows")
        ranked: list[dict[str, Any]] = raw_ranked if isinstance(raw_ranked, list) else []
        if isinstance(ranked, list) and ranked:
            for entry in ranked:
                if not isinstance(entry, dict):
                    continue
                raw = entry.get("raw_row")
                if isinstance(raw, dict) and raw:
                    clean = {k: v for k, v in raw.items() if k not in _BQ_SKIP_META and v is not None}
                else:
                    clean = {
                        k: v
                        for k, v in entry.items()
                        if k not in {"raw_row", "rank"} and v is not None
                    }
                if entry.get("rank") is not None:
                    clean.setdefault("rank", entry.get("rank"))
                if entry.get("measure_label"):
                    clean.setdefault("measure_label", entry.get("measure_label"))
                if entry.get("unit"):
                    clean.setdefault("unit", entry.get("unit"))
                if clean:
                    out.append(clean)
            continue

        row = _row_from_item(item)
        if not row:
            continue
        clean = {k: v for k, v in row.items() if k not in _BQ_SKIP_META and v is not None}
        semantics = meta.get("value_semantics")
        if isinstance(semantics, dict):
            if semantics.get("measure_label"):
                clean.setdefault("measure_label", semantics.get("measure_label"))
            if semantics.get("unit"):
                clean.setdefault("unit", semantics.get("unit"))
        if clean:
            out.append(clean)
    return out


def slugify_filename(text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^\w\s-]", "", (text or "").lower())
    slug = re.sub(r"[\s_]+", "_", slug).strip("_")
    return (slug[:max_len] or "export").strip("_")


__all__ = ["rows_from_bq_results", "slugify_filename"]
