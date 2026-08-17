"""Extract tabular rows from BQ retrieval results for export builders."""
from __future__ import annotations

import ast
import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import match_product_samples
from ml.rag.chatbot.generator import is_usable_context_item
from ml.rag.chatbot.geo_regions import detect_regions_in_text

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


_YEAR_IN_QUERY_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def slugify_filename(text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^\w\s-]", "", (text or "").lower())
    slug = re.sub(r"[\s_]+", "_", slug).strip("_")
    return (slug[:max_len] or "export").strip("_")


def report_topic(
    query: str,
    *,
    decomposition: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return (document title, filename slug) from geography, crops, and year."""
    dec = decomposition if isinstance(decomposition, dict) else {}
    regions = detect_regions_in_text(query)
    geo_label = ""
    if regions:
        geo_label = str(regions[0]).replace("_", " ").title()
    else:
        geo = dec.get("geography")
        if isinstance(geo, list):
            names = [str(g).strip() for g in geo if str(g).strip()]
            if len(names) == 1:
                geo_label = names[0]
            elif 1 < len(names) <= 3:
                geo_label = " and ".join(names)
            elif names:
                geo_label = f"{names[0]} and {len(names) - 1} others"

    blob_parts = [query or ""]
    entities_raw = dec.get("entities")
    entities: list[Any] = list(entities_raw) if isinstance(entities_raw, list) else []
    blob_parts.extend(str(e) for e in entities)
    crops = match_product_samples("stg_faostat_production", " ".join(blob_parts))[:3]
    crop_label = " and ".join(crops)

    year = ""
    te = str(dec.get("time_end") or "")[:4]
    ts = str(dec.get("time_start") or "")[:4]
    if te.isdigit() and ts.isdigit() and ts != te:
        year = f"{ts}–{te}"
    elif te.isdigit():
        year = te
    elif ts.isdigit():
        year = ts
    else:
        found = _YEAR_IN_QUERY_RE.findall(query or "")
        if found:
            year = found[-1]

    title_bits = [p for p in (geo_label, crop_label, year) if p]
    title = ", ".join(title_bits) if title_bits else ((query or "").strip()[:80] or "OpenTrace report")
    return title, slugify_filename(title, max_len=60)


__all__ = ["rows_from_bq_results", "slugify_filename", "report_topic"]
