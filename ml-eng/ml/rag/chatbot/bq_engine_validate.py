"""Engine SQL validation — no class_engines dependency."""
from __future__ import annotations

import re

from ml.rag.chatbot.bq_sql_validate import (
    validate_sql_column_allowlist,
    validate_sql_table_allowlist,
)
from ml.rag.chatbot.bundle_metrics import unsupported_grain_for_panel
from ml.rag.chatbot.geo_iso3 import validate_sql_country_iso3_subset

_EXTRACT_YEAR_WHERE_RE = re.compile(
    r"extract\s*\(\s*year\s+from\s+as_of_date\s*\)",
    re.I,
)


def validate_engine_sql(
    sql: str,
    *,
    table_id: str,
    selected_tables: list[str] | None = None,
    allowed_iso3: list[str] | None = None,
) -> tuple[bool, str]:
    if _EXTRACT_YEAR_WHERE_RE.search(sql or ""):
        return False, "EXTRACT(YEAR FROM as_of_date) forbidden in WHERE"
    if re.search(r"\bgeo_key\s*=", sql or "", re.I):
        return False, "geo_key hash filter forbidden"
    if not re.search(r"\blimit\s+\d+", sql or "", re.I):
        return False, "LIMIT required"
    iso_count = len(allowed_iso3 or [])
    if unsupported_grain_for_panel(table_id, iso_count=iso_count):
        return False, f"unsupported_grain for panel on {table_id}"
    if allowed_iso3:
        geo_err = validate_sql_country_iso3_subset(sql, allowed_iso3)
        if geo_err:
            return False, geo_err
    tables = set(selected_tables or [table_id])
    err = validate_sql_table_allowlist(sql, tables)
    if err:
        return False, err
    err = validate_sql_column_allowlist(sql, tables)
    if err:
        return False, err or ""
    return True, ""


__all__ = ["validate_engine_sql"]
