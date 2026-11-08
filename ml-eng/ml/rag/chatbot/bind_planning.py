"""Single entry point for compiling a table bind contract from a resolved request.

Applies the coverage-aware time policy to relative phrases before binding, so the
compiler never sees a calendar window the table cannot hold, and records the policy
on the contract for disclosure in the answer.
"""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import (
    TableBindContract,
    compile_table_bind_contract,
    resolve_geo_literals_for_table,
    year_column,
)
from ml.rag.chatbot.geo_iso3 import name_to_iso3
from ml.rag.chatbot.time_policy import is_relative, resolve_relative_window

_SERIES_RE = re.compile(r"\b(trend|over\s+time|since|evolution|changed?|growth|time\s*series)\b", re.I)
_SERIES_JOBS = frozenset({"trend", "compare", "time_series"})


def wants_series(facets: dict[str, Any], query: str) -> bool:
    for key in ("job", "intent", "task_mode"):
        if str(facets.get(key) or "").strip().lower() in _SERIES_JOBS:
            return True
    return bool(_SERIES_RE.search(query or ""))


def _iso3s(table_id: str, labels: list[str]) -> list[str]:
    out: list[str] = []
    for label in labels:
        text = str(label).strip()
        code = text.upper() if re.fullmatch(r"[A-Za-z]{3}", text) and text.isupper() else name_to_iso3(text)
        if code and code not in out:
            out.append(code)
    if not out:
        for lit in resolve_geo_literals_for_table(table_id, labels):
            code = lit if re.fullmatch(r"[A-Z]{3}", lit) else name_to_iso3(lit)
            if code and code not in out:
                out.append(code)
    return out


def apply_time_policy(
    table_id: str,
    facets: dict[str, Any],
    *,
    query: str,
    country_labels: list[str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (facets for binding, time policy record). Strict windows pass through unchanged."""
    windows = facets.get("time_windows") if isinstance(facets.get("time_windows"), list) else []
    if not is_relative(windows) or not year_column(table_id):
        return facets, {}
    labels = list(country_labels or facets.get("geography") or [])
    iso3s = _iso3s(table_id, labels)
    if not iso3s:
        return facets, {}
    phrase = str(facets.get("time_phrase") or "")
    window = resolve_relative_window(
        table_id,
        iso3s=iso3s,
        phrase=phrase,
        series=wants_series(facets, query),
    )
    record = window.to_dict()
    adjusted = dict(facets)
    adjusted["time_windows"] = []
    if window.start_year is None or window.end_year is None:
        # A relative phrase never becomes a calendar year the table may not hold.
        adjusted["time_start"] = None
        adjusted["time_end"] = None
        return adjusted, record
    adjusted["time_start"] = f"{window.start_year}-01-01"
    adjusted["time_end"] = f"{window.end_year}-12-31"
    adjusted["time_policy_source"] = f"policy:{window.policy}"
    return adjusted, record


def effective_time_bounds(contract: TableBindContract, facets: dict[str, Any]) -> tuple[str, str]:
    """(time_start, time_end) the contract actually binds, as ISO date prefixes."""
    policy = contract.time_policy or {}
    if not policy:
        return str(facets.get("time_start") or "")[:10], str(facets.get("time_end") or "")[:10]
    start, end = policy.get("start_year"), policy.get("end_year")
    if start is None or end is None:
        return "", ""
    return f"{start}-01-01", f"{end}-12-31"


def compile_bind_for_request(
    table_id: str,
    *,
    facets: dict[str, Any] | None,
    query: str,
    card: dict[str, Any] | None = None,
    country_labels: list[str] | None = None,
    product_labels: list[str] | None = None,
) -> TableBindContract:
    base = facets if isinstance(facets, dict) else {}
    bound_facets, policy = apply_time_policy(
        table_id, base, query=query, country_labels=country_labels
    )
    contract = compile_table_bind_contract(
        table_id,
        facets=bound_facets,
        card=card,
        query=query,
        country_labels=country_labels,
        product_labels=product_labels,
    )
    contract.time_policy = policy
    return contract


__all__ = [
    "apply_time_policy",
    "compile_bind_for_request",
    "effective_time_bounds",
    "wants_series",
]
