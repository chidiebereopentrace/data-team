"""Coverage-aware time policy for relative phrases ("lately", "now", "recent").

A Recency window computed from the calendar ("2026-01-01..today") asks annual tables
for a year they do not hold yet. For relative phrases this policy reads the mart
coverage snapshot instead: a trend or comparison takes the latest N years every
requested member holds, a point fact takes the newest common year. Explicit years
are never touched. Every decision carries a disclosure sentence for the answer.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ml.rag.chatbot.coverage_ledger import lineage_parent
from ml.rag.chatbot.mart_coverage import load_coverage

WindowStatus = Literal["ok", "partial", "no_common_period", "no_coverage"]


def trend_years() -> int:
    return max(2, int(os.environ.get("RAG_TREND_LATEST_YEARS", "5") or 5))


@dataclass(frozen=True)
class CoverageWindow:
    status: WindowStatus
    start_year: int | None
    end_year: int | None
    policy: str
    disclosure: str
    coverage_table: str
    members: dict[str, list[int] | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_relative(time_windows: list[dict[str, Any]] | None) -> bool:
    """True when every resolved window came from a relative phrase (non-strict)."""
    windows = [w for w in (time_windows or []) if isinstance(w, dict)]
    return bool(windows) and all(not bool(w.get("strict", True)) for w in windows)


def member_coverage(table_id: str, iso3s: list[str]) -> tuple[dict[str, list[int] | None], str]:
    """Per-country [min_year, max_year] for the table, else for its dbt parent fact."""
    cov = load_coverage()
    bare = table_id.split(".")[-1].lower()
    candidates = [bare]
    parent = lineage_parent(bare)
    if parent:
        candidates.append(parent)
    for source in candidates:
        found = {iso: cov.entry(source, iso) for iso in iso3s}
        if any(found.values()):
            return (
                {iso: [e.min_year, e.max_year] if e else None for iso, e in found.items()},
                source,
            )
    return {iso: None for iso in iso3s}, ""


def resolve_relative_window(
    table_id: str,
    *,
    iso3s: list[str],
    phrase: str,
    series: bool,
) -> CoverageWindow:
    members, source = member_coverage(table_id, iso3s)
    known = {iso: span for iso, span in members.items() if span}
    label = phrase or "recent"
    if not known:
        return CoverageWindow(
            status="no_coverage",
            start_year=None,
            end_year=None,
            policy="relative_unknown_coverage",
            disclosure=f"No coverage record exists for {', '.join(iso3s) or 'this request'} in {table_id}.",
            coverage_table=source,
            members=members,
        )
    common_start = max(span[0] for span in known.values())
    common_end = min(span[1] for span in known.values())
    missing = sorted(iso for iso, span in members.items() if not span)
    if common_start > common_end:
        return CoverageWindow(
            status="no_common_period",
            start_year=None,
            end_year=None,
            policy="relative_common_period",
            disclosure=(
                f"The requested places share no common period in {source}: "
                + "; ".join(f"{iso} {s[0]}-{s[1]}" for iso, s in sorted(known.items()))
            ),
            coverage_table=source,
            members=members,
        )
    if series:
        start = max(common_start, common_end - trend_years() + 1)
        policy = f"relative_latest_{trend_years()}_common_years"
        what = f"the latest {common_end - start + 1} years available for every requested place"
    else:
        start = common_end
        policy = "relative_latest_common_year"
        what = "the newest year available for every requested place"
    span_text = f"{start}" if start == common_end else f"{start}-{common_end}"
    disclosure = f"'{label}' was read as {span_text}, {what} in the warehouse."
    if source != table_id.split(".")[-1].lower():
        disclosure += f" Coverage comes from the source table {source}."
    if missing:
        disclosure += f" No coverage record for: {', '.join(missing)}."
    return CoverageWindow(
        status="partial" if missing else "ok",
        start_year=start,
        end_year=common_end,
        policy=policy,
        disclosure=disclosure,
        coverage_table=source,
        members=members,
    )


__all__ = [
    "CoverageWindow",
    "is_relative",
    "member_coverage",
    "resolve_relative_window",
    "trend_years",
]
