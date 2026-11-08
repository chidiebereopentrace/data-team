"""Result-shape contract for bind-compiled SQL: every requested member present, rows unique at the grain."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResultShape:
    missing_members: dict[str, list[str]] = field(default_factory=dict)
    duplicate_grain_rows: int = 0

    @property
    def complete(self) -> bool:
        return not self.missing_members and not self.duplicate_grain_rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_members": {k: list(v) for k, v in self.missing_members.items()},
            "duplicate_grain_rows": self.duplicate_grain_rows,
        }


def _norm(value: Any) -> str:
    return str(value if value is not None else "").strip().casefold()


def _requested_members(contract: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col_key, lit_key in (("geo_column", "geo_literals"), ("product_column", "product_literals")):
        col = str(contract.get(col_key) or "").strip()
        literals = [str(v).strip() for v in contract.get(lit_key) or [] if str(v).strip()]
        if col and len(literals) > 1:
            out[col] = literals
    return out


def check_result_shape(rows: list[dict[str, Any]], contract: dict[str, Any] | None) -> ResultShape:
    """Members the contract requested but no row carries, and rows repeating the same grain key.

    A single requested member is already enforced by the WHERE clause; only multi-member
    filters (compare, panels) can silently lose a member. Columns the SELECT did not project
    cannot be checked and are skipped.
    """
    if not rows or not isinstance(contract, dict):
        return ResultShape()
    missing: dict[str, list[str]] = {}
    for col, literals in _requested_members(contract).items():
        if not any(col in row for row in rows):
            continue
        seen = {_norm(row.get(col)) for row in rows}
        absent = [lit for lit in literals if _norm(lit) not in seen]
        if absent:
            missing[col] = absent
    measures = {str(c).lower() for c in contract.get("measure_columns") or []}
    keys = Counter(
        tuple(sorted((k, _norm(v)) for k, v in row.items() if k.lower() not in measures))
        for row in rows
    )
    duplicates = sum(n - 1 for n in keys.values() if n > 1) if measures else 0
    return ResultShape(missing_members=missing, duplicate_grain_rows=duplicates)
