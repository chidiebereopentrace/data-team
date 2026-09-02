"""Class engine base types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EngineStatus = Literal[
    "ready",
    "planned",
    "ok",
    "empty_result",
    "timeout",
    "unsupported_grain",
    "planner_error",
    "deferred",
]


@dataclass
class EngineResult:
    class_code: str
    status: EngineStatus
    table_id: str
    sql: str | None
    rows: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    value_hits: dict[str, Any] = field(default_factory=dict)
    sql_plans: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    bind_contract: dict[str, Any] | None = None
    query_intents: list[dict[str, Any]] = field(default_factory=list)
    table_hints: list[str] = field(default_factory=list)
    hints_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_code": self.class_code,
            "status": self.status,
            "table_id": self.table_id,
            "sql": self.sql,
            "rows": self.rows,
            "caveats": list(self.caveats),
            "value_hits": dict(self.value_hits),
            "sql_plans": list(self.sql_plans),
            "bind_contract": dict(self.bind_contract) if self.bind_contract else None,
            "query_intents": list(self.query_intents),
            "table_hints": list(self.table_hints),
            "hints_truncated": self.hints_truncated,
        }


class ClassEngine:
    class_code: str = ""

    def run_plan(
        self,
        query: str,
        *,
        facets: dict[str, Any],
        card: dict[str, Any],
    ) -> EngineResult:
        raise NotImplementedError


__all__ = ["ClassEngine", "EngineResult", "EngineStatus"]
