"""Internal reasoner plan: subquestions, persona gate, dual-bag composer input."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ml.rag.chatbot.turn_contract import NUMERIC_JOBS, TurnContract

LibraryKind = Literal["bq", "vector", "web"]

GLOBAL_REASONER_PLAN_TYPES = frozenset(
    {
        "government",
        "gov",
        "agribusiness",
        "agribusinesses",
        "integrated",
    }
)

# Optional escape hatch (default OFF): upgrade NGO to heavy when job in report/compare and n_geos>=3.
NGO_HEAVY_ESCAPE_HATCH = False

FORBIDDEN_JOB_NAMES = frozenset({"food_security_ipc", "land_inputs", "socio_economic"})

REASONER_JOBS = frozenset(
    {
        "fact",
        "breakdown",
        "trend",
        "rank",
        "compare",
        "list",
        "outlook",
        "diagnosis",
        "diagnose",
        "brief",
        "synthesis",
        "report",
        "clarify",
        "insufficient",
        "help",
        "social",
    }
)


def normalize_plan_type(plan_type: str | None) -> str:
    return (plan_type or "").strip().lower()


def is_heavy_plan_type(plan_type: str | None, *, job: str = "", n_geos: int = 0) -> bool:
    pt = normalize_plan_type(plan_type)
    if pt in GLOBAL_REASONER_PLAN_TYPES:
        return True
    if NGO_HEAVY_ESCAPE_HATCH and pt == "ngos":
        j = (job or "").strip().lower()
        if j in ("report", "compare") and n_geos >= 3:
            return True
    return False


def should_compile_reasoner_plan(
    contract: TurnContract | dict[str, Any] | None,
    *,
    task_mode: str = "",
    plan_type: str | None = None,
    matched_bundles: tuple[Any, ...] | None = None,
    query: str = "",
    decomposition: dict[str, Any] | None = None,
) -> bool:
    """Compile slot plans for resolved numeric/outlook turns with an explicit numeric routing signal."""
    tc = contract if isinstance(contract, TurnContract) else TurnContract.from_dict(contract)
    if tc.job in ("help", "social", "clarify"):
        return False
    if tc.serve_status == "clarify" or not tc.measure_id:
        return False
    if tc.job == "outlook":
        return True
    if tc.job not in NUMERIC_JOBS:
        return False

    mode = (task_mode or "").strip().lower()
    pt = normalize_plan_type(plan_type)
    ql = (query or "").strip().lower()
    if not matched_bundles and re.search(
        r"\b(policy|policies|regulation|strategy|framework|briefing|research say)\b",
        ql,
    ):
        if pt not in GLOBAL_REASONER_PLAN_TYPES:
            return False
    if isinstance(decomposition, dict):
        intent = str(decomposition.get("intent") or "").strip().lower()
        if intent in ("descriptive", "policy") and re.search(r"\b(policy|policies)\b", ql):
            if pt not in GLOBAL_REASONER_PLAN_TYPES:
                return False
    if pt in GLOBAL_REASONER_PLAN_TYPES:
        return True
    if matched_bundles:
        return True
    if mode in ("fact_lookup", "data_export_only", "analytical"):
        return True
    if pt in ("farmers", "ngos", "free"):
        return True
    if mode in ("research", "briefing"):
        return False
    return False


@dataclass(frozen=True)
class SubQuestion:
    id: str
    nl: str
    measure: str
    required: bool
    library: LibraryKind
    tables: tuple[str, ...] = field(default_factory=tuple)
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "nl": self.nl,
            "measure": self.measure,
            "required": self.required,
            "library": self.library,
            "tables": list(self.tables),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SubQuestion:
        tables_raw = raw.get("tables")
        tables = tuple(str(t) for t in tables_raw if str(t).strip()) if isinstance(tables_raw, list) else ()
        lib = str(raw.get("library") or "bq").strip().lower()
        if lib not in ("bq", "vector", "web"):
            lib = "bq"
        return cls(
            id=str(raw.get("id") or ""),
            nl=str(raw.get("nl") or ""),
            measure=str(raw.get("measure") or ""),
            required=bool(raw.get("required")),
            library=lib,  # type: ignore[arg-type]
            tables=tables,
            status=str(raw.get("status") or "pending"),
        )


@dataclass(frozen=True)
class ReasonerPlan:
    job: str
    plan_type: str
    export: str
    depth: str
    geos: tuple[str, ...]
    geo_grain: str
    time_start: str
    time_end: str
    subquestions: tuple[SubQuestion, ...]
    shape: str
    sections: tuple[str, ...] = field(default_factory=tuple)
    heavy_path: bool = False
    primary_measure: str = ""

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["subquestions"] = [s.to_dict() for s in self.subquestions]
        raw["geos"] = list(self.geos)
        raw["sections"] = list(self.sections)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ReasonerPlan | None:
        if not isinstance(raw, dict):
            return None
        subs = tuple(
            SubQuestion.from_dict(s) for s in (raw.get("subquestions") or []) if isinstance(s, dict)
        )
        geos = tuple(str(g) for g in (raw.get("geos") or []) if str(g).strip())
        sections = tuple(str(x) for x in (raw.get("sections") or []) if str(x).strip())
        return cls(
            job=str(raw.get("job") or "fact"),
            plan_type=str(raw.get("plan_type") or ""),
            export=str(raw.get("export") or "none"),
            depth=str(raw.get("depth") or "chat"),
            geos=geos,
            geo_grain=str(raw.get("geo_grain") or "country"),
            time_start=str(raw.get("time_start") or "")[:10],
            time_end=str(raw.get("time_end") or "")[:10],
            subquestions=subs,
            shape=str(raw.get("shape") or ""),
            sections=sections,
            heavy_path=bool(raw.get("heavy_path")),
            primary_measure=str(raw.get("primary_measure") or ""),
        )

    def bq_subquestions(self) -> tuple[SubQuestion, ...]:
        return tuple(s for s in self.subquestions if s.library == "bq")

    def vector_subquestions(self) -> tuple[SubQuestion, ...]:
        return tuple(s for s in self.subquestions if s.library == "vector")
