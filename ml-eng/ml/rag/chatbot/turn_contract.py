"""Typed turn contract: measure × geo_grain × time × job [× breakdown]."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

GeoGrain = Literal["point", "admin2", "admin1", "country", "region", "africa"]
TimeGrain = Literal["year", "year_range", "season", "latest", "panel"]
Job = Literal[
    "fact",
    "breakdown",
    "trend",
    "rank",
    "compare",
    "list",
    "outlook",
    "diagnose",
    "brief",
    "report",
    "synthesis",
    "clarify",
    "help",
    "social",
]
BreakdownDim = Literal["sex", "admin1", "admin2", "urban_rural", "product"]
ServeStatus = Literal[
    "served",
    "unsupported_grain",
    "unsupported_measure",
    "unsupported_dimension",
    "clarify",
]
PlanType = Literal["numeric", "narrative", "gap", "unsupported"]
VectorPolicy = Literal["none", "companion", "fallback_only"]
TimeRole = Literal["observation", "publication", "either", "historical"]

NUMERIC_JOBS: frozenset[str] = frozenset(
    {"fact", "breakdown", "trend", "rank", "compare", "list", "report", "synthesis"}
)
NARRATIVE_ALLOWED_JOBS: frozenset[str] = frozenset({"outlook", "diagnose", "brief"})
NON_RAG_JOBS: frozenset[str] = frozenset({"help", "social", "clarify"})
FAIL_CLOSED_STATUSES: frozenset[str] = frozenset(
    {"unsupported_grain", "unsupported_measure", "unsupported_dimension", "clarify"}
)


@dataclass
class TimeSpec:
    start: str = ""
    end: str = ""
    grain: TimeGrain = "year"
    relative: str = ""
    time_role: TimeRole = "either"
    hard_filter: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "grain": self.grain,
            "relative": self.relative,
            "time_role": self.time_role,
            "hard_filter": self.hard_filter,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> TimeSpec:
        if not isinstance(raw, dict):
            return cls()
        grain = str(raw.get("grain") or "year").strip().lower()
        if grain not in ("year", "year_range", "season", "latest", "panel"):
            grain = "year"
        time_role = str(raw.get("time_role") or "either").strip().lower()
        if time_role not in ("observation", "publication", "either", "historical"):
            time_role = "either"
        return cls(
            start=str(raw.get("start") or "").strip()[:10],
            end=str(raw.get("end") or "").strip()[:10],
            grain=grain,  # type: ignore[arg-type]
            relative=str(raw.get("relative") or "").strip(),
            time_role=time_role,  # type: ignore[arg-type]
            hard_filter=bool(raw.get("hard_filter")),
        )


@dataclass
class TurnContract:
    measure_id: str = ""
    sector: str = ""
    geo: list[str] = field(default_factory=list)
    geo_grain: GeoGrain = "country"
    time_spec: TimeSpec = field(default_factory=TimeSpec)
    job: Job = "fact"
    breakdown: list[BreakdownDim] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    answer_lang: str = "en"
    serve_status: ServeStatus = "served"
    serve_reason: str = ""
    sql_plan: dict[str, Any] = field(default_factory=dict)
    plan_type: PlanType = "numeric"
    contract_sql_only: bool = False
    skip_vector_retrieval: bool = False
    skip_bq: bool = False
    vector_policy: VectorPolicy = "none"
    vector_allow: list[str] = field(default_factory=list)
    vector_block: list[str] = field(default_factory=list)
    population: str = ""
    pathogen_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["time_spec"] = self.time_spec.to_dict()
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> TurnContract:
        if not isinstance(raw, dict):
            return cls()
        breakdown_raw = raw.get("breakdown")
        breakdown: list[BreakdownDim] = []
        if isinstance(breakdown_raw, list):
            for item in breakdown_raw:
                dim = str(item).strip().lower()
                if dim in ("sex", "admin1", "admin2", "urban_rural", "product"):
                    breakdown.append(dim)  # type: ignore[arg-type]
        geo_grain = str(raw.get("geo_grain") or "country").strip().lower()
        if geo_grain not in ("point", "admin2", "admin1", "country", "region", "africa"):
            geo_grain = "country"
        job = str(raw.get("job") or "fact").strip().lower()
        valid_jobs = (
            "fact", "breakdown", "trend", "rank", "compare", "list", "outlook",
            "diagnose", "brief", "report", "synthesis", "clarify", "help", "social",
        )
        if job not in valid_jobs:
            job = "fact"
        serve_status = str(raw.get("serve_status") or "served").strip().lower()
        valid_status = (
            "served", "unsupported_grain", "unsupported_measure",
            "unsupported_dimension", "clarify",
        )
        if serve_status not in valid_status:
            serve_status = "served"
        plan_type = str(raw.get("plan_type") or "numeric").strip().lower()
        if plan_type not in ("numeric", "narrative", "gap", "unsupported"):
            plan_type = "numeric"
        geo_raw = raw.get("geo")
        geo = [str(g).strip() for g in geo_raw if str(g).strip()] if isinstance(geo_raw, list) else []
        entities_raw = raw.get("entities")
        entities = (
            [str(e).strip() for e in entities_raw if str(e).strip()]
            if isinstance(entities_raw, list)
            else []
        )
        sql_plan = raw.get("sql_plan")
        return cls(
            measure_id=str(raw.get("measure_id") or "").strip(),
            sector=str(raw.get("sector") or "").strip(),
            geo=geo,
            geo_grain=geo_grain,  # type: ignore[arg-type]
            time_spec=TimeSpec.from_dict(raw.get("time_spec") if isinstance(raw.get("time_spec"), dict) else raw),
            job=job,  # type: ignore[arg-type]
            breakdown=breakdown,
            entities=entities,
            answer_lang=str(raw.get("answer_lang") or "en").strip() or "en",
            serve_status=serve_status,  # type: ignore[arg-type]
            serve_reason=str(raw.get("serve_reason") or "").strip(),
            sql_plan=dict(sql_plan) if isinstance(sql_plan, dict) else {},
            plan_type=plan_type,  # type: ignore[arg-type]
            contract_sql_only=bool(raw.get("contract_sql_only")),
            skip_vector_retrieval=bool(raw.get("skip_vector_retrieval")),
            skip_bq=bool(raw.get("skip_bq")),
            vector_policy=_parse_vector_policy(raw.get("vector_policy")),
            vector_allow=_parse_str_list(raw.get("vector_allow")),
            vector_block=_parse_str_list(raw.get("vector_block")),
            population=str(raw.get("population") or "").strip(),
            pathogen_id=str(raw.get("pathogen_id") or "").strip(),
        )

    def is_fail_closed(self) -> bool:
        return self.serve_status in FAIL_CLOSED_STATUSES

    def needs_numeric_pack(self) -> bool:
        return self.job in NUMERIC_JOBS and self.serve_status == "served"

    def should_retrieve_vector(self) -> bool:
        if self.skip_vector_retrieval:
            return False
        if self.job in NARRATIVE_ALLOWED_JOBS:
            return True
        if not self.measure_id:
            return self.job not in NON_RAG_JOBS
        return self.vector_policy != "none"


def _parse_vector_policy(raw: Any) -> VectorPolicy:
    val = str(raw or "none").strip().lower()
    if val in ("none", "companion", "fallback_only"):
        return val  # type: ignore[return-value]
    return "none"


def _parse_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]
