"""Resolve TurnContract against warehouse capability registry (fail closed)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ml.rag.chatbot.turn_contract import (
    BreakdownDim,
    NUMERIC_JOBS,
    TurnContract,
    VectorPolicy,
)
from ml.rag.chatbot.reasoner_plan import ReasonerPlan

_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "helpers" / "capability_registry.yaml"
_MEASURE_POLICY_PATH = Path(__file__).resolve().parents[1] / "helpers" / "measure_corpus_policy.yaml"

_GEO_GRAIN_TO_LEVEL: dict[str, str] = {
    "point": "point",
    "admin2": "admin2",
    "admin1": "admin1",
    "country": "national",
    "region": "national",
    "africa": "national",
}


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Any]:
    if not _REGISTRY_PATH.is_file():
        return {"entries": [], "geo_level_order": {}, "max_geo_level_rank": {}}
    with _REGISTRY_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def _load_measure_corpus_policy() -> dict[str, Any]:
    if not _MEASURE_POLICY_PATH.is_file():
        return {"defaults": {}}
    with _MEASURE_POLICY_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {"defaults": {}}


def _normalize_breakdown(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for item in raw:
        dim = str(item).strip().lower()
        if dim:
            out.append(dim)
    return tuple(sorted(out))


def _parse_corpus_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def measure_vector_lists(
    measure_id: str,
    *,
    policy: VectorPolicy,
) -> tuple[list[str], list[str]]:
    """Return (allow, block) corpus keys for a measure and vector policy."""
    cfg = _load_measure_corpus_policy()
    defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}
    row = cfg.get(measure_id.strip().lower()) if measure_id else None
    row = row if isinstance(row, dict) else {}

    if policy == "fallback_only":
        allow = _parse_corpus_list(row.get("fallback_allow"))
        block = _parse_corpus_list(row.get("fallback_block"))
        if not allow:
            allow = _parse_corpus_list(defaults.get("fallback_allow"))
        if not block:
            block = list({"policies", "formation", "news", "ota"})
        return allow, block

    allow = _parse_corpus_list(row.get("companion_allow"))
    block = _parse_corpus_list(row.get("companion_block"))
    if not allow:
        allow = _parse_corpus_list(defaults.get("companion_allow"))
    if not block:
        block = _parse_corpus_list(defaults.get("companion_block"))
    return allow, block


def apply_vector_policy(
    contract: TurnContract,
    *,
    entry: dict[str, Any] | None = None,
) -> None:
    """Set vector_policy / allow / block / skip flags on contract."""
    measure_id = contract.measure_id.strip().lower()

    if contract.job in ("help", "social", "clarify") or contract.serve_status == "clarify":
        contract.vector_policy = "none"
        contract.vector_allow = []
        contract.vector_block = []
        contract.skip_vector_retrieval = True
        return

    entry_policy = str((entry or {}).get("vector_policy") or "").strip().lower()
    fallback = _parse_corpus_list((entry or {}).get("fallback_corpora"))

    if contract.serve_status in ("unsupported_grain", "unsupported_measure", "unsupported_dimension"):
        if fallback or entry_policy == "fallback_only":
            contract.vector_policy = "fallback_only"
            allow, block = measure_vector_lists(measure_id, policy="fallback_only")
            if fallback:
                allow = fallback
            contract.vector_allow = allow
            contract.vector_block = block
            contract.skip_vector_retrieval = False
            contract.skip_bq = bool((entry or {}).get("skip_bq", True))
            contract.plan_type = "narrative"
            return
        contract.vector_policy = "none"
        contract.vector_allow = []
        contract.vector_block = []
        contract.skip_vector_retrieval = True
        return

    if contract.serve_status == "served" and contract.job in NUMERIC_JOBS:
        contract.vector_policy = "companion"
        allow, block = measure_vector_lists(measure_id, policy="companion")
        contract.vector_allow = allow
        contract.vector_block = block
        contract.skip_vector_retrieval = False
        return

    if contract.job in ("outlook", "diagnose", "brief"):
        contract.vector_policy = "none"
        contract.vector_allow = []
        contract.vector_block = []
        contract.skip_vector_retrieval = False
        return

    contract.vector_policy = "none"
    contract.vector_allow = []
    contract.vector_block = []
    contract.skip_vector_retrieval = False


def _entry_matches(
    entry: dict[str, Any],
    *,
    measure_id: str,
    geo_grain: str,
    time_grain: str,
    breakdown: tuple[str, ...],
    population: str = "",
) -> bool:
    if str(entry.get("measure") or "").strip().lower() != measure_id:
        return False
    if str(entry.get("geo_grain") or "").strip().lower() != geo_grain:
        return False
    entry_time = str(entry.get("time_grain") or "year").strip().lower()
    if entry_time != time_grain and not (entry_time == "year" and time_grain == "year_range"):
        if not (entry_time == "latest" and time_grain in ("latest", "year", "year_range")):
            if entry_time != time_grain:
                return False
    entry_bd = _normalize_breakdown(entry.get("breakdown"))
    if entry_bd and entry_bd != breakdown:
        return False
    if breakdown and not entry_bd and entry.get("breakdown") is not None:
        return False
    entry_pop = str(entry.get("population") or "").strip().lower()
    if entry_pop and entry_pop != population.strip().lower():
        return False
    return True


def _geo_grain_supported(entry: dict[str, Any], geo_grain: str) -> bool:
    max_level = str(entry.get("max_geo_level") or "national").strip().lower()
    registry = _load_registry()
    ranks = registry.get("max_geo_level_rank") or {}
    requested = _GEO_GRAIN_TO_LEVEL.get(geo_grain, geo_grain)
    req_rank = int(ranks.get(requested, 99))
    max_rank = int(ranks.get(max_level, 3))
    return req_rank <= max_rank


def _apply_unsupported_entry(contract: TurnContract, entry: dict[str, Any]) -> TurnContract:
    status = str(entry.get("status") or "").strip().lower()
    contract.serve_status = status  # type: ignore[assignment]
    contract.serve_reason = str(entry.get("reason") or status)
    contract.plan_type = "unsupported"
    contract.contract_sql_only = True
    apply_vector_policy(contract, entry=entry)
    return contract


def resolve_capability(contract: TurnContract) -> TurnContract:
    if contract.is_fail_closed():
        contract.skip_vector_retrieval = True
        contract.vector_policy = "none"
        contract.vector_allow = []
        contract.vector_block = []
        if contract.serve_status == "clarify":
            contract.plan_type = "gap"
        else:
            contract.plan_type = "unsupported"
        return contract

    measure_id = contract.measure_id.strip().lower()
    if not measure_id:
        contract.serve_status = "clarify"
        contract.serve_reason = "measure_unresolved"
        contract.plan_type = "gap"
        apply_vector_policy(contract)
        return contract

    registry = _load_registry()
    entries: list[dict[str, Any]] = list(registry.get("entries") or [])
    time_grain = contract.time_spec.grain
    breakdown = tuple(sorted(contract.breakdown))
    population = contract.population.strip().lower()

    for entry in entries:
        if not _entry_matches(
            entry,
            measure_id=measure_id,
            geo_grain=contract.geo_grain,
            time_grain=time_grain,
            breakdown=breakdown,
            population=population,
        ):
            continue
        status = str(entry.get("status") or "").strip().lower()
        if status == "unsupported_grain":
            return _apply_unsupported_entry(contract, entry)

    if breakdown:
        sex_only = breakdown == ("sex",) and measure_id == "employment_share"
        has_bd_row = any(
            _entry_matches(
                e,
                measure_id=measure_id,
                geo_grain=contract.geo_grain,
                time_grain=time_grain,
                breakdown=breakdown,
                population=population,
            )
            and not str(e.get("status") or "").strip()
            for e in entries
        )
        if sex_only and not has_bd_row:
            contract.serve_status = "unsupported_dimension"
            contract.serve_reason = "sex_breakdown_not_in_warehouse"
            contract.plan_type = "unsupported"
            contract.contract_sql_only = True
            apply_vector_policy(contract)
            return contract

    matched: dict[str, Any] | None = None
    for entry in entries:
        if str(entry.get("status") or "").strip():
            continue
        if not _entry_matches(
            entry,
            measure_id=measure_id,
            geo_grain=contract.geo_grain,
            time_grain=time_grain,
            breakdown=breakdown if breakdown else (),
            population=population,
        ):
            continue
        if not _geo_grain_supported(entry, contract.geo_grain):
            return _apply_unsupported_entry(
                contract,
                {
                    **entry,
                    "status": "unsupported_grain",
                    "reason": str(entry.get("reason") or f"max_geo_level={entry.get('max_geo_level')}"),
                },
            )
        matched = entry
        break

    if matched is None and breakdown and measure_id == "employment_share":
        for entry in entries:
            if str(entry.get("status") or "").strip():
                continue
            if not _entry_matches(
                entry,
                measure_id=measure_id,
                geo_grain=contract.geo_grain,
                time_grain=time_grain,
                breakdown=(),
                population=population,
            ):
                continue
            matched = entry
            contract.serve_reason = "sex_breakdown_unavailable_total_only"
            break

    if matched is None:
        contract.serve_status = "unsupported_measure"
        contract.serve_reason = f"no_registry_row:{measure_id}:{contract.geo_grain}:{time_grain}"
        contract.plan_type = "unsupported"
        contract.contract_sql_only = True
        apply_vector_policy(contract)
        return contract

    template = matched.get("template")
    contract.serve_status = "served"
    contract.sql_plan = {
        "table": str(matched.get("table") or "").strip(),
        "alt_table": str(matched.get("alt_table") or "").strip(),
        "template": str(template).strip() if template else "",
        "required_filters": list(matched.get("required_filters") or []),
        "contract_sql_only": bool(matched.get("contract_sql_only", True)),
    }
    contract.contract_sql_only = bool(matched.get("contract_sql_only", True))
    contract.plan_type = (
        "numeric" if contract.job in ("fact", "trend", "rank", "list", "compare") else contract.plan_type
    )
    apply_vector_policy(contract, entry=matched)
    return contract


def resolve_slot_capability(
    measure_id: str,
    *,
    geo_grain: str,
    time_grain: str,
    breakdown: tuple[str, ...] = (),
    population: str = "",
) -> str:
    """Per-slot warehouse capability (does not mutate turn contract)."""
    from ml.rag.chatbot.turn_contract import TimeSpec

    slot_contract = TurnContract(
        measure_id=(measure_id or "").strip().lower(),
        geo_grain=geo_grain,  # type: ignore[arg-type]
        time_spec=TimeSpec(grain=time_grain),  # type: ignore[arg-type]
        breakdown=list(breakdown),  # type: ignore[arg-type]
        population=population,
        job="fact",
    )
    resolved = resolve_capability(slot_contract)
    return str(resolved.serve_status or "served")


def apply_reasoner_to_turn(turn: TurnContract, reasoner: ReasonerPlan) -> TurnContract:
    """Slot-level capability: turn stays served if any required BQ slot is served."""
    required = [sq for sq in reasoner.bq_subquestions() if sq.required]
    if not required:
        if reasoner.primary_measure:
            turn.measure_id = reasoner.primary_measure
        return turn

    statuses = [
        resolve_slot_capability(
            sq.measure,
            geo_grain=turn.geo_grain,
            time_grain=turn.time_spec.grain,
            breakdown=tuple(turn.breakdown),
            population=turn.population,
        )
        for sq in required
    ]
    if any(st == "served" for st in statuses):
        turn.serve_status = "served"
        turn.skip_bq = False
        turn.plan_type = "numeric"
        turn.contract_sql_only = False
    elif statuses and all(str(st).startswith("unsupported") for st in statuses):
        turn.serve_status = statuses[0]  # type: ignore[assignment]
        turn.serve_reason = f"all_slots_unsupported:{','.join(sq.measure for sq in required)}"

    primary = reasoner.primary_measure or required[0].measure
    if primary:
        turn.measure_id = primary
    if reasoner.job:
        turn.job = reasoner.job  # type: ignore[assignment]
    apply_vector_policy(turn)
    return turn


def unsupported_answer_hint(contract: TurnContract) -> str:
    status = contract.serve_status
    if status == "unsupported_grain":
        if contract.measure_id == "disease_prevalence":
            pathogen = contract.pathogen_id or "this disease"
            return (
                f"National dairy-herd {pathogen} prevalence is not available in structured warehouse tables "
                f"(household×species grain only). Specify region, study year, or herd type if known."
            )
        return (
            f"We do not have structured data at {contract.geo_grain} grain for "
            f"{contract.measure_id or 'this measure'}. "
            f"Available data is at national level only."
        )
    if status == "unsupported_dimension":
        if contract.serve_reason == "sex_breakdown_unavailable_total_only":
            return (
                f"Total agricultural employment share is available; "
                f"a sex breakdown is not in the warehouse for {contract.measure_id or 'this measure'}."
            )
        return (
            f"A structured breakdown by {', '.join(contract.breakdown) or 'dimension'} "
            f"is not available in the warehouse for {contract.measure_id or 'this measure'}."
        )
    if status == "unsupported_measure":
        return "This measure is not available in the structured OpenTrace warehouse for the requested scope."
    return ""
