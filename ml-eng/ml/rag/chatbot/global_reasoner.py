"""Global reasoner: job compiler + bundles + plan enricher → ReasonerPlan."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.export_intent import detect_export_intent
from ml.rag.chatbot.intent_bundles import MatchedBundle, match_intent_bundles
from ml.rag.chatbot.plan_enricher import enrich_subquestions, primary_measure_from_slots
from ml.rag.chatbot.reasoner_plan import ReasonerPlan, is_heavy_plan_type, should_compile_reasoner_plan
from ml.rag.chatbot.task_mode import is_analytical_query
from ml.rag.chatbot.turn_contract import TurnContract

_LAST_N_YEARS_RE = re.compile(
    r"\b(?:last|past|previous)\s+(\d{1,2})\s+years?\b",
    re.I,
)
_TREND_RE = re.compile(r"\b(trend|changed|over time|year.over.year)\b", re.I)
_COMPARE_RE = re.compile(r"\b(compare|versus|vs\.?|between)\b", re.I)
_LIST_RE = re.compile(r"\b(which|list|rank|top)\b", re.I)
_REPORT_RE = re.compile(r"\b(report|briefing|assessment)\b", re.I)


def _geos_from(decomposition: dict[str, Any] | None, contract: TurnContract) -> tuple[str, ...]:
    if contract.geo:
        return tuple(contract.geo)
    if not isinstance(decomposition, dict):
        return ()
    out: list[str] = []
    for key in ("geography", "countries", "regions", "geo"):
        val = decomposition.get(key)
        if isinstance(val, list):
            out.extend(str(x).strip() for x in val if str(x).strip())
        elif isinstance(val, str) and val.strip():
            out.append(val.strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for g in out:
        gl = g.lower()
        if gl not in seen:
            seen.add(gl)
            deduped.append(g)
    return tuple(deduped)


def _pin_time_window(
    query: str,
    decomposition: dict[str, Any] | None,
    contract: TurnContract,
) -> tuple[str, str]:
    ts = contract.time_spec.start or ""
    te = contract.time_spec.end or ""
    if isinstance(decomposition, dict):
        ts = ts or str(decomposition.get("time_start") or "")[:10]
        te = te or str(decomposition.get("time_end") or "")[:10]
    m = _LAST_N_YEARS_RE.search(query or "")
    if m and not (ts and te):
        try:
            n = int(m.group(1))
            from datetime import date

            end_y = date.today().year
            start_y = max(end_y - n + 1, 1980)
            return f"{start_y}-01-01", f"{end_y}-12-31"
        except ValueError:
            pass
    return ts[:10], te[:10]


def _pin_job(query: str, contract: TurnContract, task_mode: str) -> str:
    """Deterministic job pin — memory must not override explicit user job."""
    job = (contract.job or "fact").strip().lower()
    if job in ("help", "social", "clarify"):
        return job
    q = query or ""
    if contract.breakdown and job == "fact":
        return "breakdown"
    if re.search(r"\b(outlook|lean season|ipc phase)\b", q, re.I):
        return "outlook"
    if job == "outlook":
        return "outlook"
    if _REPORT_RE.search(q) or task_mode == "analytical" and is_analytical_query(q, None):
        if job not in ("outlook", "list", "compare", "rank"):
            return "report" if _REPORT_RE.search(q) else job
    if _COMPARE_RE.search(q) and job in ("fact", "trend"):
        return "compare"
    if _LIST_RE.search(q) and job == "fact":
        return "list"
    if _TREND_RE.search(q) and job == "fact":
        return "trend"
    return job


def _shape_for_job(job: str, n_geos: int) -> str:
    j = job.strip().lower()
    if j in ("compare", "rank") and n_geos >= 2:
        return "panel_compare"
    if j == "trend":
        return "trend"
    if j == "list":
        return "list"
    if j == "outlook":
        return "outlook"
    if j == "report":
        return "report"
    if j == "breakdown":
        return "breakdown"
    return "fact"


def compile_reasoner_plan(
    query: str,
    *,
    decomposition: dict[str, Any] | None,
    turn_contract: TurnContract | dict[str, Any] | None,
    plan_type: str | None,
    task_mode: str = "chat",
    known_tables: set[str] | None = None,
    matched_bundles: tuple[MatchedBundle, ...] | None = None,
) -> ReasonerPlan | None:
    contract = (
        turn_contract
        if isinstance(turn_contract, TurnContract)
        else TurnContract.from_dict(turn_contract)
    )
    bundles = matched_bundles
    if bundles is None:
        bundles = match_intent_bundles(
            query,
            decomposition,
            breakdown=list(contract.breakdown),
        )
    if not should_compile_reasoner_plan(
        contract,
        task_mode=task_mode,
        plan_type=plan_type,
        matched_bundles=bundles,
        query=query,
        decomposition=decomposition if isinstance(decomposition, dict) else None,
    ):
        return None

    geos = _geos_from(decomposition, contract)
    heavy = is_heavy_plan_type(plan_type, job=contract.job, n_geos=len(geos))
    depth = "full" if heavy else "light"

    job = _pin_job(query, contract, task_mode)
    ts, te = _pin_time_window(query, decomposition, contract)
    export_kind = detect_export_intent(query) or "none"
    plan_depth = "report" if job in ("report", "synthesis") or export_kind != "none" else "chat"
    geo_grain = contract.geo_grain or "country"

    subquestions = enrich_subquestions(
        query=query,
        decomposition=decomposition,
        job=job,
        plan_type=str(plan_type or ""),
        geos=geos,
        geo_grain=geo_grain,
        time_start=ts,
        time_end=te,
        entities=list(contract.entities),
        breakdown=list(contract.breakdown),
        known_tables=known_tables,
        matched_bundles=bundles,
        depth=depth,
    )

    primary = primary_measure_from_slots(subquestions) or contract.measure_id or ""

    shape = _shape_for_job(job, len(geos))
    sections = ("lead", "spine", "subtopics", "interpretation", "implications", "limits", "sources")

    return ReasonerPlan(
        job=job,
        plan_type=str(plan_type or ""),
        export=export_kind or "none",
        depth=plan_depth,
        geos=geos,
        geo_grain=geo_grain,
        time_start=ts,
        time_end=te,
        subquestions=subquestions,
        shape=shape,
        sections=sections,
        heavy_path=heavy,
        primary_measure=primary,
    )


def reasoner_plan_to_bq_plan(reasoner: ReasonerPlan) -> dict[str, Any]:
    """Convert BQ subquestions to query_intents for bq_retriever."""
    intents: list[dict[str, Any]] = []
    selected: list[str] = []
    geo_filter = (
        "countries=" + ",".join(reasoner.geos[:32])
        if reasoner.geos
        else "geography from question"
    )
    y0 = (reasoner.time_start or "")[:4] or "start"
    y1 = (reasoner.time_end or "")[:4] or "end"

    for sq in reasoner.bq_subquestions():
        for tid in sq.tables:
            if tid not in selected:
                selected.append(tid)
        pattern = "time_series" if reasoner.job == "trend" else "rank_by_sum"
        if sq.measure in ("employment_share", "food_balance", "protected_area"):
            pattern = "custom"
        intents.append(
            {
                "goal": sq.nl,
                "tables": list(sq.tables) or ["fct_production"],
                "filters": f"{geo_filter}; year≈{y0}–{y1}; slot={sq.id}",
                "notes": f"reasoner_slot_{sq.id}",
                "pattern": pattern,
                "metric": "value",
                "grain": ["country_iso3", "year"]
                if reasoner.job in ("trend", "compare", "report")
                else ["country_iso3"],
                "order_by": "year DESC" if reasoner.job == "trend" else "total DESC",
                "subquestion_id": sq.id,
                "measure": sq.measure,
                "required": sq.required,
            }
        )

    from ml.rag.chatbot.analytical_bq_plan import analytical_sql_query_floor

    floor = max(len(intents), analytical_sql_query_floor())
    return {
        "selected_tables": selected,
        "query_intents": intents,
        "skip_bq": False,
        "analytical_mode": True,
        "heavy_path": reasoner.heavy_path,
        "slot_path": True,
        "max_sql_queries": floor,
        "rationale": "global_reasoner_slots",
        "reasoner_job": reasoner.job,
        "reasoner_shape": reasoner.shape,
        "crop_required": True,
        "geography_required": bool(reasoner.geos),
    }
