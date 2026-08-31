"""Qdrant planner: corpus + time rules per narrative slot (peer to BQ reasoner)."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.corpus_catalog import select_corpora
from ml.rag.chatbot.reasoner_plan import ReasonerPlan, SubQuestion
from ml.rag.chatbot.turn_contract import NUMERIC_JOBS, TurnContract

_NUMERIC_JOBS_NO_NEWS = NUMERIC_JOBS | {"report", "synthesis", "breakdown"}


def plan_vector_corpora_for_reasoner(
    *,
    query: str,
    reasoner: ReasonerPlan | None,
    turn_contract: TurnContract | dict[str, Any] | None,
    plan_type: str | None,
    decomposition: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return corpus selection overrides for heavy-path narrative slots."""
    contract = (
        turn_contract
        if isinstance(turn_contract, TurnContract)
        else TurnContract.from_dict(turn_contract)
    )
    dec = decomposition if isinstance(decomposition, dict) else {}
    job = reasoner.job if reasoner else contract.job
    measure = reasoner.primary_measure if reasoner else contract.measure_id

    selection = select_corpora(
        dec,
        plan_type=plan_type,
        query=query,
        task_mode=reasoner.depth if reasoner else None,
        contract_job=job,
        vector_allow=list(contract.vector_allow),
        vector_block=list(contract.vector_block),
        vector_policy=contract.vector_policy,
    )
    active = list(selection.active)

    if job in _NUMERIC_JOBS_NO_NEWS and "news" in active:
        active = [c for c in active if c != "news"]

    if contract.measure_id == "disease_prevalence" or any(
        sq.measure == "disease_prevalence" for sq in (reasoner.subquestions if reasoner else ())
    ):
        for preferred in ("academic_papers", "public_reports"):
            if preferred not in active:
                active.insert(0, preferred)

    if job == "outlook" or (reasoner and reasoner.job == "outlook"):
        for preferred in ("public_reports", "policies"):
            if preferred not in active:
                active.append(preferred)

    hard_time = bool(contract.time_spec.hard_filter or dec.get("time_start") or dec.get("time_end"))

    return {
        "active_corpora": active,
        "rationale": selection.rationale,
        "hard_time_filter": hard_time,
        "news_allowed": job not in _NUMERIC_JOBS_NO_NEWS,
    }


def narrative_slots(reasoner: ReasonerPlan | None) -> tuple[SubQuestion, ...]:
    if reasoner is None:
        return ()
    return reasoner.vector_subquestions()
