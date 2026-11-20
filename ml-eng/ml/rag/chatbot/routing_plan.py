"""Turn routing spine: measure + classes + corpora from decompose/enricher facets."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, TypedDict

from ml.rag.chatbot.agri_measure_ontology import MEASURES, MeasureHit, entity_is_measure_noise
from ml.rag.chatbot.analytical_intent import is_analytical_query, is_document_lookup
from ml.rag.chatbot.class_corpus_policy import corpora_for_classes
from ml.rag.chatbot.class_supervisor import SupervisorPlan
from ml.rag.chatbot.intent_bundles import MatchedBundle
from ml.rag.chatbot.slot_validator import crop_satisfied, geography_satisfied
from ml.rag.chatbot.turn_contract import FactRequirement, TurnContract, fact_scope


Source = Literal["bigquery", "qdrant"]


class RequirementDecision(TypedDict):
    requirement: str
    selected: bool
    reason: str


class SourceDecision(TypedDict):
    selected: bool
    reason: str
    requirements: list[RequirementDecision]


class SourceSelection(TypedDict):
    bigquery: SourceDecision
    qdrant: SourceDecision


def select_sources(
    turn: TurnContract, *, query: str, decomposition: dict[str, Any], task_mode: str,
) -> SourceSelection:
    """Choose source legs and retain why each requirement can or cannot run."""
    document_lookup = is_document_lookup(query)
    analytical = task_mode == "analytical" or is_analytical_query(query, decomposition)
    structured = (
        task_mode in ("fact_lookup", "data_export_only", "clarify")
        or turn.job in ("compare", "trend", "rank", "breakdown")
    )
    requirements = turn.facts
    if not requirements and turn.measure_id and (structured or analytical):
        requirements = [FactRequirement(turn.measure_id)]
    measures = [fact.measure_id for fact in requirements]

    facts: list[RequirementDecision] = []
    for fact in requirements:
        measure_id = fact.measure_id
        spec = MEASURES.get(measure_id)
        scope = fact_scope(turn, fact)
        question = fact.question or query
        commodity_scope = replace(scope, entities=[
            entity for entity in scope.entities if not entity_is_measure_noise(entity, primary_measures=[measure_id])
        ])
        if document_lookup:
            reason = "document_content_requested"
        elif spec is None or not spec.indicator_classes:
            reason = "no_warehouse_mapping"
        elif not geography_satisfied(query=question, decomposition=decomposition, contract=scope, measure=spec):
            reason = "geography_required"
        elif not crop_satisfied(
            query=question, decomposition=decomposition, contract=commodity_scope, measure=spec,
        ):
            reason = "commodity_required"
        else:
            reason = "scoped_indicator"
        facts.append({"requirement": fact.key, "selected": reason == "scoped_indicator", "reason": reason})

    has_scoped_fact = any(fact["selected"] for fact in facts)
    has_indicator_scope = has_scoped_fact or (
        bool(turn.geo) and any(fact["reason"] == "no_warehouse_mapping" for fact in facts)
    )
    discovery = (
        not document_lookup and not facts and analytical
        and bool(turn.geo or turn.geo_grain in ("africa", "region"))
        and bool(turn.entities or decomposition.get("domains"))
    )
    docs: list[RequirementDecision] = [
        {"requirement": doc.question, "selected": True, "reason": "document_evidence_requested"}
        for doc in turn.documents
    ]
    if document_lookup and not docs:
        docs.append({"requirement": query, "selected": True, "reason": "document_content_requested"})
    if docs:
        document_reason = "document_content_requested" if document_lookup else "document_evidence_requested"
    elif task_mode == "data_export_only":
        document_reason = "data_only_request"
    elif discovery or (analytical and not structured) or not (structured and (measures or task_mode == "clarify")):
        document_reason = "scoped_discovery" if analytical else "contextual_evidence"
    elif has_indicator_scope:
        document_reason = "indicator_context"
        docs.append({"requirement": query, "selected": True, "reason": "indicator_context"})
    else:
        document_reason = "insufficient_structured_scope"
    return {
        "bigquery": {
            "selected": has_scoped_fact or discovery,
            "reason": (
                "document_content_requested" if document_lookup else
                "scoped_indicators" if has_scoped_fact else
                "scoped_discovery" if discovery else
                facts[0]["reason"] if facts else "no_structured_requirement"
            ),
            "requirements": facts,
        },
        "qdrant": {
            "selected": document_reason not in ("data_only_request", "insufficient_structured_scope"),
            "reason": document_reason,
            "requirements": docs,
        },
    }


@dataclass(frozen=True)
class RoutingPlan:
    """Single control-plane object after decompose — drives supervisor, BQ, vectors."""

    primary_measure_id: str = ""
    indicator_classes: tuple[str, ...] = ()
    matched_bundles: tuple[str, ...] = ()
    job: str = "fact"
    corpus_allow: tuple[str, ...] = ()
    supervisor_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_measure_id": self.primary_measure_id,
            "indicator_classes": list(self.indicator_classes),
            "matched_bundles": list(self.matched_bundles),
            "job": self.job,
            "corpus_allow": list(self.corpus_allow),
            "supervisor_rationale": self.supervisor_rationale,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RoutingPlan | None:
        if not isinstance(raw, dict):
            return None
        return cls(
            primary_measure_id=str(raw.get("primary_measure_id") or "").strip(),
            indicator_classes=tuple(
                str(c).strip().upper() for c in (raw.get("indicator_classes") or []) if str(c).strip()
            ),
            matched_bundles=tuple(str(b).strip() for b in (raw.get("matched_bundles") or []) if str(b).strip()),
            job=str(raw.get("job") or "fact").strip(),
            corpus_allow=tuple(str(c).strip() for c in (raw.get("corpus_allow") or []) if str(c).strip()),
            supervisor_rationale=str(raw.get("supervisor_rationale") or ""),
        )


def build_routing_plan(
    *,
    turn: TurnContract,
    measure_hit: MeasureHit | None,
    supervisor_plan: SupervisorPlan,
    matched_bundles: tuple[MatchedBundle, ...] | None = None,
    primary_measures: list[str] | None = None,
) -> RoutingPlan:
    pm = ""
    if primary_measures:
        pm = str(primary_measures[0] or "").strip()
    elif measure_hit is not None:
        pm = measure_hit.measure.id
    elif turn.measure_id:
        pm = turn.measure_id

    classes = tuple(supervisor_plan.classes)
    if not classes and measure_hit is not None and measure_hit.measure.indicator_classes:
        classes = (measure_hit.measure.indicator_classes[0].upper(),)

    bundle_ids = tuple(mb.spec.id for mb in (matched_bundles or ()))
    corpus = tuple(corpora_for_classes(classes, secondary=supervisor_plan.secondary))

    return RoutingPlan(
        primary_measure_id=pm,
        indicator_classes=classes,
        matched_bundles=bundle_ids,
        job=str(turn.job or "fact"),
        corpus_allow=corpus,
        supervisor_rationale=supervisor_plan.rationale,
    )


__all__ = ["RoutingPlan", "Source", "SourceSelection", "build_routing_plan", "select_sources"]
