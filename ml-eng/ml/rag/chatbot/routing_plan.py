"""Turn routing spine: measure + classes + corpora from decompose/enricher facets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import MeasureHit
from ml.rag.chatbot.class_corpus_policy import corpora_for_classes
from ml.rag.chatbot.class_supervisor import SupervisorPlan
from ml.rag.chatbot.intent_bundles import MatchedBundle
from ml.rag.chatbot.turn_contract import TurnContract


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


__all__ = ["RoutingPlan", "build_routing_plan"]
