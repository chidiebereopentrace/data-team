"""Class supervisor: routes indicator classes for warehouse engines (never writes SQL)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from ml.rag.chatbot.intent_bundles import (
    MatchedBundle,
    bundle_primary_measure,
    has_bundle,
    match_intent_bundles,
)
from ml.rag.chatbot.mart_indicator_classes import class_for_query, facts_for_class

_FOOD_BALANCE_SHARE_RE = re.compile(
    r"\b("
    r"food\s+balance|import\s+(dependency|share|ratio|s)?|domestic\s+supply|"
    r"self[-\s]?sufficien|imports?\s+vs\s+consumption|production\s+vs\s+imports"
    r")\b",
    re.I,
)
_AGRI_ACTIVITIES_RE = re.compile(
    r"\b(agricultural\s+activities|agri\s+activities|agri\s+report|country\s+by\s+country)\b",
    re.I,
)
_OUTLOOK_RE = re.compile(
    r"\b(ipc\s+phase|lean\s+season|food\s+security\s+outlook|fews)\b",
    re.I,
)


def _slot_reasoner_active() -> bool:
    return os.environ.get("RAG_SLOT_REASONER", "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class SupervisorPlan:
    classes: tuple[str, ...]
    secondary: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    must_search_qdrant: bool = True
    forbid_insufficient_without_attempt: bool = True
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "secondary": list(self.secondary),
            "out_of_scope": list(self.out_of_scope),
            "must_search_qdrant": self.must_search_qdrant,
            "forbid_insufficient_without_attempt": self.forbid_insufficient_without_attempt,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SupervisorPlan | None:
        if not isinstance(raw, dict):
            return None
        return cls(
            classes=tuple(str(c).strip().upper() for c in (raw.get("classes") or []) if str(c).strip()),
            secondary=tuple(str(c).strip().upper() for c in (raw.get("secondary") or []) if str(c).strip()),
            out_of_scope=tuple(str(c).strip() for c in (raw.get("out_of_scope") or []) if str(c).strip()),
            must_search_qdrant=bool(raw.get("must_search_qdrant", True)),
            forbid_insufficient_without_attempt=bool(raw.get("forbid_insufficient_without_attempt", True)),
            rationale=str(raw.get("rationale") or ""),
        )


def compile_supervisor_plan(
    query: str,
    *,
    decomposition: dict[str, Any] | None = None,
    matched_bundles: tuple[MatchedBundle, ...] | None = None,
) -> SupervisorPlan:
    """Route 1-N indicator classes. Default-on unless RAG_SLOT_REASONER=1 owns BQ."""
    q = (query or "").strip()
    dec = decomposition if isinstance(decomposition, dict) else {}
    bundles = matched_bundles or match_intent_bundles(q, dec)

    if _slot_reasoner_active() and (dec.get("reasoner_job") or dec.get("matched_bundles")):
        return SupervisorPlan(
            classes=(),
            secondary=(),
            out_of_scope=("slot_reasoner_owned",),
            rationale="slot_reasoner_owned",
        )

    scored = class_for_query(q)
    primary: str | None = None
    secondary: list[str] = []
    rationale_parts: list[str] = []

    if has_bundle(bundles, "food_balance_panel") and _FOOD_BALANCE_SHARE_RE.search(q):
        primary = "FVC"
        rationale_parts.append("food_balance_share→FVC")
    elif _AGRI_ACTIVITIES_RE.search(q) or has_bundle(bundles, "agricultural_activities"):
        primary = "PROD"
        secondary = ["FVC", "PRC"]
        rationale_parts.append("agri_activities→PROD+FVC+PRC")
    elif _OUTLOOK_RE.search(q) or has_bundle(bundles, "outlook_overlay"):
        primary = "FS"
        rationale_parts.append("outlook→FS")
    elif scored:
        primary = scored[0]
        for code in scored[1:3]:
            if code != primary:
                secondary.append(code)
        rationale_parts.append(f"class_for_query={','.join(scored[:3])}")

    if not primary:
        pm = bundle_primary_measure(bundles, q)
        measure_to_class = {
            "food_balance": "FVC",
            "production": "PROD",
            "trade": "FVC",
            "food_security_ipc": "FS",
            "yield": "PROD",
            "prices": "PRC",
        }
        if pm:
            primary = measure_to_class.get(pm, scored[0] if scored else "PROD")
            rationale_parts.append(f"bundle_primary={pm}")

    if not primary:
        primary = "PROD" if scored else ""
        rationale_parts.append("fallback")

    if not primary:
        return SupervisorPlan(
            classes=(),
            secondary=(),
            out_of_scope=("no_class",),
            rationale="no_class_match",
        )

    classes = (primary,)
    sec = tuple(s for s in secondary if s != primary)[:2]

    if primary == "FVC" and _FOOD_BALANCE_SHARE_RE.search(q):
        sec = ()

    return SupervisorPlan(
        classes=classes,
        secondary=sec,
        out_of_scope=(),
        must_search_qdrant=True,
        forbid_insufficient_without_attempt=True,
        rationale="; ".join(rationale_parts) or primary,
    )


def tables_for_supervisor_plan(plan: SupervisorPlan) -> list[str]:
    """Union of default tables for routed classes (for hints / validation)."""
    seen: set[str] = set()
    out: list[str] = []
    for code in (*plan.classes, *plan.secondary):
        for tid in facts_for_class(code):
            bare = tid.split(".")[-1].lower()
            if bare not in seen:
                seen.add(bare)
                out.append(bare)
    return out


__all__ = [
    "SupervisorPlan",
    "compile_supervisor_plan",
    "tables_for_supervisor_plan",
]
