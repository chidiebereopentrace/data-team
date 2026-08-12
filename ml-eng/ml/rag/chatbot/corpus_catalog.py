"""Corpus catalog + heuristic gate/boost for six Qdrant collections.

Makes parallel retrieve aware of what each collection holds and when to
prefer or soft-skip it. Deterministic — no LLM call.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CorpusSpec:
    key: str
    collection_env: str
    default_collection: str
    doc_kind: str
    context_kind: str
    role: str
    default_boost: float


CORPUS_CATALOG: dict[str, CorpusSpec] = {
    "news": CorpusSpec(
        key="news",
        collection_env="QDRANT_COLLECTION_NEWS",
        default_collection="news_data",
        doc_kind="news_article",
        context_kind="news",
        role="Timely journalism, events, markets",
        default_boost=0.04,
    ),
    "academic_papers": CorpusSpec(
        key="academic_papers",
        collection_env="QDRANT_COLLECTION_ACADEMIC_PAPERS",
        default_collection="academic_papers",
        doc_kind="academic_article",
        context_kind="academic",
        role="Peer-reviewed research methods and evidence",
        default_boost=0.06,
    ),
    "policies": CorpusSpec(
        key="policies",
        collection_env="QDRANT_COLLECTION_POLICIES",
        default_collection="policies",
        doc_kind="policy_document",
        context_kind="policy",
        role="Laws, strategies, policy frameworks",
        default_boost=0.06,
    ),
    "public_reports": CorpusSpec(
        key="public_reports",
        collection_env="QDRANT_COLLECTION_PUBLIC_REPORTS",
        default_collection="public_reports",
        doc_kind="public_report",
        context_kind="public_report",
        role="Institutional / FAO / NGO public reports",
        default_boost=0.06,
    ),
    "formation": CorpusSpec(
        key="formation",
        collection_env="QDRANT_COLLECTION_FORMATION",
        default_collection="formation",
        doc_kind="agricultural_practise",
        context_kind="formation",
        role="Training, how-to, farmer practices",
        default_boost=0.05,
    ),
    "ota": CorpusSpec(
        key="ota",
        collection_env="QDRANT_COLLECTION_OTA_INSIGHTS",
        default_collection="OTA_insights",
        doc_kind="ota_insight",
        context_kind="ota_insight",
        role="Analyst insights, metrics, recommendations",
        default_boost=0.05,
    ),
}

ALL_CORPUS_KEYS: tuple[str, ...] = tuple(CORPUS_CATALOG.keys())

_FORMATION_RE = re.compile(
    r"\b(how\s+to|practice|practise|training|extension|farmer\s+how|"
    r"agronomic\s+practice|cultivat|planting\s+guide|best\s+practice)\b",
    re.IGNORECASE,
)
_POLICY_RE = re.compile(
    r"\b(polic(?:y|ies)|regulation|regulatory|legislation|law|strategy|"
    r"framework|act\b|decree)\b",
    re.IGNORECASE,
)
_REPORT_RE = re.compile(
    r"\b(public\s+report|assessment|outlook|situation\s+report|"
    r"institutional\s+report|fao\s+report|ngo\s+report)\b",
    re.IGNORECASE,
)
_ACADEMIC_RE = re.compile(
    r"\b(study|studies|paper|journal|peer[- ]reviewed|evidence|literature|"
    r"research\s+findings|academic)\b",
    re.IGNORECASE,
)
_OTA_RE = re.compile(
    r"\b(recommendation|ota|metric\s+dashboard|analyst\s+insight|"
    r"opentrace\s+insight)\b",
    re.IGNORECASE,
)
_INVEST_RE = re.compile(
    r"\b("
    r"invest(?:ment|ments|ing)?|opportunity|opportunities|"
    r"where\s+to|"
    r"best\s+(?:country|place|market)\s+for|"
    r"sourcing|agribusiness"
    r")\b",
    re.IGNORECASE,
)
_NEWSY_RE = re.compile(
    r"\b(news|headline|latest|breaking|this\s+week|recently|"
    r"what\s+happened|market\s+update)\b",
    re.IGNORECASE,
)


@dataclass
class CorpusSelection:
    active: list[str] = field(default_factory=list)
    boosts: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": list(self.active),
            "boosts": dict(self.boosts),
            "rationale": self.rationale,
        }


def corpus_router_enabled() -> bool:
    raw = os.environ.get("RAG_CORPUS_ROUTER", "on").strip().lower()
    return raw not in ("0", "false", "off", "no")


def default_boosts() -> dict[str, float]:
    return {k: spec.default_boost for k, spec in CORPUS_CATALOG.items()}


def _boost(base: dict[str, float], key: str, delta: float) -> None:
    base[key] = round(float(base.get(key, 0.0)) + delta, 4)


def select_corpora(
    decomposition: dict[str, Any] | None,
    *,
    plan_type: str | None = None,
    query: str = "",
    task_mode: str | None = None,
) -> CorpusSelection:
    """Heuristic gate/boost over the six corpora. Never returns an empty active set."""
    boosts = default_boosts()
    if not corpus_router_enabled():
        return CorpusSelection(
            active=list(ALL_CORPUS_KEYS),
            boosts=boosts,
            rationale="router_off",
        )

    dec = decomposition if isinstance(decomposition, dict) else {}
    q = (query or "").strip()
    intent = str(dec.get("intent") or "").strip().lower()
    plan = (plan_type or "").strip()
    mode = (task_mode or "chat").strip().lower()
    has_time = bool(str(dec.get("time_start") or "").strip() or str(dec.get("time_end") or "").strip())

    preferred: set[str] = set()
    skip: set[str] = set()
    reasons: list[str] = []

    if mode == "briefing":
        preferred.update({"news", "ota"})
        _boost(boosts, "news", 0.12)
        _boost(boosts, "ota", 0.10)
        reasons.append("task_briefing")
    elif mode in ("fact_lookup", "data_export_only"):
        # Prefer structured narrative lightly; de-emphasize long-form policy/academic.
        _boost(boosts, "news", -0.04)
        _boost(boosts, "academic_papers", -0.06)
        _boost(boosts, "policies", -0.04)
        _boost(boosts, "public_reports", -0.04)
        reasons.append(f"task_{mode}_soft_narrative")

    if _FORMATION_RE.search(q):
        preferred.add("formation")
        _boost(boosts, "formation", 0.08)
        reasons.append("formation_cues")
    if _POLICY_RE.search(q):
        preferred.add("policies")
        _boost(boosts, "policies", 0.08)
        reasons.append("policy_cues")
    if _REPORT_RE.search(q):
        preferred.add("public_reports")
        _boost(boosts, "public_reports", 0.08)
        reasons.append("report_cues")
    if _ACADEMIC_RE.search(q):
        preferred.add("academic_papers")
        _boost(boosts, "academic_papers", 0.08)
        reasons.append("academic_cues")
    if _OTA_RE.search(q):
        preferred.add("ota")
        _boost(boosts, "ota", 0.08)
        reasons.append("ota_cues")
    if _INVEST_RE.search(q):
        preferred.update({"ota", "policies", "public_reports"})
        _boost(boosts, "ota", 0.12)
        _boost(boosts, "policies", 0.04)
        _boost(boosts, "public_reports", 0.04)
        reasons.append("investment_decision_cues")
    if _NEWSY_RE.search(q) or (has_time and intent in ("monitoring", "descriptive")):
        preferred.add("news")
        _boost(boosts, "news", 0.06)
        reasons.append("news_cues")

    if plan == "Farmers":
        preferred.update({"formation", "news"})
        _boost(boosts, "formation", 0.06)
        _boost(boosts, "news", 0.04)
        if "academic_cues" not in reasons:
            skip.add("academic_papers")
            reasons.append("farmers_soft_skip_academic")
        reasons.append("plan_farmers")

    if plan in ("Agribusinesses", "Integrated"):
        preferred.add("ota")
        _boost(boosts, "ota", 0.06)
        reasons.append("plan_ota_boost")

    if intent == "decision_support":
        preferred.update({"ota", "policies", "public_reports"})
        _boost(boosts, "ota", 0.06)
        _boost(boosts, "policies", 0.04)
        _boost(boosts, "public_reports", 0.04)
        reasons.append("intent_decision_support")

    if intent == "compare":
        preferred.update({"news", "academic_papers", "public_reports", "policies"})
        reasons.append("intent_compare_keep_broad")

    # No strong cue → all six.
    if not preferred and not skip and not reasons:
        return CorpusSelection(
            active=list(ALL_CORPUS_KEYS),
            boosts=boosts,
            rationale="default_all",
        )

    active = [k for k in ALL_CORPUS_KEYS if k not in skip]
    for k in preferred:
        if k not in active:
            active.append(k)

    # Never skip more than 3 corpora.
    skipped = [k for k in ALL_CORPUS_KEYS if k not in active]
    if len(skipped) > 3:
        for k in reversed(skipped):
            if len([x for x in ALL_CORPUS_KEYS if x not in active]) <= 3:
                break
            if k not in active:
                active.append(k)

    if not active:
        active = list(ALL_CORPUS_KEYS)
        reasons.append("fallback_empty")

    active_ordered = [k for k in ALL_CORPUS_KEYS if k in set(active)]
    return CorpusSelection(
        active=active_ordered,
        boosts=boosts,
        rationale=";".join(reasons) if reasons else "default_all",
    )


def context_kind_for_corpus_key(key: str) -> str:
    spec = CORPUS_CATALOG.get(key)
    return spec.context_kind if spec else key
