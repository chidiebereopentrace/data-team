"""Advisory metacognition for Ask ADZA (plan + verify; not an agent loop).

MetacogPlan sits on top of ontology / supervisor / SQL — it advises job shape,
evidence needs, and composer tone. Post-retrieve verify is checklist-only
(no revise in MVP).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Final

from ml.rag.chatbot.capability_registry import apply_vector_policy
from ml.rag.chatbot.decompose_context import JOB_ALLOWED
from ml.rag.chatbot.turn_contract import NARRATIVE_ALLOWED_JOBS, NON_RAG_JOBS, TurnContract
from ml.rag.llm_chat import llm_chat_complete, llm_model_id

logger = logging.getLogger(__name__)

ADVISORY_MODES: Final[frozenset[str]] = frozenset(
    {
        "pass",
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
    }
)

EVIDENCE_NEEDS: Final[frozenset[str]] = frozenset({"bq", "news", "research", "ota", "web"})
MUST_GROUND_IN: Final[frozenset[str]] = frozenset({"bigquery", "narrative", "any"})
CLAIM_BUDGETS: Final[frozenset[str]] = frozenset(
    {"strict_numeric", "causal_careful", "narrative_ok"}
)

_EVIDENCE_TO_CORPORA: Final[dict[str, tuple[str, ...]]] = {
    "news": ("news",),
    "research": ("academic_papers", "policies", "public_reports"),
    "ota": ("ota",),
}

_DEFAULT_VERIFY_CHECKS: Final[tuple[str, ...]] = (
    "geo_match",
    "year_match",
    "measure_match",
    "no_invented_drivers",
)

_DIAGNOSE_CUE_RE = re.compile(
    r"\b(why|cause[sd]?|driver[s]?|what\s+led|what\s+caused|explain\s+why|"
    r"reason[s]?\s+for|behind\s+the|due\s+to|responsible\s+for)\b",
    re.I,
)
_FACT_CUE_RE = re.compile(
    r"\b(how\s+much|how\s+many|what\s+is\s+the\s+(production|yield|price|value)|"
    r"total|average|latest\s+figure)\b",
    re.I,
)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.I)

_CACHE_TTL_S = 3600.0
_CACHE_MAX = 256
_metacog_cache: dict[str, tuple[float, dict[str, Any]]] = {}

_APPLY_CONFIDENCE_MIN = 0.55

_SYSTEM = (
    "You are the metacognition planner for OpenTrace Ask ADZA "
    "(Africa-first agricultural intelligence). "
    "Return ONLY valid JSON with these keys:\n"
    '  advisory_mode: one of pass, fact, breakdown, trend, rank, compare, list, '
    "outlook, diagnose, brief, report, synthesis\n"
    "  confidence: number 0-1\n"
    "  evidence_needs: array from [bq, news, research, ota, web]\n"
    "  must_ground_in: one of bigquery, narrative, any\n"
    "  claim_budget: one of strict_numeric, causal_careful, narrative_ok\n"
    "  verify_checks: short string checklist items "
    '(e.g. geo_match, year_match, measure_match, no_invented_drivers)\n'
    "  composer_hints: up to 4 short imperative lines for answer tone\n"
    "  skip_verify: boolean\n"
    "  rationale: one short sentence\n"
    "Rules: prefer diagnose (not diagnosis) when the user asks why/causes/drivers. "
    "Prefer fact for single-value lookups. "
    "Do not invent crops, places, or years. "
    "pass means leave the compiled job unchanged."
)


@dataclass(frozen=True)
class MetacogPlan:
    advisory_mode: str = "pass"
    confidence: float = 0.0
    evidence_needs: tuple[str, ...] = ()
    must_ground_in: str = "any"
    claim_budget: str = "narrative_ok"
    verify_checks: tuple[str, ...] = _DEFAULT_VERIFY_CHECKS
    composer_hints: tuple[str, ...] = ()
    skip_verify: bool = False
    rationale: str = ""
    source: str = "none"  # none | llm | cache | heuristic

    def to_dict(self) -> dict[str, Any]:
        return {
            "advisory_mode": self.advisory_mode,
            "confidence": self.confidence,
            "evidence_needs": list(self.evidence_needs),
            "must_ground_in": self.must_ground_in,
            "claim_budget": self.claim_budget,
            "verify_checks": list(self.verify_checks),
            "composer_hints": list(self.composer_hints),
            "skip_verify": self.skip_verify,
            "rationale": self.rationale,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> MetacogPlan | None:
        if not isinstance(raw, dict):
            return None
        mode = str(raw.get("advisory_mode") or "pass").strip().lower()
        if mode == "diagnosis":
            mode = "diagnose"
        if mode not in ADVISORY_MODES:
            mode = "pass"
        needs: list[str] = []
        for item in raw.get("evidence_needs") or []:
            n = str(item).strip().lower()
            if n in EVIDENCE_NEEDS and n not in needs:
                needs.append(n)
        ground = str(raw.get("must_ground_in") or "any").strip().lower()
        if ground not in MUST_GROUND_IN:
            ground = "any"
        budget = str(raw.get("claim_budget") or "narrative_ok").strip().lower()
        if budget not in CLAIM_BUDGETS:
            budget = "narrative_ok"
        checks = tuple(
            str(c).strip() for c in (raw.get("verify_checks") or _DEFAULT_VERIFY_CHECKS) if str(c).strip()
        )[:12]
        hints = tuple(
            str(h).strip()[:160] for h in (raw.get("composer_hints") or []) if str(h).strip()
        )[:4]
        try:
            conf = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        source = str(raw.get("source") or "none").strip().lower()
        if source not in ("none", "llm", "cache", "heuristic"):
            source = "none"
        return cls(
            advisory_mode=mode,
            confidence=conf,
            evidence_needs=tuple(needs),
            must_ground_in=ground,
            claim_budget=budget,
            verify_checks=checks or _DEFAULT_VERIFY_CHECKS,
            composer_hints=hints,
            skip_verify=bool(raw.get("skip_verify")),
            rationale=str(raw.get("rationale") or "").strip()[:400],
            source=source,
        )


@dataclass(frozen=True)
class MetacogVerify:
    status: str = "ok"  # ok | partial | gap
    failed_checks: tuple[str, ...] = ()
    evidence_tier_hint: str = "strong"  # strong | partial | empty
    addendum: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failed_checks": list(self.failed_checks),
            "evidence_tier_hint": self.evidence_tier_hint,
            "addendum": self.addendum,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> MetacogVerify | None:
        if not isinstance(raw, dict):
            return None
        status = str(raw.get("status") or "ok").strip().lower()
        if status not in ("ok", "partial", "gap"):
            status = "ok"
        tier = str(raw.get("evidence_tier_hint") or "strong").strip().lower()
        if tier not in ("strong", "partial", "empty"):
            tier = "strong"
        failed = tuple(
            str(c).strip() for c in (raw.get("failed_checks") or []) if str(c).strip()
        )[:12]
        return cls(
            status=status,
            failed_checks=failed,
            evidence_tier_hint=tier,
            addendum=str(raw.get("addendum") or "").strip()[:800],
        )


def metacog_enabled() -> bool:
    """Off by default; set RAG_METACOG=1 to enable advisory metacognition."""
    raw = os.environ.get("RAG_METACOG", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def metacog_verify_enabled() -> bool:
    """Verify defaults on when metacog is on; RAG_METACOG_VERIFY=0 disables."""
    if not metacog_enabled():
        return False
    raw = os.environ.get("RAG_METACOG_VERIFY", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _metacog_model_id() -> str:
    for key in ("RAG_METACOG_MODEL_ID", "RAG_SUMMARY_MODEL_ID"):
        mid = (os.environ.get(key) or "").strip()
        if mid:
            return mid
    return llm_model_id()


def _metacog_max_tokens() -> int:
    try:
        return max(128, int(os.environ.get("RAG_METACOG_MAX_TOKENS", "400") or 400))
    except (TypeError, ValueError):
        return 400


def _normalize_cache_key(query: str, job: str, enrich: str) -> str:
    text = unicodedata.normalize("NFC", (query or "").strip())
    blob = f"{job}\n{enrich}\n{text}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> dict[str, Any] | None:
    hit = _metacog_cache.get(key)
    if not hit:
        return None
    ts, value = hit
    if time.monotonic() - ts > _CACHE_TTL_S:
        _metacog_cache.pop(key, None)
        return None
    return dict(value)


def _cache_set(key: str, value: dict[str, Any]) -> None:
    if len(_metacog_cache) >= _CACHE_MAX:
        ordered = sorted(_metacog_cache.items(), key=lambda kv: kv[1][0])
        for old_key, _ in ordered[: max(1, _CACHE_MAX // 4)]:
            _metacog_cache.pop(old_key, None)
    _metacog_cache[key] = (time.monotonic(), dict(value))


def clear_metacog_cache() -> None:
    _metacog_cache.clear()


def _extract_json_obj(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def heuristic_metacog_plan(query: str, *, current_job: str = "fact") -> MetacogPlan:
    """Deterministic fallback when LLM is off/fails — diagnosis vs fact cues."""
    q = query or ""
    job = (current_job or "fact").strip().lower()
    if job in NON_RAG_JOBS:
        return MetacogPlan(
            advisory_mode="pass",
            confidence=1.0,
            skip_verify=True,
            rationale="non_rag_job",
            source="heuristic",
        )
    if job == "diagnose" or _DIAGNOSE_CUE_RE.search(q):
        return MetacogPlan(
            advisory_mode="diagnose",
            confidence=0.72 if job != "diagnose" else 0.9,
            evidence_needs=("bq", "news", "research", "ota"),
            must_ground_in="any",
            claim_budget="causal_careful",
            verify_checks=_DEFAULT_VERIFY_CHECKS,
            composer_hints=(
                "Separate measured levels from hypothesized drivers.",
                "Do not invent causal drivers not in evidence.",
            ),
            rationale="diagnose_cues",
            source="heuristic",
        )
    if _FACT_CUE_RE.search(q) or job in ("fact", "breakdown", "trend", "rank", "compare", "list"):
        return MetacogPlan(
            advisory_mode="fact" if job == "fact" else (job if job in ADVISORY_MODES else "fact"),
            confidence=0.7,
            evidence_needs=("bq",),
            must_ground_in="bigquery",
            claim_budget="strict_numeric",
            verify_checks=("geo_match", "year_match", "measure_match"),
            composer_hints=("Lead with the measured value, unit, place, and period.",),
            rationale="fact_cues",
            source="heuristic",
        )
    return MetacogPlan(
        advisory_mode="pass",
        confidence=0.4,
        evidence_needs=("bq", "news"),
        must_ground_in="any",
        claim_budget="narrative_ok",
        rationale="no_strong_cues",
        source="heuristic",
    )


def build_metacog_plan(
    query: str,
    *,
    turn: TurnContract | None = None,
    decomposition: dict[str, Any] | None = None,
    language_enrich: str = "",
) -> MetacogPlan:
    """One structured LLM JSON call (cached); soft-fails to heuristics."""
    if not metacog_enabled():
        return MetacogPlan(source="none", rationale="metacog_disabled", skip_verify=True)

    q = (query or "").strip()
    job = str(getattr(turn, "job", None) or "fact")
    if not q:
        return MetacogPlan(source="none", rationale="empty_query", skip_verify=True)

    # The compiled contract already describes a fully specified fact lookup.
    if (
        turn is not None
        and turn.job == "fact"
        and turn.needs_numeric_pack()
        and turn.measure_id
        and len(turn.geo) == len(turn.entities) == 1
        and turn.time_spec.start
        and turn.time_spec.end
        and not turn.breakdown
        and not _DIAGNOSE_CUE_RE.search(q)
    ):
        return heuristic_metacog_plan(q, current_job=job)

    cache_key = _normalize_cache_key(q, job, language_enrich or "")
    cached = _cache_get(cache_key)
    if cached is not None:
        plan = MetacogPlan.from_dict({**cached, "source": "cache"})
        return plan or heuristic_metacog_plan(q, current_job=job)

    dec = decomposition if isinstance(decomposition, dict) else {}
    user_blob = (
        f"Query: {q}\n"
        f"English enrich (optional): {(language_enrich or '')[:400]}\n"
        f"Compiled job: {job}\n"
        f"Intent: {dec.get('intent') or ''}\n"
        f"Geography: {dec.get('geography') or []}\n"
        f"Entities: {dec.get('entities') or []}\n"
        f"Primary measures: {dec.get('primary_measures') or []}\n"
        "Return JSON now."
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_blob},
    ]
    try:
        raw = llm_chat_complete(
            messages,
            model=_metacog_model_id(),
            max_tokens=_metacog_max_tokens(),
            temperature=0.0,
            purpose="metacog.plan",
        )
    except Exception:
        logger.warning("metacog LLM failed", exc_info=True)
        return heuristic_metacog_plan(q, current_job=job)

    parsed = _extract_json_obj(raw or "")
    if not parsed:
        return heuristic_metacog_plan(q, current_job=job)

    plan = MetacogPlan.from_dict({**parsed, "source": "llm"})
    if plan is None:
        return heuristic_metacog_plan(q, current_job=job)
    _cache_set(cache_key, plan.to_dict())
    return plan


def corpora_for_evidence_needs(needs: tuple[str, ...] | list[str]) -> list[str]:
    """Map metacog evidence_needs → Qdrant corpus keys."""
    out: list[str] = []
    seen: set[str] = set()
    for need in needs:
        for key in _EVIDENCE_TO_CORPORA.get(str(need).strip().lower(), ()):
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def apply_metacog_to_turn(turn: TurnContract, plan: MetacogPlan) -> TurnContract:
    """Advisory overlay: pin diagnose/job + evidence corpora; never touch non-RAG jobs."""
    if plan.source == "none" and plan.advisory_mode == "pass":
        return turn
    if turn.job in NON_RAG_JOBS:
        return turn

    mode = plan.advisory_mode
    if (
        mode != "pass"
        and mode in JOB_ALLOWED
        and plan.confidence >= _APPLY_CONFIDENCE_MIN
    ):
        # Prefer diagnose over fact when metacog is confident; do not downgrade
        # report/outlook/compare unless upgrading to diagnose.
        if mode == "diagnose":
            turn.job = "diagnose"  # type: ignore[assignment]
        elif turn.job in ("fact", "breakdown") or turn.job == mode:
            turn.job = mode  # type: ignore[assignment]
        elif mode in ("outlook", "brief", "report", "synthesis") and turn.job == "fact":
            turn.job = mode  # type: ignore[assignment]

    if turn.job in NARRATIVE_ALLOWED_JOBS:
        turn.plan_type = "narrative"  # type: ignore[assignment]

    apply_vector_policy(turn)

    corpora = corpora_for_evidence_needs(plan.evidence_needs)
    if corpora:
        existing = list(turn.vector_allow or [])
        merged = list(dict.fromkeys([*existing, *corpora]))
        turn.vector_allow = merged
        turn.skip_vector_retrieval = False
        if turn.vector_policy == "none" and turn.job in NARRATIVE_ALLOWED_JOBS:
            turn.vector_policy = "companion"  # type: ignore[assignment]

    if "bq" in plan.evidence_needs:
        turn.skip_bq = False

    return turn


def metacog_diagnosis_addendum(plan: MetacogPlan) -> str:
    """Composer / generation addendum for diagnosis-shaped answers."""
    if plan.advisory_mode != "diagnose" and plan.claim_budget != "causal_careful":
        if not plan.composer_hints:
            return ""
    lines = ["METACOG ADVISORY:"]
    if plan.advisory_mode == "diagnose":
        lines.append(
            "Job shape: diagnosis — explain drivers only when evidence supports them."
        )
        lines.append("Lead with what is measured, then careful causal interpretation.")
    if plan.claim_budget == "causal_careful":
        lines.append("Claim budget: causal_careful — no invented drivers or mechanisms.")
    elif plan.claim_budget == "strict_numeric":
        lines.append("Claim budget: strict_numeric — numbers only from BQ or quoted passages.")
    if plan.must_ground_in == "bigquery":
        lines.append("Must ground quantities in BigQuery rows.")
    elif plan.must_ground_in == "narrative":
        lines.append("Prefer narrative corpora; flag missing warehouse rows.")
    for hint in plan.composer_hints[:4]:
        lines.append(f"- {hint}")
    return "\n".join(lines)[:900]


def _evidence_blob(
    *,
    bq_results: list[dict[str, Any]] | None,
    reranked: list[dict[str, Any]] | None,
    decomposition: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    for row in (bq_results or [])[:24]:
        if not isinstance(row, dict):
            continue
        parts.append(str(row.get("content") or ""))
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        parts.append(str(meta))
    for item in (reranked or [])[:24]:
        if not isinstance(item, dict):
            continue
        parts.append(str(item.get("content") or ""))
    if isinstance(decomposition, dict):
        parts.append(str(decomposition.get("geography") or ""))
        parts.append(str(decomposition.get("entities") or ""))
        parts.append(str(decomposition.get("time_start") or ""))
        parts.append(str(decomposition.get("time_end") or ""))
    return " ".join(parts).lower()


def verify_metacog_evidence(
    plan: MetacogPlan,
    *,
    query: str = "",
    decomposition: dict[str, Any] | None = None,
    bq_results: list[dict[str, Any]] | None = None,
    reranked_context: list[dict[str, Any]] | None = None,
) -> MetacogVerify:
    """Checklist verify after retrieve — ok / partial / gap (no revise)."""
    if plan.skip_verify or not metacog_verify_enabled():
        return MetacogVerify(status="ok", evidence_tier_hint="strong", addendum="")

    bq_n = len([r for r in (bq_results or []) if isinstance(r, dict)])
    vec_n = len([r for r in (reranked_context or []) if isinstance(r, dict)])
    blob = _evidence_blob(
        bq_results=bq_results,
        reranked=reranked_context,
        decomposition=decomposition,
    )
    failed: list[str] = []
    dec = decomposition if isinstance(decomposition, dict) else {}

    checks = plan.verify_checks or _DEFAULT_VERIFY_CHECKS
    for check in checks:
        key = str(check).strip().lower()
        if key == "geo_match":
            raw_geo = dec.get("geography")
            geos: list[Any] = list(raw_geo) if isinstance(raw_geo, list) else []
            missing = [
                g
                for g in geos
                if str(g).strip() and str(g).strip().lower() not in blob
            ]
            # Only fail when we expected geography and have some evidence bag.
            if geos and (bq_n + vec_n) > 0 and len(missing) == len(geos):
                failed.append("geo_match")
        elif key == "year_match":
            years: list[str] = []
            for field in ("time_start", "time_end"):
                raw = str(dec.get(field) or "")
                if len(raw) >= 4 and raw[:4].isdigit():
                    years.append(raw[:4])
            q_years = re.findall(r"\b(19|20)\d{2}\b", query or "")
            years.extend(q_years)
            years = list(dict.fromkeys(years))
            if years and (bq_n + vec_n) > 0 and not any(y in blob for y in years):
                failed.append("year_match")
        elif key == "measure_match":
            measures = []
            pm = dec.get("primary_measures")
            if isinstance(pm, list):
                measures.extend(str(m) for m in pm if str(m).strip())
            if measures and (bq_n + vec_n) > 0:
                if not any(m.replace("_", " ").lower() in blob or m.lower() in blob for m in measures):
                    # Soft: BQ rows often use different labels — only fail if empty-ish blob
                    if len(blob) < 40:
                        failed.append("measure_match")
        elif key == "no_invented_drivers":
            # Cannot prove inventions pre-generate; flag gap when diagnose + no evidence.
            if plan.advisory_mode == "diagnose" and bq_n + vec_n == 0:
                failed.append("no_invented_drivers")

    if bq_n + vec_n == 0:
        tier = "empty"
        status = "gap"
        if "evidence_present" not in failed:
            failed.append("evidence_present")
    elif failed:
        tier = "partial"
        status = "partial"
    else:
        tier = "strong"
        status = "ok"

    lines = ["METACOG VERIFY:"]
    lines.append(f"Status: {status}. Evidence tier: {tier}.")
    if failed:
        lines.append("Failed checks: " + ", ".join(failed[:8]) + ".")
        lines.append("Do not claim coverage for failed checks; state limits explicitly.")
    if plan.advisory_mode == "diagnose" and status != "ok":
        lines.append("Diagnosis: only cite drivers present in retrieved evidence.")
    addendum = "\n".join(lines)[:800]
    return MetacogVerify(
        status=status,
        failed_checks=tuple(dict.fromkeys(failed)),
        evidence_tier_hint=tier,
        addendum=addendum,
    )


_REVISE_DEFAULT_MODEL = "deepseek/deepseek-r1"
_SELF_CHECK_JOBS: Final[frozenset[str]] = frozenset({"diagnose", "report", "outlook"})

_REVISE_SYSTEM = (
    "You revise an Ask ADZA retrieval plan after evidence failed a checklist. "
    "Return ONLY valid JSON with keys:\n"
    '  action: "accept" or "replan"\n'
    "  advisory_mode: pass, fact, diagnose, report, outlook, or another allowed job\n"
    "  evidence_needs: array from [bq, news, research, ota, web]\n"
    "  geography: array of place names to retrieve, or empty to keep the current ones\n"
    "replan only when a different job, corpus, or place would fix the gap. "
    "accept when another retrieve will not help. Do not invent places."
)

_SELF_CHECK_SYSTEM = (
    "You check an Ask ADZA draft against retrieved evidence. "
    "Return ONLY valid JSON with keys:\n"
    '  verdict: "keep" or "revise"\n'
    "  problems: short strings naming contradictions "
    "(invented number, wrong country, causal claim with no row)\n"
    "revise only when the draft contradicts the evidence. "
    "keep when the draft already states limits or matches the rows."
)


@dataclass(frozen=True)
class MetacogReviseDecision:
    action: str = "skip"  # accept | replan | skip
    advisory_mode: str = "pass"
    evidence_needs: tuple[str, ...] = ()
    geography: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "advisory_mode": self.advisory_mode,
            "evidence_needs": list(self.evidence_needs),
            "geography": list(self.geography),
        }


@dataclass(frozen=True)
class MetacogSelfCheck:
    verdict: str = "skip"  # keep | revise | skip
    problems: tuple[str, ...] = ()
    addendum: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "problems": list(self.problems),
            "addendum": self.addendum,
        }


def metacog_revise_enabled() -> bool:
    """Off by default. One reasoning replan when verify status is gap."""
    if not metacog_enabled():
        return False
    raw = os.environ.get("RAG_METACOG_REVISE", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def metacog_self_check_enabled() -> bool:
    """Off by default. One reasoning look at the draft answer."""
    if not metacog_enabled():
        return False
    raw = os.environ.get("RAG_METACOG_SELF_CHECK", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _stage_model_id(env_key: str, default: str) -> str:
    mid = (os.environ.get(env_key) or "").strip()
    return mid or default


def _stage_max_tokens(env_key: str, default: int) -> int:
    try:
        return max(256, int(os.environ.get(env_key, str(default)) or default))
    except (TypeError, ValueError):
        return default


def evidence_sample(
    *,
    bq_results: list[dict[str, Any]] | None = None,
    reranked_context: list[dict[str, Any]] | None = None,
    limit: int = 6,
) -> str:
    lines: list[str] = []
    for row in (bq_results or [])[:limit]:
        if isinstance(row, dict):
            text = str(row.get("content") or "").strip()
            if text:
                lines.append(text[:400])
    for item in (reranked_context or [])[:limit]:
        if isinstance(item, dict):
            text = str(item.get("content") or "").strip()
            if text:
                lines.append(text[:400])
    return "\n".join(lines)[:2400]


def _clean_needs(raw: Any) -> tuple[str, ...]:
    needs: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            n = str(item).strip().lower()
            if n in EVIDENCE_NEEDS and n not in needs:
                needs.append(n)
    return tuple(needs)


def _clean_places(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    places: list[str] = []
    for item in raw:
        name = str(item).strip()
        if name and name not in places:
            places.append(name[:80])
    return tuple(places[:8])


def propose_metacog_revise(
    plan: MetacogPlan,
    verified: MetacogVerify,
    *,
    query: str = "",
    decomposition: dict[str, Any] | None = None,
    already_revised: int = 0,
    bq_results: list[dict[str, Any]] | None = None,
    reranked_context: list[dict[str, Any]] | None = None,
) -> MetacogReviseDecision:
    """One reasoning replan. Bad JSON and a second pass both accept/skip."""
    if already_revised >= 1 or not metacog_revise_enabled():
        return MetacogReviseDecision(action="skip")
    if verified.status != "gap":
        return MetacogReviseDecision(action="accept")

    dec = decomposition if isinstance(decomposition, dict) else {}
    user = (
        f"Query: {(query or '')[:500]}\n"
        f"Plan mode: {plan.advisory_mode}\n"
        f"Failed checks: {', '.join(verified.failed_checks)}\n"
        f"Geography: {dec.get('geography') or []}\n"
        f"Evidence sample:\n{evidence_sample(bq_results=bq_results, reranked_context=reranked_context)}\n"
        "Return JSON now."
    )
    try:
        raw = llm_chat_complete(
            [
                {"role": "system", "content": _REVISE_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=_stage_model_id("RAG_METACOG_REVISE_MODEL_ID", _REVISE_DEFAULT_MODEL),
            max_tokens=_stage_max_tokens("RAG_METACOG_REVISE_MAX_TOKENS", 4000),
            temperature=0.0,
            purpose="metacog.revise",
        )
    except Exception:
        logger.warning("metacog revise LLM failed", exc_info=True)
        return MetacogReviseDecision(action="accept")

    parsed = _extract_json_obj(raw or "")
    if not parsed:
        return MetacogReviseDecision(action="accept")
    action = str(parsed.get("action") or "accept").strip().lower()
    if action != "replan":
        return MetacogReviseDecision(action="accept")
    mode = str(parsed.get("advisory_mode") or "pass").strip().lower()
    if mode == "diagnosis":
        mode = "diagnose"
    if mode not in ADVISORY_MODES:
        mode = "pass"
    return MetacogReviseDecision(
        action="replan",
        advisory_mode=mode,
        evidence_needs=_clean_needs(parsed.get("evidence_needs")),
        geography=_clean_places(parsed.get("geography")),
    )


def apply_revise_decision(
    turn: TurnContract,
    plan: MetacogPlan,
    decision: MetacogReviseDecision,
) -> tuple[TurnContract, MetacogPlan, int]:
    """Apply a replan onto the turn and plan. Counter becomes 1 only on replan."""
    if decision.action != "replan":
        return turn, plan, 0
    mode = decision.advisory_mode if decision.advisory_mode != "pass" else plan.advisory_mode
    needs = decision.evidence_needs or plan.evidence_needs
    overlay = MetacogPlan(
        advisory_mode=mode,
        confidence=0.9,
        evidence_needs=needs,
        must_ground_in=plan.must_ground_in,
        claim_budget=plan.claim_budget,
        verify_checks=plan.verify_checks,
        composer_hints=plan.composer_hints,
        skip_verify=False,
        rationale="revise_replan",
        source="llm",
    )
    updated = apply_metacog_to_turn(turn, overlay)
    revised_plan = MetacogPlan(
        advisory_mode=updated.job if updated.job in ADVISORY_MODES else overlay.advisory_mode,
        confidence=overlay.confidence,
        evidence_needs=overlay.evidence_needs,
        must_ground_in=overlay.must_ground_in,
        claim_budget=overlay.claim_budget,
        verify_checks=overlay.verify_checks,
        composer_hints=overlay.composer_hints,
        skip_verify=False,
        rationale=overlay.rationale,
        source="llm",
    )
    return updated, revised_plan, 1


def self_check_applies(
    plan: MetacogPlan | None,
    verified: MetacogVerify | None,
    job: str = "",
) -> bool:
    if not metacog_self_check_enabled():
        return False
    mode = (plan.advisory_mode if plan else "") or ""
    current = (job or mode).strip().lower()
    if current in _SELF_CHECK_JOBS or mode in _SELF_CHECK_JOBS:
        return True
    return verified is not None and verified.status == "partial"


def self_check_draft(
    draft: str,
    *,
    query: str = "",
    evidence: str = "",
) -> MetacogSelfCheck:
    """One reasoning check of the draft. Bad JSON keeps the draft."""
    try:
        raw = llm_chat_complete(
            [
                {"role": "system", "content": _SELF_CHECK_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {(query or '')[:500]}\n"
                        f"Draft:\n{(draft or '')[:2000]}\n"
                        f"Evidence:\n{(evidence or '')[:2000]}\n"
                        "Return JSON now."
                    ),
                },
            ],
            model=_stage_model_id("RAG_METACOG_SELF_CHECK_MODEL_ID", _REVISE_DEFAULT_MODEL),
            max_tokens=_stage_max_tokens("RAG_METACOG_SELF_CHECK_MAX_TOKENS", 4000),
            temperature=0.0,
            purpose="metacog.self_check",
        )
    except Exception:
        logger.warning("metacog self-check LLM failed", exc_info=True)
        return MetacogSelfCheck(verdict="keep")

    parsed = _extract_json_obj(raw or "")
    if not parsed:
        return MetacogSelfCheck(verdict="keep")
    verdict = str(parsed.get("verdict") or "keep").strip().lower()
    if verdict != "revise":
        return MetacogSelfCheck(verdict="keep")
    problems = tuple(
        str(p).strip()[:180]
        for p in (parsed.get("problems") or [])
        if str(p).strip()
    )[:6]
    lines = ["METACOG SELF-CHECK: Rewrite the answer.", "Fix these problems:"]
    for problem in problems:
        lines.append(f"- {problem}")
    if not problems:
        lines.append("- The draft contradicts the retrieved evidence.")
    return MetacogSelfCheck(
        verdict="revise",
        problems=problems,
        addendum="\n".join(lines)[:800],
    )


def complete_self_check(
    draft: str,
    *,
    plan: MetacogPlan | None,
    verified: MetacogVerify | None,
    query: str,
    job: str,
    evidence: str,
    regenerate,
) -> tuple[str, MetacogSelfCheck]:
    """Regenerate at most once. The second answer is not checked again."""
    if not self_check_applies(plan, verified, job):
        return draft, MetacogSelfCheck(verdict="skip")
    check = self_check_draft(draft, query=query, evidence=evidence)
    if check.verdict != "revise":
        return draft, check
    second = str(regenerate(check.addendum) or "")
    return second or draft, check


__all__ = [
    "ADVISORY_MODES",
    "MetacogPlan",
    "MetacogReviseDecision",
    "MetacogSelfCheck",
    "MetacogVerify",
    "apply_metacog_to_turn",
    "apply_revise_decision",
    "build_metacog_plan",
    "clear_metacog_cache",
    "complete_self_check",
    "corpora_for_evidence_needs",
    "evidence_sample",
    "heuristic_metacog_plan",
    "metacog_diagnosis_addendum",
    "metacog_enabled",
    "metacog_revise_enabled",
    "metacog_self_check_enabled",
    "metacog_verify_enabled",
    "propose_metacog_revise",
    "self_check_applies",
    "self_check_draft",
    "verify_metacog_evidence",
]
