"""Post-retrieval generation strategy: answer shape and evidence priority."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ml.rag.chatbot.agri_measure_ontology import MEASURES, MeasureHit, resolve_measure
from ml.rag.chatbot.context_diversity import normalize_context_kind
from ml.rag.chatbot.generator import (
    classify_evidence_tier,
    filter_context_items,
    is_comparative_bq_query,
    is_numeric_data_query,
    is_ranking_numeric_query,
    is_usable_context_item,
    is_usable_structured_bq_row,
)
from ml.rag.chatbot.retrieval_contract import RetrievalContract
from ml.rag.chatbot.turn_contract import NUMERIC_JOBS, TurnContract
from ml.rag.chatbot.stakeholder_prompts import (
    CategorySource,
    prose_register_for_persona,
    resolve_effective_category,
)
from ml.rag.chatbot.output_format import (
    OutputType,
    answer_shape_from_output_type,
    answer_subtopics,
    grain_window_line,
    output_type_from_answer_shape,
    output_type_from_contract,
    output_type_from_job,
)

AnswerShape = Literal[
    "numeric_fact",
    "ranking",
    "comparison",
    "trend",
    "breakdown",
    "briefing_digest",
    "research_synthesis",
    "policy_narrative",
    "export_table",
    "gap_ack",
]

EvidenceTier = Literal["strong", "partial", "empty"]

LeadWith = Literal["structured_value", "narrative_context"]
MustGroundIn = Literal["bigquery", "narrative", "any"]

_DEFAULT_NARRATIVE_PRIORITY = (
    "news",
    "public_report",
    "policy",
    "academic",
    "ota_insight",
    "formation",
    "web",
    "bigquery",
)

_MEASURE_EVIDENCE_PRIORITY: dict[str, tuple[str, ...]] = {
    "production": ("bigquery", "public_report", "news", "academic", "ota_insight"),
    "yield": ("bigquery", "public_report", "academic", "news"),
    "trade": ("bigquery", "public_report", "news"),
    "market_price": ("bigquery", "news", "public_report", "ota_insight"),
    "food_security_ipc": ("public_report", "news", "bigquery", "ota_insight", "policy"),
    "climate": ("public_report", "academic", "bigquery", "news"),
    "soil": ("academic", "formation", "public_report", "bigquery"),
    "socio_economic": ("bigquery", "public_report", "news", "policy"),
    "employment_share": ("bigquery", "public_report", "news"),
    "investment": ("ota_insight", "news", "public_report", "bigquery"),
    "investor_best_country": ("bigquery", "ota_insight", "public_report", "news"),
    "land_inputs": ("bigquery", "public_report", "academic"),
    "emissions": ("bigquery", "public_report", "academic"),
    "livestock": ("bigquery", "public_report", "news", "academic"),
    "spatial_vegetation": ("bigquery", "academic", "formation"),
    "research_meta": ("academic", "public_report", "policy"),
    "news_briefing": ("news", "ota_insight", "public_report", "policy"),
    "research_synthesis": ("academic", "policy", "public_report", "news"),
    "data_export_panel": ("bigquery", "news", "public_report"),
}


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class GenerationPlanOntology:
    measure_id: str = ""
    measure_label: str = ""
    unit_hint: str = ""
    geography: list[str] = field(default_factory=list)
    time_start: str = ""
    time_end: str = ""
    companion_measure_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReportSection:
    title: str
    guidance: str
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "guidance": self.guidance, "optional": self.optional}


@dataclass(frozen=True)
class GenerationPlan:
    answer_shape: AnswerShape
    evidence_priority: tuple[str, ...]
    lead_with: LeadWith
    must_ground_in: MustGroundIn
    ontology: GenerationPlanOntology
    synthesis_notes: tuple[str, ...]
    deprioritize_kinds: tuple[str, ...]
    rationale: str
    report_sections: tuple[ReportSection, ...] = field(default_factory=tuple)
    effective_category: str = ""
    category_source: CategorySource = "none"
    use_bullet_layout: bool = False
    evidence_tier: EvidenceTier = "strong"
    plan_type: str = "numeric"
    serve_status: str = "served"
    contract_job: str = ""
    output_type: OutputType = "fact"
    grain_window_line: str = ""
    answer_subtopics: tuple[str, ...] = field(default_factory=tuple)
    has_usable_spine: bool = False

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["ontology"] = asdict(self.ontology)
        raw["report_sections"] = [s.to_dict() for s in self.report_sections]
        return raw


def _geo_from_decomposition(decomposition: dict[str, Any] | None) -> list[str]:
    if not isinstance(decomposition, dict):
        return []
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
    return deduped


def _time_from_decomposition(decomposition: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(decomposition, dict):
        return "", ""
    ts = str(decomposition.get("time_start") or decomposition.get("start_date") or "").strip()[:10]
    te = str(decomposition.get("time_end") or decomposition.get("end_date") or "").strip()[:10]
    return ts, te


def _context_kind_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items or []:
        if not is_usable_context_item(item):
            continue
        kind = normalize_context_kind(item)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _has_usable_bq(items: list[dict[str, Any]]) -> bool:
    return any(
        is_usable_context_item(item) and normalize_context_kind(item) == "bigquery"
        for item in items or []
    )


def _ontology_from_hit(
    hit: MeasureHit | None,
    decomposition: dict[str, Any] | None,
) -> GenerationPlanOntology:
    if not hit:
        companions: list[str] = []
        if isinstance(decomposition, dict):
            raw = decomposition.get("companion_measures")
            if isinstance(raw, list):
                companions = [str(x) for x in raw if str(x).strip()]
        geo = _geo_from_decomposition(decomposition)
        ts, te = _time_from_decomposition(decomposition)
        return GenerationPlanOntology(
            geography=geo,
            time_start=ts,
            time_end=te,
            companion_measure_ids=companions,
        )
    spec = hit.measure
    companions = list(spec.companions)
    if isinstance(decomposition, dict):
        raw = decomposition.get("companion_measures")
        if isinstance(raw, list):
            for c in raw:
                cs = str(c).strip()
                if cs and cs not in companions:
                    companions.append(cs)
    geo = _geo_from_decomposition(decomposition)
    ts, te = _time_from_decomposition(decomposition)
    unit_hint = ""
    if "tonne" in spec.filter_hints.lower() or "production" in spec.id:
        unit_hint = "tonnes"
    elif "price" in spec.id:
        unit_hint = "local currency or USD"
    return GenerationPlanOntology(
        measure_id=spec.id,
        measure_label=spec.id.replace("_", " "),
        unit_hint=unit_hint,
        geography=geo,
        time_start=ts,
        time_end=te,
        companion_measure_ids=companions,
    )


_PLANTING_SEASON_RE = re.compile(
    r"\b(rainy season|rainy seasons|planting|planting window|onset|when does.*start)\b",
    re.IGNORECASE,
)


def _shape_from_contract_job(job: str, *, breakdown: list[str] | None = None) -> AnswerShape | None:
    """Legacy answer_shape alias from TurnContract job."""
    out = output_type_from_job(job)
    if job == "fact" and breakdown and "sex" in breakdown:
        return "breakdown"
    return answer_shape_from_output_type(out)  # type: ignore[return-value]


def _flatten_slot_rows(rows_by_subquestion: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(rows_by_subquestion, dict):
        return []
    out: list[dict[str, Any]] = []
    for bag in rows_by_subquestion.values():
        if isinstance(bag, list):
            out.extend(r for r in bag if isinstance(r, dict))
    return out


def _reasoner_has_bq_slots(reasoner_plan: dict[str, Any] | None) -> bool:
    if not isinstance(reasoner_plan, dict):
        return False
    subs = reasoner_plan.get("subquestions")
    if not isinstance(subs, list):
        return False
    return any(
        isinstance(s, dict) and str(s.get("library") or "bq").strip().lower() == "bq"
        for s in subs
    )


def _shape_from_reasoner_plan(
    reasoner_plan: dict[str, Any] | None,
    *,
    task_mode: str,
) -> AnswerShape | None:
    if not isinstance(reasoner_plan, dict) or not _reasoner_has_bq_slots(reasoner_plan):
        return None
    job = str(reasoner_plan.get("job") or "fact").strip().lower()
    export_raw = reasoner_plan.get("export")
    export = str(export_raw).strip().lower() if isinstance(export_raw, str) else "none"
    if task_mode == "data_export_only" or export not in ("", "none"):
        return "export_table"
    by_job: dict[str, AnswerShape] = {
        "compare": "comparison",
        "rank": "ranking",
        "trend": "trend",
        "list": "ranking",
        "breakdown": "breakdown",
        "outlook": "briefing_digest",
        "report": "comparison",
        "synthesis": "research_synthesis",
        "brief": "briefing_digest",
        "diagnosis": "trend",
        "diagnose": "trend",
        "fact": "numeric_fact",
    }
    return by_job.get(job, "comparison")


def _base_shape_from_task_mode(task_mode: str) -> AnswerShape:
    mode = (task_mode or "chat").strip().lower()
    if mode == "fact_lookup":
        return "numeric_fact"
    if mode == "briefing":
        return "briefing_digest"
    if mode == "research":
        return "research_synthesis"
    if mode == "data_export_only":
        return "export_table"
    if mode == "analytical":
        return "comparison"
    return "policy_narrative"


def _refine_shape_from_query(
    query: str,
    decomposition: dict[str, Any] | None,
    base: AnswerShape,
) -> AnswerShape:
    if isinstance(decomposition, dict):
        intent = str(decomposition.get("intent") or "").strip().lower()
        if intent == "diagnostic":
            return "trend"
    if is_ranking_numeric_query(query):
        return "ranking"
    if is_comparative_bq_query(query, decomposition):
        return "comparison"
    if base == "numeric_fact" and _PLANTING_SEASON_RE.search(query or ""):
        if not is_ranking_numeric_query(query) and not is_numeric_data_query(query, decomposition):
            return "policy_narrative"
    if base == "numeric_fact" and not is_numeric_data_query(query, decomposition):
        return "policy_narrative"
    if base == "policy_narrative" and is_numeric_data_query(query, decomposition):
        return "numeric_fact"
    intent = ""
    if isinstance(decomposition, dict):
        intent = str(decomposition.get("intent") or "").strip().lower()
    if intent == "compare" and base not in ("export_table", "briefing_digest", "research_synthesis"):
        return "comparison"
    if intent in ("diagnostic", "predictive") and base == "policy_narrative":
        return "trend"
    return base


def _evidence_priority_for_measure(measure_id: str) -> tuple[str, ...]:
    if measure_id in _MEASURE_EVIDENCE_PRIORITY:
        return _MEASURE_EVIDENCE_PRIORITY[measure_id]
    spec = MEASURES.get(measure_id)
    if spec and "research" in " ".join(spec.corpus_domains).lower():
        return ("academic", "policy", "public_report", "news", "bigquery")
    return _DEFAULT_NARRATIVE_PRIORITY


def _apply_context_fingerprint(
    priority: tuple[str, ...],
    counts: dict[str, int],
    *,
    has_bq: bool,
    numeric_query: bool,
) -> tuple[str, ...]:
    present = [k for k, n in counts.items() if n > 0]
    if not present:
        return priority
    if has_bq and numeric_query:
        ordered = ["bigquery"] + [k for k in priority if k != "bigquery"]
        for k in present:
            if k not in ordered:
                ordered.append(k)
        return tuple(dict.fromkeys(ordered))
    if not has_bq and present:
        present_set = set(present)
        ordered: list[str] = []
        for k in priority:
            if k != "bigquery" and k in present_set and k not in ordered:
                ordered.append(k)
        for k in present:
            if k not in ordered:
                ordered.append(k)
        return tuple(ordered)
    merged: list[str] = []
    for k in priority:
        if k in present and k not in merged:
            merged.append(k)
    for k in present:
        if k not in merged:
            merged.append(k)
    return tuple(merged)


def _lead_with_for_shape(shape: AnswerShape, has_bq: bool) -> LeadWith:
    if shape in ("numeric_fact", "ranking", "export_table", "breakdown") and has_bq:
        return "structured_value"
    if shape in ("briefing_digest", "research_synthesis", "policy_narrative"):
        return "narrative_context"
    if has_bq and shape in ("comparison", "trend"):
        return "structured_value"
    return "narrative_context"


def _must_ground_in_for_shape(shape: AnswerShape, has_bq: bool) -> MustGroundIn:
    if shape == "gap_ack":
        return "any"
    if shape in ("numeric_fact", "ranking", "export_table", "breakdown") and has_bq:
        return "bigquery"
    if shape in ("briefing_digest", "research_synthesis") and not has_bq:
        return "narrative"
    return "any"


def _must_ground_in_for_output_type(output_type: OutputType, has_bq: bool) -> MustGroundIn:
    if output_type == "insufficient":
        return "any"
    if output_type in ("fact", "trend", "compare", "list", "export") and has_bq:
        return "bigquery"
    if output_type in ("outlook", "diagnosis", "brief") and not has_bq:
        return "narrative"
    if output_type == "outlook":
        return "narrative"
    return "any"


def _synthesis_notes(
    shape: AnswerShape,
    *,
    has_bq: bool,
    ontology: GenerationPlanOntology,
    priority: tuple[str, ...],
) -> tuple[str, ...]:
    notes: list[str] = []
    if shape == "numeric_fact" and has_bq:
        notes.append("Lead with the structured value, unit, country, and year.")
    if shape == "breakdown" and has_bq:
        notes.append("Give each breakdown series (e.g. male/female) with value, unit, and year from structured rows only.")
    if shape == "ranking":
        notes.append("State the ranked entities and values; prefer structured rows for ordering.")
    if shape == "comparison":
        notes.append("Organize by region or country; contrast structured and narrative sources explicitly.")
    if shape == "briefing_digest":
        notes.append("Use 3–6 bullets; prefer recent news and OTA insights.")
    if shape == "research_synthesis":
        notes.append("Synthesize academic and policy evidence; cite mechanisms, not raw table IDs.")
    if shape == "export_table":
        notes.append("Write a short caption only; numbers come from the export artifact.")
    if shape == "gap_ack":
        notes.append("Acknowledge the data gap; do not invent numeric facts.")
    if ontology.measure_id:
        notes.append(f"Primary measure: {ontology.measure_label or ontology.measure_id}.")
    if ontology.geography:
        notes.append(f"Scope geography: {', '.join(ontology.geography[:4])}.")
    if priority:
        notes.append(f"Prefer evidence from: {', '.join(priority[:4])}.")
    return tuple(notes[:3])


def _deprioritize_kinds(priority: tuple[str, ...], counts: dict[str, int]) -> tuple[str, ...]:
    if not priority:
        return ()
    top = set(priority[:3])
    return tuple(k for k, n in counts.items() if n > 0 and k not in top)


def _has_usable_spine(
    output_type: OutputType,
    *,
    has_bq: bool,
    evidence_tier: EvidenceTier,
) -> bool:
    if evidence_tier == "empty":
        return False
    if output_type in ("fact", "trend", "compare", "list", "export"):
        return has_bq
    return True


def format_template_for_shape(shape: str, *, use_bullets: bool = False) -> str:
    """Compact format contract for non-analytical chat modes."""
    if use_bullets:
        return (
            "FORMAT: Use 3–6 plain bullet points — what is happening, what it means, "
            "and one next step when evidence supports it. No ## headings."
        )
    templates: dict[str, str] = {
        "numeric_fact": (
            "FORMAT: One lead sentence with value, unit, geography, and year; "
            "optional one-sentence context. No headings."
        ),
        "ranking": (
            "FORMAT: Lead sentence naming #1; then numbered list "
            "'1. Country — value unit (year)'."
        ),
        "comparison": (
            "FORMAT: Use ## Comparison with structured figures or a small table when Context supports it."
        ),
        "trend": (
            "FORMAT: Use ## Trend for direction and magnitude when in Context; "
            "## Context for drivers from narrative sources."
        ),
        "breakdown": (
            "FORMAT: Lead with each breakdown series on its own line — "
            "label, value, unit, year. No policy essay."
        ),
    }
    return templates.get(shape, "")


def build_generation_plan(
    query: str,
    *,
    task_mode: str = "chat",
    decomposition: dict[str, Any] | None = None,
    measure_hit: MeasureHit | None = None,
    retrieval_contract: RetrievalContract | dict[str, Any] | None = None,
    reranked_context: list[dict[str, Any]] | None = None,
    plan_type: str | None = None,
    category: str | None = None,
    measure_id: str | None = None,
    turn_contract: dict[str, Any] | None = None,
    reasoner_plan: dict[str, Any] | None = None,
    rows_by_subquestion: dict[str, Any] | None = None,
) -> GenerationPlan:
    """Deterministic post-retrieval strategy for generation."""
    tc = TurnContract.from_dict(turn_contract)
    slot_reasoner = _reasoner_has_bq_slots(reasoner_plan)
    use_contract_shape = bool(turn_contract and tc.measure_id) and not slot_reasoner
    effective_category, category_source = resolve_effective_category(
        category=category,
        plan_type=plan_type,
        query=query,
        decomposition=decomposition,
    )
    usable = filter_context_items(list(reranked_context or []))
    slot_rows = _flatten_slot_rows(rows_by_subquestion)
    if slot_reasoner and slot_rows:
        usable = filter_context_items(list(usable) + slot_rows)
    counts = _context_kind_counts(usable)
    has_bq = _has_usable_bq(usable)

    hit = measure_hit
    if hit is None and measure_id and measure_id in MEASURES:
        hit = MeasureHit(MEASURES[measure_id], score=100, matched_alias=measure_id)
    if hit is None:
        hit = resolve_measure(query, decomposition)

    mid = ""
    if tc.measure_id:
        mid = tc.measure_id
    elif hit:
        mid = hit.measure.id
    elif isinstance(retrieval_contract, RetrievalContract):
        mid = retrieval_contract.primary_measures[0] if retrieval_contract.primary_measures else ""
    elif isinstance(retrieval_contract, dict):
        pm = retrieval_contract.get("primary_measures")
        if isinstance(pm, list) and pm:
            mid = str(pm[0])

    ontology = _ontology_from_hit(hit, decomposition)

    has_structured_numeric = any(is_usable_structured_bq_row(i) for i in usable)
    has_usable = bool(usable)

    numeric_fail_closed = (
        tc.is_fail_closed()
        and tc.serve_status != "clarify"
        and tc.job in NUMERIC_JOBS
        and not slot_reasoner
    )
    if not has_usable or numeric_fail_closed:
        evidence_tier: EvidenceTier = "empty"
        shape: AnswerShape = "gap_ack"
        output_type: OutputType = "insufficient"
        priority: tuple[str, ...] = ()
        rationale = "gap_no_usable_context" if not has_usable else f"contract_{tc.serve_status}"
    else:
        evidence_tier = classify_evidence_tier(
            query,
            usable,
            decomposition,
            structured_bq_numeric_available=has_structured_numeric,
        )
        if evidence_tier == "empty":
            shape = "gap_ack"
            output_type = "insufficient"
            priority = ()
            rationale = "gap_no_on_topic_evidence"
        else:
            reasoner_shape = _shape_from_reasoner_plan(reasoner_plan, task_mode=task_mode)
            if slot_reasoner and reasoner_shape:
                shape = reasoner_shape
                output_type = output_type_from_job(
                    str(reasoner_plan.get("job") or tc.job or "fact")  # type: ignore[union-attr]
                )
            elif use_contract_shape:
                output_type = output_type_from_contract(
                    tc,
                    evidence_tier=evidence_tier,
                    has_usable_context=has_usable,
                    task_mode=task_mode,
                )
                contract_shape = _shape_from_contract_job(tc.job, breakdown=list(tc.breakdown))
                shape = contract_shape or answer_shape_from_output_type(output_type)  # type: ignore[assignment]
            else:
                contract_shape = None
                base = _base_shape_from_task_mode(task_mode)
                if task_mode == "analytical" and isinstance(decomposition, dict):
                    intent = str(decomposition.get("intent") or "").strip().lower()
                    if intent == "diagnostic":
                        base = "trend"
                    elif intent in ("compare", "decision_support"):
                        base = "comparison"
                if mid in ("news_briefing",):
                    base = "briefing_digest"
                elif mid in ("research_synthesis", "research_meta"):
                    base = "research_synthesis"
                if contract_shape is None:
                    shape = _refine_shape_from_query(query, decomposition, base)
                else:
                    shape = base
                output_type = output_type_from_contract(
                    tc if tc.measure_id else None,
                    evidence_tier=evidence_tier,
                    has_usable_context=has_usable,
                    task_mode=task_mode,
                )
                if not use_contract_shape:
                    output_type = output_type_from_answer_shape(shape)
            if task_mode == "data_export_only":
                output_type = "export"
                shape = "export_table"
            priority = _evidence_priority_for_measure(mid) if mid else _DEFAULT_NARRATIVE_PRIORITY
            if output_type == "outlook" or mid == "food_security_ipc":
                priority = ("public_report", "news", "ota_insight", "bigquery", "policy")
            priority = _apply_context_fingerprint(
                priority,
                counts,
                has_bq=has_bq,
                numeric_query=is_numeric_data_query(query, decomposition),
            )
            rationale = (
                f"task_{task_mode or 'chat'}|measure_{mid or 'none'}|"
                f"output_{output_type}|shape_{shape}|tier_{evidence_tier}"
            )

    lead = _lead_with_for_shape(shape, has_bq)
    ground = _must_ground_in_for_output_type(output_type, has_bq)
    if tc.vector_policy == "companion" and has_bq:
        lead = "structured_value"
    elif tc.vector_policy == "fallback_only":
        lead = "narrative_context"
        ground = "narrative"
    notes = _synthesis_notes(shape, has_bq=has_bq, ontology=ontology, priority=priority)
    if tc.vector_policy == "companion" and has_bq:
        notes = notes + (
            "Use narrative Context only for drivers or background; numeric claims must come from structured rows.",
        )
    elif tc.vector_policy == "fallback_only":
        notes = notes + ("Do not state a numeric prevalence unless explicitly supported by literature Context.",)
    deprior = _deprioritize_kinds(priority, counts)
    gw_line = grain_window_line(tc if tc.measure_id else None, asdict(ontology))
    subtopics = answer_subtopics(
        decomposition,
        tc if tc.measure_id else None,
        output_type,
    )
    has_spine = _has_usable_spine(output_type, has_bq=has_bq, evidence_tier=evidence_tier)
    reg = prose_register_for_persona(effective_category or None)
    mode_l = (task_mode or "chat").strip().lower()
    use_bullets = bool(reg and reg.use_bullet_layout and mode_l == "analytical")

    return GenerationPlan(
        answer_shape=shape,
        evidence_priority=priority,
        lead_with=lead,
        must_ground_in=ground,
        ontology=ontology,
        synthesis_notes=notes,
        deprioritize_kinds=deprior,
        rationale=rationale,
        report_sections=(),
        effective_category=effective_category or "",
        category_source=category_source,
        use_bullet_layout=use_bullets,
        evidence_tier=evidence_tier,
        plan_type=tc.plan_type,
        serve_status=tc.serve_status,
        contract_job=tc.job,
        output_type=output_type,
        grain_window_line=gw_line,
        answer_subtopics=subtopics,
        has_usable_spine=has_spine,
    )


def generation_plan_addendum(plan: GenerationPlan | dict[str, Any] | None) -> str:
    """Render a short system-prompt addendum from a generation plan."""
    if plan is None:
        return ""
    if isinstance(plan, dict):
        try:
            ont_raw = _dict_or_empty(plan.get("ontology"))
            plan_obj = GenerationPlan(
                answer_shape=str(plan.get("answer_shape") or "policy_narrative"),  # type: ignore[arg-type]
                evidence_priority=tuple(plan.get("evidence_priority") or ()),
                lead_with=str(plan.get("lead_with") or "narrative_context"),  # type: ignore[arg-type]
                must_ground_in=str(plan.get("must_ground_in") or "any"),  # type: ignore[arg-type]
                ontology=GenerationPlanOntology(
                    measure_id=str(ont_raw.get("measure_id") or ""),
                    measure_label=str(ont_raw.get("measure_label") or ""),
                    unit_hint=str(ont_raw.get("unit_hint") or ""),
                    geography=list(ont_raw.get("geography") or []),
                    time_start=str(ont_raw.get("time_start") or ""),
                    time_end=str(ont_raw.get("time_end") or ""),
                    companion_measure_ids=list(ont_raw.get("companion_measure_ids") or []),
                ),
                synthesis_notes=tuple(plan.get("synthesis_notes") or ()),
                deprioritize_kinds=tuple(plan.get("deprioritize_kinds") or ()),
                rationale=str(plan.get("rationale") or ""),
                report_sections=tuple(
                    ReportSection(
                        title=str(s.get("title") or ""),
                        guidance=str(s.get("guidance") or ""),
                        optional=bool(s.get("optional")),
                    )
                    for s in (plan.get("report_sections") or [])
                    if isinstance(s, dict)
                ),
                effective_category=str(plan.get("effective_category") or ""),
                category_source=str(plan.get("category_source") or "none"),  # type: ignore[arg-type]
                use_bullet_layout=bool(plan.get("use_bullet_layout")),
                evidence_tier=str(plan.get("evidence_tier") or "strong"),  # type: ignore[arg-type]
                output_type=str(plan.get("output_type") or "fact"),  # type: ignore[arg-type]
                grain_window_line=str(plan.get("grain_window_line") or ""),
                answer_subtopics=tuple(plan.get("answer_subtopics") or ()),
                has_usable_spine=bool(plan.get("has_usable_spine")),
            )
        except (TypeError, ValueError):
            return ""
    else:
        plan_obj = plan

    parts: list[str] = [
        "GENERATION STRATEGY:",
        f"Output type: {plan_obj.output_type}.",
        f"Answer shape: {plan_obj.answer_shape.replace('_', ' ')}.",
        f"Lead with: {'structured numeric facts' if plan_obj.lead_with == 'structured_value' else 'narrative context'}.",
    ]
    if plan_obj.evidence_priority:
        parts.append(f"Evidence priority: {', '.join(plan_obj.evidence_priority[:5])}.")
    if plan_obj.must_ground_in == "bigquery":
        parts.append("Ground quantitative claims in OpenTrace figures when present.")
    elif plan_obj.must_ground_in == "narrative":
        parts.append("Ground claims in news, policy, research, or reports; do not invent structured totals.")
    for note in plan_obj.synthesis_notes[:2]:
        parts.append(note)
    text = "\n".join(parts)
    return text[:400]
