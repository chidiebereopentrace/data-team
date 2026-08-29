"""Post-retrieval generation strategy: answer shape and evidence priority."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ml.rag.chatbot.agri_measure_ontology import MEASURES, MeasureHit, resolve_measure
from ml.rag.chatbot.context_diversity import normalize_context_kind
from ml.rag.chatbot.generator import (
    filter_context_items,
    is_comparative_bq_query,
    is_numeric_data_query,
    is_ranking_numeric_query,
    is_usable_context_item,
)
from ml.rag.chatbot.retrieval_contract import RetrievalContract
from ml.rag.chatbot.stakeholder_prompts import (
    CategorySource,
    format_outline_for_persona,
    resolve_effective_category,
)

AnswerShape = Literal[
    "numeric_fact",
    "ranking",
    "comparison",
    "trend",
    "briefing_digest",
    "research_synthesis",
    "policy_narrative",
    "export_table",
    "gap_ack",
]

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
    if shape in ("numeric_fact", "ranking", "export_table") and has_bq:
        return "structured_value"
    if shape in ("briefing_digest", "research_synthesis", "policy_narrative"):
        return "narrative_context"
    if has_bq and shape in ("comparison", "trend"):
        return "structured_value"
    return "narrative_context"


def _must_ground_in_for_shape(shape: AnswerShape, has_bq: bool) -> MustGroundIn:
    if shape == "gap_ack":
        return "any"
    if shape in ("numeric_fact", "ranking", "export_table") and has_bq:
        return "bigquery"
    if shape in ("briefing_digest", "research_synthesis") and not has_bq:
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


_COMPANION_SECTION_TITLES: dict[str, str] = {
    "production": "Production and trade",
    "market_price": "Markets and prices",
    "trade": "Trade flows",
    "food_security_ipc": "Food security pressure",
    "socio_economic": "Economics and development",
    "investment": "Investment and research",
    "climate": "Climate context",
}


def _section(title: str, guidance: str, *, optional: bool = False) -> ReportSection:
    return ReportSection(title=title, guidance=guidance, optional=optional)


def build_analytical_report_outline(
    *,
    query: str,
    task_mode: str,
    answer_shape: AnswerShape,
    ontology: GenerationPlanOntology,
    has_bq: bool,
    decomposition: dict[str, Any] | None,
    category: str | None = None,
) -> tuple[tuple[ReportSection, ...], bool]:
    """Build dynamic report sections from retrieval fingerprint and question shape."""
    mode = (task_mode or "chat").strip().lower()
    if answer_shape == "gap_ack":
        return (), False
    if mode != "analytical" and answer_shape not in ("comparison", "trend", "ranking"):
        return (), False

    sections: list[ReportSection] = []
    mid = ontology.measure_id
    multi_geo = len(ontology.geography) != 1
    companions = list(ontology.companion_measure_ids or [])

    if answer_shape == "ranking":
        sections.append(
            _section(
                "Lead findings",
                "Open with the top-ranked entity and value from structured Context when present.",
            )
        )
        sections.append(
            _section(
                "Ranked detail",
                "List ranked countries or entities with values, units, and years from Context.",
            )
        )
        if has_bq:
            sections.append(
                _section(
                    "Trend note",
                    "Note direction or bracketing-year trend when Context includes it.",
                    optional=True,
                )
            )
    elif answer_shape == "trend":
        sections.append(
            _section(
                "Trend summary",
                "State direction, magnitude, and period using structured or narrative Context.",
            )
        )
        sections.append(
            _section(
                "Historical context",
                "Place the trend in recent years or seasons supported by Context.",
                optional=not has_bq,
            )
        )
        sections.append(
            _section(
                "Drivers",
                "Explain drivers only where Context supports them; do not invent causation.",
                optional=True,
            )
        )
        sections.append(
            _section(
                "Monitoring",
                "What to watch next based on available evidence.",
                optional=True,
            )
        )
    elif answer_shape == "comparison" or multi_geo:
        sections.append(
            _section(
                "Executive summary",
                "One-paragraph planning or decision takeaway with the strongest contrast from Context.",
            )
        )
        sections.append(
            _section(
                "Geographic comparison",
                "Compare countries or regions with structured figures when BQ rows are present.",
            )
        )
    else:
        sections.append(
            _section(
                "Key findings",
                "Lead with the strongest decision-relevant conclusions supported by Context.",
            )
        )
        sections.append(
            _section(
                "Regional picture",
                "Synthesise geographic patterns when Context provides regional material.",
                optional=not multi_geo,
            )
        )

    if mid == "investor_best_country":
        sections = [
            _section(
                "Investment signal",
                "Lead with relative country strength on available multi-signal evidence.",
            ),
            _section(
                "Production and trade",
                "Summarise production and trade signals from Context.",
                optional=not has_bq,
            ),
            _section(
                "Economics and prices",
                "Cover GDP, prices, or macro signals when present in Context.",
                optional=True,
            ),
            _section(
                "Food security risk",
                "Note IPC or food-security downside when Context includes it.",
                optional=True,
            ),
        ]
    elif mid == "food_security_ipc":
        sections = [
            _section(
                "Situation",
                "Summarise the food-security picture from IPC/FEWS or narrative Context.",
            ),
            _section(
                "Affected populations",
                "Who and where is most affected using geography and population figures in Context.",
            ),
            _section(
                "Market and production context",
                "Cross-domain companions: prices and staple production when Context includes them.",
                optional=True,
            ),
            _section(
                "Program implications",
                "Operational targeting or monitoring cues for program staff.",
                optional=True,
            ),
        ]
    else:
        seen_titles = {s.title.lower() for s in sections}
        for cid in companions[:4]:
            title = _COMPANION_SECTION_TITLES.get(cid, cid.replace("_", " ").title())
            if title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            sections.append(
                _section(
                    title,
                    f"Cover {cid.replace('_', ' ')} evidence from Context when present.",
                    optional=True,
                )
            )

    sections.append(
        _section(
            "Data notes",
            "Short neutral bullets on coverage limits only — never lead with gaps.",
            optional=False,
        )
    )

    sections = sections[:10]
    raw = [s.to_dict() for s in sections]
    formatted, use_bullets = format_outline_for_persona(category, raw)
    out = tuple(
        ReportSection(
            title=str(s.get("title") or ""),
            guidance=str(s.get("guidance") or ""),
            optional=bool(s.get("optional")),
        )
        for s in formatted
    )
    return out, use_bullets


ANALYTICAL_VOICE_RULES = (
    "ANALYTICAL VOICE RULES:\n"
    "Write as an external intelligence analyst, not as a system explaining limitations.\n"
    "- Lead with the strongest available findings. Never open with gaps or caveats.\n"
    "- Put coverage limits only in the final data-notes section (or a brief closing line for bullets).\n"
    "- Never write that structured data is unavailable, that OpenTrace lacks data, or that this is a data gap.\n"
    "- In data notes use neutral wording such as: "
    "'Coverage for [year/region/commodity] remains limited in available sources.'\n"
    "- Do not invent production totals, rankings, yields, or year values not in Context.\n"
    "- Tone: clear, decisive, concise. No academic padding or restating the question."
)


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
    }
    return templates.get(shape, "")


def analytical_outline_addendum(
    plan: GenerationPlan | dict[str, Any] | None,
) -> str:
    """Render dynamic report outline for analytical generation."""
    if plan is None:
        return ""
    sections: list[ReportSection] = []
    use_bullets = False
    if isinstance(plan, dict):
        use_bullets = bool(plan.get("use_bullet_layout"))
        for raw in plan.get("report_sections") or []:
            if isinstance(raw, dict):
                sections.append(
                    ReportSection(
                        title=str(raw.get("title") or ""),
                        guidance=str(raw.get("guidance") or ""),
                        optional=bool(raw.get("optional")),
                    )
                )
    else:
        sections = list(plan.report_sections)
        use_bullets = plan.use_bullet_layout

    if not sections:
        return ""

    if use_bullets:
        lines = [
            "REPORT OUTLINE (bullet layout — cover these topics in order; "
            "skip optional topics with no supporting Context):",
        ]
        for sec in sections:
            opt = " (optional)" if sec.optional else ""
            lines.append(f"- {sec.title}{opt}: {sec.guidance}")
        return "\n".join(lines)

    lines = [
        "REPORT OUTLINE (use these ## headings in order; "
        "skip optional sections with no supporting Context):",
    ]
    for sec in sections:
        opt = " [optional]" if sec.optional else ""
        lines.append(f"## {sec.title}{opt}")
        lines.append(f"- {sec.guidance}")
    return "\n".join(lines)


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
) -> GenerationPlan:
    """Deterministic post-retrieval strategy for generation."""
    effective_category, category_source = resolve_effective_category(
        category=category,
        plan_type=plan_type,
        query=query,
        decomposition=decomposition,
    )
    usable = filter_context_items(list(reranked_context or []))
    counts = _context_kind_counts(usable)
    has_bq = _has_usable_bq(usable)

    hit = measure_hit
    if hit is None and measure_id and measure_id in MEASURES:
        hit = MeasureHit(MEASURES[measure_id], score=100, matched_alias=measure_id)
    if hit is None:
        hit = resolve_measure(query, decomposition)

    mid = ""
    if hit:
        mid = hit.measure.id
    elif isinstance(retrieval_contract, RetrievalContract):
        mid = retrieval_contract.primary_measures[0] if retrieval_contract.primary_measures else ""
    elif isinstance(retrieval_contract, dict):
        pm = retrieval_contract.get("primary_measures")
        if isinstance(pm, list) and pm:
            mid = str(pm[0])

    ontology = _ontology_from_hit(hit, decomposition)

    if not usable:
        shape: AnswerShape = "gap_ack"
        priority: tuple[str, ...] = ()
        rationale = "gap_no_usable_context"
    else:
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
        shape = _refine_shape_from_query(query, decomposition, base)
        priority = _evidence_priority_for_measure(mid) if mid else _DEFAULT_NARRATIVE_PRIORITY
        priority = _apply_context_fingerprint(
            priority,
            counts,
            has_bq=has_bq,
            numeric_query=is_numeric_data_query(query, decomposition),
        )
        rationale = f"task_{task_mode or 'chat'}|measure_{mid or 'none'}|shape_{shape}"

    lead = _lead_with_for_shape(shape, has_bq)
    ground = _must_ground_in_for_shape(shape, has_bq)
    notes = _synthesis_notes(shape, has_bq=has_bq, ontology=ontology, priority=priority)
    deprior = _deprioritize_kinds(priority, counts)

    report_sections, use_bullets = build_analytical_report_outline(
        query=query,
        task_mode=task_mode,
        answer_shape=shape,
        ontology=ontology,
        has_bq=has_bq,
        decomposition=decomposition,
        category=effective_category,
    )

    return GenerationPlan(
        answer_shape=shape,
        evidence_priority=priority,
        lead_with=lead,
        must_ground_in=ground,
        ontology=ontology,
        synthesis_notes=notes,
        deprioritize_kinds=deprior,
        rationale=rationale,
        report_sections=report_sections,
        effective_category=effective_category or "",
        category_source=category_source,
        use_bullet_layout=use_bullets,
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
            )
        except (TypeError, ValueError):
            return ""
    else:
        plan_obj = plan

    parts: list[str] = [
        "GENERATION STRATEGY:",
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
