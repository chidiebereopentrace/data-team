"""Typed output geometry: one structure per job, Perplexity-style slots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ml.rag.chatbot.capability_registry import unsupported_answer_hint
from ml.rag.chatbot.turn_contract import NUMERIC_JOBS, TurnContract

OutputType = Literal[
    "fact",
    "trend",
    "compare",
    "list",
    "outlook",
    "diagnosis",
    "brief",
    "insufficient",
    "export",
]

EvidenceTier = Literal["strong", "partial", "empty"]

_JOB_TO_OUTPUT: dict[str, OutputType] = {
    "fact": "fact",
    "trend": "trend",
    "rank": "compare",
    "compare": "compare",
    "list": "list",
    "outlook": "outlook",
    "diagnose": "diagnosis",
    "brief": "brief",
    "clarify": "insufficient",
    "help": "insufficient",
    "social": "insufficient",
}

_SHAPE_TO_OUTPUT: dict[str, OutputType] = {
    "numeric_fact": "fact",
    "breakdown": "fact",
    "trend": "trend",
    "comparison": "compare",
    "ranking": "compare",
    "briefing_digest": "brief",
    "research_synthesis": "diagnosis",
    "policy_narrative": "diagnosis",
    "gap_ack": "insufficient",
    "export_table": "export",
}

_OUTPUT_TO_SHAPE: dict[OutputType, str] = {
    "fact": "numeric_fact",
    "trend": "trend",
    "compare": "comparison",
    "list": "ranking",
    "outlook": "trend",
    "diagnosis": "research_synthesis",
    "brief": "briefing_digest",
    "insufficient": "gap_ack",
    "export": "export_table",
}

_FORBIDDEN_DEFAULT_HEADINGS = frozenset(
    {
        "Key Findings",
        "Regional Picture",
        "Regional & Country Picture",
        "Drivers",
        "Drivers & Context",
        "Executive summary",
        "Regional overview",
        "Monitoring",
        "Historical context",
    }
)


@dataclass(frozen=True)
class OutputSlots:
    lead: str
    spine: str
    interpretation: str | None
    implications: str | None
    limits: str
    allow_headings: frozenset[str] = frozenset()


SLOT_TEMPLATES: dict[OutputType, OutputSlots] = {
    "fact": OutputSlots(
        lead="Sentence 1: VALUE + unit + place + time with [N] when citing.",
        spine="Optional one-line definition of what the number measures.",
        interpretation=None,
        implications=None,
        limits="One caveat line only if Context requires it; then Sources.",
        allow_headings=frozenset(),
    ),
    "trend": OutputSlots(
        lead="Sentence 1: direction + magnitude + period (start → end).",
        spine="Small table or numbered list: year/period × value (same unit throughout).",
        interpretation="2–4 lines on what changed (YoY or total %) only if in Context.",
        implications=None,
        limits="One definition/source note if series mix differs.",
        allow_headings=frozenset(),
    ),
    "compare": OutputSlots(
        lead="Sentence 1: who/what is higher (or ranked #1) on the metric.",
        spine="Table: entity × metric(s) × year — same series per column.",
        interpretation="Gap commentary only when Context supports it.",
        implications=None,
        limits="Do-not-mix-series note if units or years differ.",
        allow_headings=frozenset(),
    ),
    "list": OutputSlots(
        lead="Sentence 1: N places met threshold T in period (or cannot list at requested grain).",
        spine="Numbered list or table: place, anomaly/value, year, [N].",
        interpretation=None,
        implications=None,
        limits="If only national data exists, say so in the lead.",
        allow_headings=frozenset(),
    ),
    "outlook": OutputSlots(
        lead="One sentence on current food-security / market phase if in Context.",
        spine="Use ## Now and ## Next lean season (or next window) only when Context supports phases.",
        interpretation="Assumptions (rain, assistance) as bullets only from Context.",
        implications=None,
        limits="Watch items (2–3 bullets) from outlook sources only.",
        allow_headings=frozenset({"Now", "Next lean season", "Next window", "Watch items"}),
    ),
    "diagnosis": OutputSlots(
        lead="One-sentence claim grounded in Context.",
        spine="2–5 evidence bullets, each cited.",
        interpretation="Mechanisms only if explicitly in sources.",
        implications=None,
        limits="What would falsify this (1 line) if evidence is partial.",
        allow_headings=frozenset(),
    ),
    "brief": OutputSlots(
        lead="Decision-relevant sentence for the audience.",
        spine="Table or 3 bullets with the same figures as other jobs.",
        interpretation=None,
        implications="4–8 lines: implications for THIS audience only — no new numbers.",
        limits="Limits: 1–3 lines.",
        allow_headings=frozenset(),
    ),
    "insufficient": OutputSlots(
        lead="Cannot ground TYPE at GRAIN for PLACE × TIME — then what exists instead.",
        spine="Do not invent a table; state available grain or breakdown.",
        interpretation=None,
        implications=None,
        limits="How to rephrase the question (1–2 lines). No Sources unless truly cited.",
        allow_headings=frozenset(),
    ),
    "export": OutputSlots(
        lead="2–5 sentence caption summarizing the downloadable table/chart.",
        spine="Do not paste the full table in chat.",
        interpretation=None,
        implications=None,
        limits="Mention export artifact; no report headings.",
        allow_headings=frozenset(),
    ),
}


def output_type_from_job(job: str) -> OutputType:
    return _JOB_TO_OUTPUT.get((job or "").strip().lower(), "fact")


def output_type_from_answer_shape(shape: str) -> OutputType:
    return _SHAPE_TO_OUTPUT.get((shape or "").strip().lower(), "diagnosis")


def answer_shape_from_output_type(output_type: OutputType) -> str:
    return _OUTPUT_TO_SHAPE.get(output_type, "policy_narrative")


def output_type_from_plan(plan: dict[str, Any] | None) -> OutputType:
    if not isinstance(plan, dict):
        return "fact"
    raw = str(plan.get("output_type") or "").strip().lower()
    if raw in SLOT_TEMPLATES:
        return raw  # type: ignore[return-value]
    return output_type_from_answer_shape(str(plan.get("answer_shape") or ""))


def output_type_from_contract(
    contract: TurnContract | None,
    *,
    evidence_tier: EvidenceTier = "strong",
    has_usable_context: bool = True,
    task_mode: str = "chat",
) -> OutputType:
    if contract is None:
        mode = (task_mode or "chat").strip().lower()
        if mode == "data_export_only":
            return "export"
        if mode == "briefing":
            return "brief"
        if mode == "research":
            return "diagnosis"
        return "fact"
    if evidence_tier == "empty":
        return "insufficient"
    if (
        contract.is_fail_closed()
        and contract.serve_status != "clarify"
        and contract.job in NUMERIC_JOBS
    ):
        return "insufficient"
    if not has_usable_context and contract.job in NUMERIC_JOBS:
        return "insufficient"
    return output_type_from_job(contract.job)


_BREAKDOWN_SUBTOPIC_LABELS: dict[str, tuple[str, ...]] = {
    "sex": ("Male", "Female"),
}


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


def answer_subtopics(
    decomposition: dict[str, Any] | None,
    contract: TurnContract | None,
    output_type: OutputType,
) -> tuple[str, ...]:
    """Decomposition-driven ### subheads for the answer block (compare geo, fact breakdown)."""
    if output_type in ("outlook", "insufficient", "export", "diagnosis", "brief"):
        return ()

    geo: list[str] = []
    if contract and contract.geo:
        geo = [g for g in contract.geo if g]
    if not geo:
        geo = _geo_from_decomposition(decomposition)

    if output_type == "compare" and len(geo) >= 2:
        return tuple(geo[:6])

    if output_type == "fact" and contract and contract.breakdown:
        labels: list[str] = []
        for dim in contract.breakdown:
            preset = _BREAKDOWN_SUBTOPIC_LABELS.get(dim)
            if preset:
                labels.extend(preset)
            else:
                labels.append(dim.replace("_", " ").title())
        return tuple(dict.fromkeys(labels))

    return ()


_PERSONA_IMPLICATIONS: dict[str, str] = {
    "Government": (
        "4–8 lines on planning, fiscal, or monitoring implications — same spine numbers only; "
        "no new statistics."
    ),
    "NGOs": (
        "4–8 lines on program targeting or operational monitoring — same spine numbers only; "
        "no new statistics."
    ),
    "Agribusinesses": (
        "4–8 lines on market signal, sourcing, or exposure for commercial decisions — "
        "same spine numbers only; no new statistics."
    ),
    "Farmers": (
        "3–5 lines on what this means for farm decisions this season — same numbers, plain language."
    ),
}

_COMPARE_PERSONA_IMPLICATIONS: dict[str, str] = {
    "Government": (
        "4–8 lines on policy or fiscal implications from the comparison table — "
        "same figures only."
    ),
    "NGOs": (
        "4–8 lines on where program priorities differ across geographies — same table figures only."
    ),
    "Agribusinesses": (
        "4–8 lines on sourcing, basis-risk, or market exposure implied by the comparison table — "
        "same figures only; do not infer volatility from news alone."
    ),
    "Farmers": (
        "3–5 lines on what the contrast means locally — same table figures, plain language."
    ),
}

_INSUFFICIENT_REPHRASE: dict[str, str] = {
    "Government": (
        "Try national grain with a single year, or ask for IPC phase at country level."
    ),
    "NGOs": "Specify region or program geography; national-only figures may be available.",
    "Agribusinesses": (
        "Name countries and a calendar window; I won't rank volatility from news alone."
    ),
    "Farmers": "Name your district or country and crop for a localized answer.",
}


def persona_implications_block(
    category: str,
    output_type: OutputType,
    *,
    has_spine: bool,
) -> str | None:
    """Persona-wrapped implications after spine — same numbers, audience-specific framing."""
    if not has_spine or output_type == "insufficient":
        return None
    cat = (category or "").strip()
    if not cat:
        return None
    if output_type == "compare":
        return _COMPARE_PERSONA_IMPLICATIONS.get(cat) or _PERSONA_IMPLICATIONS.get(cat)
    if output_type == "brief":
        return _PERSONA_IMPLICATIONS.get(cat)
    return _PERSONA_IMPLICATIONS.get(cat)


def persona_insufficient_rephrase(category: str | None) -> str:
    cat = (category or "").strip()
    return _INSUFFICIENT_REPHRASE.get(cat, "")


def grain_window_line(contract: TurnContract | None, ontology: dict[str, Any] | None = None) -> str:
    """One-line scope: grain, time window, product."""
    ont = ontology if isinstance(ontology, dict) else {}
    parts: list[str] = []
    if contract and contract.geo_grain:
        parts.append(contract.geo_grain.replace("_", " "))
    elif ont.get("geography"):
        geo = ont.get("geography")
        if isinstance(geo, list) and geo:
            parts.append(", ".join(str(g) for g in geo[:3]))
    ts = contract.time_spec if contract else None
    start = str(ont.get("time_start") or (ts.start if ts else "") or "").strip()[:10]
    end = str(ont.get("time_end") or (ts.end if ts else "") or "").strip()[:10]
    if start and end and start != end:
        parts.append(f"calendar {start[:4]}–{end[:4]}")
    elif start:
        parts.append(f"calendar {start[:4]}")
    elif ts and ts.grain == "latest":
        parts.append("latest available")
    entities = contract.entities if contract else []
    if not entities and ont.get("measure_id"):
        mid = str(ont.get("measure_id") or "")
        if mid and mid not in ("socio_economic",):
            parts.append(mid.replace("_", " "))
    elif entities:
        parts.append(", ".join(entities[:2]))
    if contract and contract.measure_id:
        parts.insert(0, contract.measure_id.replace("_", " "))
    return "; ".join(dict.fromkeys(p for p in parts if p))


def format_prompt_for_type(
    output_type: OutputType,
    *,
    persona: str = "",
    grain_window_line: str = "",
    include_implications: bool = False,
    answer_subtopics: tuple[str, ...] = (),
    implications_text: str = "",
) -> str:
    slots = SLOT_TEMPLATES.get(output_type, SLOT_TEMPLATES["fact"])
    lines = [
        f"OUTPUT TYPE: {output_type}",
        "GEOMETRY (strict — no default report chapters):",
        f"- Lead: {slots.lead}",
    ]
    if answer_subtopics:
        labels = ", ".join(answer_subtopics)
        lines.extend(
            [
                "Answer block (after lead, before spine):",
                "- First sentence = overall answer when there is one.",
                f"- Optional ### subheads ONLY for: {labels}",
                "- No subhead without packed evidence for that facet.",
            ]
        )
    lines.append(f"- Spine: {slots.spine}")
    if grain_window_line:
        lines.append(f"- Scope line (after lead): {grain_window_line}")
    if slots.interpretation:
        lines.append(f"- Interpretation (optional): {slots.interpretation}")
    if implications_text and include_implications:
        lines.append(f"- Implications (after spine): {implications_text}")
    elif include_implications and persona and output_type != "insufficient":
        impl = persona_implications_block(persona, output_type, has_spine=True)
        if impl:
            lines.append(f"- Implications (after spine): {impl}")
    elif slots.implications and output_type == "brief" and not include_implications:
        lines.append("- Skip implications block (no persona selected).")
    lines.append(f"- Limits: {slots.limits}")
    allowed_heading_parts: list[str] = []
    if slots.allow_headings:
        allowed_heading_parts.extend(sorted(slots.allow_headings))
    if answer_subtopics:
        allowed_heading_parts.extend(f"### {s}" for s in answer_subtopics)
    if allowed_heading_parts:
        lines.append(f"- Allowed headings only: {', '.join(allowed_heading_parts)}")
    else:
        lines.append("- No ## headings unless allowed above.")
    lines.append("- Cite inline only sources you actually use.")
    return "\n".join(lines)


def render_insufficient(
    contract: TurnContract,
    *,
    query: str = "",
    academic_count: int = 0,
    job_label: str = "",
    category: str = "",
) -> str:
    """Deterministic insufficient answer — lead with miss, then alternatives."""
    job = job_label or contract.job or "answer"
    grain = contract.geo_grain or "requested grain"
    place = ", ".join(contract.geo[:2]) if contract.geo else "requested place"
    ts = contract.time_spec
    time_part = ""
    if ts.start and ts.end:
        time_part = f"{ts.start[:4]}–{ts.end[:4]}"
    elif ts.start:
        time_part = ts.start[:4]
    elif ts.grain == "latest":
        time_part = "latest"

    scope = grain_window_line(contract)
    lead_parts = [f"Cannot ground a {job} at {grain}"]
    if place:
        lead_parts.append(f"for {place}")
    if time_part:
        lead_parts.append(f"× {time_part}")
    lead = " ".join(lead_parts) + "."

    if contract.measure_id == "disease_prevalence" and contract.serve_status == "unsupported_grain":
        pathogen = contract.pathogen_id or "this pathogen"
        papers = (
            f" Searched {academic_count} academic/public report passages."
            if academic_count > 0
            else ""
        )
        body = (
            f"No national dairy-herd {pathogen} prevalence figure is in structured warehouse tables "
            f"(household×species grain only).{papers}"
        )
        return f"{lead} {body} Specify region, study year, or herd type for a literature-based answer."

    alt = unsupported_answer_hint(contract)
    if contract.serve_reason == "sex_breakdown_unavailable_total_only":
        alt = (
            "Total agricultural employment share may be available; "
            "sex breakdown is not in the warehouse."
        )

    lines = [lead]
    if alt:
        lines.append(alt)
    elif scope:
        lines.append(f"Available scope: {scope}.")
    else:
        lines.append(
            "Try national grain, a single year, or drop breakdown dimensions."
        )
    if contract.job in NUMERIC_JOBS and contract.serve_status == "served":
        lines.append(
            "No structured figure matched this question in Context for the requested scope."
        )
    rephrase = persona_insufficient_rephrase(category)
    if rephrase:
        lines.append(rephrase)
    return " ".join(lines)


def forbidden_headings_for_type(
    output_type: OutputType,
    *,
    allowed_subtopics: tuple[str, ...] = (),
) -> frozenset[str]:
    slots = SLOT_TEMPLATES.get(output_type, SLOT_TEMPLATES["fact"])
    allowed = {h.lower() for h in slots.allow_headings}
    allowed |= {s.lower() for s in allowed_subtopics}
    return frozenset(h for h in _FORBIDDEN_DEFAULT_HEADINGS if h.lower() not in allowed)


def export_caption_instruction() -> str:
    return format_prompt_for_type("export")
