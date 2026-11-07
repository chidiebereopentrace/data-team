"""System and user prompt for answer generation."""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from ml.rag.chatbot.answer_language import (
    detect_answer_language,
    is_english_answer_lang,
    language_instruction,
)
from ml.rag.chatbot.answer_outcome import DOCUMENTS_ONLY_INSTRUCTION, OutcomeKind
from ml.rag.chatbot.output_format import (
    export_caption_instruction,
    format_prompt_for_type,
    output_type_from_plan,
    persona_implications_block,
)
from ml.rag.chatbot.plan_policy import (
    instruction_for_category,
    plan_generation_addendum,
)
from ml.rag.chatbot.stakeholder_prompts import prose_register_addendum
from ml.rag.chatbot.time_resolver import utc_today

EvidenceTier = Literal["strong", "partial", "empty"]

_EXPORT_CAPTION_BLOCK = (
    "\n\nARTIFACT EXPORT MODE: "
    + export_caption_instruction().replace("\n", " ")
)

_SLIM_BASE_PROMPT = (
    "You are Ask ADZA, OpenTrace Africa's agricultural intelligence assistant. "
    "Lead with the direct answer. Be precise and compact. "
    "Use only evidence in Context from numbered [Source N] chunks. "
    "Never invent statistics, rankings, or dates. "
    "Never mention warehouses, datasets, table IDs, SQL, vector DBs, or pipelines. "
    "Attribute structured figures to public sources (FAOSTAT, FEWS NET, ILRI, WFP, etc.) "
    "or the neutral label OpenTrace agricultural data (country=…, year=…, product=…). "
    "Put a blank line before every ## heading. "
    "If evidence is insufficient, say so in 2–4 sentences and stop — no report skeleton. "
    "Never open with Unfortunately, Based on the context, or The context provided."
)

def _current_date_block(reference_date: date | None = None) -> str:
    """
    ML-057 (Sprint 2 Week 2): tell the generator what today's date is.

    The query decomposer already resolves time ranges against ``date.today()``
    (see ``decompose_context.today_iso``), but the generation prompt never
    carried a date -- so the LLM fell back on its training cut-off and could
    not tell an upcoming event from one in progress or already past.

    Built at call time rather than baked into ``_SLIM_BASE_PROMPT`` because a
    module-level constant would freeze the date at import and go stale on a
    long-running Railway process.
    """
    today = reference_date or utc_today()
    return (
        f"\n\nCURRENT DATE: Today is {today.isoformat()} (year {today.year}). "
        f"Ground every time-sensitive statement in this date: an event before "
        f"it has already happened, an event after it has not happened yet. "
        f"Do not describe {today.year - 1} or earlier figures as current -- "
        f"state the year the figure belongs to. "
        f"Never infer the present year from your training data."
    )


_EVIDENCE_INTERPRETATION_RULES = (
    "\n\nEVIDENCE INTERPRETATION: Distinguish formal policies, laws, and strategies from "
    "programmes, projects, and interventions; do not describe a programme as a policy or "
    "assume a policy was implemented unless Context supports that claim. When GEDSI or "
    "inclusion is requested or materially relevant, consider gender, equity, disability, "
    "youth, and broader social inclusion. Cover only dimensions supported by Context and "
    "state evidence gaps rather than assume impacts. Treat Wikipedia as a tertiary, "
    "community-edited source; attribute it clearly and do not present it as equivalent to "
    "peer-reviewed research, official statistics, or primary policy documents. For numeric "
    "claims, use only units explicitly supported by Context; if the unit is missing, say so. "
    "Respect the requested geographic level: when only a broader level is available, name "
    "that level and the limitation instead of presenting it as local data. State the year of "
    "historical figures and do not describe them as current; use the current date supplied in "
    "Context when interpreting time-sensitive events."
)

_MARKET_PRICE_EVIDENCE_RULES = (
    "\n\nMARKET-PRICE QUESTIONS: When Context contains a structured price figure, state the "
    "value with its provided currency/unit, product, geography or market, and recorded date. "
    "Describe it as the latest available observation, not a live or current price, unless the "
    "source explicitly establishes that it is current. If the date basis is loaded_at, do not "
    "present it as the market observation date. If Context has no price figure, say that no "
    "price figure is available; do not invent one. Do not give a numeric price forecast or "
    "quantified price movement unless Context contains that forecast or movement. "
    "Keep each product name, unit, and currency exactly as the rows label them; never average "
    "across different units or currencies. Never call one price higher, lower, cheaper, or more "
    "expensive than another when their currencies or units differ; report each in its own "
    "currency and say they are not directly comparable without an exchange rate. Label the "
    "period from the row's period_start and latest_date, not from the year asked. If a row "
    "carries a coverage note, repeat it. "
    "Close with one line on what the evidence means for the market: price direction or level, "
    "any demand signal in Context, and the implication for what to grow, stock, or sell, "
    "citing the evidence it rests on."
)

_PARTIAL_EVIDENCE_RULES = (
    "\n\nPARTIAL EVIDENCE: Only state claims clearly supported by Context. "
    "One short closing line on years or coverage limits is enough. "
    "No ## Key Findings or multi-section report headings."
)

_YIELD_ONLY_RULE = (
    "\n\nYIELD DATA ONLY: Context rows are crop yields (t/ha), not production tonnage. "
    "Do not describe a single yield observation as a multi-year production trend. "
    "State yield for the specific year shown."
)

def _build_prompt(
    query: str,
    context_block: str,
    decomposition: dict[str, Any] | None = None,
    memory_block: str = "",
    category: str = "",
    plan_type: str = "",
    answer_lang: str | None = None,
    outcome: OutcomeKind = "rows",
    structured_bq_numeric_available: bool = False,
    structured_bq_comparative_available: bool = False,
    analytical_mode: bool = False,
    task_mode: str = "chat",
    measure_id: str | None = None,
    recency_tier: str | None = None,
    context_source_kinds: list[str] | None = None,
    inline_citations: bool = False,
    generation_plan: dict[str, Any] | None = None,
    export_intent: str | None = None,
    evidence_tier: EvidenceTier = "strong",
    yield_only: bool = False,
    composer_addendum: str = "",
    reference_date: date | None = None,
) -> list[dict[str, str]]:
    lang = (answer_lang or "").strip() or detect_answer_language(query)
    mode = (task_mode or ("analytical" if analytical_mode else "chat")).strip().lower()
    if analytical_mode and mode == "chat":
        mode = "analytical"

    system = _SLIM_BASE_PROMPT + language_instruction(lang, inline_citations=inline_citations)
    # ML-057: date grounding goes directly after the base prompt so every
    # downstream addendum is read in the context of the correct year.
    system += _current_date_block(reference_date)
    system += _EVIDENCE_INTERPRETATION_RULES
    if re.search(r"\b(?:prices?|costs?|expensive|cheap(?:er|est)?|volatile|volatility)\b", query or "", re.IGNORECASE):
        system += _MARKET_PRICE_EVIDENCE_RULES

    if evidence_tier == "partial":
        system += _PARTIAL_EVIDENCE_RULES
    elif evidence_tier == "empty":
        system += (
            "\n\nCRITICAL: Context is empty. Reply in 2–4 sentences that you cannot answer "
            "from the sources provided. Do not use report headings or invent facts."
        )

    if inline_citations:
        system += (
            "\n\nCITATION HYGIENE: Use plain [N] only — never [Source N], [Source 1, 3], "
            "or [Wikipedia | …]. Do not repeat the same [N] twice in a row. "
            "Blank line before every ## heading."
        )
    else:
        system += (
            "\n\nDo NOT insert [N] or [Source N] footnote markers — structured citations "
            "are returned separately for the client."
        )

    effective_category = (category or "").strip()
    answer_shape = ""
    use_bullet_layout = False
    if generation_plan and isinstance(generation_plan, dict):
        ec = str(generation_plan.get("effective_category") or "").strip()
        if ec:
            effective_category = ec
        answer_shape = str(generation_plan.get("answer_shape") or "")
        use_bullet_layout = bool(generation_plan.get("use_bullet_layout"))

    register_block = prose_register_addendum(
        effective_category or None,
        task_mode=mode,
        answer_shape=answer_shape,
        inline_citations=inline_citations,
    )
    if register_block:
        system += "\n\n" + register_block

    cat_tone = instruction_for_category(
        effective_category or None,
        measure_id=measure_id,
        recency_tier=recency_tier,
    ) if (effective_category or measure_id or recency_tier) else ""
    if cat_tone:
        system += "\n\n" + cat_tone

    plan_addendum = plan_generation_addendum(plan_type) if plan_type else ""
    if plan_addendum:
        system += "\n\n" + plan_addendum

    if generation_plan:
        # generation_plan imports generator, which imports this module.
        from ml.rag.chatbot.generation_plan import generation_plan_addendum

        gen_plan_addendum = generation_plan_addendum(generation_plan)
        if gen_plan_addendum:
            system += "\n\n" + gen_plan_addendum

    if composer_addendum:
        system += "\n\n" + composer_addendum

    # Same cycle as generation_plan_addendum above.
    from ml.rag.chatbot.generation_plan import format_template_for_shape

    if mode == "fact_lookup":
        system += (
            "\n\nFACT LOOKUP: First sentence = what was measured (product or indicator and place) "
            "+ number/fact + unit + period + [N] when citing. "
            "No report headings."
        )
    elif mode == "briefing":
        system += (
            "\n\nBRIEFING: 3–6 bullet points from the most recent narrative sources; "
            "one closing line on limits."
        )
    elif mode == "research":
        system += (
            "\n\nRESEARCH: Synthesize academic/policy evidence; do not invent citations."
        )
    elif mode == "data_export_only":
        system += _EXPORT_CAPTION_BLOCK.replace("ARTIFACT EXPORT MODE", "DATA EXPORT MODE")
    elif answer_shape and evidence_tier != "empty":
        out_type = output_type_from_plan(generation_plan)
        gw = str((generation_plan or {}).get("grain_window_line") or "").strip()
        answer_subtopics = tuple((generation_plan or {}).get("answer_subtopics") or ())
        has_spine = bool((generation_plan or {}).get("has_usable_spine"))
        persona = effective_category
        plan_type_s = str((generation_plan or {}).get("plan_type") or "").strip()
        include_impl = bool(
            (persona or plan_type_s)
            and evidence_tier != "empty"
            and has_spine
            and out_type != "insufficient"
        )
        impl_text = ""
        if include_impl and persona:
            impl = persona_implications_block(persona, out_type, has_spine=has_spine)
            if impl:
                impl_text = impl
        type_fmt = format_prompt_for_type(
            out_type,
            persona=persona,
            grain_window_line=gw,
            include_implications=include_impl and bool(impl_text),
            answer_subtopics=answer_subtopics,
            implications_text=impl_text,
        )
        if type_fmt:
            system += "\n\n" + type_fmt
        else:
            shape_fmt = format_template_for_shape(answer_shape, use_bullets=use_bullet_layout)
            if shape_fmt:
                system += "\n\n" + shape_fmt

    if export_intent and mode != "data_export_only":
        system += _EXPORT_CAPTION_BLOCK

    if yield_only:
        system += _YIELD_ONLY_RULE

    if effective_category and cat_tone and not is_english_answer_lang(lang) and lang not in ("unknown", ""):
        system += (
            "\n\nAnswer in the user's language while keeping the category audience rules."
        )

    kinds = [str(k).strip().lower() for k in (context_source_kinds or []) if str(k).strip()]
    unique_kinds = list(dict.fromkeys(kinds))
    if evidence_tier == "strong" and len(unique_kinds) >= 2:
        system += (
            "\n\nCross-source synthesis: weave structured figures with narrative sources "
            "when both are relevant; do not invent numbers."
        )

    if structured_bq_numeric_available:
        system = (
            "CRITICAL: Context includes authoritative OpenTrace figures with units. "
            "Use them for numeric answers; do not override with conflicting narrative.\n\n"
        ) + system
    elif structured_bq_comparative_available:
        system = (
            "Context includes OpenTrace structured data — use it for comparisons; "
            "use narrative sources for drivers only.\n\n"
        ) + system
    elif outcome == "documents_only":
        system = DOCUMENTS_ONLY_INSTRUCTION + "\n\n" + system

    mb = (memory_block.strip() + "\n\n") if memory_block.strip() else ""
    user = f"{mb}Context:\n{context_block}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
