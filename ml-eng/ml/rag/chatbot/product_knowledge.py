"""
Product knowledge layer for OpenTrace / Ask ADZA questions.

Loads canonical product copy from opentrace_product.json and routes queries that
ask about OpenTrace's mission, pillars, trust model, etc. without retrieval.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from ml.rag.chatbot.assistant_identity import META_ANSWER_FOOTER, _append_footer
from ml.rag.chatbot.answer_language import (
    detect_answer_language,
    is_english_answer_lang,
    language_instruction,
)

_DEFAULT_KB_PATH = Path(__file__).resolve().parent / "data" / "opentrace_product.json"

# Brand / product tokens (applied after _normalize_for_product_gate)
_BRAND_PATTERNS: tuple[str, ...] = (
    r"\bopentrace\b",
    r"\bask adza\b",
    r"\baskadza\b",
    r"\badza\b",
    r"\bofia\b",
    r"\bacf\b",
    r"\badza confidence framework\b",
    r"\bconfidence framework\b",
    r"\bfederated intelligence\b",
)

# Product-intent phrases (no brand required if combined with brand elsewhere)
_PRODUCT_INTENT_PATTERNS: tuple[str, ...] = (
    r"\b(?:aim|purpose|mission|goal)s?\b.*\b(?:opentrace|adza|ofia|acf)\b",
    r"\b(?:opentrace|adza|ofia|acf)\b.*\b(?:aim|purpose|mission|goal)s?\b",
    r"\bwhat(?:'s| is) (?:the )?(?:aim|purpose|mission|goal)\b",
    r"\bwhy partner\b",
    r"\bhow (?:do you|does opentrace) work\b",
    r"\bwho is (?:this|it) for\b",
    r"\b(?:tell me|explain).*(?:about )?opentrace\b",
    r"\bwhat is opentrace\b",
    r"\bwhat is ask adza\b",
    r"\bwho is ask adza\b",
    r"\bwhat is ofia\b",
    r"\bwhat is acf\b",
    r"\bexplain.*confidence framework\b",
    r"\bexplain.*pillars?\b",
    r"\b(?:moat|traction|sovereignty)\b",
    r"\binfrastructure company\b",
    # Localized product intents
    r"\bc['']est\s+quoi\s+opentrace\b",
    r"\bqu['']est[- ]ce\s+que\s+opentrace\b",
    r"\bmission\s+d['']opentrace\b",
    r"\bopentrace\s+ni\s+nini\b",
    r"\blengo\s+la\s+opentrace\b",
    r"\bmenene\s+opentrace\b",
    r"\bo\s+que\s+[ée]\s+opentrace\b",
    r"\bo\s+que\s+e\s+opentrace\b",
    r"\bwetin\s+be\s+opentrace\b",
)

# Capability / help phrasing — assistant use, onboarding, question menu
_CAPABILITY_PATTERNS: tuple[str, ...] = (
    r"\bwhat(?:'s| is| are) your (?:use|purpose|role|function)\b",
    r"\bwhat can i use (?:you|ask adza|adza|opentrace) for\b",
    r"\bhow (?:can|do) i use (?:you|ask adza|adza|opentrace)\b",
    r"\bwhat (?:can|do) you (?:do|help with|answer)\b",
    r"\bwhat can ask adza help(?: me)? with\b",
    r"\bwhat questions can i ask\b",
    r"\bwhat kind of questions?\b",
    r"\bwhat kinds of questions?\b",
    r"\bwhat type of questions?\b",
    r"\bwhat data do you (?:have|cover)\b",
    r"\bwhat indicators? do you cover\b",
    r"\bhelp me (?:get started|use) (?:ask adza|opentrace|you)\b",
    r"\bwhat are you (?:good|useful) for\b",
    r"\bhow does (?:ask adza|this|opentrace) work\b",
    r"\bhow (?:can|do) i (?:get started|start)\b.*\b(?:you|ask adza|opentrace)\b",
    r"\bwhat (?:can|should) i ask (?:you|ask adza)\b",
    # French
    r"\bà quoi (?:tu\s+)?sers\b",
    r"\bà quoi sert ask adza\b",
    r"\bcomment (?:t'|te )?utiliser\b",
    # Swahili
    r"\bnaweza kutumia ask adza vipi\b",
    r"\bask adza inafanya nini\b",
    # Nigerian Pidgin
    r"\bwetin i fit use you for\b",
    r"\bwetin ask adza dey do\b",
)

_ASSISTANT_REF_RE = re.compile(r"\b(?:you|your)\b", re.IGNORECASE)
_METHODOLOGY_INDICATORS: frozenset[str] = frozenset({
    "ipc", "ndvi", "fews", "gdd", "evi", "spi", "vci", "chirps", "era5",
})
_METHODOLOGY_WORK_RE = re.compile(r"\bhow does (\w+) work\b", re.IGNORECASE)

_BRAND_RE = re.compile("|".join(_BRAND_PATTERNS), re.IGNORECASE)
_PRODUCT_INTENT_RE = re.compile("|".join(_PRODUCT_INTENT_PATTERNS), re.IGNORECASE)
_CAPABILITY_RE = re.compile("|".join(_CAPABILITY_PATTERNS), re.IGNORECASE)

CAPABILITY_STATIC_ANSWER = (
    "Ask ADZA is OpenTrace Africa's natural-language interface for African agricultural "
    "intelligence. You can ask about crops, markets, climate, food security, policy and "
    "trade impacts, and related topics across Africa. Answers are grounded in OpenTrace "
    "structured data and curated evidence when available, with transparency about "
    "confidence and limits.\n\n"
    "Example questions:\n"
    "- What were maize yields in Kenya in 2020?\n"
    "- How have rice prices trended in West Africa recently?"
)

CAPABILITY_TEMPLATES: dict[str, str] = {
    "en": CAPABILITY_STATIC_ANSWER,
    "fr": (
        "Ask ADZA est l'interface en langage naturel d'OpenTrace Africa pour "
        "l'intelligence agricole africaine. Vous pouvez poser des questions sur les "
        "cultures, les marchés, le climat, la sécurité alimentaire, les politiques et "
        "les impacts commerciaux en Afrique. Les réponses s'appuient sur les données "
        "structurées OpenTrace et des preuves sélectionnées lorsque disponibles.\n\n"
        "Exemples :\n"
        "- Quels étaient les rendements de maïs au Kenya en 2020 ?\n"
        "- Comment évoluent les prix du riz en Afrique de l'Ouest ?"
    ),
    "sw": (
        "Ask ADZA ni kiolesura cha lugha ya asili cha OpenTrace Africa kwa akili ya "
        "kilimo cha Afrika. Unaweza kuuliza kuhusu mazao, masoko, hali ya hewa, "
        "usalama wa chakula, sera na athari za biashara barani Afrika. Majibu "
        "yanategemea data iliyopangwa ya OpenTrace na ushahidi uliochaguliwa inapopatikana.\n\n"
        "Mifano:\n"
        "- Mazao ya mahindi Kenya mwaka 2020 yalikuwa nini?\n"
        "- Bei za mchele Magharibi mwa Afrika zimebadilika vipi hivi karibuni?"
    ),
    "pcm": (
        "Ask ADZA na OpenTrace Africa natural-language interface for African "
        "agricultural intelligence. You fit ask about crops, market, climate, food "
        "security, policy and trade impact for Africa. Answers dey grounded for "
        "OpenTrace structured data and curated evidence when e dey available.\n\n"
        "Example questions:\n"
        "- Wetin be maize yield for Kenya for 2020?\n"
        "- How rice price dey trend for West Africa recently?"
    ),
}

# Agricultural data signals — if present with geography, force RAG
_AG_ENTITY_TOKENS: frozenset[str] = frozenset({
    "maize", "rice", "wheat", "sorghum", "millet", "cassava", "yam", "cocoa", "coffee",
    "tea", "cotton", "groundnut", "soybean", "bean", "cowpea", "livestock", "cattle",
    "poultry", "fish", "crop", "harvest", "planting", "irrigation", "drought", "flood",
    "rainfall", "rain", "yield", "production", "productivity", "market price", "food price",
    "food security", "fertilizer", "pesticide", "livestock", "pasture", "rangeland",
})

_AG_DATA_VERBS: tuple[str, ...] = (
    r"\bdata on\b",
    r"\bshow me\b",
    r"\btrend(?:s)? (?:of|in|for)\b",
    r"\byield(?:s)? (?:in|for|of)\b",
    r"\bproduction (?:in|for|of)\b",
    r"\bcompare\b",
    r"\bhow (?:have|has|did)\b.*\b(?:changed|trended|varied)\b",
)

_AG_DATA_VERB_RE = re.compile("|".join(_AG_DATA_VERBS), re.IGNORECASE)

PRODUCT_SYSTEM_PROMPT = (
    "You are Ask ADZA, the natural-language interface for OpenTrace Africa. "
    "Answer ONLY using facts in the Product Knowledge provided below. "
    "Write clear, direct prose for decision-makers — not a meta-commentary about sources. "
    "Never open with phrases like 'The context provided...', 'Unfortunately...', or 'Based on the knowledge...'. "
    "Lead with the answer. If the Product Knowledge does not cover a detail, say so briefly after your answer. "
    "Do not invent statistics, partnerships, or capabilities not stated in the Product Knowledge. "
    "Do not append a Citations block. Keep answers concise and chat-UI ready."
)


def _kb_path() -> Path:
    override = os.environ.get("RAG_PRODUCT_KB_PATH", "").strip()
    if override:
        return Path(override)
    return _DEFAULT_KB_PATH


@lru_cache(maxsize=1)
def load_product_kb() -> dict[str, Any]:
    """Load and cache the product knowledge JSON."""
    path = _kb_path()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Product KB must be a JSON object: {path}")
    return data


def format_product_kb_for_prompt(kb: dict[str, Any] | None = None) -> str:
    """Format product KB as compact sectioned text for LLM context."""
    kb = kb or load_product_kb()
    sections: list[str] = []

    tagline = kb.get("tagline") or ""
    subtitle = kb.get("subtitle") or ""
    mission = kb.get("mission") or ""
    aim = kb.get("aim") or ""
    if tagline or mission or aim:
        block = f"## OpenTrace Africa\n{tagline}"
        if subtitle:
            block += f"\n{subtitle}"
        if aim:
            block += f"\nAim: {aim}"
        if mission:
            block += f"\n{mission}"
        sections.append(block)

    current = kb.get("current_state") or {}
    if isinstance(current, dict) and current.get("summary"):
        sections.append(f"## Current state\n{current.get('summary', '')}")

    core = kb.get("core_problem") or {}
    if isinstance(core, dict) and core.get("headline"):
        sections.append(f"## Core problem\n{core.get('headline')}\n{core.get('summary', '')}")

    opportunity = kb.get("opportunity") or {}
    if isinstance(opportunity, dict) and opportunity.get("headline"):
        sections.append(
            f"## The opportunity\n{opportunity.get('headline')}\n{opportunity.get('summary', '')}"
        )

    challenge = kb.get("challenge") or {}
    if isinstance(challenge, dict) and challenge.get("headline"):
        sections.append(
            f"## The challenge\n{challenge.get('headline')}\n{challenge.get('summary', '')}"
        )

    why = kb.get("why_it_persists") or {}
    if isinstance(why, dict):
        costs = why.get("systemic_costs") or []
        if costs:
            lines = [f"- {c.get('name', '')}: {c.get('description', '')}" for c in costs if isinstance(c, dict)]
            block = f"## Why it persists\n{why.get('headline', '')}\n{why.get('summary', '')}\n" + "\n".join(lines)
            without = why.get("without_reliable_intelligence") or []
            if without:
                block += "\nWithout reliable intelligence: " + "; ".join(str(x) for x in without)
            sections.append(block)

    gap = kb.get("agricultural_data_gap") or {}
    if isinstance(gap, dict) and gap.get("headline"):
        sections.append(f"## Agricultural data gap\n{gap.get('headline')}\n{gap.get('summary', '')}")

    connects = kb.get("what_opentrace_connects") or {}
    if isinstance(connects, dict):
        domains = connects.get("data_domains") or []
        if domains:
            sections.append(
                "## What OpenTrace connects\n"
                + connects.get("summary", "")
                + "\n"
                + ", ".join(str(d) for d in domains)
            )

    trusted = kb.get("trusted_intelligence") or {}
    if isinstance(trusted, dict):
        principles = trusted.get("principles") or []
        if principles:
            sections.append(
                "## Trusted intelligence\n" + "\n".join(f"- {p}" for p in principles)
            )

    pillars = kb.get("pillars") or {}
    if isinstance(pillars, dict):
        for key in ("ask_adza", "acf", "ofia"):
            p = pillars.get(key)
            if not isinstance(p, dict):
                continue
            name = p.get("name") or key
            role = p.get("role") or ""
            desc = p.get("description") or ""
            block = f"### {name} ({role})\n{desc}"
            does = p.get("does")
            if isinstance(does, list) and does:
                block += "\nDoes: " + "; ".join(str(x) for x in does)
            does_not = p.get("does_not")
            if isinstance(does_not, list) and does_not:
                block += "\nDoes not: " + "; ".join(str(x) for x in does_not)
            sections.append(block)

    capabilities = kb.get("capabilities") or {}
    if isinstance(capabilities, dict):
        for cap_key in ("data_reconstruction", "system_intelligence"):
            cap = capabilities.get(cap_key)
            if not isinstance(cap, dict):
                continue
            name = cap.get("name") or cap_key
            block = f"### {name}\n{cap.get('summary', '')}"
            does = cap.get("does") or cap.get("domains") or []
            if does:
                block += "\n" + "\n".join(f"- {d}" for d in does)
            examples = cap.get("examples") or []
            if examples:
                block += "\nExamples: " + "; ".join(str(e) for e in examples)
            sections.append(block)

    tiers = kb.get("acf_tiers") or []
    if tiers:
        tier_lines = [
            f"- Tier {t.get('tier')}: {t.get('weight_pct')}% {t.get('label')} — {t.get('examples', '')}"
            for t in tiers
            if isinstance(t, dict)
        ]
        sections.append("## ACF evidence tiers\n" + "\n".join(tier_lines))

    fed = kb.get("federation") or {}
    if isinstance(fed, dict):
        vert = ", ".join(str(x) for x in (fed.get("vertical_levels") or []))
        horiz = ", ".join(str(x) for x in (fed.get("horizontal_sectors") or []))
        sections.append(f"## Federation\nVertical: {vert}\nHorizontal: {horiz}\n{fed.get('summary', '')}")

    trust = kb.get("trust_and_sovereignty") or {}
    if isinstance(trust, dict):
        commits = trust.get("commitments") or []
        never = trust.get("never_do") or []
        block = f"## Trust & sovereignty\n{trust.get('summary', '')}"
        if commits:
            block += "\nCommitments: " + "; ".join(str(x) for x in commits)
        if never:
            block += "\nNever: " + "; ".join(str(x) for x in never)
        if trust.get("monetization"):
            block += f"\n{trust['monetization']}"
        sections.append(block)

    how = kb.get("how_we_work") or {}
    if isinstance(how, dict):
        steps = how.get("steps") or []
        step_lines = [
            f"{s.get('step')}. {s.get('name')}: {s.get('description', '')}"
            for s in steps
            if isinstance(s, dict)
        ]
        block = f"## How we work\n{how.get('summary', '')}\n" + "\n".join(step_lines)
        if how.get("pricing_model"):
            block += f"\n{how['pricing_model']}"
        sections.append(block)

    audiences = kb.get("audiences") or []
    if audiences:
        aud_lines = [
            f"- {a.get('segment', '')}: {a.get('value', '')}"
            for a in audiences
            if isinstance(a, dict)
        ]
        sections.append("## Who this is for\n" + "\n".join(aud_lines))

    details = kb.get("stakeholder_details") or {}
    if isinstance(details, dict) and details:
        detail_blocks: list[str] = []
        for _key, detail in details.items():
            if not isinstance(detail, dict):
                continue
            pts = detail.get("points") or []
            if not pts:
                continue
            detail_blocks.append(
                f"### {detail.get('headline', _key)}\n"
                + "\n".join(f"- {p}" for p in pts)
            )
        if detail_blocks:
            sections.append("## Stakeholder use cases\n" + "\n\n".join(detail_blocks))

    biz = kb.get("business_model") or {}
    if isinstance(biz, dict):
        streams = biz.get("revenue_streams") or []
        if streams:
            sections.append(
                f"## Business model\n{biz.get('summary', '')}\n"
                + "\n".join(f"- {s}" for s in streams)
            )

    traction = kb.get("traction") or {}
    if isinstance(traction, dict):
        stats = traction.get("stats") or {}
        if stats:
            stat_str = ", ".join(f"{k}: {v}" for k, v in stats.items())
            sections.append(f"## Traction\n{stat_str}\n{traction.get('partnerships', '')}")

    growth = kb.get("growth") or {}
    if isinstance(growth, dict):
        markets = ", ".join(str(x) for x in (growth.get("priority_markets") or []))
        phases = growth.get("phases") or []
        phase_lines = [
            f"Phase {p.get('phase')}: {p.get('name')} — {p.get('description', '')}"
            for p in phases
            if isinstance(p, dict)
        ]
        sections.append(f"## Growth\n{growth.get('strategy', '')}\nPriority markets: {markets}\n" + "\n".join(phase_lines))

    why_partner = kb.get("why_partner") or []
    if why_partner:
        wp_lines = [
            f"- {w.get('title', '')}: {w.get('description', '')}"
            for w in why_partner
            if isinstance(w, dict)
        ]
        sections.append("## Why partner\n" + "\n".join(wp_lines))

    ecosystem = kb.get("partnership_ecosystem") or {}
    if isinstance(ecosystem, dict):
        types = ecosystem.get("partner_types") or []
        if types:
            sections.append(
                f"## Partnership ecosystem\n{ecosystem.get('summary', '')}\n"
                + "\n".join(f"- {t}" for t in types)
            )

    advantage = kb.get("competitive_advantage") or {}
    if isinstance(advantage, dict):
        factors = advantage.get("factors") or []
        if factors:
            sections.append(
                f"## Competitive advantage\n{advantage.get('summary', '')}\n"
                + "\n".join(f"- {f}" for f in factors)
            )

    vision = kb.get("vision") or {}
    if isinstance(vision, dict) and vision.get("headline"):
        sections.append(f"## Vision\n{vision.get('headline')}\n{vision.get('summary', '')}")

    contact = kb.get("contact") or {}
    if isinstance(contact, dict):
        sites = [
            contact.get("website", ""),
            contact.get("product_website", ""),
            contact.get("product_website_alt", ""),
        ]
        site_str = " | ".join(s for s in sites if s)
        sections.append(f"## Contact\n{site_str} | Email: {contact.get('email', '')}")

    text = "\n\n".join(s for s in sections if s.strip())
    cap = int(os.environ.get("RAG_PRODUCT_KB_PROMPT_MAX_CHARS", "24000") or 24000)
    if len(text) > cap:
        return text[:cap] + "\n\n[Product knowledge truncated for context budget.]"
    return text


def _has_ag_data_intent(query: str) -> bool:
    """Return True if query asks for agricultural data, not product info."""
    q = query.lower()
    if _AG_DATA_VERB_RE.search(q):
        return True
    tokens = set(re.findall(r"[a-z]{3,}", q))
    if tokens & _AG_ENTITY_TOKENS:
        return True
    return False


def _has_geography(decomposition: dict[str, Any] | None) -> bool:
    if not decomposition:
        return False
    geo = decomposition.get("geography")
    return isinstance(geo, list) and len(geo) > 0


def _has_ag_entities_in_decomposition(decomposition: dict[str, Any] | None) -> bool:
    if not decomposition:
        return False
    entities = decomposition.get("entities") or []
    if not isinstance(entities, list):
        return False
    for ent in entities:
        ent_l = str(ent).lower()
        if any(tok in ent_l for tok in _AG_ENTITY_TOKENS):
            return True
    return False


def _normalize_for_product_gate(query: str) -> str:
    """Normalize query text for brand/capability pattern matching."""
    text = unicodedata.normalize("NFC", (query or "").strip())
    text = re.sub(r"\bwuestions\b", "questions", text, flags=re.IGNORECASE)
    text = re.sub(r"\bask\s*adza\b", "ask adza", text, flags=re.IGNORECASE)
    text = re.sub(r"\baskadza\b", "ask adza", text, flags=re.IGNORECASE)
    return text.casefold()


def render_internal_indicator_catalog(*, max_classes: int = 8) -> str:
    """Internal/debug catalog with table names and do-not-mix notes (not for public users)."""
    from ml.rag.chatbot.ontology_context import list_indicator_class_contexts

    lines = [
        "Ontology indicator catalog (internal):",
        "",
    ]
    for ctx in list_indicator_class_contexts(max_classes=max_classes):
        facts = ", ".join(ctx.primary_facts[:3]) or "see mart index"
        lines.append(f"{ctx.code} — {ctx.name}")
        lines.append(f"Primary tables: {facts}")
        for claim in ctx.example_claims[:2]:
            lines.append(f"- Example: “{claim}”")
        if ctx.do_not_mix_notes:
            lines.append(f"- Note: {ctx.do_not_mix_notes[0][:120]}")
        lines.append("")
    return "\n".join(lines)


def render_public_capability_answer() -> str:
    """User-facing help: topic sections with plain-language example questions."""
    from ml.rag.chatbot.ontology_context import list_public_capability_contexts

    lines = [
        "Ask ADZA is OpenTrace Africa's natural-language interface for agricultural "
        "intelligence across Africa. Ask in everyday language—we find the right data "
        "behind the scenes.",
        "",
        "You can ask about topics like:",
        "",
    ]
    for ctx in list_public_capability_contexts():
        lines.append(ctx.name)
        for question in ctx.public_examples[:2]:
            lines.append(f"- {question}")
        lines.append("")
    lines.append(
        "Tip: include a country or region, the crop or topic when relevant, "
        "and a year or time period for the clearest answers."
    )
    return "\n".join(lines)


def render_indicator_catalog_answer(*, max_classes: int = 8) -> str:
    """Deprecated alias — use render_public_capability_answer for user help."""
    _ = max_classes
    return render_public_capability_answer()


def _is_methodology_question(normalized: str, raw_query: str) -> bool:
    """Return True when query asks how an ag indicator works (not product help)."""
    m = _METHODOLOGY_WORK_RE.search(normalized)
    if not m:
        return False
    indicator = m.group(1).casefold()
    if indicator not in _METHODOLOGY_INDICATORS:
        return False
    if _BRAND_RE.search(normalized) or _ASSISTANT_REF_RE.search(raw_query):
        return False
    return True


def _passes_product_negative_guards(
    query: str,
    decomposition: dict[str, Any] | None,
) -> bool:
    """Return False when agricultural data focus should force full RAG."""
    if _has_geography(decomposition):
        return False
    if _has_ag_entities_in_decomposition(decomposition):
        return False
    if _has_ag_data_intent(query):
        return False
    return True


def is_help_query(query: str, decomposition: dict[str, Any] | None = None) -> bool:
    """
    Return True for capability / onboarding / use questions about Ask ADZA.

    Does not require a brand token when assistant-reference phrasing is present.
    """
    if not query or not query.strip():
        return False
    raw = query.strip()
    normalized = _normalize_for_product_gate(raw)
    if _is_methodology_question(normalized, raw):
        return False
    if not _CAPABILITY_RE.search(normalized):
        return False
    return _passes_product_negative_guards(raw, decomposition)


def classify_product_subroute(query: str) -> Literal["help", "product"] | None:
    """Classify product-path subroute for observability."""
    if is_help_query(query):
        return "help"
    raw = query.strip()
    normalized = _normalize_for_product_gate(raw)
    has_brand = bool(_BRAND_RE.search(normalized))
    has_product_intent = bool(_PRODUCT_INTENT_RE.search(normalized))
    if (has_brand or has_product_intent) and _passes_product_negative_guards(raw, None):
        return "product"
    return None


def static_capability_answer(query: str = "", answer_lang: str | None = None) -> str:
    """Return indicator-class capability catalog for known languages."""
    lang = (answer_lang or detect_answer_language(query)).strip().lower()
    if lang in CAPABILITY_TEMPLATES and lang != "en":
        text = CAPABILITY_TEMPLATES.get(lang) or CAPABILITY_STATIC_ANSWER
    else:
        text = render_public_capability_answer()
    return _append_footer(text.strip())


def is_product_query(query: str, decomposition: dict[str, Any] | None = None) -> bool:
    """
    Return True if the query is about OpenTrace product/mission or assistant capability.

    Agricultural data queries that mention OpenTrace (e.g. 'OpenTrace data on Kenya maize')
    return False so they route to full RAG.
    """
    if not query or not query.strip():
        return False

    if is_help_query(query, decomposition):
        return True

    raw = query.strip()
    normalized = _normalize_for_product_gate(raw)
    has_brand = bool(_BRAND_RE.search(normalized))
    has_product_intent = bool(_PRODUCT_INTENT_RE.search(normalized))

    if not has_brand and not has_product_intent:
        return False

    return _passes_product_negative_guards(raw, decomposition)


def generate_product_answer(query: str, **kwargs: Any) -> str:
    """Produce an answer from product KB + LLM. No retrieval, no Citations block."""
    lang = detect_answer_language(query)
    if is_help_query(query):
        if is_english_answer_lang(lang) or lang in CAPABILITY_TEMPLATES:
            return static_capability_answer(query, answer_lang=lang)

    from ml.rag.chatbot.generator import _call_llama, _resolve_memory_block

    kb_block = format_product_kb_for_prompt()
    memory_block = _resolve_memory_block(**kwargs)
    from ml.rag.chatbot.plan_policy import instruction_for_category, plan_generation_addendum

    category = (kwargs.get("category") or "").strip()
    plan_type = (kwargs.get("plan_type") or "").strip()
    tone = instruction_for_category(category) if category else ""
    plan_addendum = plan_generation_addendum(plan_type) if plan_type else ""

    system = PRODUCT_SYSTEM_PROMPT + "\n\n" + language_instruction(detect_answer_language(query))
    if tone:
        system = system + "\n\n" + tone
    if plan_addendum:
        system = system + "\n\n" + plan_addendum

    user_parts: list[str] = []
    if memory_block.strip():
        user_parts.append(memory_block.strip())
    user_parts.append(f"Product Knowledge:\n{kb_block}")
    user_parts.append(f"Question: {query}")
    user = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = _call_llama(messages, purpose="generate_product")
    answer = raw.strip() if raw else (
        "OpenTrace Africa builds Africa's agricultural intelligence layer — "
        "federating fragmented data into decision intelligence via OFIA, ACF, and Ask ADZA."
    )
    return _append_footer(answer)


__all__ = [
    "load_product_kb",
    "format_product_kb_for_prompt",
    "is_help_query",
    "is_product_query",
    "classify_product_subroute",
    "static_capability_answer",
    "generate_product_answer",
    "CAPABILITY_STATIC_ANSWER",
    "PRODUCT_SYSTEM_PROMPT",
]
