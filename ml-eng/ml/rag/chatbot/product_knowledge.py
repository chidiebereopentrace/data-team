"""
Product knowledge layer for OpenTrace / Ask ADZA questions.

Loads canonical product copy from opentrace_product.json and routes queries that
ask about OpenTrace's mission, pillars, trust model, etc. without retrieval.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ml.rag.chatbot.assistant_identity import META_ANSWER_FOOTER, _append_footer

_DEFAULT_KB_PATH = Path(__file__).resolve().parent / "data" / "opentrace_product.json"

# Brand / product tokens
_BRAND_PATTERNS: tuple[str, ...] = (
    r"\bopentrace\b",
    r"\bask adza\b",
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
)

_BRAND_RE = re.compile("|".join(_BRAND_PATTERNS), re.IGNORECASE)
_PRODUCT_INTENT_RE = re.compile("|".join(_PRODUCT_INTENT_PATTERNS), re.IGNORECASE)

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


def is_product_query(query: str, decomposition: dict[str, Any] | None = None) -> bool:
    """
    Return True if the query is about OpenTrace product/mission (not agricultural data).

    Agricultural data queries that mention OpenTrace (e.g. 'OpenTrace data on Kenya maize')
    return False so they route to full RAG.
    """
    if not query or not query.strip():
        return False

    q = query.strip()
    has_brand = bool(_BRAND_RE.search(q))
    has_product_intent = bool(_PRODUCT_INTENT_RE.search(q))

    if not has_brand and not has_product_intent:
        return False

    # Negative signals: geographic or agricultural data focus → RAG
    if _has_geography(decomposition):
        return False
    if _has_ag_entities_in_decomposition(decomposition):
        return False
    if _has_ag_data_intent(q):
        return False

    return True


def generate_product_answer(query: str, **kwargs: Any) -> str:
    """Produce an answer from product KB + LLM. No retrieval, no Citations block."""
    from ml.rag.chatbot.generator import _call_llama, _resolve_memory_block

    kb_block = format_product_kb_for_prompt()
    memory_block = _resolve_memory_block(**kwargs)
    from ml.rag.chatbot.plan_policy import instruction_for_category, plan_generation_addendum

    category = (kwargs.get("category") or "").strip()
    plan_type = (kwargs.get("plan_type") or "").strip()
    tone = instruction_for_category(category) if category else ""
    plan_addendum = plan_generation_addendum(plan_type) if plan_type else ""

    system = PRODUCT_SYSTEM_PROMPT
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
    raw = _call_llama(messages)
    answer = raw.strip() if raw else (
        "OpenTrace Africa builds Africa's agricultural intelligence layer — "
        "federating fragmented data into decision intelligence via OFIA, ACF, and Ask ADZA."
    )
    return _append_footer(answer)


__all__ = [
    "load_product_kb",
    "format_product_kb_for_prompt",
    "is_product_query",
    "generate_product_answer",
    "PRODUCT_SYSTEM_PROMPT",
]
