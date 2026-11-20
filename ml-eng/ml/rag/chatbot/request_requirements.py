"""Ground indicator requirements in the clauses that actually requested them."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from ml.rag.chatbot.agri_entities import CROP_ENTITY_RE
from ml.rag.chatbot.agri_measure_ontology import MEASURES, explicit_measure_ids
from ml.rag.chatbot.geo_iso3 import (
    countries_named_in_text,
    iso3_to_canonical_name,
    name_to_iso3,
)
from ml.rag.chatbot.time_resolver import (
    parse_iso_date,
    resolve_relative_time,
    utc_today,
)
from ml.rag.chatbot.turn_contract import FactRequirement, TurnContract

_INVENTORY = re.compile(
    r"\b(?:itemiz\w*|by (?:crop|product|commodity|item)|par (?:culture|produit)|"
    r"(?:table|tableau|list\w*|breakdown|ventilation).*?\b(?:crops?|products?|commodit\w*|cultures?|produits?|cereals?|cereales))\b",
    re.I,
)


def grounded_fact_phrases(raw: Any, query: str) -> list[tuple[str, str]]:
    """Accept canonical measures only with a verbatim span of the planning question."""
    result = []
    if not isinstance(raw, list):
        return result
    for item in raw[:20]:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("measure_id") or "")
        phrase = str(item.get("question") or "").strip()
        if (mid in MEASURES and len(phrase) >= 3 and phrase.casefold() in query.casefold()
                and (mid, phrase) not in result):
            result.append((mid, phrase))
    return result


def compile_fact_requirements(
    turn: TurnContract, query: str, decomposition: dict[str, Any],
) -> list[FactRequirement]:
    phrases = grounded_fact_phrases(decomposition.get("fact_requirements"), query)
    clauses = re.split(r"(?<=[.!?;])\s+", query)
    # A local place or period binds the indicators in that clause. Later instructions
    # about units or attribution must not overwrite those scopes.
    for mid in explicit_measure_ids(query):
        if any(measure == mid for measure, _ in phrases):
            continue
        matches = [clause for clause in clauses if mid in explicit_measure_ids(clause)]
        scoped = [clause for clause in matches if countries_named_in_text(clause) or re.search(r"\b(?:19|20)\d{2}\b", clause)]
        for clause in scoped or matches[:1]:
            phrases.append((mid, clause))

    reference = parse_iso_date(str(decomposition.get("reference_date") or "")) or utc_today()
    counts: dict[str, int] = {}
    for mid, _ in phrases:
        counts[mid] = counts.get(mid, 0) + 1
    seen: dict[str, int] = {}
    facts = []
    for mid, phrase in phrases:
        seen[mid] = seen.get(mid, 0) + 1
        original = turn.original_question or query
        inventory = bool(_INVENTORY.search(original))
        member_query = original if inventory and len(phrases) == 1 else phrase
        scope_phrase = member_query if inventory else phrase
        places = sorted({iso3_to_canonical_name(name_to_iso3(place) or "") or place
                         for place in countries_named_in_text(scope_phrase)})
        windows = resolve_relative_time(scope_phrase, reference)
        time = turn.time_spec
        if windows:
            time = replace(
                time, windows=windows, start=min(w.start for w in windows).isoformat(),
                end=max(w.end for w in windows).isoformat(),
                grain="year_range" if len(windows) > 1 or windows[0].start.year != windows[0].end.year else "year",
            )
        crops = list(dict.fromkeys(match.group().lower() for match in CROP_ENTITY_RE.finditer(phrase)))
        # One inventory request may be expanded into many suggested examples.
        # Preserve its original member scope instead of filtering to that basket.
        facts.append(FactRequirement(
            mid, question=phrase, geo=places or list(turn.geo), time_spec=time,
            entities=crops or list(turn.entities),
            requirement_id=mid if counts[mid] == 1 else f"{mid}_{seen[mid]}",
            member_query=member_query,
            enumerate_members=mid != "market_price" and (
                inventory or (MEASURES[mid].crop_required and len({m.group().casefold() for m in CROP_ENTITY_RE.finditer(member_query)}) > 1)
            ),
        ))
    return facts
