"""The clarify gate and table choice follow contracts, not one saved sentence."""
from __future__ import annotations

from ml.rag.chatbot.agri_measure_ontology import MEASURES, resolve_measure
from ml.rag.chatbot.class_table_router import select_table_plans
from ml.rag.chatbot.mart_indicator_classes import (
    all_class_codes,
    class_for_query,
    families_for_class,
    family_declared_tokens,
    family_match_score,
    family_question_phrase,
)
from ml.rag.chatbot.retrieval_contract import choose_agg_vs_fact
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.task_mode import needs_clarify

# Production and prices keep their own planners. Every other class uses the declaration.
_BRANCH_OWNED = frozenset({"PROD", "PRC"})


def _bare(table_id: str) -> str:
    return table_id.strip().split(".")[-1].lower()


def _numeric_question(measure_id: str, aliases: tuple[str, ...]) -> str | None:
    """A numeric question the resolver assigns to this measure, when it can."""
    for alias in sorted(aliases, key=len):
        question = f"What was the {alias}?"
        hit = resolve_measure(question, {"geography": [], "entities": []})
        if hit is not None and hit.measure.id == measure_id:
            return question
    return None


def test_clarify_gate_follows_the_job_then_the_measure_slots() -> None:
    checked = 0
    for spec in MEASURES.values():
        if spec.default_task_mode != "fact_lookup" or spec.country_is_answer:
            continue
        numeric = _numeric_question(spec.id, spec.aliases)
        if numeric is None:
            continue
        checked += 1
        document = f"What is the policy of {spec.aliases[0]} in West Africa?"
        empty = {"geography": [], "entities": []}
        requires_slot = spec.geography_required or spec.crop_required
        assert needs_clarify(numeric, empty) is requires_slot, spec.id
        assert needs_clarify(document, {"geography": ["West Africa"], "entities": []}) is False, spec.id
        if not requires_slot:
            continue
        filled = f"What was the {spec.aliases[0]} of maize in Kenya in 2020?"
        assert (
            needs_clarify(
                filled,
                {
                    "geography": ["Kenya"],
                    "entities": ["maize"],
                    "time_start": "2020-01-01",
                    "time_end": "2020-12-31",
                },
            )
            is False
        ), spec.id
    assert checked >= 4


def test_non_numeric_jobs_do_not_enter_the_clarify_gate() -> None:
    jobs = (
        "Compare maize production in Kenya and Ghana from 2015 to 2020",
        "Give me a briefing on maize production in Kenya",
        "What does research say about maize production in Kenya",
        "Explain maize production in Kenya",
        "What is the outlook for maize production in Kenya",
    )
    for question in jobs:
        assert needs_clarify(question, {"geography": ["Kenya"], "entities": ["maize"]}) is False, question


def test_family_terms_select_the_declared_table() -> None:
    facets = {
        "geography": ["Kenya"],
        "time_start": "2020-01-01",
        "time_end": "2020-12-31",
    }
    for code in all_class_codes():
        card = load_schema_card(code) or {}
        for family in families_for_class(code):
            table = str(family.get("table") or "").strip()
            if not table:
                continue
            phrase = family_question_phrase(family)
            question = f"What was the {phrase} in Kenya in 2020?"
            assert family_match_score(family, question) > 0, (code, family.get("id"))
            subjects, _grain = family_declared_tokens(family)
            if subjects:
                assert code in class_for_query(question), (code, family.get("id"))
            if code in _BRANCH_OWNED:
                continue
            plans = select_table_plans(
                code,
                query=question,
                facets=facets,
                bundles=(),
                card=card,
                iso_list=["KEN"],
            )
            assert plans, (code, family.get("id"))
            expected = choose_agg_vs_fact(
                _bare(table),
                query=question,
                multi_country=False,
                year_hint="2020",
                single_country=True,
                iso_count=1,
            )
            assert plans[0].table_id == expected, (code, family.get("id"), plans[0].table_id)


def test_coarse_branches_keep_their_tables() -> None:
    facets = {"geography": ["Kenya"], "time_start": "2020-01-01", "time_end": "2020-12-31"}
    prices = select_table_plans(
        "FVC",
        query="rice prices in Kenya",
        facets=facets,
        bundles=(),
        card=load_schema_card("FVC") or {},
        iso_list=["KEN"],
    )
    balance = select_table_plans(
        "FVC",
        query="food balance of wheat in Ghana",
        facets={"geography": ["Ghana"]},
        bundles=(),
        card=load_schema_card("FVC") or {},
        iso_list=["GHA"],
    )
    assert prices[0].table_id == "agg_prices_country_month"
    assert balance[0].table_id == "agg_food_balance_country_year"
