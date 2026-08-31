"""Property tests B1–B10 for Optimal Reasoner (dictionary + bundles + slots)."""
from __future__ import annotations

from ml.rag.chatbot.agri_measure_ontology import MEASURES, get_measure, resolve_measure
from ml.rag.chatbot.analytical_bq_plan import (
    _should_use_food_security_plan,
    build_analytical_bq_plan,
    build_food_security_bq_plan,
)
from ml.rag.chatbot.bq_sql_reasoner import reason_bq_sql_plan
from ml.rag.chatbot.capability_registry import apply_reasoner_to_turn, resolve_slot_capability
from ml.rag.chatbot.composer import is_sentinel_row, partition_bq_by_subquestion
from ml.rag.chatbot.agri_measure_ontology import resolve_measure
from ml.rag.chatbot.facet_compiler import compile_turn_contract
from ml.rag.chatbot.fact_bq_plan import build_fact_bq_plan
from ml.rag.chatbot.global_reasoner import compile_reasoner_plan, reasoner_plan_to_bq_plan
from ml.rag.chatbot.intent_bundles import bundle_required_measures, match_intent_bundles
from ml.rag.chatbot.plan_enricher import enrich_subquestions
from ml.rag.chatbot.reasoner_plan import ReasonerPlan, SubQuestion, should_compile_reasoner_plan
from ml.rag.chatbot.turn_contract import TurnContract
from ml.rag.tests.chatbot.fixtures.synthetic_facets import (
    heavy_facet_decomposition,
    synthetic_heavy_reasoner_facets,
)


def _facet(id_: str) -> dict:
    for f in synthetic_heavy_reasoner_facets():
        if f["id"] == id_:
            return f
    raise KeyError(id_)


def test_b1_agricultural_activities_production_and_trade_not_fs_primary() -> None:
    facet = _facet("agri_activities_panel")
    dec = heavy_facet_decomposition(facet)
    bundles = match_intent_bundles(facet["query"], dec)
    dec["matched_bundles"] = [b.spec.id for b in bundles]
    dec["primary_measures"] = list(bundle_required_measures(bundles))
    turn = compile_turn_contract(
        facet["query"],
        dec,
        matched_bundles=tuple(bundles),
    )
    rp = compile_reasoner_plan(
        facet["query"],
        decomposition=dec,
        turn_contract=turn,
        plan_type=facet["plan_type"],
        matched_bundles=tuple(bundles),
    )
    assert rp is not None
    measures = {sq.measure for sq in rp.bq_subquestions()}
    assert "production" in measures
    assert "trade" in measures
    assert turn.measure_id != "food_security_ipc"
    assert rp.primary_measure != "food_security_ipc"


def test_b2_protected_area_uses_fct_protected_areas_not_land_inputs() -> None:
    facet = _facet("protected_wdpa")
    dec = heavy_facet_decomposition(facet)
    bundles = match_intent_bundles(facet["query"], dec)
    subs = enrich_subquestions(
        query=facet["query"],
        decomposition=dec,
        job="compare",
        plan_type=facet["plan_type"],
        geos=tuple(facet["geos"]),
        geo_grain="country",
        time_start="",
        time_end="",
        matched_bundles=tuple(bundles),
    )
    protected = [s for s in subs if s.measure == "protected_area"]
    assert protected
    assert "fct_protected_areas" in protected[0].tables
    land_tables = {"fct_fertilizer", "fct_pesticide", "fct_land_use"}
    assert not any(t in land_tables for s in subs for t in s.tables)


def test_b3_fs_alias_hint_does_not_override_activities_bundle() -> None:
    q = (
        "Food security IPC outlook and agricultural activities country by country "
        "for West Africa — production and trade."
    )
    dec = {"geography": list(_facet("agri_activities_panel")["geos"][:4])}
    bundles = match_intent_bundles(q, dec)
    dec["matched_bundles"] = [b.spec.id for b in bundles]
    dec["primary_measures"] = list(bundle_required_measures(bundles))
    hint = resolve_measure(q, dec)
    assert hint is not None
    turn = compile_turn_contract(q, dec, measure_hit=hint, matched_bundles=tuple(bundles))
    rp = compile_reasoner_plan(
        q,
        decomposition=dec,
        turn_contract=turn,
        plan_type="government",
        matched_bundles=tuple(bundles),
    )
    assert rp is not None
    measures = {sq.measure for sq in rp.bq_subquestions()}
    assert "production" in measures and "trade" in measures
    plan = reasoner_plan_to_bq_plan(rp)
    assert plan.get("slot_path") is True
    assert len(plan.get("query_intents") or []) >= 2


def test_b4_one_slot_unsupported_other_required_still_served() -> None:
    rp = ReasonerPlan(
        job="report",
        plan_type="government",
        export="none",
        depth="chat",
        geos=("Nigeria",),
        geo_grain="country",
        time_start="2020-01-01",
        time_end="2024-12-31",
        subquestions=(
            SubQuestion(
                id="prod_panel",
                nl="Production panel",
                measure="production",
                required=True,
                library="bq",
                tables=("fct_production",),
            ),
            SubQuestion(
                id="bad_slot",
                nl="Unknown measure slot",
                measure="nonexistent_measure_xyz",
                required=True,
                library="bq",
                tables=("fct_production",),
            ),
        ),
        shape="report",
        primary_measure="production",
    )
    assert resolve_slot_capability("nonexistent_measure_xyz", geo_grain="country", time_grain="year").startswith(
        "unsupported"
    )
    turn = TurnContract(job="report", geo=["Nigeria"], measure_id="food_security_ipc")
    turn.serve_status = "unsupported_measure"
    turn = apply_reasoner_to_turn(turn, rp)
    assert turn.serve_status == "served"
    assert turn.measure_id == "production"


def test_b5_trend_last_n_years_compiled_before_sql() -> None:
    facet = _facet("trend_last_years")
    dec = heavy_facet_decomposition(facet)
    dec.pop("time_start", None)
    dec.pop("time_end", None)
    hit = resolve_measure(facet["query"], dec)
    turn = compile_turn_contract(facet["query"], dec, measure_hit=hit, task_mode_hint="analytical")
    turn.job = "trend"  # type: ignore[assignment]
    rp = compile_reasoner_plan(
        facet["query"],
        decomposition=dec,
        turn_contract=turn,
        plan_type=facet["plan_type"],
        task_mode="analytical",
    )
    assert rp is not None
    assert rp.job == "trend"
    assert rp.time_start and rp.time_end
    plan = reasoner_plan_to_bq_plan(rp)
    filters_blob = " ".join(str(i.get("filters") or "") for i in plan.get("query_intents") or [])
    assert rp.time_start[:4] in filters_blob or rp.time_end[:4] in filters_blob


def test_b6_farmers_numeric_fact_light_reasoner_not_fact_bq_plan() -> None:
    facet = _facet("farmers_simple_fact")
    dec = heavy_facet_decomposition(facet)
    hit = resolve_measure(facet["query"], dec)
    turn = compile_turn_contract(facet["query"], dec, measure_hit=hit)
    rp = compile_reasoner_plan(
        facet["query"],
        decomposition=dec,
        turn_contract=turn,
        plan_type=facet["plan_type"],
    )
    assert rp is not None
    assert rp.heavy_path is False
    assert len(rp.bq_subquestions()) == 1
    dec_with_reasoner = {**dec, "reasoner_job": rp.job}
    assert build_fact_bq_plan(
        facet["query"],
        decomposition=dec_with_reasoner,
        known_tables=set(MEASURES["production"].candidate_tables),
        task_mode="fact_lookup",
    ) is None


def test_b7_sentinels_never_in_composer_partition() -> None:
    from ml.rag.chatbot.test_pipeline_integration import _maize_bq_chunk

    rp = compile_reasoner_plan(
        "Compare production in Nigeria and Ghana.",
        decomposition={"geography": ["Nigeria", "Ghana"], "primary_measures": ["production"]},
        turn_contract=TurnContract(job="compare", geo=["Nigeria", "Ghana"], measure_id="production"),
        plan_type="government",
    )
    assert rp is not None
    good = _maize_bq_chunk(country="Nigeria", year=2022)
    good_meta = dict(good.get("metadata") or {})
    good_meta["subquestion_id"] = "prod_panel"
    good = {**good, "metadata": good_meta}
    rows = [
        good,
        {
            "content": "[BQ no_valid_sql]",
            "source": "bigquery",
            "metadata": {"status": "no_valid_sql", "subquestion_id": "trade_panel"},
        },
    ]
    partitioned = partition_bq_by_subquestion(rows, rp)
    flat = [r for bag in partitioned.values() for r in bag]
    assert flat
    assert not any(is_sentinel_row(r) for r in flat)


def test_b8_enricher_tables_match_measure_dictionary() -> None:
    facet = _facet("food_balance")
    dec = heavy_facet_decomposition(facet)
    bundles = match_intent_bundles(facet["query"], dec)
    subs = enrich_subquestions(
        query=facet["query"],
        decomposition=dec,
        job="compare",
        plan_type=facet["plan_type"],
        geos=tuple(facet["geos"]),
        geo_grain="country",
        time_start="2015-01-01",
        time_end="2024-12-31",
        matched_bundles=tuple(bundles),
    )
    for sq in subs:
        if sq.library != "bq" or not sq.measure:
            continue
        spec = get_measure(sq.measure)
        assert spec is not None
        allowed = {str(t).split(".")[-1].lower() for t in spec.candidate_tables}
        for tid in sq.tables:
            assert tid in allowed, f"{tid} not in dictionary for {sq.measure}"


def test_b9_outlook_ipc_vector_ag_activities_no_forced_ipc_bq_spine() -> None:
    ag_facet = _facet("agri_activities_panel")
    ag_dec = heavy_facet_decomposition(ag_facet)
    ag_dec["matched_bundles"] = ["agricultural_activities"]
    assert _should_use_food_security_plan(ag_facet["query"], ag_dec) is False
    assert build_analytical_bq_plan(
        ag_facet["query"],
        decomposition={**ag_dec, "reasoner_job": "report"},
        known_tables=set(),
    ) is None
    fs_direct = build_food_security_bq_plan(
        ag_facet["query"],
        decomposition=ag_dec,
        known_tables={"fct_food_security", "fct_production"},
    )
    assert fs_direct is None or _should_use_food_security_plan(ag_facet["query"], ag_dec) is False

    outlook_facet = _facet("outlook_ipc")
    outlook_dec = heavy_facet_decomposition(outlook_facet)
    bundles = match_intent_bundles(outlook_facet["query"], outlook_dec)
    subs = enrich_subquestions(
        query=outlook_facet["query"],
        decomposition=outlook_dec,
        job="outlook",
        plan_type=outlook_facet["plan_type"],
        geos=tuple(outlook_facet["geos"]),
        geo_grain="country",
        time_start="",
        time_end="",
        matched_bundles=tuple(bundles),
    )
    vector_ipc = [s for s in subs if s.library == "vector" and "food_security" in s.measure]
    assert vector_ipc or any(s.measure == "food_security_ipc" for s in subs)


def test_b10_social_help_no_reasoner_or_legacy_bq_plans() -> None:
    for job in ("help", "social"):
        turn = TurnContract(job=job)  # type: ignore[arg-type]
        assert should_compile_reasoner_plan(turn) is False
        assert turn.should_retrieve_vector() is False
        assert compile_reasoner_plan(
            "How do I use this chatbot?",
            decomposition={},
            turn_contract=turn,
            plan_type="farmers",
        ) is None
    legacy = reason_bq_sql_plan(
        "Hello",
        decomposition={"intent": "help"},
        task_mode="chat",
    )
    assert legacy.get("skip_bq") or not legacy.get("query_intents")
