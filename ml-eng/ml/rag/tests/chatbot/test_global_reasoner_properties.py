"""Property tests T1–T10 for Global Reasoner + plan enricher (synthetic facets)."""
from __future__ import annotations

from ml.rag.chatbot.composer import (
    composer_addendum,
    is_sentinel_row,
    partition_bq_by_subquestion,
)
from ml.rag.chatbot.agri_measure_ontology import resolve_measure
from ml.rag.chatbot.facet_compiler import compile_turn_contract
from ml.rag.chatbot.global_reasoner import compile_reasoner_plan
from ml.rag.chatbot.output_format import render_insufficient
from ml.rag.chatbot.intent_bundles import match_intent_bundles
from ml.rag.chatbot.plan_enricher import enrich_subquestions
from ml.rag.chatbot.qdrant_planner import plan_vector_corpora_for_reasoner
from ml.rag.chatbot.reasoner_plan import ReasonerPlan, is_heavy_plan_type
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


def test_t1_agri_activities_production_and_trade_slots() -> None:
    facet = _facet("agri_activities_panel")
    dec = heavy_facet_decomposition(facet)
    bundles = match_intent_bundles(facet["query"], dec)
    turn = compile_turn_contract(facet["query"], dec, matched_bundles=tuple(bundles))
    rp = compile_reasoner_plan(
        facet["query"],
        decomposition=dec,
        turn_contract=turn,
        plan_type=facet["plan_type"],
    )
    assert rp is not None
    assert rp.heavy_path is True
    measures = {sq.measure for sq in rp.bq_subquestions()}
    assert "production" in measures
    assert "trade" in measures
    assert rp.primary_measure != "food_security_ipc"
    assert rp.job in ("report", "compare", "synthesis")


def test_t2_optional_ipc_unsupported_does_not_drop_prod_trade() -> None:
    subs = enrich_subquestions(
        query="Agricultural activities report for West Africa with optional IPC outlook overlay.",
        decomposition={"geography": list(_facet("agri_activities_panel")["geos"])},
        job="report",
        plan_type="government",
        geos=tuple(_facet("agri_activities_panel")["geos"]),
        geo_grain="country",
        time_start="2015-01-01",
        time_end="2024-12-31",
        known_tables=set(),
    )
    bq_ids = {s.id for s in subs if s.library == "bq"}
    assert "prod_panel" in bq_ids
    assert "trade_panel" in bq_ids
    vector = [s for s in subs if s.library == "vector"]
    assert vector  # outlook narrative may be optional


def test_t3_trend_last_n_years_time_window() -> None:
    facet = _facet("trend_last_years")
    dec = heavy_facet_decomposition(facet)
    dec.pop("time_start", None)
    dec.pop("time_end", None)
    dec["primary_measures"] = ["production"]
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
    assert int(rp.time_end[:4]) >= int(rp.time_start[:4])
    assert int(rp.time_end[:4]) - int(rp.time_start[:4]) + 1 <= 6


def test_t4_compare_geos_full_list_in_plan_and_gap_copy() -> None:
    facet = _facet("protected_wdpa")
    geos = list(facet["geos"])
    dec = heavy_facet_decomposition(facet)
    bundles = match_intent_bundles(facet["query"], dec)
    turn = compile_turn_contract(
        facet["query"],
        dec,
        matched_bundles=tuple(bundles),
    )
    turn.geo = geos
    rp = compile_reasoner_plan(
        facet["query"],
        decomposition=dec,
        turn_contract=turn,
        plan_type=facet["plan_type"],
    )
    assert rp is not None
    assert len(rp.geos) == len(geos)
    gap = render_insufficient(
        TurnContract.from_dict({**turn.to_dict(), "geo": geos, "job": "compare"}),
        query=facet["query"],
    )
    for g in geos:
        assert g in gap


def test_t5_protected_area_not_fertilizer() -> None:
    facet = _facet("protected_wdpa")
    subs = enrich_subquestions(
        query=facet["query"],
        decomposition=heavy_facet_decomposition(facet),
        job="compare",
        plan_type=facet["plan_type"],
        geos=tuple(facet["geos"]),
        geo_grain="country",
        time_start="",
        time_end="",
    )
    protected = [s for s in subs if s.measure == "protected_area"]
    assert protected
    assert "fct_protected_areas" in protected[0].tables
    assert all("fertilizer" not in t for s in subs for t in s.tables)
    assert all("pesticide" not in t for s in subs for t in s.tables)


def test_t6_sentinels_never_in_composer_partition() -> None:
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
        {"content": "timeout", "value": "timeout", "metadata": {}},
    ]
    partitioned = partition_bq_by_subquestion(rows, rp)
    flat = [r for bag in partitioned.values() for r in bag]
    assert len(flat) == 1
    assert not any(is_sentinel_row(r) for r in flat)


def test_t7_empty_required_slot_coverage_miss_line() -> None:
    q = "Agricultural activities report for West Africa country by country."
    dec = {"geography": ["Nigeria", "Ghana"]}
    bundles = match_intent_bundles(q, dec)
    turn = compile_turn_contract(q, dec, matched_bundles=tuple(bundles))
    rp = compile_reasoner_plan(
        q,
        decomposition=dec,
        turn_contract=turn,
        plan_type="government",
        matched_bundles=tuple(bundles),
    )
    assert rp is not None
    rows_by = {sq.id: [] for sq in rp.bq_subquestions()}
    addendum = composer_addendum(rp, rows_by)
    assert "explicit miss" in addendum.lower() or "No structured" in addendum
    assert "invent" in addendum.lower()


def test_t8_farmers_light_slot_path_not_heavy() -> None:
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
    assert len(rp.bq_subquestions()) >= 1
    assert is_heavy_plan_type(facet["plan_type"]) is False


def test_t9_numeric_job_news_off_in_qdrant_planner() -> None:
    rp = compile_reasoner_plan(
        "Compare maize production Nigeria vs Kenya 2022.",
        decomposition={"geography": ["Nigeria", "Kenya"]},
        turn_contract=TurnContract(
            job="compare",
            geo=["Nigeria", "Kenya"],
            measure_id="production",
        ),
        plan_type="agribusiness",
    )
    assert rp is not None
    planner = plan_vector_corpora_for_reasoner(
        query="Compare maize production Nigeria vs Kenya 2022.",
        reasoner=rp,
        turn_contract=TurnContract(job="compare"),
        plan_type="agribusiness",
        decomposition={"geography": ["Nigeria", "Kenya"]},
    )
    assert planner.get("news_allowed") is False
    assert "news" not in planner.get("active_corpora", [])


def test_t10_social_help_no_vector_retrieve() -> None:
    turn = TurnContract(job="help")
    assert turn.should_retrieve_vector() is False
    turn = TurnContract(job="social")
    assert turn.should_retrieve_vector() is False
