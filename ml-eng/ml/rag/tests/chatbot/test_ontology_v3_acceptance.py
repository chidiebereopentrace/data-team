"""Ontology v3 offline acceptance (§11 cases a–f). BQ / Qdrant / LLM mocked at seams."""
from __future__ import annotations

import logging
import os
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("opentrace_ontology")

import opentrace_ontology as oo
from ml.rag.chatbot import ontology_routing as orouting
from ml.rag.chatbot.acf_scoring import ACFResult, cap_acf_band
from ml.rag.chatbot.answer_outcome import decide_outcome
from ml.rag.chatbot.geo_policy import (
    apply_profile_geography_default,
    sanitize_profile_country,
)
from ml.rag.chatbot.query_decomposer import (
    _extract_countries,
    resolve_retrieval_geographies,
)
from ml.rag.chatbot.query_normalize import normalize_query_text
from ml.rag.chatbot.time_resolver import resolve_relative_time
from ml.rag.retrievers.bq_retriever import _format_query_constraints
from ml.rag.retrievers.web_retriever import needs_web_fallback

_CASSAVA_Q = "What is the current state of casssava in Nigeria"
_MUSHROOM_Q = "mushroom production in Ghana"
_ZORBLAX_Q = "zorblax yields in Togo"
_MAIZE_YEAR_Q = "maize prices this year"

_ML_ENG = Path(__file__).resolve().parents[4]
_COVERAGE_SCRIPT = (
    _ML_ENG / "opentrace_ontology" / "ontology_build" / "check_value_index_coverage.py"
)
_MVI = _ML_ENG / "ml" / "rag" / "value_index" / "mart_value_index.json"

# Warm plan_query p95 target: <1 ms preferred; Windows CI may need modest slack.
_P95_BUDGET_S = 0.001 if sys.platform != "win32" else 0.005


@pytest.fixture
def ontology_flags_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADZA_ONTOLOGY_ROUTING", "1")
    monkeypatch.setenv("ADZA_ONTOLOGY_BQ_VALUES", "1")
    monkeypatch.setenv("ADZA_ACF_SOURCE_BASIS_CAP", "1")
    monkeypatch.setenv("ADZA_ONTOLOGY_QUERY_VIEW", "1")
    monkeypatch.setenv("ADZA_QDRANT_ONTOLOGY_FILTER", "1")


def _modes(plan: dict) -> dict[str, str]:
    return {
        e["entity_id"]: e["mode"]
        for e in plan["entities"]
        if e.get("entity_id")
    }


# ---------------------------------------------------------------------------
# (a) casssava Nigeria, Farmers, profile Albania
# ---------------------------------------------------------------------------


def test_a_cassava_nigeria_farmers_profile_albania(ontology_flags_on) -> None:
    # Farmers profile Albania → dropped (non-African)
    assert sanitize_profile_country("Albania") is None

    normalized = normalize_query_text(_CASSAVA_Q)
    countries = _extract_countries(normalized)
    assert countries == ["Nigeria"]
    dec = apply_profile_geography_default(
        {"geography": countries, "geography_source": "query"},
        "Albania",
    )
    assert dec["geography"] == ["Nigeria"]
    assert "Albania" not in (dec.get("geography") or [])
    geos = resolve_retrieval_geographies(
        geo_override="",
        geography=dec["geography"],
    )
    assert geos == ["Nigeria"]

    plan = orouting.ontology_plan_for_state({"query": _CASSAVA_Q})
    assert plan is not None
    # Ontology geography is ISO3
    assert plan["countries"] == ["NGA"]
    assert plan["african_countries"] == ["NGA"]

    modes = _modes(plan)
    assert modes.get("OT:CROP:cassava") == "bq+qdrant"
    assert modes.get("OT:GEO:NGA") == "bq+qdrant"
    assert plan["source_basis"] == "structured"

    filters = orouting.bq_value_filters(plan)
    block = _format_query_constraints(
        geo_country="NGA",
        time_start=None,
        time_end=None,
        entities=None,
        domains=None,
        ontology_value_filters=filters,
    )
    assert "ALLOWED FILTER LITERALS" in block
    flat_vals = {v for vals in filters.values() for v in vals}
    assert "Cassava" in flat_vals
    assert "NGA" in flat_vals
    assert "NG" in flat_vals

    # No screen-date leak without time_start_override
    ref = date(2026, 10, 14)
    windows = resolve_relative_time(_CASSAVA_Q, ref)
    assert windows
    assert all(w.start != date(2024, 4, 5) for w in windows)
    assert "2024-04-05" not in "".join(w.start.isoformat() for w in windows)

# ---------------------------------------------------------------------------
# (b) mushroom Ghana — documents_only / qdrant_only
# ---------------------------------------------------------------------------


def test_b_mushroom_ghana_documents_only(ontology_flags_on) -> None:
    plan = orouting.ontology_plan_for_state({"query": _MUSHROOM_Q})
    assert plan is not None
    assert plan["source_basis"] == "documents_only"
    assert plan["acf_ceiling"] == "moderate"
    assert not plan["run_bigquery"]
    assert plan["run_qdrant"]
    assert plan["countries"] == ["GHA"]

    subjects = [
        e
        for e in plan["entities"]
        if e.get("entity_type") != "country" and e.get("entity_id")
    ]
    assert subjects
    assert all(e["mode"] == "qdrant_only" for e in subjects)

    view = orouting.qdrant_expansion_view(plan)
    assert view is not None
    assert "mushroom" in view.lower() or "fungi" in view.lower()

    filters = orouting.bq_value_filters(plan)
    mushroom_vals = [
        v
        for vals in filters.values()
        for v in vals
        if "mushroom" in str(v).lower()
    ]
    assert mushroom_vals == []

    ceiling = orouting.acf_band_ceiling(plan, {"qdrant", "vector"})
    assert ceiling == "moderate"
    capped = cap_acf_band(
        ACFResult(
            band="strong",
            band_label="Strong confidence",
            score=80,
            explanation="base",
            note="base",
        ),
        ceiling,
    )
    assert capped.band == "moderate"

    items = [
        {
            "content": "Mushroom cultivation notes for Ghana smallholders.",
            "_context_kind": "news",
            "metadata": {"context_kind": "news"},
        }
    ]
    assert decide_outcome(items).kind == "documents_only"


# ---------------------------------------------------------------------------
# (c) zorblax Togo — web_fallback after empty BQ+Qdrant
# ---------------------------------------------------------------------------


def test_c_zorblax_togo_web_fallback(ontology_flags_on) -> None:
    plan = orouting.ontology_plan_for_state({"query": _ZORBLAX_Q})
    assert plan is not None
    assert plan["web_fallback_terms"] == ["zorblax"]
    assert plan["countries"] == ["TGO"]
    assert plan["source_basis"] == "web_only"
    assert plan["acf_ceiling"] == "limited"
    # Unknown subject is web_fallback; geography may still attempt BQ
    unknown = [e for e in plan["entities"] if e.get("entity_type") == "unknown"]
    assert unknown and unknown[0]["mode"] == "web_fallback"

    # Web when BQ+Qdrant empty and warehouse did not produce a blocking attempt
    assert needs_web_fallback(
        [],
        enabled=True,
        task_mode="chat",
        has_usable_bq=False,
        bq_warehouse_attempted=False,
    )
    # With Qdrant hits, do not force web
    hits = [
        {
            "content": "Togo agriculture overview",
            "_context_kind": "news",
            "metadata": {"context_kind": "news"},
            "_rerank_score": 0.9,
        }
        for _ in range(3)
    ]
    assert not needs_web_fallback(
        hits,
        enabled=True,
        task_mode="chat",
        has_usable_bq=False,
        bq_warehouse_attempted=False,
    )

    ceiling = orouting.acf_band_ceiling(plan, {"web"})
    assert ceiling == "limited"
    capped = cap_acf_band(
        ACFResult(
            band="strong",
            band_label="Strong confidence",
            score=80,
            explanation="base",
            note="base",
        ),
        ceiling,
    )
    assert capped.band == "limited"


# ---------------------------------------------------------------------------
# (d) maize prices this year — profile soft default
# ---------------------------------------------------------------------------


def test_d_maize_prices_this_year_profile_defaults(
    ontology_flags_on, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="ml.rag.chatbot.geo_policy"):
        assert sanitize_profile_country("Albania") is None
    assert any(
        "profile_country_dropped_non_african" in r.getMessage() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]

    # Albania → no country default when query has no geography
    dec_alb = apply_profile_geography_default({"geography": []}, "Albania")
    assert not dec_alb.get("geography")

    # NG → NGA soft default
    assert sanitize_profile_country("NG") == "NGA"
    dec_ng = apply_profile_geography_default({"geography": []}, "NG")
    assert dec_ng["geography"] == ["NGA"]
    assert dec_ng["geography_source"] == "profile"

    plan = oo.plan_query(_MAIZE_YEAR_Q)
    assert plan["source_basis"] == "structured"
    modes = _modes(plan)
    assert any(k and "maize" in k for k in modes)
    assert plan["countries"] == []

# ---------------------------------------------------------------------------
# (e) all flags off — existing behaviour
# ---------------------------------------------------------------------------


def test_e_all_flags_off_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ADZA_ONTOLOGY_ROUTING",
        "ADZA_ONTOLOGY_BQ_VALUES",
        "ADZA_ACF_SOURCE_BASIS_CAP",
        "ADZA_ONTOLOGY_QUERY_VIEW",
        "ADZA_QDRANT_ONTOLOGY_FILTER",
    ):
        monkeypatch.setenv(key, "0")
    assert orouting.ontology_plan_for_state({"query": _CASSAVA_Q}) is None
    assert orouting.qdrant_expansion_view(
        {"entities": [{"mode": "qdrant_only", "entity_type": "product", "qdrant_terms": ["x"]}]}
    ) is None


# ---------------------------------------------------------------------------
# (f) latency / coverage gate
# ---------------------------------------------------------------------------


def test_f_plan_query_warm_p95_and_coverage() -> None:
    oo.plan_query(_CASSAVA_Q)  # warm
    samples: list[float] = []
    for _ in range(200):
        t0 = time.perf_counter()
        oo.plan_query(_CASSAVA_Q)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    p95 = samples[int(0.95 * (len(samples) - 1))]
    assert p95 < _P95_BUDGET_S, (
        f"plan_query warm p95={p95 * 1000:.3f} ms exceeds "
        f"{_P95_BUDGET_S * 1000:.1f} ms budget "
        f"(platform={sys.platform}; prefer <1 ms)"
    )

    assert _COVERAGE_SCRIPT.is_file(), f"missing {_COVERAGE_SCRIPT}"
    assert _MVI.is_file(), f"missing {_MVI}"
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_ML_ENG) + (os.pathsep + prev if prev else "")
    proc = subprocess.run(
        [sys.executable, str(_COVERAGE_SCRIPT), "--mvi", str(_MVI)],
        cwd=str(_ML_ENG),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    out = (proc.stdout or "").strip()
    # values=5358 … allowlisted=25 unmapped=0
    assert "values=5358" in out
    assert "allowlisted=25" in out
    assert "unmapped=0" in out
    # mean latency documented for PR paste
    mean_ms = statistics.mean(samples) * 1000
    assert mean_ms >= 0  # keep sample list used
