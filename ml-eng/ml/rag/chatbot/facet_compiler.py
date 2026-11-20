"""Compile raw query + decomposition into a typed TurnContract."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import (
    MEASURES,
    MeasureHit,
    MeasureSpec,
    explicit_measure_ids,
    query_locks_production,
)
from ml.rag.chatbot.analytical_intent import (
    asks_about_causal_effect,
    requests_document_content,
)
from ml.rag.chatbot.continental_scope import (
    CONTINENTAL_COUNT_RE,
    decomposition_has_africa_scope,
    is_continental_rank_query,
)
from ml.rag.chatbot.decompose_context import JOB_ALLOWED
from ml.rag.chatbot.geo_iso3 import infer_country_iso3_from_query, name_to_iso3
from ml.rag.chatbot.intent_bundles import (
    MatchedBundle,
    bundle_primary_measure,
    bundle_required_measures,
    bundles_block_primary,
)
from ml.rag.chatbot.request_requirements import compile_fact_requirements
from ml.rag.chatbot.slot_validator import validate_turn_slots
from ml.rag.chatbot.time_resolver import (
    TimeWindow,
    parse_iso_date,
    resolve_relative_time,
    retrieval_envelope,
    utc_today,
)
from ml.rag.chatbot.turn_contract import (
    BreakdownDim,
    DocumentRequirement,
    GeoGrain,
    Job,
    TimeGrain,
    TimeSpec,
    TurnContract,
)

_ADMIN2_RE = re.compile(
    r"\b(districts?|wards?|counties|sub[- ]?counties|localities)\b",
    re.IGNORECASE,
)
_ADMIN1_RE = re.compile(
    r"\b(regions?|provinces|states|departments|governorates)\b",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"\b(which|list|name\s+the|show\s+me\s+the)\b.*\b(districts?|regions?|counties|areas)\b",
    re.IGNORECASE,
)
_TREND_RE = re.compile(
    r"\b(trend|over\s+time|how\s+has|changed|evolution|growth|since|past|last\s+\d+\s+years?)\b",
    re.IGNORECASE,
)
_RANK_RE = re.compile(
    r"\b(highest|lowest|top|rank|ranking|most|least|best|worst|which\s+country)\b",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(r"\b(compare|versus|vs\.?|compared\s+to)\b", re.IGNORECASE)
_AGRI_PANEL_REPORT_RE = re.compile(
    r"\b(agricultural activities|agri activities|country.by.country|country by country)\b",
    re.IGNORECASE,
)
_OUTLOOK_RE = re.compile(
    r"\b(outlook|projected|projection|forecast|likely|next\s+lean|lean\s+season)\b",
    re.IGNORECASE,
)
_DIAGNOSE_RE = re.compile(r"\b(why|driver|drivers|cause|causes|because|explain)\b", re.IGNORECASE)
_BRIEF_RE = re.compile(r"\b(brief|report|briefing|digest|summary\s+report)\b", re.IGNORECASE)
_SEX_BREAKDOWN_RE = re.compile(
    r"\b(by\s+(?:men\s+and\s+women|women\s+and\s+men|sex|gender)|"
    r"gender[- ]disaggregated|male\s+and\s+female|men\s+and\s+women)\b",
    re.IGNORECASE,
)
_URBAN_RURAL_RE = re.compile(r"\b(by\s+)?(urban|rural|urban/rural)\b", re.IGNORECASE)
_EMPLOYMENT_SHARE_RE = re.compile(
    r"\b(share\s+of\s+(?:employment|workers|jobs|labou?r)\s+in\s+agricultur(?:e|al)|"
    r"agricultural\s+employment\s+share|employment\s+in\s+agricultur(?:e|al))\b",
    re.IGNORECASE,
)
_AGRICULTURE_SECTOR_RE = re.compile(r"\bagricultur(?:e|al)\b", re.IGNORECASE)
_LEAN_SEASON_RE = re.compile(r"\b(next\s+lean\s+season|lean\s+season)\b", re.IGNORECASE)
_HISTORICAL_RE = re.compile(r"\b(how\s+did|what\s+happened|famine|drought\s+of|crisis\s+of)\b", re.IGNORECASE)
_PREVALENCE_RE = re.compile(
    r"\b(prevalence|seroprevalence|incidence|infection\s+rate|disease\s+rate)\b",
    re.IGNORECASE,
)
_ECF_RE = re.compile(
    r"\b(east\s+coast\s+fever|ecf\b|theileria\s+parva|theileriosis)\b",
    re.IGNORECASE,
)
_DAIRY_POP_RE = re.compile(r"\b(smallholder\s+dairy|dairy\s+herd|dairy\s+cattle|dairy\s+farm)\b", re.IGNORECASE)


_CITED_EXPLANATION_RE = re.compile(
    r"\b(?:use|using|cite|citing|based on)\b[^.!?]*\b(?:evaluations?|research|studies|assessments?|reports?)\b"
    r"[^.!?]*\b(?:explain|assess|evaluate|discuss|affected|influenced)\b",
    re.IGNORECASE,
)


def compile_time_spec(query: str, decomposition: dict[str, Any] | None) -> TimeSpec:
    dec = decomposition if isinstance(decomposition, dict) else {}
    ts = str(dec.get("time_start") or "").strip()[:10]
    te = str(dec.get("time_end") or "").strip()[:10]
    q = (query or "").strip()
    ql = q.lower()

    hard_filter = False
    time_role: str = "either"

    ref_raw = str(dec.get("reference_date") or "").strip()[:10]
    ref = parse_iso_date(ref_raw) or utc_today()
    raw_windows = dec.get("time_windows")
    windows = [
        window for item in raw_windows if isinstance(item, dict)
        if (window := TimeWindow.from_dict(item)) is not None
    ] if isinstance(raw_windows, list) else []
    if not windows:
        windows = resolve_relative_time(q, ref, infer_country_iso3_from_query(q))
    if _LEAN_SEASON_RE.search(ql):
        return TimeSpec(
            start=ts, end=te, grain="season", relative="next_lean_season",
            hard_filter=True, time_role="observation", windows=windows,
        )
    if windows and not ts and not te:
        envelope = retrieval_envelope(windows)
        if envelope:
            ts = envelope[0].isoformat()
            te = envelope[1].isoformat()

    grain: TimeGrain = "year"
    relative = ""
    if ts and te and ts[:4] != te[:4]:
        grain = "year_range"
    if len(windows) == 1 and windows[0].grain == "season":
        grain = "season"
    if len(windows) >= 2:
        grain = "year_range"
        hard_filter = True
        time_role = "observation"
    if re.search(r"\b(?:past|last|over)\s+\d{1,2}\s+years?\b", ql):
        relative = "last_n_years"
        grain = "year_range"
        hard_filter = True
        time_role = "observation"
    if re.search(r"\b(?:in|during|for)\s+(?:20\d{2})\b", ql) or (ts and te and ts[:4] == te[:4] and ts[:4].isdigit()):
        hard_filter = True
        time_role = "observation"
    if _HISTORICAL_RE.search(ql) and (ts or te):
        time_role = "historical"
        hard_filter = True
    if windows and all(not w.strict for w in windows):
        hard_filter = False
        relative = relative or "latest_available"
    if str(dec.get("time_source") or "") == "ui":
        hard_filter = False

    return TimeSpec(
        start=ts,
        end=te,
        grain=grain,
        relative=relative,
        hard_filter=hard_filter,
        time_role=time_role,  # type: ignore[arg-type]
        windows=windows,
    )


def compile_geo_grain(query: str, decomposition: dict[str, Any] | None) -> GeoGrain:
    dec = decomposition if isinstance(decomposition, dict) else {}
    q = (query or "").strip()
    geo_scope = str(dec.get("geo_scope") or "").strip().lower()
    if geo_scope == "continent":
        return "africa"
    if geo_scope == "region":
        return "region"
    if geo_scope == "multi_country":
        return "country"
    if geo_scope == "country":
        return "country"
    if dec.get("africa_panel"):
        return "africa"
    if dec.get("africa_default"):
        return "africa"
    if decomposition_has_africa_scope(dec) and is_continental_rank_query(q):
        return "africa"
    if _ADMIN2_RE.search(q) or _LIST_RE.search(q):
        return "admin2"
    if _ADMIN1_RE.search(q):
        return "admin1"
    geo = dec.get("geography")
    if isinstance(geo, list) and len(geo) >= 2:
        return "country" if all(name_to_iso3(str(place)) for place in geo) else "region"
    return "country"


def compile_breakdown(query: str) -> list[BreakdownDim]:
    q = (query or "").strip()
    out: list[BreakdownDim] = []
    if _SEX_BREAKDOWN_RE.search(q):
        out.append("sex")
    if _URBAN_RURAL_RE.search(q):
        out.append("urban_rural")
    if _ADMIN1_RE.search(q) and "by" in q.lower():
        out.append("admin1")
    if _ADMIN2_RE.search(q) and "by" in q.lower():
        out.append("admin2")
    seen: set[str] = set()
    deduped: list[BreakdownDim] = []
    for dim in out:
        if dim not in seen:
            seen.add(dim)
            deduped.append(dim)
    return deduped


def compile_job(
    query: str,
    decomposition: dict[str, Any] | None,
    *,
    task_mode_hint: str = "",
) -> Job:
    q = (query or "").strip()
    ql = q.lower()
    intent = str((decomposition or {}).get("intent") or "").strip().lower()
    mode = (task_mode_hint or "").strip().lower()

    if mode == "analytical" and asks_about_causal_effect(q):
        return "diagnose"

    model_job = str((decomposition or {}).get("job") or "").strip().lower()
    if model_job in JOB_ALLOWED:
        return model_job  # type: ignore[return-value]

    if mode == "clarify":
        return "clarify"
    if mode in ("briefing",):
        return "brief"
    if mode == "research":
        return "diagnose"
    if _LIST_RE.search(q):
        return "list"
    if _OUTLOOK_RE.search(q) or intent == "predictive":
        return "outlook"
    if _DIAGNOSE_RE.search(q) or intent == "diagnostic":
        return "diagnose"
    if _AGRI_PANEL_REPORT_RE.search(q):
        return "report"
    if _BRIEF_RE.search(q):
        return "brief"
    if _COMPARE_RE.search(q) or intent == "compare":
        return "compare"
    if _RANK_RE.search(q) or CONTINENTAL_COUNT_RE.search(q):
        return "rank"
    if _TREND_RE.search(q) or intent in ("monitoring", "descriptive") and ("since" in ql or "from 20" in ql):
        return "trend"
    if mode in ("fact_lookup", "data_export_only"):
        return "fact"
    if intent == "compare":
        return "compare"
    return "fact"


def compile_population(query: str) -> str:
    q = (query or "").strip()
    if _DAIRY_POP_RE.search(q):
        return "dairy"
    return ""


def compile_pathogen(query: str) -> str:
    q = (query or "").strip()
    if _ECF_RE.search(q):
        return "ecf"
    return ""


def compile_measure(
    query: str,
    decomposition: dict[str, Any] | None,
    measure_hit: MeasureHit | None,
    *,
    breakdown: list[BreakdownDim],
    matched_bundles: tuple[MatchedBundle, ...] | None = None,
) -> tuple[str, str]:
    q = (query or "").strip()
    sector = ""
    bundles = matched_bundles or ()
    bundle_measures = bundle_required_measures(bundles)
    if bundle_measures:
        mid = bundle_primary_measure(bundles, q) or bundle_measures[0]
        if mid == "employment_share":
            sector = "agriculture"
        return mid, sector
    if _PREVALENCE_RE.search(q) or _ECF_RE.search(q):
        return "disease_prevalence", sector
    if re.search(r"\b(rainfall|rainfall anomaly|precipitation)\b", q, re.I):
        return "climate", sector
    if _EMPLOYMENT_SHARE_RE.search(q) or (
        _AGRICULTURE_SECTOR_RE.search(q) and re.search(r"\b(employment|share|workers?|labou?r)\b", q, re.I)
    ):
        if _AGRICULTURE_SECTOR_RE.search(q):
            sector = "agriculture"
        return "employment_share", sector
    dec = decomposition if isinstance(decomposition, dict) else {}
    if query_locks_production(q, dec):
        return "production", sector
    if measure_hit is not None:
        mid = measure_hit.measure.id
        if bundles_block_primary(mid, bundles) and bundle_measures:
            return bundle_measures[0], sector
        if mid == "socio_economic" and (
            re.search(r"\b(employment|share\s+of)\b", q, re.I) or "sex" in breakdown
        ):
            if _AGRICULTURE_SECTOR_RE.search(q):
                sector = "agriculture"
            return "employment_share", sector
        if re.search(r"\bhdi\b", q, re.I):
            return "hdi", sector
        if re.search(r"\bgdp\b", q, re.I):
            return "gdp", sector
        return mid, sector
    raw_pm = dec.get("primary_measures")
    if isinstance(raw_pm, list) and raw_pm:
        mid = str(raw_pm[0]).strip()
        if mid in MEASURES:
            return mid, sector
    return "", sector


def _apply_clarify_if_incomplete(
    contract: TurnContract,
    measure_spec: MeasureSpec | None,
    *,
    query: str = "",
    decomposition: dict[str, Any] | None = None,
    matched_bundles: tuple[MatchedBundle, ...] | None = None,
) -> TurnContract:
    if contract.job in ("help", "social", "clarify"):
        contract.serve_status = "clarify"
        contract.plan_type = "gap"
        contract.skip_vector_retrieval = True
        return contract
    bundle_measures = bundle_required_measures(matched_bundles or ())
    multi_measure_panel = len(bundle_measures) >= 2 or contract.job in (
        "report",
        "synthesis",
        "compare",
        "list",
    )
    validation = validate_turn_slots(
        contract,
        measure_spec,
        query=query,
        decomposition=decomposition,
        multi_measure_panel=multi_measure_panel,
    )
    if validation.outcome == "clarify":
        contract.serve_status = "clarify"
        contract.serve_reason = validation.reason or "incomplete_slots"
        contract.plan_type = "gap"
        contract.job = "clarify"
        return contract
    if contract.measure_id == "employment_share" and contract.job in ("diagnose", "brief"):
        contract.job = "fact"
    if contract.measure_id == "disease_prevalence" and contract.pathogen_id == "ecf" and not contract.population:
        contract.population = "dairy"
    return contract


def compile_turn_contract(
    query: str,
    decomposition: dict[str, Any] | None,
    *,
    answer_lang: str = "en",
    measure_hit: MeasureHit | None = None,
    task_mode_hint: str = "",
    matched_bundles: tuple[MatchedBundle, ...] | None = None,
    original_question: str | None = None,
) -> TurnContract:
    dec = decomposition if isinstance(decomposition, dict) else {}
    breakdown = compile_breakdown(query)
    measure_id, sector = compile_measure(
        query,
        dec,
        measure_hit,
        breakdown=breakdown,
        matched_bundles=matched_bundles,
    )
    pathogen_id = compile_pathogen(query)
    population = compile_population(query)
    job = compile_job(query, dec, task_mode_hint=task_mode_hint)
    geo: list[str] = []
    raw_geo = dec.get("geography")
    if isinstance(raw_geo, list):
        geo = [str(g).strip() for g in raw_geo if str(g).strip()]
    entities_raw = dec.get("entities")
    entities = (
        [str(e).strip() for e in entities_raw if str(e).strip()]
        if isinstance(entities_raw, list)
        else []
    )
    time_spec = compile_time_spec(query, dec)
    geo_grain = compile_geo_grain(query, dec)
    original = query if original_question is None else original_question
    requested = explicit_measure_ids(query)
    if original != query:
        requested = list(dict.fromkeys([*requested, *explicit_measure_ids(original)]))
    explicit_comparison = bool(requested and _COMPARE_RE.search(query))
    compares_countries = (
        len(geo) >= 2 and all(name_to_iso3(place) for place in geo)
        and str(dec.get("geo_scope") or "") in ("", "country", "multi_country")
    )
    if explicit_comparison and (len(time_spec.windows) >= 2 or compares_countries):
        job = "compare"
        if compares_countries:
            geo_grain = "country"
        if measure_id not in requested:
            measure_id = requested[0]
    asks_for_documents = (
        requests_document_content(query) or asks_about_causal_effect(query)
        or _CITED_EXPLANATION_RE.search(query)
        or re.search(r"\b(?:assessments?|studies|reports|evidence)\b", query, re.I)
        or (
            re.search(r"\b(?:explain|contribut\w+|impacts?)\b", query, re.I)
            and re.search(r"\b(?:polic(?:y|ies)|program(?:me)?s?|reforms?)\b", query, re.I)
        )
    )
    topics: list[str] = []
    if asks_for_documents:
        topics_raw = dec.get("evidence_topics")
        if isinstance(topics_raw, list):
            topics = [topic.strip() for topic in topics_raw if isinstance(topic, str) and topic.strip()]
        if not topics:
            sentences = re.split(r"(?<=[.!?])\s+", query)
            topics = [sentence for sentence in sentences if _CITED_EXPLANATION_RE.search(sentence)] or [query]

    plan_type: str = "numeric"
    if job in ("diagnose", "brief", "outlook"):
        plan_type = "narrative"
    elif job in ("help", "social", "clarify"):
        plan_type = "gap"

    contract = TurnContract(
        measure_id=measure_id,
        sector=sector,
        geo=geo,
        geo_grain=geo_grain,
        time_spec=time_spec,
        job=job,
        breakdown=breakdown,
        entities=entities,
        answer_lang=answer_lang or "en",
        plan_type=plan_type,  # type: ignore[arg-type]
        pathogen_id=pathogen_id,
        population=population,
        original_question=original,
        documents=[DocumentRequirement(topic) for topic in dict.fromkeys(topics)],
    )
    contract.facts = compile_fact_requirements(contract, query, dec)
    if any(fact.enumerate_members for fact in contract.facts):
        # Suggested regional breakdowns in enrichment do not replace the
        # dimensions the user actually requested.
        contract.breakdown = compile_breakdown(original)

    measure_spec = MEASURES.get(measure_id) if measure_id else None
    if measure_hit is not None and measure_spec is None:
        measure_spec = measure_hit.measure
    contract = _apply_clarify_if_incomplete(
        contract,
        measure_spec,
        query=query,
        decomposition=dec,
        matched_bundles=matched_bundles,
    )
    return contract
