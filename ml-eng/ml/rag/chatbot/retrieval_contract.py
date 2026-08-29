"""Decomposition-driven retrieval contract: multi-measure → tables + corpus domains."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import (
    MEASURES,
    MeasureHit,
    effective_tables,
    resolve_measures,
)
from ml.rag.chatbot.analytical_bq_plan import build_food_security_bq_plan
from ml.rag.chatbot.bq_table_schema_yaml import (
    list_mart_table_index,
    load_mart_table_schema,
    pack_mart_table_hints,
    year_column,
)
from ml.rag.chatbot.facet_enrich import enrich_decomposition_facets
from ml.rag.chatbot.mart_indicator_classes import class_for_query, do_not_mix_tables


@dataclass
class RetrievalContract:
    """Control-plane contract: what to search in BQ and which corpus domains to prefer."""

    primary_measures: list[str] = field(default_factory=list)
    companion_measures: list[str] = field(default_factory=list)
    measure_hits: list[MeasureHit] = field(default_factory=list)
    bq_tables: list[str] = field(default_factory=list)
    bq_intents: list[dict[str, Any]] = field(default_factory=list)
    corpus_domain_tags: list[str] = field(default_factory=list)
    geography: list[str] = field(default_factory=list)
    time_start: str = ""
    time_end: str = ""
    skip_bq: bool = False
    rationale: str = "contract_from_entities"

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_measures": list(self.primary_measures),
            "companion_measures": list(self.companion_measures),
            "bq_tables": list(self.bq_tables),
            "bq_intents": list(self.bq_intents),
            "corpus_domain_tags": list(self.corpus_domain_tags),
            "geography": list(self.geography),
            "time_start": self.time_start,
            "time_end": self.time_end,
            "skip_bq": self.skip_bq,
            "rationale": self.rationale,
            "measure_id": self.primary_measures[0] if self.primary_measures else None,
        }


def _geo_list(decomposition: dict[str, Any]) -> list[str]:
    raw = decomposition.get("geography")
    if not isinstance(raw, list):
        return []
    return [str(g).strip() for g in raw if str(g).strip()]


def _time_bounds_filter(
    table_id: str,
    *,
    time_start: str,
    time_end: str,
    year_hint: str,
) -> str:
    ts = (time_start or "")[:10]
    te = (time_end or "")[:10]
    if ts and te:
        schema = load_mart_table_schema(table_id)
        names = {
            str(c.get("name") or "").strip().lower()
            for c in (schema or {}).get("columns") or []
            if str(c.get("name") or "").strip()
        }
        if "as_of_date" in names:
            return f"as_of_date BETWEEN '{ts}' AND '{te}'"
        ycol = year_column(table_id) or "year"
        if ts[:4].isdigit() and te[:4].isdigit():
            return f"{ycol} BETWEEN {ts[:4]} AND {te[:4]}"
    return f"year≈{year_hint}"


def _intent_for_table(
    table_id: str,
    *,
    measure_id: str,
    geo_filter: str,
    year_hint: str,
    multi_country: bool,
    time_start: str = "",
    time_end: str = "",
) -> dict[str, Any]:
    time_filter = _time_bounds_filter(
        table_id,
        time_start=time_start,
        time_end=time_end,
        year_hint=year_hint,
    )
    filters = f"{geo_filter}; {time_filter}"
    pattern = "custom"
    grain = ["country_iso3"] if multi_country else ["country_iso3", "year"]
    order_by = "value DESC"
    if table_id == "fct_production" or "agg_production" in table_id:
        filters = f"production_grain='physical'; {filters}"
        pattern = "rank_by_sum" if multi_country else "custom"
        order_by = "total DESC" if multi_country else "value DESC"
    elif table_id == "fct_prices":
        filters = f"price_source from question; {filters}"
    elif table_id == "fct_trade":
        filters = f"trade_grain from question; {filters}"
        pattern = "rank_by_sum" if multi_country else "custom"
        order_by = "total DESC" if multi_country else "value DESC"
    elif table_id == "fct_food_security":
        filters = f"measure_type from question (population vs classification); {filters}"
    elif table_id == "fct_yield":
        filters = f"season_key/harvest_year; {filters}"
    elif table_id == "fct_employment":
        filters = f"JOIN dim_indicator; unit=% vs headcount; {filters}"
    return {
        "goal": f"{measure_id} signal from {table_id}",
        "tables": [table_id],
        "filters": filters,
        "notes": f"contract_{measure_id}_{table_id}",
        "pattern": pattern,
        "metric": "value",
        "grain": grain,
        "order_by": order_by,
    }


def choose_agg_vs_fact(
    table_id: str,
    *,
    query: str,
    multi_country: bool,
    year_hint: str,
) -> str:
    """Prefer agg_* for national annual rollups when question scope fits."""
    q = (query or "").lower()
    national = multi_country or "national" in q or "country" in q
    annual = "month" not in q and "season" not in q and "fnid" not in q
    if table_id == "fct_production" and national and annual:
        return "agg_production_annual"
    if table_id == "fct_food_security" and national and "month" in q:
        return "agg_food_security_monthly"
    if table_id == "fct_prices" and national and "market" not in q:
        return "agg_prices_country_month"
    return table_id


def _validate_table_bundle(tables: list[str], *, analytical: bool = False) -> tuple[list[str], str]:
    """Drop do-not-mix pairs unless analytical comparison mode."""
    if analytical or len(tables) < 2:
        return tables, ""
    kept: list[str] = []
    notes: list[str] = []
    for tid in tables:
        conflict = False
        for existing in kept:
            reason = do_not_mix_tables(existing, tid)
            if reason:
                notes.append(f"{existing}+{tid}: {reason}")
                conflict = True
                break
        if not conflict:
            kept.append(tid)
    return kept, "; ".join(notes)


def build_retrieval_contract(
    query: str,
    *,
    decomposition: dict[str, Any] | None,
    known_tables: set[str] | None = None,
) -> RetrievalContract:
    """
    Build a multi-measure retrieval contract from decomposition facets.

    Food-security (and any other measure) is activated only when entities/aliases
    score it — never as a global default.
    """
    known = known_tables or set()
    enriched = enrich_decomposition_facets(query, decomposition)
    hits = resolve_measures(query, enriched)
    geo = _geo_list(enriched)
    ts = str(enriched.get("time_start") or "")[:10]
    te = str(enriched.get("time_end") or "")[:10]
    year_hint = (te or ts or "")[:4] or "year from question"
    multi = len(geo) != 1
    geo_filter = (
        f"country_iso3 in ({', '.join(geo[:16])})"
        if geo
        else "geography from question (country_iso3 or join dim_geography)"
    )

    indicator_classes = class_for_query(query)
    analytical = str(enriched.get("task_mode") or "") == "analytical"

    bq_hits = [
        h
        for h in hits
        if h.measure.candidate_tables or h.measure.bq_index_domains
    ]

    primary_ids: list[str] = []
    companion_ids: list[str] = []
    if hits:
        primary_ids.append(hits[0].measure.id)
        declared = set(hits[0].measure.companions)
        for h in hits[1:]:
            if h.measure.id in declared or h.matched_alias.startswith("companion_of_"):
                companion_ids.append(h.measure.id)

    # Prefer specialized multi-intent builder when food_security is the top activated measure.
    bq_tables: list[str] = []
    bq_intents: list[dict[str, Any]] = []
    if hits and hits[0].measure.id == "food_security_ipc":
        fs = build_food_security_bq_plan(query, decomposition=enriched, known_tables=known)
        if fs is not None and not fs.get("skip_bq"):
            bq_tables = [str(t) for t in (fs.get("selected_tables") or []) if str(t).strip()]
            bq_intents = list(fs.get("query_intents") or [])

    if not bq_intents and bq_hits:
        seen_tables: set[str] = set()
        for h in bq_hits[:1]:
            for tid in effective_tables(h):
                routed = choose_agg_vs_fact(
                    tid, query=query, multi_country=multi, year_hint=year_hint
                )
                if known and routed not in known:
                    continue
                if routed in seen_tables:
                    continue
                seen_tables.add(routed)
                bq_tables.append(routed)
                bq_intents.append(
                    _intent_for_table(
                        routed,
                        measure_id=h.measure.id,
                        geo_filter=geo_filter,
                        year_hint=year_hint,
                        multi_country=multi,
                        time_start=ts,
                        time_end=te,
                    )
                )
                if len(bq_tables) >= 6:
                    break
            if len(bq_tables) >= 6:
                break

    bq_tables, mix_note = _validate_table_bundle(bq_tables, analytical=analytical)
    if mix_note and not analytical:
        bq_intents = [i for i in bq_intents if i.get("tables", [None])[0] in bq_tables]

    corpus_tags: list[str] = []
    seen_tags: set[str] = set()
    for h in hits:
        for tag in h.measure.corpus_domains:
            tl = tag.strip()
            if tl and tl.lower() not in seen_tags:
                seen_tags.add(tl.lower())
                corpus_tags.append(tl)
        if h.child_measure_id and h.child_measure_id in MEASURES:
            for tag in MEASURES[h.child_measure_id].corpus_domains:
                tl = tag.strip()
                if tl and tl.lower() not in seen_tags:
                    seen_tags.add(tl.lower())
                    corpus_tags.append(tl)
    for d in enriched.get("domains") or []:
        tl = str(d).strip()
        if tl and tl.lower() not in seen_tags:
            seen_tags.add(tl.lower())
            corpus_tags.append(tl)

    skip_bq = bool(hits) and not bq_hits

    rationale = "contract_from_entities"
    if mix_note:
        rationale = f"{rationale}; do_not_mix={mix_note[:120]}"
    if indicator_classes:
        rationale = f"{rationale}; classes={','.join(indicator_classes[:4])}"
    if primary_ids:
        rationale = f"{rationale}:{','.join(primary_ids[:4])}"

    return RetrievalContract(
        primary_measures=primary_ids,
        companion_measures=companion_ids,
        measure_hits=hits,
        bq_tables=bq_tables,
        bq_intents=bq_intents,
        corpus_domain_tags=corpus_tags,
        geography=geo,
        time_start=ts,
        time_end=te,
        skip_bq=skip_bq,
        rationale=rationale,
    )


def contract_to_bq_plan(
    contract: RetrievalContract,
    *,
    query: str,
    decomposition: dict[str, Any],
    index_truncated: bool = False,
) -> dict[str, Any] | None:
    """Materialize a BQ reasoner plan dict from a retrieval contract."""
    if contract.skip_bq or not contract.bq_tables:
        return {
            "selected_tables": [],
            "query_intents": [],
            "skip_bq": True,
            "rationale": contract.rationale,
            "measure_id": contract.primary_measures[0] if contract.primary_measures else None,
            "primary_measures": list(contract.primary_measures),
            "companion_measures": list(contract.companion_measures),
            "corpus_domain_tags": list(contract.corpus_domain_tags),
            "index_truncated": index_truncated,
            "table_hints": [],
            "hints_truncated": False,
        }
    terms: list[str] = [query[:80], *contract.primary_measures, *contract.geography[:5]]
    for e in (decomposition.get("entities") or [])[:6]:
        if str(e).strip():
            terms.append(str(e).strip())
    hints, hints_truncated = pack_mart_table_hints(contract.bq_tables, query_terms=terms)
    return {
        "selected_tables": list(contract.bq_tables),
        "query_intents": list(contract.bq_intents),
        "skip_bq": False,
        "rationale": contract.rationale,
        "measure_id": contract.primary_measures[0] if contract.primary_measures else None,
        "primary_measures": list(contract.primary_measures),
        "companion_measures": list(contract.companion_measures),
        "corpus_domain_tags": list(contract.corpus_domain_tags),
        "max_sql_queries": max(3, len(contract.bq_intents)),
        "index_truncated": index_truncated,
        "table_hints": hints,
        "hints_truncated": hints_truncated,
    }


__all__ = [
    "RetrievalContract",
    "build_retrieval_contract",
    "choose_agg_vs_fact",
    "contract_to_bq_plan",
]
