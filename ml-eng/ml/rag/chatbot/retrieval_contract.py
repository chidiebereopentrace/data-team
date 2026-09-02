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
    compile_intent_for_table,
    list_mart_table_index,
    pack_mart_table_hints,
)
from ml.rag.chatbot.facet_enrich import enrich_decomposition_facets
from ml.rag.chatbot.intent_bundles import bundle_primary_measures, bundles_from_ids
from ml.rag.chatbot.mart_indicator_classes import class_for_query, do_not_mix_tables
from ml.rag.chatbot.sql_compiler import sql_compiler_enabled


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


def _intent_for_table(
    table_id: str,
    *,
    measure_id: str,
    geo_labels: list[str],
    year_hint: str,
    multi_country: bool,
    query: str = "",
    time_start: str = "",
    time_end: str = "",
    africa_panel: bool = False,
) -> dict[str, Any]:
    return compile_intent_for_table(
        table_id,
        measure_id=measure_id,
        query=query,
        geo_labels=geo_labels,
        year_hint=year_hint,
        multi_country=multi_country,
        africa_panel=africa_panel,
        time_start=time_start,
        time_end=time_end,
    )


def choose_agg_vs_fact(
    table_id: str,
    *,
    query: str,
    multi_country: bool,
    year_hint: str,
    single_country: bool = False,
    iso_count: int = 0,
) -> str:
    """Prefer agg_* for national annual rollups when question scope fits."""
    q = (query or "").lower()
    national = (
        multi_country
        or single_country
        or iso_count >= 1
        or "national" in q
        or "country" in q
    )
    annual = "month" not in q and "season" not in q and "fnid" not in q
    if table_id == "fct_production" and national and annual:
        if multi_country or iso_count >= 2:
            return "agg_production_country_year"
        return "agg_production_country_year"
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
    africa_panel = bool(enriched.get("africa_panel"))

    indicator_classes = class_for_query(query)
    analytical = str(enriched.get("task_mode") or "") == "analytical"
    compiler_bq_disabled = sql_compiler_enabled() and not analytical

    bq_hits = [
        h
        for h in hits
        if h.measure.candidate_tables or h.measure.bq_index_domains
    ]

    primary_ids: list[str] = []
    companion_ids: list[str] = []
    declared_pm = enriched.get("primary_measures")
    matched_bundles_raw = enriched.get("matched_bundles")
    matched_bundle_ids = (
        [str(b).strip() for b in matched_bundles_raw if str(b).strip()]
        if isinstance(matched_bundles_raw, list)
        else []
    )
    if isinstance(declared_pm, list) and declared_pm:
        primary_ids = [str(m).strip().lower() for m in declared_pm if str(m).strip()]
    elif matched_bundle_ids:
        bundles = bundles_from_ids(matched_bundle_ids)
        primary_ids = list(bundle_primary_measures(bundles, query))
    elif hits:
        primary_ids.append(hits[0].measure.id)
        declared = set(hits[0].measure.companions)
        for h in hits[1:]:
            if h.measure.id in declared or h.matched_alias.startswith("companion_of_"):
                companion_ids.append(h.measure.id)

    matched_bundles = enriched.get("matched_bundles")
    ag_activities = (
        isinstance(matched_bundles, list) and "agricultural_activities" in matched_bundles
    )

    # Prefer specialized multi-intent builder when food_security is the top activated measure.
    bq_tables: list[str] = []
    bq_intents: list[dict[str, Any]] = []
    if not compiler_bq_disabled and (
        not ag_activities
        and not enriched.get("reasoner_job")
        and hits
        and hits[0].measure.id == "food_security_ipc"
    ):
        fs = build_food_security_bq_plan(query, decomposition=enriched, known_tables=known)
        if fs is not None and not fs.get("skip_bq"):
            bq_tables = [str(t) for t in (fs.get("selected_tables") or []) if str(t).strip()]
            bq_intents = list(fs.get("query_intents") or [])

    if not compiler_bq_disabled and not bq_intents and bq_hits:
        seen_tables: set[str] = set()
        target_measures = set(primary_ids + companion_ids) if primary_ids else None
        for h in bq_hits:
            if target_measures is not None and h.measure.id not in target_measures:
                continue
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
                        geo_labels=geo,
                        year_hint=year_hint,
                        multi_country=multi,
                        query=query,
                        time_start=ts,
                        time_end=te,
                        africa_panel=africa_panel,
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

    skip_bq = bool(hits) and not bq_hits and not bq_tables

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
