"""Canonical agri measure ontology: aliases → slots → candidate BQ tables.

Aids the SQL reasoner (scoped prompts / fallback plans). Does not replace the LLM.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

RecencyTier = Literal["live", "near_term", "historical_ok", "point_in_time"]
TaskModeName = Literal[
    "clarify",
    "analytical",
    "fact_lookup",
    "briefing",
    "data_export_only",
    "research",
    "chat",
]


@dataclass(frozen=True)
class MeasureSpec:
    id: str
    aliases: tuple[str, ...]
    corpus_domains: tuple[str, ...]
    bq_index_domains: tuple[str, ...]
    candidate_tables: tuple[str, ...]
    filter_hints: str = ""
    crop_required: bool = False
    geography_required: bool = True
    country_is_answer: bool = False
    default_task_mode: TaskModeName = "fact_lookup"
    recency_tier: RecencyTier = "historical_ok"
    companions: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class MeasureHit:
    measure: MeasureSpec
    score: int
    matched_alias: str = ""
    child_measure_id: str | None = None  # for data_export_panel wrapping


MEASURES: dict[str, MeasureSpec] = {
    "production": MeasureSpec(
        id="production",
        aliases=(
            "production",
            "output",
            "harvest volume",
            "tonnes produced",
            "produced",
            "produces",
            "produce the most",
        ),
        corpus_domains=("Agricultural Production & Yield", "agriculture"),
        bq_index_domains=("faostat", "production"),
        candidate_tables=(
            "stg_faostat_production",
            "stg_faostat_food_balances",
            "stg_yield_raw_data",
        ),
        filter_hints="element='Production'; filter product_name/item; country_name; year",
        crop_required=True,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
    ),
    "yield": MeasureSpec(
        id="yield",
        aliases=(
            "yield",
            "yields",
            "productivity",
            "t/ha",
            "tonnes per hectare",
            "tons per hectare",
        ),
        corpus_domains=("Agricultural Production & Yield",),
        bq_index_domains=("faostat", "production"),
        candidate_tables=("stg_faostat_production", "stg_yield_raw_data"),
        filter_hints="element='Yield' (NOT Production); product_name; country_name; year",
        crop_required=True,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
    ),
    "trade": MeasureSpec(
        id="trade",
        aliases=(
            "export",
            "exports",
            "import",
            "imports",
            "trade",
            "traded",
            "export volume",
            "import volume",
        ),
        corpus_domains=("Agricultural International Trade (Exports & Imports)",),
        bq_index_domains=("faostat", "fews"),
        candidate_tables=("stg_faostat_trade", "stg_fews_cross_border_trade"),
        filter_hints="use trade tables; Import/Export quantity or value — never stg_faostat_production for export",
        crop_required=True,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
    ),
    "market_price": MeasureSpec(
        id="market_price",
        aliases=(
            "price",
            "prices",
            "retail price",
            "retail prices",
            "market price",
            "farm gate",
            "commodity price",
            "producer price",
        ),
        corpus_domains=(
            "Agricultural Economics",
            "Agricultural Food Systems & Value Chain",
            "Agricultural Market Access & Infrastructure",
        ),
        bq_index_domains=("fews", "market_prices", "faostat"),
        candidate_tables=(
            "stg_fews_market_prices",
            "stg_wfp_vampire_prices",
            "stg_faostat_prices",
        ),
        filter_hints="retail→stg_fews_market_prices price_type=Retail; producer→stg_faostat_prices",
        crop_required=True,
        default_task_mode="fact_lookup",
        recency_tier="near_term",
    ),
    "food_security_ipc": MeasureSpec(
        id="food_security_ipc",
        aliases=(
            "ipc",
            "food insecurity",
            "food security",
            "food security phase",
            "famine risk",
            "crisis phase",
            "humanitarian food",
        ),
        corpus_domains=(
            "Agricultural Nutrition & Food Security",
            "Agricultural Humanitarian & Agricultural Emergency",
        ),
        bq_index_domains=("fews", "ilri", "faostat"),
        candidate_tables=(
            "stg_fews_food_security",
            "stg_fews_market_prices",
            "stg_faostat_production",
            "stg_ilri_household_food_security",
        ),
        filter_hints="IPC / FEWS spine; companions: retail prices, staple production, ILRI; crop not required",
        crop_required=False,
        default_task_mode="briefing",
        recency_tier="live",
        companions=("market_price", "production"),
    ),
    "climate": MeasureSpec(
        id="climate",
        aliases=(
            "rainfall",
            "rainy season",
            "precipitation",
            "temperature",
            "drought",
            "weather",
            "climate",
        ),
        corpus_domains=("Agricultural Environmental & Climate",),
        bq_index_domains=("climate",),
        candidate_tables=(
            "stg_nasa_power",
            "stg_copernicus_era5",
            "stg_climatewatch_health",
        ),
        filter_hints="climate/weather tables; geography required; crop optional",
        crop_required=False,
        default_task_mode="fact_lookup",
        recency_tier="near_term",
    ),
    "soil": MeasureSpec(
        id="soil",
        aliases=(
            "soil",
            "organic matter",
            "fertility",
            "soil health",
            "nutrients",
            "degradation",
            "soil organic",
        ),
        corpus_domains=("Land Use & Soil Health",),
        bq_index_domains=("soil_and_land",),
        candidate_tables=(
            "stg_isric_africa_soil",
            "stg_isda_soil_enriched",
            "stg_cifor_icraf",
            "stg_unccd_land_degradation",
            "stg_s4a_field_surveys",
        ),
        filter_hints="ISRIC/iSDA soil; crop not required",
        crop_required=False,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
    ),
    "socio_economic": MeasureSpec(
        id="socio_economic",
        aliases=(
            "gdp",
            "economy",
            "economies",
            "hdi",
            "development",
            "employment",
            "population",
        ),
        corpus_domains=("Agricultural Economics", "Agricultural Policy & Institutional"),
        bq_index_domains=("socio_economic", "faostat"),
        candidate_tables=(
            "stg_africa_gdp_ppp",
            "stg_africa_hdi",
            "stg_faostat_sdg_hdi",
            "stg_faostat_population_employment",
        ),
        filter_hints="GDP/HDI/population; crop not required",
        crop_required=False,
        default_task_mode="analytical",
        recency_tier="historical_ok",
    ),
    "investment": MeasureSpec(
        id="investment",
        aliases=(
            "asti",
            "r&d spend",
            "agri finance",
            "agricultural research spending",
            "investment in agriculture research",
        ),
        corpus_domains=(
            "Agricultural Investment Readiness & Enterprise",
            "Agricultural Technology & Innovation",
        ),
        bq_index_domains=("faostat",),
        candidate_tables=("stg_faostat_investment_asti",),
        filter_hints="ASTI investment; crop optional",
        crop_required=False,
        default_task_mode="analytical",
        recency_tier="near_term",
    ),
    "investor_best_country": MeasureSpec(
        id="investor_best_country",
        aliases=(
            "best country for agricultural investment",
            "best african country for agricultural investment",
            "best country for agri investment",
            "agri investment attractiveness",
            "where to invest in agri",
            "where to invest in agriculture",
            "agricultural investment",
            "best for agricultural investment",
            "best for agri investment",
        ),
        corpus_domains=(
            "Agricultural Investment Readiness & Enterprise",
            "Agricultural Economics",
            "Agricultural Production & Yield",
            "Agricultural International Trade (Exports & Imports)",
            "Agricultural Market Access & Infrastructure",
        ),
        bq_index_domains=("faostat", "socio_economic", "fews"),
        candidate_tables=(
            "stg_faostat_production",
            "stg_faostat_trade",
            "stg_africa_gdp_ppp",
            "stg_faostat_investment_asti",
            "stg_fews_market_prices",
            "stg_fews_food_security",
        ),
        filter_hints=(
            "multi-signal continental country ranking; crop optional; "
            "country is the answer — do not ask which country"
        ),
        crop_required=False,
        geography_required=False,
        country_is_answer=True,
        default_task_mode="analytical",
        recency_tier="near_term",
        companions=(
            "production",
            "trade",
            "socio_economic",
            "investment",
            "market_price",
            "food_security_ipc",
        ),
        notes="Composite investment attractiveness across African countries",
    ),
    "land_inputs": MeasureSpec(
        id="land_inputs",
        aliases=(
            "fertilizer",
            "land use",
            "cropland",
            "agricultural land",
            "inputs",
        ),
        corpus_domains=("Land Use & Soil Health", "Agricultural Production & Yield"),
        bq_index_domains=("faostat",),
        candidate_tables=("stg_faostat_land_inputs",),
        filter_hints="land/fertilizer inputs",
        crop_required=False,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
    ),
    "emissions": MeasureSpec(
        id="emissions",
        aliases=(
            "emissions",
            "ghg",
            "greenhouse gas",
            "carbon from agriculture",
        ),
        corpus_domains=("Agricultural Environmental & Climate",),
        bq_index_domains=("faostat",),
        candidate_tables=("stg_faostat_emissions",),
        filter_hints="FAOSTAT emissions",
        crop_required=False,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
    ),
    "livestock": MeasureSpec(
        id="livestock",
        aliases=(
            "livestock",
            "dairy",
            "animal health",
            "feed",
            "livestock insurance",
            "cattle",
            "poultry",
        ),
        corpus_domains=(
            "agriculture",
            "Agricultural Nutrition & Food Security",
            "Agricultural Food Systems & Value Chain",
        ),
        bq_index_domains=("ilri", "production"),
        candidate_tables=(
            "stg_ilri_animal_health",
            "stg_ilri_dairy_genetics",
            "stg_ilri_vegetation_feed",
            "stg_ilri_i4i_livestock_insurance",
            "stg_ilri_household_food_security",
            "stg_ilri_food_hazards",
            "stg_ilri_vendor_consumer",
            "stg_ilri_other_surveys",
        ),
        filter_hints="ILRI livestock suite; crop not required",
        crop_required=False,
        default_task_mode="research",
        recency_tier="historical_ok",
    ),
    "spatial_vegetation": MeasureSpec(
        id="spatial_vegetation",
        aliases=(
            "ndvi",
            "vegetation",
            "remote sensing",
            "germplasm",
            "biodiversity",
            "protected areas",
        ),
        corpus_domains=(
            "Agricultural Technology & Innovation",
            "Land Use & Soil Health",
            "Agricultural Environmental & Climate",
        ),
        bq_index_domains=("spatial",),
        candidate_tables=(
            "stg_vegetation_ndvi",
            "stg_germplasm",
            "stg_biodiversity",
            "stg_protected_areas",
        ),
        filter_hints="spatial/NDVI/biodiversity",
        crop_required=False,
        default_task_mode="analytical",
        recency_tier="near_term",
    ),
    "research_meta": MeasureSpec(
        id="research_meta",
        aliases=(
            "research projects",
            "publications",
            "organisations",
            "openaire",
            "who studies",
        ),
        corpus_domains=(
            "Agricultural Technology & Innovation",
            "Agricultural Policy & Institutional",
        ),
        bq_index_domains=("research",),
        candidate_tables=(
            "stg_openaire_projects",
            "stg_openaire_organisations",
            "stg_openaire_persons",
            "stg_openaire_product_links",
            "stg_openaire_data_sources",
        ),
        filter_hints="OpenAIRE bibliographic/project facts; prefer PDF corpus for synthesis",
        crop_required=False,
        geography_required=False,
        default_task_mode="research",
        recency_tier="near_term",
    ),
    "news_briefing": MeasureSpec(
        id="news_briefing",
        aliases=(
            "news",
            "latest news",
            "briefing",
            "headlines",
            "what's new",
            "situation update",
        ),
        corpus_domains=("agriculture",),
        bq_index_domains=(),
        candidate_tables=(),
        filter_hints="corpus-first news briefing; optional thin BQ companions",
        crop_required=False,
        geography_required=False,
        default_task_mode="briefing",
        recency_tier="live",
    ),
    "research_synthesis": MeasureSpec(
        id="research_synthesis",
        aliases=(
            "what does research say",
            "literature on",
            "according to research",
            "studies show",
            "academic research",
        ),
        corpus_domains=("agriculture",),
        bq_index_domains=(),
        candidate_tables=(),
        filter_hints="corpus PDF/research synthesis; BQ optional",
        crop_required=False,
        geography_required=False,
        default_task_mode="research",
        recency_tier="near_term",
    ),
    "data_export_panel": MeasureSpec(
        id="data_export_panel",
        aliases=(
            "copy",
            "numbers only",
            "table of",
            "just the numbers",
            "data only",
            "all african countries",
            "by african country",
            "every african country",
        ),
        corpus_domains=("Agricultural Production & Yield",),
        bq_index_domains=("faostat",),
        candidate_tables=("stg_faostat_production",),
        filter_hints="full African country panel export; inherits child measure tables when resolved",
        crop_required=False,
        geography_required=False,
        country_is_answer=False,
        default_task_mode="data_export_only",
        recency_tier="historical_ok",
        notes="Wrapper — resolve child measure for tables/filters",
    ),
}

# Prefer longer / more specific measures when scoring aliases.
_PRIORITY_IDS: tuple[str, ...] = (
    "investor_best_country",
    "data_export_panel",
    "research_synthesis",
    "news_briefing",
    "food_security_ipc",
    "market_price",
    "spatial_vegetation",
    "research_meta",
    "socio_economic",
    "land_inputs",
    "emissions",
    "livestock",
    "climate",
    "soil",
    "investment",
    "trade",
    "yield",
    "production",
)

_LIVE_RE = re.compile(
    r"\b(right\s+now|currently|today|this\s+week|latest|lately|recent(?:ly)?)\b",
    re.IGNORECASE,
)
_PANEL_RE = re.compile(
    r"\b("
    r"all\s+african\s+countr(?:y|ies)|"
    r"every\s+african\s+countr(?:y|ies)|"
    r"by\s+african\s+countr(?:y|ies)|"
    r"across\s+(?:all\s+)?african\s+countr(?:y|ies)|"
    r"for\s+all\s+african\s+countr(?:y|ies)|"
    r"african\s+countr(?:y|ies)\s+panel"
    r")\b",
    re.IGNORECASE,
)
_COPY_PANEL_RE = re.compile(
    r"\b("
    r"copy|numbers?\s+only|just\s+(the\s+)?(numbers?|data|table)|"
    r"data\s+only|table\s+of|give\s+me\s+(the\s+)?(numbers?|data)"
    r")\b",
    re.IGNORECASE,
)


def wants_africa_panel(query: str) -> bool:
    """Full ~54-country panel (values per country), not which-country ranking."""
    return bool(_PANEL_RE.search(query or ""))


def wants_data_export_panel(query: str) -> bool:
    q = query or ""
    return bool(_COPY_PANEL_RE.search(q) or wants_africa_panel(q))


def get_measure(measure_id: str) -> MeasureSpec | None:
    return MEASURES.get(measure_id)


def _alias_score(query_lower: str, alias: str) -> int:
    a = alias.lower().strip()
    if not a:
        return 0
    if " " in a or "/" in a:
        if a in query_lower:
            return 10 + len(a)
        return 0
    if re.search(rf"\b{re.escape(a)}\b", query_lower):
        return 8 + min(len(a), 12)
    return 0


def _score_measure(
    mid: str,
    *,
    query_lower: str,
    entity_blob: str,
    domain_blob: str,
) -> MeasureHit | None:
    if mid not in MEASURES:
        return None
    if mid in ("data_export_panel", "investor_best_country"):
        return None
    spec = MEASURES[mid]
    score = 0
    matched = ""
    for alias in spec.aliases:
        s = _alias_score(query_lower, alias)
        if s > score:
            score = s
            matched = alias
    for alias in spec.aliases:
        al = alias.lower()
        if al and al in entity_blob:
            score = max(score, 6 + min(len(al), 8))
            matched = matched or alias
    for tag in spec.corpus_domains:
        tl = tag.lower()
        if tl and (tl in domain_blob or tl in query_lower):
            score = max(score, 5)
            matched = matched or tag
        # Partial token overlap for long domain labels.
        for token in re.findall(r"[a-z]{4,}", tl):
            if token in entity_blob or token in domain_blob or token in query_lower:
                score = max(score, 4)
                matched = matched or tag
                break
    if score <= 0:
        return None
    return MeasureHit(spec, score=score, matched_alias=matched)


def _investor_forced_hit(query_lower: str) -> MeasureHit | None:
    if any(
        phrase in query_lower
        for phrase in (
            "best country for agricultural investment",
            "best african country for agricultural investment",
            "best country for agri investment",
            "agri investment attractiveness",
            "where to invest in agri",
            "where to invest in agriculture",
            "best for agricultural investment",
            "best for agri investment",
        )
    ) or (
        "investment" in query_lower
        and re.search(r"\b(best|which)\b", query_lower)
        and re.search(r"\b(country|countries|african)\b", query_lower)
        and re.search(r"\b(agri|agricultur)", query_lower)
    ):
        return MeasureHit(MEASURES["investor_best_country"], score=100, matched_alias="investment")
    return None


def resolve_measures(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    top_k: int | None = None,
) -> list[MeasureHit]:
    """
    Multi-label measure resolution from query + decomposition entities/domains.

    Returns up to ``top_k`` scored hits (default 3, env ``RAG_MEASURE_TOP_K``).
    Companion measure ids declared on the primary hit are appended when scorable
    or always included at a soft floor score so spines stay multi-domain.
    """
    q = (query or "").strip()
    if not q:
        return []
    ql = q.lower()
    dec = decomposition if isinstance(decomposition, dict) else {}
    try:
        k = int(top_k) if top_k is not None else int(os.environ.get("RAG_MEASURE_TOP_K", "3") or 3)
    except ValueError:
        k = 3
    k = max(1, min(k, 6))

    forced = _investor_forced_hit(ql)
    if forced is not None:
        return [forced]

    raw_entities = dec.get("entities")
    entities: list[Any] = raw_entities if isinstance(raw_entities, list) else []
    entity_blob = " ".join(str(e).lower() for e in entities)
    raw_domains = dec.get("domains")
    domains: list[Any] = raw_domains if isinstance(raw_domains, list) else []
    domain_blob = " ".join(str(d).lower() for d in domains)

    scored: list[MeasureHit] = []
    for mid in _PRIORITY_IDS:
        hit = _score_measure(mid, query_lower=ql, entity_blob=entity_blob, domain_blob=domain_blob)
        if hit is not None:
            scored.append(hit)
    scored.sort(key=lambda h: (-h.score, _PRIORITY_IDS.index(h.measure.id) if h.measure.id in _PRIORITY_IDS else 99))

    if wants_data_export_panel(q) and scored:
        child = scored[0]
        panel = MEASURES["data_export_panel"]
        return [
            MeasureHit(
                panel,
                score=max(child.score, 20),
                matched_alias="africa panel" if wants_africa_panel(q) else "export panel",
                child_measure_id=child.measure.id,
            )
        ]
    if wants_africa_panel(q) and not scored:
        fallback_id = "yield" if re.search(r"\byields?\b", ql) else "production"
        return [
            MeasureHit(
                MEASURES["data_export_panel"],
                score=15,
                matched_alias="all african countries",
                child_measure_id=fallback_id,
            )
        ]

    primary = scored[:k]
    if not primary:
        return []

    # Attach declared companions of the primary measure (multi-domain spine).
    out: list[MeasureHit] = list(primary)
    seen = {h.measure.id for h in out}
    for companion_id in primary[0].measure.companions:
        if companion_id in seen or companion_id not in MEASURES:
            continue
        c_hit = _score_measure(
            companion_id, query_lower=ql, entity_blob=entity_blob, domain_blob=domain_blob
        )
        if c_hit is None:
            c_hit = MeasureHit(
                MEASURES[companion_id],
                score=3,
                matched_alias=f"companion_of_{primary[0].measure.id}",
            )
        out.append(c_hit)
        seen.add(companion_id)
    return out[: max(k, len(out))]


def resolve_measure(
    query: str,
    decomposition: dict[str, Any] | None = None,
) -> MeasureHit | None:
    """Best single measure (compatibility wrapper around ``resolve_measures``)."""
    hits = resolve_measures(query, decomposition, top_k=1)
    return hits[0] if hits else None


def effective_tables(hit: MeasureHit) -> list[str]:
    if hit.child_measure_id and hit.child_measure_id in MEASURES:
        return list(MEASURES[hit.child_measure_id].candidate_tables)
    return list(hit.measure.candidate_tables)


def effective_filter_hints(hit: MeasureHit) -> str:
    parts: list[str] = []
    if hit.child_measure_id and hit.child_measure_id in MEASURES:
        child = MEASURES[hit.child_measure_id]
        parts.append(child.filter_hints)
        parts.append(f"child_measure={child.id}")
    parts.append(hit.measure.filter_hints)
    return "; ".join(p for p in parts if p)


def effective_crop_required(hit: MeasureHit) -> bool:
    if hit.child_measure_id and hit.child_measure_id in MEASURES:
        return MEASURES[hit.child_measure_id].crop_required
    return hit.measure.crop_required


def resolve_recency_tier(query: str, hit: MeasureHit | None) -> RecencyTier:
    base: RecencyTier = hit.measure.recency_tier if hit else "historical_ok"
    if _LIVE_RE.search(query or ""):
        if base in ("historical_ok", "point_in_time"):
            return "near_term"
        return "live" if base == "live" else "near_term"
    return base


def reasoner_scope(hit: MeasureHit) -> dict[str, Any]:
    """Shrink reasoner prompt: candidate tables, domains, filters, slot policy."""
    spec = hit.measure
    child = MEASURES.get(hit.child_measure_id or "")
    tables = effective_tables(hit)
    domains = list(spec.bq_index_domains)
    if child:
        domains = list(dict.fromkeys([*child.bq_index_domains, *domains]))
    return {
        "measure_id": spec.id,
        "child_measure_id": hit.child_measure_id,
        "candidate_tables": tables,
        "index_domains": domains,
        "filter_hints": effective_filter_hints(hit),
        "crop_required": effective_crop_required(hit),
        "geography_required": spec.geography_required,
        "country_is_answer": spec.country_is_answer,
        "task_mode": spec.default_task_mode if spec.id != "data_export_panel" else "data_export_only",
        "recency_tier": spec.recency_tier,
        "companions": list(spec.companions),
        "notes": spec.notes or (child.notes if child else ""),
    }


def fallback_plan(
    hit: MeasureHit,
    *,
    query: str,
    decomposition: dict[str, Any],
    known_tables: set[str],
    task_mode: str = "fact_lookup",
) -> dict[str, Any] | None:
    """Last-resort forced plan from MeasureSpec after reasoner retries fail."""
    from ml.rag.chatbot.bq_table_schema_yaml import pack_selected_table_hints

    tables = [t for t in effective_tables(hit) if t in known_tables]
    if not tables and hit.measure.candidate_tables:
        return None
    if not tables:
        # Corpus-only measures: skip BQ honestly.
        return {
            "selected_tables": [],
            "query_intents": [],
            "skip_bq": True,
            "rationale": f"ontology_fallback_corpus_only_{hit.measure.id}",
            "measure_id": hit.measure.id,
            "task_mode": task_mode,
        }

    raw_geo = decomposition.get("geography")
    geo: list[Any] = raw_geo if isinstance(raw_geo, list) else []
    countries = [str(g).strip() for g in geo if str(g).strip()]
    africa_panel = bool(decomposition.get("africa_panel")) or wants_africa_panel(query)
    africa_default = bool(decomposition.get("africa_default"))
    if africa_panel or (hit.measure.country_is_answer and not countries):
        geo_filter = "Africa continental panel GROUP BY country_name (~54 countries)"
    elif countries:
        geo_filter = f"country_name in ({', '.join(countries[:16])})"
    else:
        geo_filter = "geography from question"

    ts = str(decomposition.get("time_start") or "")[:10]
    te = str(decomposition.get("time_end") or "")[:10]
    year_hint = (te or ts or "")[:4] or "year from question"
    raw_entities = decomposition.get("entities")
    entities: list[Any] = raw_entities if isinstance(raw_entities, list) else []
    item_hint = ", ".join(str(e).strip() for e in entities[:4] if str(e).strip()) or "entity from question"
    hints = effective_filter_hints(hit)
    primary = tables[0]

    intents: list[dict[str, Any]] = [
        {
            "goal": f"{hit.measure.id} fact/rank for asked scope",
            "tables": [primary],
            "filters": f"{hints}; {geo_filter}; year≈{year_hint}; item≈{item_hint}",
            "notes": f"ontology_fallback_{hit.measure.id}",
            "pattern": (
                "rank_by_sum"
                if (africa_default or africa_panel or hit.measure.country_is_answer or len(countries) != 1)
                else "custom"
            ),
            "metric": "value",
            "grain": ["country_name"]
            if (africa_default or africa_panel or hit.measure.country_is_answer or len(countries) != 1)
            else ["country_name", "year"],
            "order_by": "value DESC",
        }
    ]

    # Multi-signal investor / analytical composite.
    if hit.measure.id == "investor_best_country":
        intents = []
        for tid in tables[:6]:
            intents.append(
                {
                    "goal": f"Investment signal from {tid}",
                    "tables": [tid],
                    "filters": f"{geo_filter}; recent years; {hints}",
                    "notes": f"ontology_fallback_investor_{tid}",
                    "pattern": "rank_by_sum" if "faostat_production" in tid or "trade" in tid else "custom",
                    "metric": "value",
                    "grain": ["country_name"],
                    "order_by": "value DESC",
                }
            )

    # FEWS / IPC food security — spine + cross-domain companions (not production-only pad).
    if hit.measure.id == "food_security_ipc":
        ordered = [
            "stg_fews_food_security",
            "stg_fews_market_prices",
            "stg_faostat_production",
            "stg_ilri_household_food_security",
        ]
        ordered_present = [t for t in ordered if t in tables] or tables[:6]
        intents = []
        for tid in ordered_present:
            intents.append(
                {
                    "goal": f"Food security signal from {tid}",
                    "tables": [tid],
                    "filters": (
                        f"{hints}; {geo_filter}; year≈{year_hint}"
                        + ("; price_type='Retail'" if "market_prices" in tid else "")
                        + (
                            "; element='Production'"
                            if "faostat_production" in tid
                            else ""
                        )
                    ),
                    "notes": f"ontology_fallback_food_security_{tid}",
                    "pattern": (
                        "rank_by_sum"
                        if "faostat_production" in tid
                        and (africa_default or africa_panel or len(countries) != 1)
                        else "custom"
                    ),
                    "metric": "value",
                    "grain": ["country_name"]
                    if (africa_default or africa_panel or len(countries) != 1)
                    else ["country_name", "year"],
                    "order_by": "value DESC",
                }
            )
        tables = ordered_present

    if task_mode == "data_export_only" or hit.measure.id == "data_export_panel":
        intents.append(
            {
                "goal": "Tabular multi-row export / full panel",
                "tables": [primary],
                "filters": f"{hints}; {geo_filter}; year around {year_hint}",
                "notes": "ontology_fallback_export",
                "pattern": "custom",
                "metric": "value",
                "grain": ["country_name", "year"],
                "order_by": "year ASC",
            }
        )

    selected = tables[:6]
    plan: dict[str, Any] = {
        "selected_tables": selected,
        "query_intents": intents,
        "skip_bq": False,
        "rationale": f"ontology_fallback_{hit.measure.id}",
        "measure_id": hit.measure.id,
        "child_measure_id": hit.child_measure_id,
        "task_mode": task_mode,
        "max_sql_queries": max(3, len(intents)),
    }
    terms = [query[:80], item_hint, *countries[:5], hit.measure.id]
    packed, hints_truncated = pack_selected_table_hints(selected, query_terms=terms)
    plan["table_hints"] = packed
    plan["index_truncated"] = False
    plan["hints_truncated"] = hints_truncated
    return plan


__all__ = [
    "MEASURES",
    "MeasureHit",
    "MeasureSpec",
    "effective_crop_required",
    "effective_filter_hints",
    "effective_tables",
    "fallback_plan",
    "get_measure",
    "reasoner_scope",
    "resolve_measure",
    "resolve_measures",
    "resolve_recency_tier",
    "wants_africa_panel",
    "wants_data_export_panel",
]
