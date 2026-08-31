"""Canonical agri measure ontology: aliases → slots → candidate BQ tables.

Aids the SQL reasoner (scoped prompts / fallback plans). Does not replace the LLM.
``resolve_measure()`` is a **hint API** for NL2SQL context — not turn authority.
Turn planning uses intent bundles + job compiler + plan enricher (see intent_bundles.py).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ml.rag.chatbot.mart_indicator_classes import class_for_query, facts_for_classes

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
    indicator_classes: tuple[str, ...] = ()


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
        bq_index_domains=("production", "faostat"),
        candidate_tables=(
            "fct_production",
            "agg_production_annual",
        ),
        filter_hints="production_grain='physical'; filter metric/element; country_iso3; year; never mix with fct_yield",
        crop_required=True,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
        indicator_classes=("PROD",),
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
        bq_index_domains=("production",),
        candidate_tables=("fct_yield",),
        filter_hints="fct_yield FNID-season grain; harvest_year/season_key; never fct_production for subnational yield",
        crop_required=True,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
        indicator_classes=("PROD",),
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
        bq_index_domains=("trade", "fvc"),
        candidate_tables=("fct_trade",),
        filter_hints="filter trade_grain; Import/Export — use fct_trade not fct_production",
        crop_required=True,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
        indicator_classes=("FVC",),
    ),
    "food_balance": MeasureSpec(
        id="food_balance",
        aliases=(
            "food balance",
            "food balance sheet",
            "fbs",
            "consumption",
            "domestic supply",
            "import dependency",
            "self-sufficiency",
        ),
        corpus_domains=("Agricultural Nutrition & Food Security",),
        bq_index_domains=("food_balance", "fvc"),
        candidate_tables=("fct_food_balance",),
        filter_hints="production vs imports vs consumption; country_iso3; year",
        crop_required=False,
        default_task_mode="analytical",
        recency_tier="historical_ok",
        indicator_classes=("FVC", "PROD"),
    ),
    "protected_area": MeasureSpec(
        id="protected_area",
        aliases=(
            "protected area",
            "protected areas",
            "wdpa",
            "protected planet",
            "terrestrial protected",
            "national park",
            "park coverage",
        ),
        corpus_domains=("Land Use & Soil Health", "Agricultural Environmental & Climate"),
        bq_index_domains=("protected", "biodiversity"),
        candidate_tables=("fct_protected_areas",),
        filter_hints="WDPA terrestrial protected area; never fertilizer/pesticide/machinery",
        crop_required=False,
        default_task_mode="analytical",
        recency_tier="historical_ok",
        indicator_classes=("ENV", "BIO"),
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
        bq_index_domains=("prices", "fvc"),
        candidate_tables=(
            "fct_prices",
            "agg_prices_country_month",
        ),
        filter_hints="filter price_source (fews/wfp/faostat) and price_type; country_iso3; month 1-12",
        crop_required=True,
        default_task_mode="fact_lookup",
        recency_tier="near_term",
        indicator_classes=("PRC", "FVC"),
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
        bq_index_domains=("food_security", "fews"),
        candidate_tables=(
            "fct_food_security",
            "agg_food_security_monthly",
            "fct_humanitarian",
        ),
        filter_hints="measure_type population vs classification — never union; IPC/FEWS spine",
        crop_required=False,
        default_task_mode="briefing",
        recency_tier="live",
        companions=("market_price", "production"),
        indicator_classes=("FS",),
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
        candidate_tables=("fct_climate",),
        filter_hints="filter climate_grain; JOIN dim_indicator for named series; country_iso3",
        crop_required=False,
        default_task_mode="fact_lookup",
        recency_tier="near_term",
        indicator_classes=("CLIM",),
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
            "fct_soil_health",
            "fct_land_degradation",
            "fct_land_inputs",
        ),
        filter_hints="iSDA vs ISRIC source_key; input_grain on land_inputs",
        crop_required=False,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
        indicator_classes=("SOIL", "INP"),
    ),
    "socio_economic": MeasureSpec(
        id="socio_economic",
        aliases=(
            "gdp",
            "economy",
            "economies",
            "hdi",
            "development",
            "population",
        ),
        corpus_domains=("Agricultural Economics", "Agricultural Policy & Institutional"),
        bq_index_domains=("socio_economic", "hdi"),
        candidate_tables=(
            "fct_hdi",
            "fct_economics",
            "agg_hdi_latest",
        ),
        filter_hints="GDP/HDI macro; country_iso3 + year",
        crop_required=False,
        default_task_mode="analytical",
        recency_tier="historical_ok",
        indicator_classes=("EL", "HDI"),
    ),
    "employment_share": MeasureSpec(
        id="employment_share",
        aliases=(
            "employment",
            "employment share",
            "share of employment",
            "agricultural employment",
            "employment in agriculture",
            "agricultural employment share",
            "share of workers in agriculture",
            "labour in agriculture",
            "labor in agriculture",
        ),
        corpus_domains=("Agricultural Economics",),
        bq_index_domains=("socio_economic",),
        candidate_tables=("fct_employment",),
        filter_hints="fct_employment; filter unit='%' and indicator for agriculture share; sex when requested",
        crop_required=False,
        geography_required=True,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
        indicator_classes=("EL",),
    ),
    "hdi": MeasureSpec(
        id="hdi",
        aliases=("hdi", "human development index", "human development"),
        corpus_domains=("Agricultural Economics",),
        bq_index_domains=("hdi", "socio_economic"),
        candidate_tables=("fct_hdi", "agg_hdi_latest"),
        filter_hints="country_iso3 + year",
        crop_required=False,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
        indicator_classes=("HDI",),
    ),
    "gdp": MeasureSpec(
        id="gdp",
        aliases=("gdp", "gross domestic product", "economy size"),
        corpus_domains=("Agricultural Economics",),
        bq_index_domains=("socio_economic",),
        candidate_tables=("fct_economics",),
        filter_hints="measurement_form for share vs current USD; country_iso3 + year",
        crop_required=False,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
        indicator_classes=("EL",),
    ),
    "rainfall": MeasureSpec(
        id="rainfall",
        aliases=(
            "rainfall",
            "rain",
            "precipitation",
            "rainfall anomaly",
            "climate rainfall",
        ),
        corpus_domains=("Climate & Weather",),
        bq_index_domains=("climate",),
        candidate_tables=(),
        filter_hints="National climate tables only; admin2 rainfall not in mart",
        crop_required=False,
        geography_required=True,
        default_task_mode="fact_lookup",
        recency_tier="near_term",
        indicator_classes=("CLIM",),
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
        bq_index_domains=("research",),
        candidate_tables=("fct_research_expenditure", "fct_investment", "fct_researchers"),
        filter_hints="ASTI/research expenditure; JOIN dim_indicator",
        crop_required=False,
        default_task_mode="analytical",
        recency_tier="near_term",
        indicator_classes=("RES",),
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
        bq_index_domains=("production", "socio_economic", "trade", "prices"),
        candidate_tables=(
            "fct_production",
            "fct_trade",
            "fct_economics",
            "fct_research_expenditure",
            "fct_prices",
            "fct_food_security",
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
        indicator_classes=("PROD", "EL", "FVC", "FS", "RES"),
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
        bq_index_domains=("faostat", "inputs"),
        candidate_tables=("fct_land_inputs", "fct_fertilizer", "fct_pesticide", "fct_machinery"),
        filter_hints="input_grain for fertilizer use vs trade; country_iso3",
        crop_required=False,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
        indicator_classes=("INP", "SOIL"),
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
        bq_index_domains=("emissions",),
        candidate_tables=("fct_emissions",),
        filter_hints="filter element total vs intensity; source_key",
        crop_required=False,
        default_task_mode="fact_lookup",
        recency_tier="historical_ok",
        indicator_classes=("ENV",),
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
        bq_index_domains=("livestock",),
        candidate_tables=(
            "fct_animal_health",
            "fct_food_hazards",
            "fct_insurance",
            "fct_household",
        ),
        filter_hints="ILRI/animal health mart facts; farm CDS vs study prevalence",
        crop_required=False,
        default_task_mode="research",
        recency_tier="historical_ok",
        indicator_classes=("AH",),
    ),
    "disease_prevalence": MeasureSpec(
        id="disease_prevalence",
        aliases=(
            "disease prevalence",
            "prevalence",
            "seroprevalence",
            "east coast fever",
            "ecf",
            "theileriosis",
            "theileria parva",
        ),
        corpus_domains=("Animal Health",),
        bq_index_domains=("livestock", "animal_health"),
        candidate_tables=("fct_animal_health",),
        filter_hints="household×species grain; national herd prevalence not in mart",
        crop_required=False,
        default_task_mode="research",
        recency_tier="historical_ok",
        indicator_classes=("AH",),
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
        bq_index_domains=("spatial", "vegetation"),
        candidate_tables=(
            "fct_vegetation",
            "fct_germplasm",
            "fct_biodiversity",
            "fct_protected_areas",
        ),
        filter_hints="vegetation_grain ndvi_grid vs ilri_site; occurrence vs index",
        crop_required=False,
        default_task_mode="analytical",
        recency_tier="near_term",
        indicator_classes=("VEG", "BIO"),
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
            "dim_research_project",
            "dim_organisation",
            "dim_person",
            "fct_investment",
        ),
        filter_hints="Research dims + ASTI facts; prefer PDF corpus for synthesis",
        crop_required=False,
        geography_required=False,
        default_task_mode="research",
        recency_tier="near_term",
        indicator_classes=("RES",),
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
        bq_index_domains=("production",),
        candidate_tables=("fct_production", "agg_production_annual"),
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
    "food_balance",
    "protected_area",
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
    iclasses = list(spec.indicator_classes)
    if child and child.indicator_classes:
        iclasses = list(dict.fromkeys([*child.indicator_classes, *iclasses]))
    class_tables = facts_for_classes(iclasses) if iclasses else []
    if class_tables:
        tables = list(dict.fromkeys([*tables, *[t for t in class_tables if t.startswith("fct_") or t.startswith("agg_")]]))[:8]
    return {
        "measure_id": spec.id,
        "child_measure_id": hit.child_measure_id,
        "candidate_tables": tables,
        "index_domains": domains,
        "indicator_classes": iclasses,
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
    from ml.rag.chatbot.bq_table_schema_yaml import compile_intent_for_table, pack_mart_table_hints

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
        geo_filter = "Africa continental panel GROUP BY country_iso3 (~54 countries)"
    elif countries:
        geo_filter = f"country_iso3 in ({', '.join(countries[:16])})"
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

    multi_country = africa_panel or africa_default or hit.measure.country_is_answer or len(countries) != 1

    intents: list[dict[str, Any]] = [
        compile_intent_for_table(
            primary,
            measure_id=hit.measure.id,
            query=query,
            geo_labels=countries,
            year_hint=year_hint,
            multi_country=multi_country,
            africa_panel=africa_panel or africa_default,
            time_start=ts,
            time_end=te,
            extra_filters=hints,
        )
    ]
    intents[0]["goal"] = f"{hit.measure.id} fact/rank for asked scope"
    intents[0]["notes"] = f"ontology_fallback_{hit.measure.id}"
    intents[0]["filters"] = f"{hints}; {intents[0]['filters']}"

    # Multi-signal investor / analytical composite.
    if hit.measure.id == "investor_best_country":
        intents = []
        for tid in tables[:6]:
            extra = hints
            intent = compile_intent_for_table(
                tid,
                measure_id=hit.measure.id,
                query=query,
                geo_labels=countries,
                year_hint=year_hint,
                multi_country=True,
                africa_panel=africa_panel or africa_default,
                time_start=ts,
                time_end=te,
                extra_filters=extra,
            )
            intent["goal"] = f"Investment signal from {tid}"
            intent["notes"] = f"ontology_fallback_investor_{tid}"
            intent["filters"] = f"{extra}; recent years; {intent['filters']}"
            intents.append(intent)

    # FEWS / IPC food security — spine + cross-domain companions (not production-only pad).
    if hit.measure.id == "food_security_ipc":
        ordered = [
            "fct_food_security",
            "fct_prices",
            "fct_production",
            "fct_household",
        ]
        ordered_present = [t for t in ordered if t in tables] or tables[:6]
        intents = []
        for tid in ordered_present:
            extra = hints
            if tid == "fct_prices":
                extra = f"{hints}; price_source='fews'"
            elif tid == "fct_production":
                extra = f"{hints}; production_grain='physical'"
            elif tid == "fct_food_security":
                extra = f"{hints}; measure_type='population'"
            intent = compile_intent_for_table(
                tid,
                measure_id=hit.measure.id,
                query=query,
                geo_labels=countries,
                year_hint=year_hint,
                multi_country=africa_default or africa_panel or len(countries) != 1,
                africa_panel=africa_panel or africa_default,
                time_start=ts,
                time_end=te,
                extra_filters=extra,
            )
            intent["goal"] = f"Food security signal from {tid}"
            intent["notes"] = f"ontology_fallback_food_security_{tid}"
            intents.append(intent)
        tables = ordered_present

    if task_mode == "data_export_only" or hit.measure.id == "data_export_panel":
        export_intent = compile_intent_for_table(
            primary,
            measure_id=hit.measure.id,
            query=query,
            geo_labels=countries,
            year_hint=year_hint,
            multi_country=multi_country,
            africa_panel=africa_panel or africa_default,
            time_start=ts,
            time_end=te,
            extra_filters=hints,
        )
        export_intent["goal"] = "Tabular multi-row export / full panel"
        export_intent["notes"] = "ontology_fallback_export"
        export_intent["pattern"] = "time_series"
        export_intent["order_by"] = "year ASC"
        export_intent["filters"] = f"{hints}; {export_intent['filters']}"
        intents.append(export_intent)

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
    packed, hints_truncated = pack_mart_table_hints(selected, query_terms=terms)
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
