"""
Decompose a user query into semantic facets for retrieval routing and filtering.

Model-first: when an LLM backend is configured, every non-empty query is decomposed via
the model with full session/plan context (see ``decompose_context``). Heuristics remain
as validators, provisional hints, and fallbacks for time ranges and geo grounding.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from typing import Any

from ml.rag.chatbot.time_resolver import (
    TimeWindow,
    parse_iso_date,
    resolve_relative_time,
    retrieval_envelope,
    utc_today,
)

from ml.rag.llm_chat import llm_chat_complete, llm_model_id
from ml.rag.chatbot.decompose_context import (
    DECOMPOSE_MEASURE_HINTS,
    DecomposeContext,
    GEO_SCOPE_ALLOWED,
    INTENT_ALLOWED,
    JOB_ALLOWED,
    format_decompose_system_prompt,
    format_decompose_user_prompt,
)
from ml.rag.chatbot.query_normalize import normalize_query_text
from ml.rag.chatbot.agri_entities import CROP_ENTITY_RE as _CROP_ENTITY_RE
from ml.rag.chatbot.agri_measure_ontology import MEASURES
from ml.rag.chatbot.continental_scope import (
    CONTINENTAL_PANEL_RE as _AFRICA_PANEL_RE,
    CONTINENTAL_RANK_RE as _RANKING_SCOPE_RE,
    wants_africa_default_scope as _continental_wants_default,
    wants_africa_panel_scope as _continental_wants_panel,
)
from ml.rag.chatbot.geo_iso3 import infer_country_iso3_from_query
from ml.rag.chatbot.geo_regions import all_non_country_geo_labels
from ml.rag.observability import trace_elapsed_ms

# Re-export for callers that imported from here historically.
_INTENT_LLM_LINES = (
    "descriptive: what happened, levels, trends, summaries from data or documents",
    "diagnostic: why, drivers, contributing factors (ground claims carefully)",
    "predictive: forward-looking (forecast, likely, outlook, scenario) — note uncertainty",
    "monitoring: ongoing watch, worsening, emerging risks, what to track",
    "compare: versus, rank, best/worst, regional or group comparison",
    "locate: where to focus, priority areas, geographic targeting",
    "decision_support: planning, funding, sourcing, investment, policy choices",
)

_INTENT_ALIASES: dict[str, str] = {
    "general": "descriptive",
    "data_analytics": "descriptive",
    "schema_lookup": "descriptive",
    "news": "descriptive",
    "academic": "descriptive",
    "prediction": "predictive",
    "forecasting": "predictive",
    "comparison": "compare",
    "ranking": "compare",
    "monitor": "monitoring",
    "tracking": "monitoring",
    "diagnosis": "diagnostic",
    "causal": "diagnostic",
    "decision": "decision_support",
    "recommendation": "decision_support",
    "location": "locate",
    "prioritization": "locate",
}

# Common country / region tokens for agriculture news (extend as needed)
_COUNTRY_ALIASES = {
    # African countries and their common aliases - full list (as of 2023)
    "algeria": "Algeria",
    "angola": "Angola",
    "benin": "Benin",
    "botswana": "Botswana",
    "burkina faso": "Burkina Faso",
    "burundi": "Burundi",
    "cabo verde": "Cabo Verde",
    "cape verde": "Cabo Verde",
    "cameroon": "Cameroon",
    "central african republic": "Central African Republic",
    "car": "Central African Republic",
    "chad": "Chad",
    "comoros": "Comoros",
    "congo": "Republic of the Congo",
    "congo-brazzaville": "Republic of the Congo",
    "congo republic": "Republic of the Congo",
    "republic of the congo": "Republic of the Congo",
    "congo-kinshasa": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",
    "democratic republic of the congo": "Democratic Republic of the Congo",
    "djibouti": "Djibouti",
    "egypt": "Egypt",
    "equatorial guinea": "Equatorial Guinea",
    "eritrea": "Eritrea",
    "eswatini": "Eswatini",
    "swaziland": "Eswatini",
    "ethiopia": "Ethiopia",
    "gabon": "Gabon",
    "gambia": "Gambia",
    "ghana": "Ghana",
    "guinea": "Guinea",
    "guinea-bissau": "Guinea-Bissau",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "côte d'ivoire": "Côte d'Ivoire",
    "kenya": "Kenya",
    "lesotho": "Lesotho",
    "liberia": "Liberia",
    "libya": "Libya",
    "madagascar": "Madagascar",
    "malawi": "Malawi",
    "mali": "Mali",
    "mauritania": "Mauritania",
    "mauritius": "Mauritius",
    "morocco": "Morocco",
    "mozambique": "Mozambique",
    "namibia": "Namibia",
    "niger": "Niger",
    "nigeria": "Nigeria",
    "rwanda": "Rwanda",
    "sao tome and principe": "Sao Tome and Principe",
    "senegal": "Senegal",
    "seychelles": "Seychelles",
    "sierra leone": "Sierra Leone",
    "somalia": "Somalia",
    "south africa": "South Africa",
    "south sudan": "South Sudan",
    "sudan": "Sudan",
    "tanzania": "Tanzania",
    "togo": "Togo",
    "tunisia": "Tunisia",
    "uganda": "Uganda",
    "zambia": "Zambia",
    "zimbabwe": "Zimbabwe",
    # Demonyms ("Nigerian markets", "Kenyan maize")
    "algerian": "Algeria",
    "angolan": "Angola",
    "beninese": "Benin",
    "burkinabe": "Burkina Faso",
    "burundian": "Burundi",
    "cameroonian": "Cameroon",
    "chadian": "Chad",
    "congolese": "Democratic Republic of the Congo",
    "egyptian": "Egypt",
    "eritrean": "Eritrea",
    "ethiopian": "Ethiopia",
    "gambian": "Gambia",
    "ghanaian": "Ghana",
    "guinean": "Guinea",
    "ivorian": "Côte d'Ivoire",
    "kenyan": "Kenya",
    "liberian": "Liberia",
    "libyan": "Libya",
    "malagasy": "Madagascar",
    "malawian": "Malawi",
    "malian": "Mali",
    "mauritanian": "Mauritania",
    "moroccan": "Morocco",
    "mozambican": "Mozambique",
    "namibian": "Namibia",
    "nigerien": "Niger",
    "nigerian": "Nigeria",
    "rwandan": "Rwanda",
    "senegalese": "Senegal",
    "sierra leonean": "Sierra Leone",
    "somali": "Somalia",
    "south african": "South Africa",
    "south sudanese": "South Sudan",
    "sudanese": "Sudan",
    "tanzanian": "Tanzania",
    "togolese": "Togo",
    "tunisian": "Tunisia",
    "ugandan": "Uganda",
    "zambian": "Zambia",
    "zimbabwean": "Zimbabwe",
}

# Continent / region / agroeco zone tokens — derived from geo_regions catalog.
_NON_COUNTRY_GEO: frozenset[str] = all_non_country_geo_labels()

# LLM often echoes these from "which country…" — never treat as geo filters.
_GEO_STOPWORDS: frozenset[str] = frozenset(
    {
        "country",
        "countries",
        "nation",
        "nations",
        "region",
        "regions",
        "world",
        "global",
        "worldwide",
        "international",
        "place",
        "places",
        "location",
        "locations",
        "area",
        "areas",
        "state",
        "states",
        "continent",
        "continents",
    }
)

_AGRI_SCOPE_RE = re.compile(
    r"\b("
    r"agricultur(?:e|al)|farming|crop|production|yield|livestock|"
    r"food\s+security|agribusiness|output"
    r")\b",
    re.IGNORECASE,
)

_DOMAIN_KEYWORDS = (
    "yield",
    "crop",
    "livestock",
    "food security",
    "drought",
    "rainfall",
    "climate",
    "soil",
    "fertilizer",
    "trade",
    "export",
    "import",
    "policy",
    "subsidy",
    "smallholder",
    "irrigation",
)


def _extract_countries(text: str) -> list[str]:
    q = text.lower()
    found: list[str] = []
    for key, canonical in sorted(_COUNTRY_ALIASES.items(), key=lambda x: -len(x[0])):
        # Word boundaries avoid matching "niger" inside "nigeria".
        if re.search(rf"\b{re.escape(key)}\b", q):
            if canonical not in found:
                found.append(canonical)
    return found


def normalize_geography_for_filter(geography: list[str] | None) -> list[str]:
    """Drop continent/region tokens and geo stopwords that break exact country filters."""
    out: list[str] = []
    for g in geography or []:
        s = str(g).strip()
        if not s:
            continue
        low = s.lower()
        if low in _NON_COUNTRY_GEO or low in _GEO_STOPWORDS:
            continue
        if s not in out:
            out.append(s)
    return out


def wants_africa_default_scope(query: str) -> bool:
    """
    True for unscoped which-country / ranking / count questions.

    OpenTrace is Africa-first: when the user does not name a country, continental
    rankings default to African agricultural intelligence.
    """
    return _continental_wants_default(query, extract_countries=bool(_extract_countries(query)))


def wants_africa_panel_scope(query: str) -> bool:
    """True when the user wants values for every African country (~54-country panel)."""
    return _continental_wants_panel(query)


def apply_africa_default_scope(decomposition: dict[str, Any], query: str) -> dict[str, Any]:
    """Annotate decomposition for Africa ranking and/or full continental panels."""
    out = dict(decomposition or {})
    panel = wants_africa_panel_scope(query)
    ranking = wants_africa_default_scope(query)
    if not panel and not ranking:
        out.pop("africa_default", None)
        out.pop("africa_panel", None)
        return out
    expanded = list(out.get("expanded_regions") or [])
    if not any(str(r).strip().lower() in ("africa", "african") for r in expanded):
        expanded.append("Africa")
    out["expanded_regions"] = expanded
    if panel:
        out["africa_panel"] = True
        # Panel is not a missing-country clarify case; keep geography empty for GROUP BY.
        out.pop("africa_default", None)
    if ranking:
        out["africa_default"] = True
        entities = list(out.get("entities") or [])
        if not any(str(e).strip().lower() == "africa" for e in entities):
            entities.append("Africa")
        out["entities"] = entities
    return out


def facet_grounded_in_query(value: str, query: str) -> bool:
    """True when a geography/entity string is evidenced in the raw user query.

    Public wrapper so downstream callers (e.g. reasoner-plan geography
    reassignment in graph.py) can re-verify a facet against the *raw* user
    query before accepting it, instead of trusting an upstream source blindly.
    """
    return _facet_grounded_in_query(value, query)


def _facet_grounded_in_query(value: str, query: str) -> bool:
    """True when a geography/entity string is evidenced in the raw user query."""
    q = (query or "").lower()
    v = (value or "").strip()
    if not v or not q:
        return False
    vl = v.lower()
    if vl in q:
        return True
    # Canonical country name → any known alias present in the query.
    for alias, canonical in _COUNTRY_ALIASES.items():
        if canonical.lower() == vl or canonical == v:
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return True
    # Region / continent labels (kept in entities; geography filter drops them later).
    if vl in _NON_COUNTRY_GEO and vl in q:
        return True
    for region in _NON_COUNTRY_GEO:
        if region == vl and re.search(rf"\b{re.escape(region)}\b", q):
            return True
    return False


def _ground_facets_in_query(
    values: list[str] | None,
    query: str,
) -> list[str]:
    out: list[str] = []
    for raw in values or []:
        s = str(raw).strip()
        if not s:
            continue
        if _facet_grounded_in_query(s, query) and s not in out:
            out.append(s)
    return out


def resolve_news_geo(*, geo_override: str, geography: list[str] | None) -> str | None:
    """Pick at most one country for news filtering; skip continents/regions."""
    countries = resolve_retrieval_geographies(geo_override=geo_override, geography=geography)
    return countries[0] if countries else None


def resolve_retrieval_geographies(
    *,
    geo_override: str,
    geography: list[str] | None,
) -> list[str]:
    """Country list for vector/BQ filters (compare queries may include 2+ countries).

    Decomposition geography wins. ``geo_override`` is used only when the query
    did not name a country.
    """
    countries = normalize_geography_for_filter(geography)
    if countries:
        return countries
    override = (geo_override or "").strip()
    if not override or override.lower() in _NON_COUNTRY_GEO:
        return []
    return [override]


def _resolution_note(windows: list[TimeWindow]) -> str:
    lines = []
    for window in windows:
        lines.append(
            f"The user's time phrase '{window.phrase}' resolves to "
            f"{window.start.isoformat()}..{window.end.isoformat()}. "
            "Use exactly these dates; do not invent other dates."
        )
    return " ".join(lines)


def _extract_year_range(
    text: str,
    ref_date: date | None = None,
) -> tuple[str | None, str | None]:
    """Single window as ISO dates. Two compared windows are not merged."""
    windows = resolve_relative_time(text or "", ref_date or utc_today())
    if len(windows) != 1:
        return None, None
    return windows[0].start.isoformat(), windows[0].end.isoformat()


def _normalize_intent(value: str | None) -> str:
    """Map free-text or legacy intent to exactly one of INTENT_ALLOWED; default descriptive."""
    if not value or not str(value).strip():
        return "descriptive"
    raw = str(value).strip().lower().replace(" ", "_")
    if raw in INTENT_ALLOWED:
        return raw
    if raw in _INTENT_ALIASES:
        return _INTENT_ALIASES[raw]
    # Substring / fuzzy: e.g. "predictive_analysis"
    return "descriptive"


def _infer_intent(text: str) -> str:
    """
    Classify query by insight type (first match wins). Order: predictive → monitoring → compare
    → locate → decision_support → diagnostic → descriptive.
    """
    tl = text.lower()
    padded = f" {tl} "

    pred_kw = (
        "likely",
        "forecast",
        "predict",
        "projection",
        "outlook",
        "scenario",
        "what if",
        "expect",
    )
    if any(k in tl for k in pred_kw) or re.search(r"\bwill\b", tl):
        return "predictive"

    mon_kw = ("monitor", "track", "worsening", "emerging", "watch", "ongoing", "alert")
    if any(k in tl for k in mon_kw):
        return "monitoring"

    if (
        "compare" in tl
        or "versus" in tl
        or " vs " in padded
        or " rank " in padded
        or re.search(r"\b(top|bottom|best|worst|highest|lowest)\b", tl)
    ):
        return "compare"

    loc_kw = (
        "where should",
        "which regions",
        "priority areas",
        "focus on",
        "where to",
        "which areas",
        "priority region",
    )
    if any(k in tl for k in loc_kw):
        return "locate"

    if (
        "should we" in tl
        or "what should" in tl
        or "recommend" in tl
        or "allocate" in tl
        or "prioritize" in tl
        or "what to prioritize" in tl
        or "policy design" in tl
        or re.search(r"\binvest(?:ing|ment)?\b", tl)
        or re.search(r"\bfund(?:ing)?\b", tl)
    ):
        return "decision_support"

    diag_kw = ("why", "because", "driver", "cause", "explain", "due to", "reason")
    if any(k in tl for k in diag_kw):
        return "diagnostic"

    desc_kw = (
        "trend",
        "changed",
        "over time",
        "how has",
        "what happened",
        "summary",
        "levels",
        "pattern",
        "show me",
        "describe",
    )
    if any(k in tl for k in desc_kw):
        return "descriptive"

    return "descriptive"


def _infer_domains(text: str) -> list[str]:
    tl = text.lower()
    out: list[str] = []
    for kw in _DOMAIN_KEYWORDS:
        if kw in tl and kw not in out:
            out.append(kw)
    return out[:8]


_BRIEFING_CUES_RE = re.compile(
    r"\b("
    r"brief(?:\s+me)?|briefing|latest|headline|headlines|what'?s\s+new|"
    r"this\s+week|this\s+month|news\s+update|quick\s+update|situation\s+update|"
    r"\bnews\b"
    r")\b",
    re.IGNORECASE,
)

_LLM_REQUIRED_INTENTS = frozenset(
    {"compare", "decision_support", "diagnostic", "predictive", "locate", "monitoring"}
)

_JOB_ALIASES: dict[str, str] = {
    "diagnostic": "diagnose",
    "monitoring": "brief",
    "decision_support": "report",
    "locate": "list",
    "descriptive": "fact",
    "predictive": "outlook",
}

_ENTITY_CANONICAL_ALIASES: dict[str, str] = {
    "corn": "maize",
    "paddy": "rice",
    "soy": "soybean",
    "groundnuts": "groundnut",
    "ground nuts": "groundnut",
    "cow peas": "cowpea",
}


def _decompose_backend_configured() -> bool:
    return bool(os.environ.get("HF_API_TOKEN") or os.environ.get("RAG_LLM_BASE_URL", "").strip())


def decompose_model_id() -> str:
    """Dedicated decompose model when set; otherwise the global chat model."""
    return (os.environ.get("RAG_DECOMPOSE_MODEL_ID") or "").strip() or llm_model_id()


def should_use_llm_decompose(query: str) -> bool:
    """
    Legacy skip gate: True when the old hybrid path would have invoked the LLM.

    Kept for tests and observability (``_legacy_would_skip_llm``). ``decompose_query``
    now calls the model whenever a backend is configured regardless of this flag.
    """
    q = (query or "").strip()
    if not q:
        return False
    if wants_africa_panel_scope(q):
        return True
    intent = _infer_intent(q)
    if intent in _LLM_REQUIRED_INTENTS:
        return True
    countries = _extract_countries(q)
    if len(countries) >= 2:
        return True
    windows = resolve_relative_time(q, utc_today())
    has_year = bool(windows)
    open_or_since = any(
        (not w.strict) or bool(re.search(r"\b(?:since|depuis|desde)\b", w.phrase, re.I))
        for w in windows
    )
    has_crop = bool(_CROP_ENTITY_RE.search(q))
    if _BRIEFING_CUES_RE.search(q):
        return False
    if has_crop and countries and has_year and intent == "descriptive" and not open_or_since:
        return False
    if (
        intent == "descriptive"
        and countries
        and (has_crop or _infer_domains(q))
        and not open_or_since
    ):
        return False
    return True


def _normalize_job(value: str | None) -> str:
    if not value or not str(value).strip():
        return ""
    raw = str(value).strip().lower().replace(" ", "_")
    if raw in JOB_ALLOWED:
        return raw
    return _JOB_ALIASES.get(raw, "")


def _normalize_geo_scope(value: str | None) -> str:
    if not value or not str(value).strip():
        return ""
    raw = str(value).strip().lower().replace(" ", "_")
    if raw in GEO_SCOPE_ALLOWED:
        return raw
    return ""


def _normalize_measure_hints(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    allowed = set(DECOMPOSE_MEASURE_HINTS) | set(MEASURES.keys())
    for item in raw[:6]:
        mid = str(item).strip().lower()
        if mid and mid in allowed and mid not in out:
            out.append(mid)
    return out


def _entity_evidence_in_query(canonical: str, query: str) -> bool:
    q = (query or "").lower()
    cl = (canonical or "").strip().lower()
    if not cl or not q:
        return False
    if _facet_grounded_in_query(canonical, query):
        return True
    for alias, canon in _ENTITY_CANONICAL_ALIASES.items():
        if canon.lower() == cl and re.search(rf"\b{re.escape(alias)}\b", q):
            return True
    return False


def _ground_entities_with_normalization(
    values: list[str] | None,
    query: str,
) -> list[str]:
    out: list[str] = []
    for raw in values or []:
        s = str(raw).strip()
        if not s:
            continue
        sl = s.lower()
        canonical = _ENTITY_CANONICAL_ALIASES.get(sl, s)
        if _entity_evidence_in_query(canonical, query):
            pick = canonical if sl in _ENTITY_CANONICAL_ALIASES else s
            if pick not in out:
                out.append(pick)
    return out


def build_provisional_decompose_hints(
    query: str,
    ref_date: date | None = None,
) -> dict[str, Any]:
    """Heuristic pre-parse passed to the decompose LLM for confirm/correct."""
    q = (query or "").strip()
    ts, te = _extract_year_range(q, ref_date)
    crops = [m.group(0).lower() for m in _CROP_ENTITY_RE.finditer(q)]
    return {
        "countries": _extract_countries(q),
        "intent_heuristic": _infer_intent(q),
        "domains": _infer_domains(q),
        "time_start": ts or "",
        "time_end": te or "",
        "crops": crops[:8],
    }


def _call_llama_decompose(
    query: str,
    *,
    context: DecomposeContext | None = None,
    provisional: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not _decompose_backend_configured():
        return None
    ctx = context or DecomposeContext.from_query(query)
    prov = provisional if provisional is not None else build_provisional_decompose_hints(query)
    system = format_decompose_system_prompt()
    user = format_decompose_user_prompt(ctx, prov)
    try:
        raw = llm_chat_complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=decompose_model_id(),
            max_tokens=int(os.environ.get("RAG_DECOMPOSE_MAX_TOKENS", "600") or 600),
            temperature=0.0,
            timeout_s=float(os.environ.get("RAG_DECOMPOSE_TIMEOUT_S", "45") or 45),
            purpose="decompose",
        )
        if not raw:
            return None
        if "```" in raw:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if m:
                raw = m.group(1).strip()
        return json.loads(raw)
    except Exception:
        return None


def _merge_llm_decompose(
    out: dict[str, Any],
    llm: dict[str, Any],
    query: str,
    *,
    ref_date: date,
) -> None:
    if isinstance(llm.get("intent"), str) and llm["intent"].strip():
        out["intent"] = llm["intent"].strip()
    job = _normalize_job(llm.get("job"))
    if job:
        out["job"] = job
    geo_scope = _normalize_geo_scope(llm.get("geo_scope"))
    if geo_scope:
        out["geo_scope"] = geo_scope
    measure_hints = _normalize_measure_hints(llm.get("primary_measure_hints"))
    if measure_hints:
        out["primary_measure_hints"] = measure_hints
    if isinstance(llm.get("entities"), list):
        out["entities"] = [str(x) for x in llm["entities"][:20]]
    if isinstance(llm.get("geography"), list) and llm["geography"]:
        geo = [str(x).strip() for x in llm["geography"] if str(x).strip()]
        for c in geo:
            if c not in out["geography"]:
                out["geography"].append(c)
    if isinstance(llm.get("domains"), list) and llm["domains"]:
        for d in llm["domains"][:10]:
            ds = str(d).strip().lower()
            if ds and ds not in out["domains"]:
                out["domains"].append(ds)
    # A window resolved from the question is exact. The model must not replace it.
    if not resolve_relative_time(query, ref_date):
        t0 = llm.get("time_start") or ""
        t1 = llm.get("time_end") or ""
        if isinstance(t0, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", t0.strip()):
            out["time_start"] = t0.strip()
        if isinstance(t1, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", t1.strip()):
            out["time_end"] = t1.strip()
    if llm.get("africa_panel") is True:
        out["africa_panel"] = True
    if llm.get("africa_default") is True:
        out["africa_default"] = True


def decompose_query(
    query: str,
    *,
    use_llm: bool = True,
    context: DecomposeContext | None = None,
    reference_date: date | None = None,
) -> dict[str, Any]:
    """
    Return facets: intent, entities, geography, domains, time_start, time_end,
    plus optional job, geo_scope, primary_measure_hints.

    Internal keys ``_decompose_llm_ms``, ``_skipped_decompose_llm``,
    ``_decompose_llm_used``, and ``_legacy_would_skip_llm`` are attached for
    observability; callers should strip them before downstream use.
    """
    q = normalize_query_text((query or "").strip())
    ctx = context or DecomposeContext.from_query(q)
    ref = reference_date or date.fromisoformat(ctx.today_iso)
    if not q:
        return {
            "intent": "descriptive",
            "entities": [],
            "geography": [],
            "domains": [],
            "time_start": "",
            "time_end": "",
            "_decompose_llm_ms": 0.0,
            "_skipped_decompose_llm": True,
            "_decompose_llm_used": False,
            "_legacy_would_skip_llm": True,
        }

    llm_attempted = bool(use_llm and _decompose_backend_configured())
    windows = resolve_relative_time(q, ref, infer_country_iso3_from_query(q))
    if windows:
        ctx.time_resolution_note = _resolution_note(windows)
    provisional = build_provisional_decompose_hints(q, ref)
    llm: dict[str, Any] | None = None
    llm_t0 = time.perf_counter()
    if llm_attempted:
        llm = _call_llama_decompose(q, context=ctx, provisional=provisional)
    decompose_llm_ms = trace_elapsed_ms(llm_t0) if llm_attempted else 0.0
    llm_used = bool(llm and isinstance(llm, dict))

    countries = _extract_countries(q)
    domains = _infer_domains(q)
    intent = _infer_intent(q)
    if len(windows) >= 2 and intent == "descriptive":
        intent = "compare"
    single = windows[0] if len(windows) == 1 else None
    envelope = retrieval_envelope(windows)
    out: dict[str, Any] = {
        "intent": intent,
        "entities": [],
        "geography": countries,
        "domains": domains,
        "time_start": envelope[0].isoformat() if envelope else "",
        "time_end": envelope[1].isoformat() if envelope else "",
        "time_windows": [window.as_dict() for window in windows],
        "time_phrase": (
            single.phrase
            if single
            else " and ".join(w.phrase for w in windows)
            if windows
            else ""
        ),
        "time_forecast": any(w.forecast for w in windows),
        "time_from_report": any(w.from_report for w in windows),
        "reference_date": ref.isoformat(),
    }
    if windows:
        out["time_source"] = "query"

    if llm_used and llm is not None:
        _merge_llm_decompose(out, llm, q, ref_date=ref)

    if not windows:
        screen_start = parse_iso_date(ctx.time_start_override)
        screen_end = parse_iso_date(ctx.time_end_override)
        if screen_start or screen_end:
            out["time_start"] = (screen_start or screen_end).isoformat()  # type: ignore[union-attr]
            out["time_end"] = (screen_end or screen_start).isoformat()  # type: ignore[union-attr]
            out["time_source"] = "ui"
            out["time_phrase"] = ""

    # Strict geography grounding; permissive entity normalization.
    out["geography"] = _ground_facets_in_query(out.get("geography"), q)
    out["entities"] = _ground_entities_with_normalization(out.get("entities"), q)
    crop_hits = [m.group(0).lower() for m in _CROP_ENTITY_RE.finditer(q)]
    if crop_hits:
        seen = {str(e).strip().lower() for e in out["entities"]}
        for crop in crop_hits:
            if crop not in seen:
                out["entities"].append(crop)
                seen.add(crop)
    out["geography"] = normalize_geography_for_filter(out.get("geography"))
    out["intent"] = _normalize_intent(out.get("intent"))
    out = apply_africa_default_scope(out, q)
    out["_decompose_llm_ms"] = decompose_llm_ms
    out["_skipped_decompose_llm"] = not llm_attempted
    out["_decompose_llm_used"] = llm_used
    out["_legacy_would_skip_llm"] = not should_use_llm_decompose(q)
    return out
