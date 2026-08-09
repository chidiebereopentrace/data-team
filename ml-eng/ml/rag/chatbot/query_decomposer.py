"""
Decompose a user query into semantic facets for retrieval routing and filtering.

Intent is stakeholder-oriented (descriptive, diagnostic, predictive, monitoring, compare,
locate, decision_support), not database- or channel-specific. Heuristics run by default;
optional LLM enrichment when HF_API_TOKEN is set. Output is JSON-serializable for graph state.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any

from ml.rag.llm_chat import llm_chat_complete, llm_model_id

# Stakeholder-oriented insight intents (not DB/channel labels). Used in heuristics, LLM prompt, and normalization.
INTENT_ALLOWED: tuple[str, ...] = (
    "descriptive",
    "diagnostic",
    "predictive",
    "monitoring",
    "compare",
    "locate",
    "decision_support",
)

# One-line definitions for the LLM (no database vocabulary).
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
}

# Continent / region tokens — not valid news geo_country filters (too broad or non-exact).
_NON_COUNTRY_GEO: frozenset[str] = frozenset(
    {
        "africa",
        "sub-saharan africa",
        "sub saharan africa",
        "east africa",
        "west africa",
        "southern africa",
        "north africa",
        "central africa",
        "african continent",
        "global",
        "worldwide",
        "international",
    }
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
    """Drop continent/region tokens that break exact country news filters."""
    out: list[str] = []
    for g in geography or []:
        s = str(g).strip()
        if not s or s.lower() in _NON_COUNTRY_GEO:
            continue
        if s not in out:
            out.append(s)
    return out


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
    """Country list for vector/BQ filters (compare queries may include 2+ countries)."""
    if geo_override.strip():
        g = geo_override.strip()
        return [] if g.lower() in _NON_COUNTRY_GEO else [g]
    return normalize_geography_for_filter(geography)


def _is_open_ended_time(text: str) -> bool:
    """True when the query asks for a range ending at the present (not a fixed year)."""
    tl = (text or "").lower()
    patterns = (
        r"\btill\s+now\b",
        r"\buntil\s+now\b",
        r"\bto\s+date\b",
        r"\bto\s+present\b",
        r"\bup\s+to\s+now\b",
        r"\bas\s+of\s+now\b",
        r"\bpresent\s+day\b",
        r"\bcurrently\b",
    )
    return any(re.search(p, tl) for p in patterns)


def _extract_since_year(text: str) -> int | None:
    """Parse 'since 2015', 'from 2013', 'starting 2020'."""
    m = re.search(
        r"\b(?:since|from|starting|after)\s+(?:the\s+)?(?:year\s+)?((?:19|20)\d{2})\b",
        (text or "").lower(),
    )
    if not m:
        return None
    return int(m.group(1))


def _extract_relative_year_range(text: str) -> tuple[str | None, str | None]:
    """Parse 'past 7 years', 'last 5 years', etc."""
    m = re.search(r"\b(?:past|last)\s+(\d{1,2})\s+years?\b", (text or "").lower())
    if not m:
        return None, None
    n = max(1, min(50, int(m.group(1))))
    end = date.today()
    start = date(end.year - n, end.month, min(end.day, 28))
    return start.isoformat(), end.isoformat()


def _extract_year_range(text: str) -> tuple[str | None, str | None]:
    """Return (start_iso, end_iso) as YYYY-MM-DD or (None, None)."""
    rel_start, rel_end = _extract_relative_year_range(text)
    if rel_start and rel_end:
        return rel_start, rel_end

    today = date.today().isoformat()
    tl = (text or "").lower()
    open_ended = _is_open_ended_time(text)
    since_y = _extract_since_year(text)

    if since_y is not None:
        start = f"{since_y}-01-01"
        end = today if open_ended else None
        if end:
            return start, end
        # "since 2015" without explicit open end still implies ongoing range for trend queries
        if re.search(r"\b(?:trend|over\s+time|how\s+has|changed|evolution|growth)\b", tl):
            return start, today

    years = [int(m.group(0)) for m in re.finditer(r"\b(?:19|20)\d{2}\b", text or "")]
    if not years:
        return None, None
    years = sorted(set(years))
    if len(years) >= 2:
        y0, y1 = years[0], years[-1]
        return f"{y0}-01-01", f"{y1}-12-31"
    y = years[0]
    if open_ended or since_y == y:
        return f"{y}-01-01", today
    if re.search(rf"\b(?:in|during|for)\s+{y}\b", tl):
        return f"{y}-01-01", f"{y}-12-31"
    # Lone year with trend language → from that year to today
    if re.search(r"\b(?:trend|since|from|over\s+time|how\s+has)\b", tl):
        return f"{y}-01-01", today
    return f"{y}-01-01", f"{y}-12-31"


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


def _call_llama_decompose(query: str) -> dict[str, Any] | None:
    model_id = llm_model_id()
    if not os.environ.get("HF_API_TOKEN") and not os.environ.get("RAG_LLM_BASE_URL", "").strip():
        return None
    intent_block = "\n".join(f"  - {line}" for line in _INTENT_LLM_LINES)
    allowed_csv = ", ".join(INTENT_ALLOWED)
    prompt = (
        "Users are government, NGOs, agribusiness, finance, and rural communities (often via programs). "
        "They rarely mention databases. Extract structured fields for retrieval and answering.\n\n"
        f"Return ONLY valid JSON with keys: intent (string, EXACTLY one of: {allowed_csv}), "
        "entities (array of strings), geography (array of country names or empty), "
        "domains (array of short topic tags), time_start (YYYY-MM-DD or empty string), "
        "time_end (YYYY-MM-DD or empty string).\n\n"
        "For time: use time_end = today's date when the question says 'till now', 'to date', "
        "'until now', or 'since YEAR' with no fixed end year. Use empty time_start/time_end only "
        "when no time period is implied.\n\n"
        "intent must be one of these values; meanings:\n"
        f"{intent_block}\n\n"
        "No markdown, no extra keys.\n\nQuestion: "
        + query
    )
    try:
        raw = llm_chat_complete(
            [{"role": "user", "content": prompt}],
            model=model_id,
            max_tokens=400,
            temperature=0.0,
            timeout_s=45,
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


def decompose_query(query: str) -> dict[str, Any]:
    """
    Return facets: intent (one of INTENT_ALLOWED), entities, geography, domains,
    time_start, time_end (ISO dates or "").
    """
    q = (query or "").strip()
    if not q:
        return {
            "intent": "descriptive",
            "entities": [],
            "geography": [],
            "domains": [],
            "time_start": "",
            "time_end": "",
        }

    llm = _call_llama_decompose(q)
    countries = _extract_countries(q)
    domains = _infer_domains(q)
    ts, te = _extract_year_range(q)
    intent = _infer_intent(q)

    out: dict[str, Any] = {
        "intent": intent,
        "entities": [],
        "geography": countries,
        "domains": domains,
        "time_start": ts or "",
        "time_end": te or "",
    }

    if llm and isinstance(llm, dict):
        if isinstance(llm.get("intent"), str) and llm["intent"].strip():
            out["intent"] = llm["intent"].strip()
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
        t0 = llm.get("time_start") or ""
        t1 = llm.get("time_end") or ""
        if isinstance(t0, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", t0.strip()):
            out["time_start"] = t0.strip()
        if isinstance(t1, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", t1.strip()):
            out["time_end"] = t1.strip()
        elif isinstance(t0, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", t0.strip()) and not str(t1).strip():
            if _is_open_ended_time(q) or _extract_since_year(q) is not None:
                out["time_end"] = date.today().isoformat()

    # Fill missing fields from heuristics (do not narrow an open-ended LLM start to one calendar year)
    if not out["time_start"] and ts:
        out["time_start"] = ts
    if not out["time_end"] and te:
        out["time_end"] = te
    elif (
        out["time_start"]
        and out["time_end"]
        and out["time_end"] == f"{out['time_start'][:4]}-12-31"
        and (_is_open_ended_time(q) or _extract_since_year(q) is not None)
    ):
        out["time_end"] = date.today().isoformat()

    # Open-ended / since-year: heuristics win over LLM stale or single-year end dates.
    if (_is_open_ended_time(q) or _extract_since_year(q) is not None) and ts and te:
        out["time_start"] = ts
        out["time_end"] = te

    # Drop LLM-hallucinated places/entities not evidenced in the user text.
    out["geography"] = _ground_facets_in_query(out.get("geography"), q)
    out["entities"] = _ground_facets_in_query(out.get("entities"), q)
    out["geography"] = normalize_geography_for_filter(out.get("geography"))
    out["intent"] = _normalize_intent(out.get("intent"))
    return out
