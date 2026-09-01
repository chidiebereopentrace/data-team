"""Route indicator-class queries to one or more mart_dev fact/agg tables."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import known_mart_table_names
from ml.rag.chatbot.bundle_metrics import is_agri_activities_panel, is_multi_country_panel
from ml.rag.chatbot.intent_bundles import MatchedBundle, has_bundle
from ml.rag.chatbot.mart_indicator_classes import facts_for_class, families_for_class
from ml.rag.chatbot.retrieval_contract import choose_agg_vs_fact

_YIELD_RE = re.compile(r"\b(yield|season|fnid|harvest\s+season)\b", re.I)
_SHARE_RE = re.compile(
    r"\b(import\s+(share|dependency|ratio)|domestic\s+supply|self[-\s]?sufficien|food\s+balance)\b",
    re.I,
)
_TRADE_RE = re.compile(r"\b(trade|imports?|exports?)\b", re.I)
_PRICE_RE = re.compile(r"\b(price|prices|market|retail|wholesale)\b", re.I)
_MARKET_DETAIL_RE = re.compile(r"\b(market|bamako|kano|nairobi|retail|wholesale|urban)\b", re.I)

_TABLE_PREFIX_SKIP = ("dim_", "bridge_")


def _bare_table(table_id: str) -> str:
    return (table_id or "").strip().split(".")[-1].lower()


def _is_skippable_table(table_id: str) -> bool:
    bare = _bare_table(table_id)
    return not bare or bare.startswith(_TABLE_PREFIX_SKIP)


@dataclass(frozen=True)
class TablePlan:
    table_id: str
    family_id: str
    role: str  # panel | series | companion


def _card_tables(card: dict[str, Any]) -> list[str]:
    do_not_use = {_bare_table(str(t)) for t in (card.get("do_not_use") or [])}
    out: list[str] = []
    seen: set[str] = set()
    for raw in card.get("tables") or []:
        bare = _bare_table(str(raw))
        if not bare or bare in do_not_use or _is_skippable_table(bare):
            continue
        if bare not in seen:
            seen.add(bare)
            out.append(bare)
    default = _bare_table(str(card.get("default_table") or ""))
    if default and default not in seen and default not in do_not_use and not _is_skippable_table(default):
        out.insert(0, default)
    return out


def _candidate_tables(class_code: str, card: dict[str, Any]) -> list[str]:
    facts = {_bare_table(t) for t in facts_for_class(class_code)}
    known = known_mart_table_names()
    out: list[str] = []
    seen: set[str] = set()
    for tid in _card_tables(card):
        if tid not in facts or tid not in known:
            continue
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def _family_matches_query(family: dict[str, Any], query: str) -> bool:
    fid = str(family.get("id") or "")
    q = query or ""
    if fid in ("fnid_season_yield",) and _YIELD_RE.search(q):
        return True
    if fid in ("faostat_country_year_physical", "faostat_index_gross_value") and not _YIELD_RE.search(q):
        if "production" in q.lower() or "output" in q.lower() or "harvest" in q.lower():
            return fid == "faostat_country_year_physical"
    if fid == "trade_grain" and _TRADE_RE.search(q) and not _SHARE_RE.search(q):
        return True
    if fid == "food_losses" and _SHARE_RE.search(q):
        return True
    if fid == "agg_country_year" and not _YIELD_RE.search(q):
        return True
    if fid == "agg_production_country_year_panel":
        return True
    return False


def _bundle_required_measures(bundles: tuple[MatchedBundle, ...]) -> set[str]:
    out: set[str] = set()
    for b in bundles:
        for m in b.spec.required_measures:
            out.add(str(m).lower())
    return out


def select_table_plans(
    class_code: str,
    *,
    query: str,
    facets: dict[str, Any],
    bundles: tuple[MatchedBundle, ...],
    card: dict[str, Any],
    iso_list: list[str] | None = None,
) -> list[TablePlan]:
    """Pick mart tables for a class engine turn (never union unlike tables)."""
    code = class_code.upper()
    candidates = _candidate_tables(code, card)
    if not candidates:
        default = _bare_table(str(card.get("default_table") or ""))
        if default and default in known_mart_table_names():
            candidates = [default]

    iso = iso_list or []
    multi = is_multi_country_panel(iso)
    panel = is_agri_activities_panel(query, facets, bundles=bundles)
    measures = _bundle_required_measures(bundles)
    known = known_mart_table_names()

    def _plan(table_id: str, *, family_id: str = "", role: str = "series") -> TablePlan | None:
        bare = _bare_table(table_id)
        if bare not in known:
            return None
        return TablePlan(table_id=bare, family_id=family_id or bare, role=role)

    plans: list[TablePlan] = []

    if code == "PROD":
        if panel or multi:
            p = _plan("agg_production_country_year", family_id="agg_production_country_year_panel", role="panel")
            if p:
                plans.append(p)
        elif "production" in measures or "trade" not in measures:
            routed = choose_agg_vs_fact(
                "fct_production",
                query=query,
                multi_country=multi,
                year_hint=str(facets.get("time_end") or facets.get("time_start") or "")[:4],
                single_country=len(iso) == 1,
                iso_count=len(iso),
            )
            fam_id = "agg_country_year" if routed.startswith("agg_") else "faostat_country_year_physical"
            p = _plan(routed, family_id=fam_id, role="series")
            if p:
                plans.append(p)
        if not plans:
            for fam in families_for_class(code):
                if _family_matches_query(fam, query):
                    p = _plan(str(fam.get("table") or ""), family_id=str(fam.get("id") or ""))
                    if p:
                        plans.append(p)
                        break
        if not plans:
            default = _bare_table(str(card.get("default_table") or "fct_production"))
            p = _plan(default)
            if p:
                plans.append(p)
        return plans

    if code == "PRC":
        if _MARKET_DETAIL_RE.search(query) and len(iso) == 1:
            p = _plan("fct_prices", family_id="fews_market")
            if p:
                return [p]
        p = _plan("agg_prices_country_month", family_id="faostat_national")
        if p:
            return [p]
        default = _bare_table(str(card.get("default_table") or "fct_prices"))
        p = _plan(default)
        return [p] if p else []

    if code == "FVC":
        if panel or (multi and has_bundle(bundles, "agricultural_activities")):
            fb = _plan("fct_food_balance", family_id="food_losses", role="panel")
            tr = _plan("fct_trade", family_id="trade_grain", role="companion")
            if fb:
                plans.append(fb)
            if tr and ("trade" in measures or panel):
                plans.append(tr)
            return plans
        if _SHARE_RE.search(query):
            p = _plan("fct_food_balance", family_id="food_losses")
            if p:
                return [p]
        if _TRADE_RE.search(query) and not _SHARE_RE.search(query):
            p = _plan("fct_trade", family_id="trade_grain")
            if p:
                return [p]
        if _PRICE_RE.search(query):
            p = _plan("fct_prices", family_id="prices")
            if p:
                return [p]
        for fam in families_for_class(code):
            if _family_matches_query(fam, query):
                p = _plan(str(fam.get("table") or ""), family_id=str(fam.get("id") or ""))
                if p:
                    return [p]
        default = _bare_table(str(card.get("default_table") or "fct_food_balance"))
        p = _plan(default)
        return [p] if p else []

    # Generic classes: family match then default
    for fam in families_for_class(code):
        if _family_matches_query(fam, query):
            p = _plan(str(fam.get("table") or ""), family_id=str(fam.get("id") or ""))
            if p:
                return [p]
    for tid in candidates:
        p = _plan(tid)
        if p:
            return [p]
    default = _bare_table(str(card.get("default_table") or ""))
    p = _plan(default)
    return [p] if p else []


__all__ = ["TablePlan", "select_table_plans"]
