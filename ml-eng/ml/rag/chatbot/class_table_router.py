"""Route indicator-class queries to one or more mart_dev fact/agg tables."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from ml.rag.chatbot.bq_table_schema_yaml import known_mart_table_names
from ml.rag.chatbot.bundle_metrics import is_agri_activities_panel, is_multi_country_panel
from ml.rag.chatbot.intent_bundles import MatchedBundle, has_bundle
from ml.rag.chatbot.mart_indicator_classes import (
    best_declared_family,
    companion_facts_for_class,
    facts_for_class,
)
from ml.rag.chatbot.retrieval_contract import choose_agg_vs_fact

_YIELD_RE = re.compile(r"\b(yield|season|fnid|harvest\s+season)\b", re.I)
_PRODUCT_DERIVED_SEASON_RE = re.compile(r"^(.+?)\s+season$", re.I)
_SEASON_PRODUCTION_TABLE = "agg_production_country_season"
_SHARE_RE = re.compile(
    r"\b(import\s+(share|dependency|ratio)|domestic\s+supply|self[-\s]?sufficien|food\s+balance)\b",
    re.I,
)
_TRADE_RE = re.compile(r"\b(trade|imports?|exports?)\b", re.I)
_PRICE_RE = re.compile(r"\b(price|prices|market|retail|wholesale)\b", re.I)
_MARKET_DETAIL_RE = re.compile(r"\b(market|bamako|kano|nairobi|retail|wholesale|urban)\b", re.I)

_TABLE_PREFIX_SKIP = ("dim_", "bridge_", "gold_dim_")
_MAX_COMPANIONS = 2
# A single subject token scores 10. Two subject tokens score 40.
# A full grain phrase adds 50. These floors keep the coarse branches.
_COARSE_BRANCH_SCORE = 15
_COARSE_SHARE_SCORE = 45


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


def _coarse_fvc_score(query: str) -> int:
    if _SHARE_RE.search(query or ""):
        return _COARSE_SHARE_SCORE
    if _TRADE_RE.search(query or "") or _PRICE_RE.search(query or ""):
        return _COARSE_BRANCH_SCORE
    return 0


def _declared_plan(
    code: str,
    query: str,
    *,
    plan: Callable[..., TablePlan | None],
    minimum_score: int,
) -> TablePlan | None:
    declared = best_declared_family(code, query)
    if declared is None or declared[1] <= minimum_score:
        return None
    family, _score = declared
    return plan(str(family.get("table") or ""), family_id=str(family.get("id") or ""))


def _bundle_required_measures(bundles: tuple[MatchedBundle, ...]) -> set[str]:
    out: set[str] = set()
    for b in bundles:
        for m in b.spec.required_measures:
            out.add(str(m).lower())
    return out


def _season_grain_required(query: str, facets: dict[str, Any]) -> bool:
    """True when the question needs country-season production, not a crop-named dim sample."""
    if _YIELD_RE.search(query or ""):
        return True
    roles = facets.get("entity_roles")
    if not isinstance(roles, dict):
        return False
    raw_seasons = roles.get("season")
    if not isinstance(raw_seasons, list) or not raw_seasons:
        return False
    crop_stems: set[str] = set()
    for key in ("item", "product"):
        vals = roles.get(key)
        if isinstance(vals, list):
            crop_stems.update(str(v).strip().lower() for v in vals if str(v).strip())
    ents = facets.get("entities")
    if isinstance(ents, list):
        crop_stems.update(str(e).strip().lower() for e in ents if str(e).strip())
    for raw in raw_seasons:
        label = str(raw).strip()
        if not label:
            continue
        derived = _PRODUCT_DERIVED_SEASON_RE.match(label)
        if derived:
            stem = derived.group(1).strip().lower()
            if stem and any(stem in crop or crop.startswith(stem) for crop in crop_stems):
                continue
        return True
    return False


def _with_taxonomy_companions(
    plans: list[TablePlan],
    *,
    class_code: str,
    card: dict[str, Any],
    known: set[str],
    max_companions: int = _MAX_COMPANIONS,
) -> list[TablePlan]:
    """Append up to max_companions mart fact/agg tables from companion_facts."""
    if not plans:
        return plans
    primary_ids = {p.table_id for p in plans}
    do_not_use = {_bare_table(str(t)) for t in (card.get("do_not_use") or [])}
    added: list[TablePlan] = []
    for tid in companion_facts_for_class(class_code):
        if tid in primary_ids or tid in do_not_use or tid not in known:
            continue
        if any(p.table_id == tid for p in added):
            continue
        added.append(TablePlan(table_id=tid, family_id=tid, role="companion"))
        if len(added) >= max_companions:
            break
    return list(plans) + added


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
        routed = choose_agg_vs_fact(
            bare,
            query=query,
            multi_country=multi,
            year_hint=str(facets.get("time_end") or facets.get("time_start") or "")[:4],
            single_country=len(iso) == 1,
            iso_count=len(iso),
        )
        if routed in known:
            bare = routed
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
            declared = _declared_plan(code, query, plan=_plan, minimum_score=0)
            if declared:
                plans.append(declared)
        if not plans:
            default = _bare_table(str(card.get("default_table") or "fct_production"))
            p = _plan(default)
            if p:
                plans.append(p)
        if _season_grain_required(query, facets):
            season = _plan(
                _SEASON_PRODUCTION_TABLE,
                family_id="agg_production_country_season_panel",
                role="companion",
            )
            if season and season.table_id not in {p.table_id for p in plans}:
                plans.append(season)
        return _with_taxonomy_companions(plans, class_code=code, card=card, known=known)

    if code == "PRC":
        if _MARKET_DETAIL_RE.search(query) and len(iso) == 1:
            p = _plan("fct_prices", family_id="fews_market")
            if p:
                plans = [p]
        else:
            p = _plan("agg_prices_country_month", family_id="faostat_national")
            if p:
                plans = [p]
            else:
                default = _bare_table(str(card.get("default_table") or "fct_prices"))
                p = _plan(default)
                plans = [p] if p else []
        return _with_taxonomy_companions(plans, class_code=code, card=card, known=known)

    if code == "FVC":
        # Explicit panel+trade companions; taxonomy companion_facts are dims only.
        if panel or (multi and has_bundle(bundles, "agricultural_activities")):
            fb = _plan("fct_food_balance", family_id="food_losses", role="panel")
            tr = _plan("fct_trade", family_id="trade_grain", role="companion")
            if fb:
                plans.append(fb)
            if tr and ("trade" in measures or panel):
                plans.append(tr)
            return plans
        declared = _declared_plan(code, query, plan=_plan, minimum_score=_coarse_fvc_score(query))
        if declared:
            return _with_taxonomy_companions([declared], class_code=code, card=card, known=known)
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
        default = _bare_table(str(card.get("default_table") or "fct_food_balance"))
        p = _plan(default)
        return _with_taxonomy_companions([p] if p else [], class_code=code, card=card, known=known)

    if code == "FS":
        declared = _declared_plan(code, query, plan=_plan, minimum_score=0)
        if declared:
            plans = [declared]
        if not plans:
            for tid in candidates:
                p = _plan(tid)
                if p:
                    plans = [p]
                    break
        if not plans:
            default = _bare_table(str(card.get("default_table") or "fct_food_security"))
            p = _plan(default)
            plans = [p] if p else []
        return _with_taxonomy_companions(plans, class_code=code, card=card, known=known)

    # Generic classes: the family's own terms, then the first card table.
    declared = _declared_plan(code, query, plan=_plan, minimum_score=0)
    if declared:
        return _with_taxonomy_companions([declared], class_code=code, card=card, known=known)
    for tid in candidates:
        p = _plan(tid)
        if p:
            return _with_taxonomy_companions([p], class_code=code, card=card, known=known)
    default = _bare_table(str(card.get("default_table") or ""))
    p = _plan(default)
    return _with_taxonomy_companions([p] if p else [], class_code=code, card=card, known=known)


__all__ = ["TablePlan", "select_table_plans"]
