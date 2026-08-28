"""Exhaustive coverage: countries, agricultural products, and domain labels."""
from __future__ import annotations

import re

import pytest

from ml.rag.chatbot.bq_sql_templates import _CROP_ALIASES
from ml.rag.chatbot.facet_enrich import _ENTITY_EXPAND, enrich_decomposition_facets
from ml.rag.chatbot.query_decomposer import _COUNTRY_ALIASES, _extract_countries
from ml.rag.chatbot.retrieval_contract import build_retrieval_contract
from ml.rag.text_processors.domain_taxonomy import DOMAIN_KEYWORDS

_D_PROD = "Agricultural Production & Yield"

_LIVESTOCK_FISH_ENTITIES = frozenset(
    {
        "livestock",
        "cattle",
        "goats",
        "sheep",
        "camels",
        "donkeys",
        "horses",
        "pigs",
        "poultry",
        "dairy",
        "beekeeping",
        "pastoralism",
        "feed",
        "animal health",
        "aquaculture",
        "fisheries",
        "tilapia",
        "catfish",
        "nile perch",
        "shrimp",
        "seaweed",
    }
)

_KNOWN_TABLES = {
    "stg_fews_food_security",
    "stg_fews_market_prices",
    "stg_faostat_production",
    "stg_ilri_household_food_security",
    "stg_faostat_trade",
    "stg_faostat_investment_asti",
}

_SMOKE_COUNTRIES = ("Kenya", "Nigeria", "Ethiopia", "Mali", "South Africa")
_SMOKE_STAPLES = (
    "maize",
    "rice",
    "wheat",
    "cassava",
    "sorghum",
    "coffee",
    "cocoa",
    "livestock",
)

_CROP_ALIAS_EXPECTED_ENTITY = {
    "maize": "maize",
    "corn": "maize",
    "rice": "rice",
    "wheat": "wheat",
    "millet": "millet",
    "sorghum": "sorghum",
    "cassava": "cassava",
    "yam": "yam",
    "soy": "soybean",
    "groundnut": "groundnut",
    "peanut": "groundnut",
}


def _id_trigger(trigger: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", trigger.strip().lower()).strip("_")
    return (cleaned or "empty")[:60]


def _product_rows() -> list[tuple[str, str, str | None]]:
    rows: list[tuple[str, str, str | None]] = []
    for trigger, entity, domain in _ENTITY_EXPAND:
        if domain == _D_PROD or entity.lower() in _LIVESTOCK_FISH_ENTITIES:
            rows.append((trigger, entity, domain))
    return rows


def _domain_representative_triggers() -> list[tuple[str, str]]:
    """One (domain, trigger) pair per taxonomy domain."""
    by_domain: dict[str, str] = {}
    for trigger, _entity, domain in _ENTITY_EXPAND:
        if domain and domain not in by_domain:
            by_domain[domain] = trigger
    return [(domain, trigger) for domain, trigger in sorted(by_domain.items())]


# ---------------------------------------------------------------------------
# Domain fidelity
# ---------------------------------------------------------------------------


def test_expand_domains_subset_of_taxonomy() -> None:
    expand_domains = {d for _, _, d in _ENTITY_EXPAND if d}
    assert expand_domains <= set(DOMAIN_KEYWORDS.keys())


def test_every_taxonomy_domain_has_expand_row() -> None:
    covered = {d for _, _, d in _ENTITY_EXPAND if d}
    missing = sorted(set(DOMAIN_KEYWORDS.keys()) - covered)
    assert missing == [], f"DOMAIN_KEYWORDS lacking _ENTITY_EXPAND coverage: {missing}"


# ---------------------------------------------------------------------------
# All _ENTITY_EXPAND rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trigger,entity,domain",
    _ENTITY_EXPAND,
    ids=[_id_trigger(t) for t, _, _ in _ENTITY_EXPAND],
)
def test_every_expand_row_grounds(trigger: str, entity: str, domain: str | None) -> None:
    out = enrich_decomposition_facets(trigger, {"entities": [], "domains": []})
    ent_l = {e.lower() for e in out["entities"]}
    assert entity.lower() in ent_l, f"trigger={trigger!r} missing entity={entity!r}"
    if domain:
        dom_l = {d.lower() for d in out["domains"]}
        assert domain.lower() in dom_l, f"trigger={trigger!r} missing domain={domain!r}"


@pytest.mark.parametrize(
    "trigger,entity,domain",
    _product_rows(),
    ids=[_id_trigger(t) for t, _, _ in _product_rows()],
)
def test_agricultural_product_rows(trigger: str, entity: str, domain: str | None) -> None:
    out = enrich_decomposition_facets(trigger, {"entities": [], "domains": []})
    assert entity.lower() in {e.lower() for e in out["entities"]}
    if domain:
        assert domain.lower() in {d.lower() for d in out["domains"]}


@pytest.mark.parametrize(
    "alias,expected_entity",
    list(_CROP_ALIAS_EXPECTED_ENTITY.items()),
    ids=list(_CROP_ALIAS_EXPECTED_ENTITY.keys()),
)
def test_bq_crop_alias_bridge(alias: str, expected_entity: str) -> None:
    assert any(a == alias for a, _ in _CROP_ALIASES), f"alias {alias!r} not in _CROP_ALIASES"
    out = enrich_decomposition_facets(alias, {"entities": [], "domains": []})
    assert expected_entity.lower() in {e.lower() for e in out["entities"]}
    assert _D_PROD.lower() in {d.lower() for d in out["domains"]}


# ---------------------------------------------------------------------------
# All country aliases
# ---------------------------------------------------------------------------


_COUNTRY_ALIAS_CASES = sorted(_COUNTRY_ALIASES.items(), key=lambda x: (-len(x[0]), x[0]))
_CANONICAL_COUNTRIES = sorted(set(_COUNTRY_ALIASES.values()))


@pytest.mark.parametrize(
    "alias,canonical",
    _COUNTRY_ALIAS_CASES,
    ids=[_id_trigger(a) for a, _ in _COUNTRY_ALIAS_CASES],
)
def test_every_country_alias_extracts(alias: str, canonical: str) -> None:
    found = _extract_countries(f"maize production in {alias}")
    assert canonical in found, f"alias={alias!r} expected {canonical!r} in {found!r}"


@pytest.mark.parametrize(
    "canonical",
    _CANONICAL_COUNTRIES,
    ids=[_id_trigger(c) for c in _CANONICAL_COUNTRIES],
)
def test_every_canonical_country_extracts(canonical: str) -> None:
    found = _extract_countries(f"agriculture in {canonical}")
    assert canonical in found
    assert found, f"empty geography for {canonical!r}"


def test_nigeria_does_not_match_niger() -> None:
    assert _extract_countries("agriculture in nigeria") == ["Nigeria"]
    assert "Niger" not in _extract_countries("products nigeria produces")


def test_canonical_country_count() -> None:
    # AU / African states present in alias table (unique canonicals).
    assert len(_CANONICAL_COUNTRIES) >= 50


# ---------------------------------------------------------------------------
# Sparse retrieval-contract smoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("country", _SMOKE_COUNTRIES)
@pytest.mark.parametrize("staple", _SMOKE_STAPLES)
def test_contract_smoke_country_staple(country: str, staple: str) -> None:
    q = f"{staple} yield in {country}"
    dec = {
        "geography": [country],
        "entities": [staple],
        "domains": [],
        "time_end": "2020-12-31",
    }
    enriched = enrich_decomposition_facets(q, dec)
    contract = build_retrieval_contract(q, decomposition=enriched, known_tables=_KNOWN_TABLES)
    assert contract is not None
    if staple == "livestock":
        assert "livestock" in contract.primary_measures or not contract.skip_bq
        return
    assert not contract.skip_bq
    assert set(contract.primary_measures) & {"yield", "production", "livestock"}


@pytest.mark.parametrize(
    "domain,trigger",
    _domain_representative_triggers(),
    ids=[_id_trigger(d) for d, _ in _domain_representative_triggers()],
)
def test_contract_smoke_per_domain(domain: str, trigger: str) -> None:
    q = f"{trigger} in Kenya"
    enriched = enrich_decomposition_facets(
        q,
        {"geography": ["Kenya"], "entities": [], "domains": []},
    )
    assert domain.lower() in {d.lower() for d in enriched["domains"]}
    contract = build_retrieval_contract(q, decomposition=enriched, known_tables=_KNOWN_TABLES)
    assert contract is not None
    # Briefing / weak-signal domains may skip BQ; only require no raise.
    _ = contract.primary_measures
    _ = contract.bq_tables
