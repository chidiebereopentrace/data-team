"""Unit tests for Qdrant filter building."""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from ml.rag.retrievers.vector_retriever import (
    VectorRetriever,
    _publication_years_in_range,
    build_qdrant_filter,
    score_metadata_constraints,
)


def test_publication_years_match_any_not_range() -> None:
    years = _publication_years_in_range("2020-01-01", "2026-05-27")
    assert years[0] == "2020"
    assert years[-1] == "2026"
    f = build_qdrant_filter(
        published_at_from="2020-01-01",
        published_at_to="2026-05-27",
        indexed_fields=frozenset({"publication_year", "doc_kind"}),
        doc_kinds=["academic_article"],
    )
    assert f is not None
    payload = f.model_dump() if hasattr(f, "model_dump") else f.dict()
    must = payload.get("must") or []
    year_conds = [m for m in must if isinstance(m, dict) and m.get("key") == "publication_year"]
    assert year_conds
    match = year_conds[0].get("match") or {}
    assert "any" in match or "Any" in str(match)


def test_multi_country_geo_filter() -> None:
    f = build_qdrant_filter(
        geo_countries=["Nigeria", "Rwanda"],
        indexed_fields=frozenset(
            {"geo_country_primary", "country", "geo_countries", "doc_kind"}
        ),
        doc_kind="news_article",
    )
    assert f is not None


def test_soft_metadata_score_signs() -> None:
    meta_ok = {
        "geo_country_primary": "Kenya",
        "published_at": "2020-06-01",
        "domains": "food_security;trade",
    }
    s = score_metadata_constraints(
        meta_ok,
        geo_list=["Kenya"],
        time_from="2020-01-01",
        time_to="2020-12-31",
        domains_substring="food_security",
    )
    assert s > 0

    meta_bad = {"geo_country_primary": "Nigeria", "published_at": "2015-01-01"}
    s_bad = score_metadata_constraints(
        meta_bad,
        geo_list=["Kenya"],
        time_from="2020-01-01",
        time_to="2020-12-31",
    )
    assert s_bad < 0


def test_metadata_passes_filters_niger_not_nigeria() -> None:
    vr = VectorRetriever.__new__(VectorRetriever)
    assert not vr._metadata_passes_filters(
        {"geo_country_primary": "Nigeria"},
        doc_kind=None,
        geo_country="Niger",
        published_at_from=None,
        published_at_to=None,
        domains_substring=None,
    )
    assert vr._metadata_passes_filters(
        {"geo_country_primary": "Niger"},
        doc_kind=None,
        geo_country="Niger",
        published_at_from=None,
        published_at_to=None,
        domains_substring=None,
    )


if __name__ == "__main__":
    test_publication_years_match_any_not_range()
    test_multi_country_geo_filter()
    print("ok")
