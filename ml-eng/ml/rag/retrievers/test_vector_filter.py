"""Unit tests for Qdrant filter building."""
from __future__ import annotations

from ml.rag.retrievers.vector_retriever import _publication_years_in_range, build_qdrant_filter


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


if __name__ == "__main__":
    test_publication_years_match_any_not_range()
    test_multi_country_geo_filter()
    print("ok")
