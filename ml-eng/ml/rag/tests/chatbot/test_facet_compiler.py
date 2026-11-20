"""Tests for facet_compiler job and geo_scope wiring from decompose."""
from __future__ import annotations

from ml.rag.chatbot.facet_compiler import compile_geo_grain, compile_job


def test_compile_job_prefers_model_job() -> None:
    dec = {"job": "rank", "intent": "descriptive"}
    assert compile_job("maize production in kenya 2020", dec) == "rank"


def test_compile_job_falls_back_to_regex_when_job_invalid() -> None:
    dec = {"job": "invalid_job", "intent": "descriptive"}
    assert compile_job("maize production in kenya 2020", dec) == "fact"


def test_compile_geo_grain_prefers_geo_scope_continent() -> None:
    dec = {"geo_scope": "continent", "geography": ["Kenya"]}
    assert compile_geo_grain("production in kenya", dec) == "africa"


def test_compile_geo_grain_prefers_geo_scope_country() -> None:
    dec = {"geo_scope": "country", "geography": ["Kenya"]}
    assert compile_geo_grain("production in kenya", dec) == "country"


def test_compile_geo_grain_multi_country_with_two_geos() -> None:
    dec = {"geo_scope": "multi_country", "geography": ["Kenya", "Ghana"]}
    assert compile_geo_grain("compare kenya and ghana", dec) == "country"
