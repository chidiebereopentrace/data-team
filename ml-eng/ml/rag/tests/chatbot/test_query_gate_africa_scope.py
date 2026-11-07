"""Scope gates: non-African places and crop-husbandry how-to route away from full RAG."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.query_decomposer import _extract_countries
from ml.rag.chatbot.query_gate import (
    agronomy_redirect_answer,
    early_non_rag_route,
    generate_social_answer,
    is_agronomy_howto_query,
    is_non_african_scope_query,
)


@pytest.mark.parametrize(
    "query",
    [
        "What is the average olive oil price in Albania in 2023?",
        "Compare wheat prices in France and Germany for 2022.",
        "What was maize production in Iowa, USA in 2023?",
        "Average tomato prices in Spain last year?",
        "How much rice does Japan produce annually?",
        "Olive harvest yields in southern Italy 2021?",
    ],
)
def test_non_african_place_routes_out_of_scope(query: str) -> None:
    assert is_non_african_scope_query(query)
    assert early_non_rag_route(query) == "out_of_scope"


@pytest.mark.parametrize(
    "query",
    [
        "Compare maize prices in Kenya and France in 2023",
        "Average tomato price in Nigerian markets for 2023",
        "Turkey and chicken prices in Ghana",
        "What is the chile pepper price in Nigeria?",
        "What is the price of maize?",
    ],
)
def test_african_or_unscoped_query_stays_in_scope(query: str) -> None:
    assert not is_non_african_scope_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "How do I grow tomatoes in Kano?",
        "What is the right planting spacing for maize?",
        "How much fertilizer should I apply per hectare of rice?",
        "How to control fall armyworm on maize in Kenya",
        "What fertilizer schedule should I use for cassava?",
    ],
)
def test_agronomy_howto_redirects_to_market_mandate(query: str) -> None:
    assert is_agronomy_howto_query(query)
    assert early_non_rag_route(query) == "out_of_scope"
    answer = generate_social_answer("out_of_scope", query, answer_lang="en")
    assert "market conditions" in answer


@pytest.mark.parametrize(
    "query",
    [
        "Is it profitable to grow tomatoes in Kano given current prices?",
        "How do fertilizer policy notes for Nigeria interact with maize prices?",
        "When does the main rainy season start in western Kenya for planting maize?",
    ],
)
def test_market_angle_questions_are_not_redirected(query: str) -> None:
    assert not is_agronomy_howto_query(query)


def test_redirect_names_the_crop() -> None:
    assert "maize price" in agronomy_redirect_answer("What is the right planting spacing for maize?")


def test_demonyms_resolve_to_countries() -> None:
    assert _extract_countries("Average cassava price in Ghanaian markets") == ["Ghana"]
    assert _extract_countries("Nigerian tomato prices") == ["Nigeria"]
    assert _extract_countries("Kenyan maize") == ["Kenya"]
