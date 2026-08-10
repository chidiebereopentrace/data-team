"""Unit tests for plan_type tier gates."""

from __future__ import annotations

from ml.rag.chatbot.plan_policy import (
    allows_cross_country,
    apply_category_domain_hints,
    apply_plan_decomposition_gates,
    instruction_for_category,
    is_valid_category,
    is_valid_plan_type,
    plan_generation_addendum,
)


def test_valid_enums() -> None:
    assert is_valid_plan_type("Free")
    assert is_valid_plan_type("Integrated")
    assert is_valid_category("NGOs")
    assert not is_valid_plan_type("enterprise")
    assert not is_valid_category("Entrepreneurs")


def test_cross_country_only_agribusiness_and_integrated() -> None:
    assert not allows_cross_country("Free")
    assert not allows_cross_country("Government")
    assert allows_cross_country("Agribusinesses")
    assert allows_cross_country("Integrated")


def test_export_only_agribusiness_and_integrated() -> None:
    from ml.rag.chatbot.plan_policy import allows_export

    assert allows_export("Agribusinesses")
    assert allows_export("Integrated")
    assert not allows_export("NGOs")


def test_decomposition_clamps_geography_for_government() -> None:
    dec = {
        "intent": "compare",
        "geography": ["Nigeria", "Ghana"],
        "time_start": "2010",
        "time_end": "2024",
    }
    out = apply_plan_decomposition_gates(dec, "Government", "Nigeria")
    assert out["geography"] == ["Nigeria"]
    assert out["intent"] == "descriptive"


def test_decomposition_allows_multi_country_for_agribusiness() -> None:
    dec = {"intent": "compare", "geography": ["Nigeria", "Ghana"]}
    out = apply_plan_decomposition_gates(dec, "Agribusinesses", None)
    assert out["geography"] == ["Nigeria", "Ghana"]
    assert out["intent"] == "compare"


def test_free_prefers_profile_country() -> None:
    dec = {"intent": "compare", "geography": ["Nigeria", "Ghana"]}
    out = apply_plan_decomposition_gates(dec, "Free", "Ghana")
    assert out["geography"] == ["Ghana"]
    assert out["intent"] == "descriptive"


def test_category_instruction_nonempty() -> None:
    assert "government" in instruction_for_category("Government").lower()
    assert "plain" in instruction_for_category("Farmers").lower()


def test_category_domain_soft_fill() -> None:
    filled = apply_category_domain_hints({"domains": []}, "NGOs")
    assert "climate" in filled["domains"]
    kept = apply_category_domain_hints({"domains": ["prices"]}, "NGOs")
    assert kept["domains"] == ["prices"]


def test_plan_addendum_free_is_brief() -> None:
    text = plan_generation_addendum("Free")
    assert "concise" in text.lower() or "top-line" in text.lower()
