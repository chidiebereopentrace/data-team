"""Persona inference, prose register, and outline reframing."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.stakeholder_prompts import (
    format_outline_for_persona,
    infer_category_from_query,
    prose_register_addendum,
    prose_register_for_persona,
    resolve_effective_category,
)


def test_infer_farmers_from_plain_language() -> None:
    assert infer_category_from_query("What should I plant this season on my farm?") == "Farmers"


def test_infer_government_from_policy_language() -> None:
    q = "National budget IPC indicators and YoY maize production by sub-national region"
    assert infer_category_from_query(q) == "Government"


def test_infer_ngo_from_program_language() -> None:
    q = "Which districts show humanitarian overlap for our field program targeting?"
    assert infer_category_from_query(q) == "NGOs"


def test_resolve_explicit_category_wins() -> None:
    cat, src = resolve_effective_category(
        category="Government",
        plan_type="Farmers",
        query="what should I plant on my farm",
    )
    assert cat == "Government"
    assert src == "explicit"


def test_resolve_query_when_no_explicit() -> None:
    cat, src = resolve_effective_category(
        category="",
        plan_type="Integrated",
        query="What should I plant this season on my cooperative farm?",
    )
    assert cat == "Farmers"
    assert src == "query"


def test_resolve_plan_type_fallback() -> None:
    cat, src = resolve_effective_category(
        category="",
        plan_type="NGOs",
        query="maize production Ghana",
    )
    assert cat == "NGOs"
    assert src == "plan_type"


@pytest.mark.parametrize(
    ("category", "needle"),
    [
        ("Government", "markdown tables"),
        ("Farmers", "Do not use markdown tables"),
        ("NGOs", "operational"),
        ("Agribusinesses", "market"),
    ],
)
def test_prose_register_addendum_per_persona(category: str, needle: str) -> None:
    block = prose_register_addendum(category, task_mode="analytical", answer_shape="comparison")
    assert needle.lower() in block.lower()


def test_farmers_register_uses_bullet_layout() -> None:
    reg = prose_register_for_persona("Farmers")
    assert reg is not None
    assert reg.use_bullet_layout is True
    block = prose_register_addendum("Farmers", task_mode="analytical")
    assert "bullet" in block.lower()


def test_format_outline_for_farmers_bullets() -> None:
    sections = [{"title": "Key findings", "guidance": "Lead", "optional": False}]
    _, use_bullets = format_outline_for_persona("Farmers", sections)
    assert use_bullets is True


def test_format_outline_renames_government_titles() -> None:
    sections = [{"title": "Key findings", "guidance": "Lead", "optional": False}]
    out, _ = format_outline_for_persona("Government", sections)
    assert out[0]["title"] == "Planning takeaway"
