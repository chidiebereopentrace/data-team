"""Research is a paid plan and an independently selectable audience perspective."""
import pytest
from ml.rag.api_schemas import UserProfile
from ml.rag.app.api import app
from ml.rag.chatbot.plan_policy import PLAN_ROUTE_SLUGS, allows_export, default_category_for_plan
from ml.rag.chatbot.stakeholder_prompts import CATEGORIES, instruction_for_category, prose_register_for_persona
from ml.rag.request_context import resolve_request_context
from ml.rag.chatbot.plan_policy import model_for_plan, allows_cross_country
from ml.rag.chatbot.reasoner_plan import is_heavy_plan_type
from ml.rag.chatbot.stakeholder_prompts import resolve_effective_category


def test_research_profile_route_and_perspective_are_registered():
    profile = UserProfile(plan_type="Research", category="Research")
    assert PLAN_ROUTE_SLUGS["research"] == "Research"
    assert "/query/research" in {getattr(route, "path", None) for route in app.routes}
    assert allows_export(profile.plan_type)
    assert default_category_for_plan("Research") == "Research"
    assert next(c["label"] for c in CATEGORIES if c["id"] == "Research") == "Research & Academia"
    register = prose_register_for_persona("Research")
    assert register and register.allow_markdown_tables and register.prefer_inline_source_refs
    assert "method" in instruction_for_category("Research").lower()


def test_research_perspective_does_not_change_free_route_permissions():
    context = resolve_request_context(
        user_profile=UserProfile(plan_type="Research", category="Research"), injected_plan_type="Free",
    )
    assert context.category == "Research" and context.plan_type == "Free"
    assert not allows_export(context.plan_type)


def test_research_gets_analytical_planning_and_shared_model(monkeypatch):
    assert is_heavy_plan_type("Research")
    assert allows_cross_country("Research")
    monkeypatch.setenv("RAG_LLM_MODEL_RESEARCH", "provider/research-model")
    assert model_for_plan("Research") == "provider/research-model"
    assert resolve_effective_category(category=None, plan_type="Research", query="What does the evidence show?") == ("Research", "plan_type")


@pytest.mark.parametrize("plan", ["Free", "Integrated", "Research"])
def test_explicit_research_perspective_reaches_generation(plan):
    from ml.rag.chatbot.generation_plan import build_generation_plan

    result = build_generation_plan("Compare crop evidence", task_mode="analytical", plan_type=plan, category="Research")
    assert result.effective_category == "Research" and result.category_source == "explicit"
    from ml.rag.chatbot.generation_prompt import _build_prompt

    messages = _build_prompt("Compare crop evidence", "Verified context", category="Research", plan_type=plan,
                             task_mode="analytical", inline_citations=True, generation_plan=result.to_dict())
    system = messages[0]["content"]
    assert "Audience: Research & Academia" in system and "Vocabulary: academic" in system
    assert "do not invent publications" in system
    if plan == "Free":
        assert "Free explorer" in system
