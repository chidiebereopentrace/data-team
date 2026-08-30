"""Deterministic empty-answer templates for unsupported / no-SQL turns."""
from __future__ import annotations

from ml.rag.chatbot.output_format import render_insufficient
from ml.rag.chatbot.turn_contract import TurnContract


def empty_answer_for_contract(
    contract: TurnContract,
    *,
    query: str = "",
    academic_count: int = 0,
    category: str = "",
) -> str | None:
    """Return a template answer when pack is empty and LLM should not invent numbers."""
    if contract.serve_status == "clarify":
        return None
    if contract.vector_policy == "fallback_only" and academic_count > 0:
        return None
    return render_insufficient(
        contract,
        query=query,
        academic_count=academic_count,
        category=category,
    )
