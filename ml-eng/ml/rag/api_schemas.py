"""Shared Pydantic models for RAG API responses (query + v1 chat)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# ACF (ADZA Confidence Framework) — surfaced on every response (Sprint 1 Wk2)
# ---------------------------------------------------------------------------

ACFBandLiteral = Literal["high", "medium", "low", "no_evidence"]


class ACFSignal(BaseModel):
    """Confidence signal attached to every RAG response."""

    band: ACFBandLiteral = Field(
        ...,
        description=(
            "Confidence band: high, medium, low, or no_evidence. "
            "Derived from retrieval quality, not LLM fluency."
        ),
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Composite confidence score (0.0–1.0).",
    )
    note: str = Field(
        ...,
        description="Plain-language explanation of the confidence level.",
    )

PlanType = Literal[
    "Free",
    "Farmers",
    "Government",
    "NGOs",
    "Agribusinesses",
    "Integrated",
]

Category = Literal[
    "Government",
    "NGOs",
    "Agribusinesses",
    "Farmers",
]


class CitationItem(BaseModel):
    id: int
    kind: str
    text: str
    url: str | None = None


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country: str | None = None
    plan_type: PlanType
    category: Category


class UsageStats(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_usage_dict(cls, raw: dict | None) -> UsageStats:
        if not raw:
            return cls()
        inp = int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0)
        out = int(raw.get("output_tokens") or raw.get("completion_tokens") or 0)
        total = int(raw.get("total_tokens") or (inp + out))
        return cls(input_tokens=inp, output_tokens=out, total_tokens=total)
