"""Shared Pydantic models for RAG API responses (query + v1 chat)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# ACF (ADZA Confidence Framework) — Path B (0–100)
# ---------------------------------------------------------------------------

ACFBandLiteral = Literal[
    "very_strong",
    "strong",
    "moderate",
    "limited",
    "low",
    "no_evidence",
]


class ACFSignal(BaseModel):
    """Confidence signal attached to every RAG response (Path B)."""

    band: ACFBandLiteral = Field(
        ...,
        description=(
            "Confidence band from open-trace ACF: very_strong, strong, moderate, "
            "limited, low, or no_evidence."
        ),
    )
    band_label: str = Field(
        ...,
        description="Human-readable band label (e.g. 'Strong confidence').",
    )
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Composite confidence score (0–100).",
    )
    explanation: str = Field(
        ...,
        description="One-sentence rationale for the score.",
    )
    note: str | None = Field(
        None,
        description="Alias of explanation (backward compatible).",
    )
    components: dict[str, Any] | None = Field(
        None,
        description="Optional T/A/F/G component breakdown.",
    )
    applied_ceiling: str | None = Field(
        None,
        description="Ceiling safeguard applied, if any.",
    )
    config_version: str | None = Field(
        None,
        description="ACF config version used for scoring.",
    )
    claim_level: str | None = None
    question_type: str | None = None


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
