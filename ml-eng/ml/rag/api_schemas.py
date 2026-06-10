"""Shared Pydantic models for RAG API responses (query + v1 chat)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
