"""Shared Pydantic models for RAG API responses (query + v1 chat)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    id: int
    kind: str
    text: str
    url: str | None = None


class UserProfile(BaseModel):
    country: str | None = None
    stakeholder_type: str | None = None
    audience_instructions: str | None = Field(None, max_length=4000)


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_tokens: int = Field(default=0, description="Alias for prompt_tokens")
    output_tokens: int = Field(default=0, description="Alias for completion_tokens")

    @classmethod
    def from_usage_dict(cls, raw: dict | None) -> UsageStats:
        if not raw:
            return cls()
        prompt = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
        completion = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
        total = int(raw.get("total_tokens") or (prompt + completion))
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            input_tokens=prompt,
            output_tokens=completion,
        )
