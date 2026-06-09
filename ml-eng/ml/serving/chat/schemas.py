from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ml.rag.api_schemas import CitationItem, UsageStats, UserProfile

StakeholderType = Literal[
    "government_public",
    "development_partners",
    "private_sector",
    "farmers_communities",
    "entrepreneurs_ecosystem",
]


class SessionCreateRequest(BaseModel):
    stakeholder_type: StakeholderType


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str
    stakeholder_type: StakeholderType


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str | None = Field(None, description="User message (alias for query)")
    query: str | None = Field(None, description="User message (alias for message)")
    session_id: str | None = None
    stakeholder_type: StakeholderType | None = Field(
        None,
        description="Deprecated bootstrap field. Prefer user_profile.stakeholder_type.",
    )
    user_profile: UserProfile | None = Field(
        None,
        description="User profile: country, stakeholder_type, audience_instructions.",
    )
    chat_history: list[ChatMessage] | None = Field(
        None,
        description="Prior turns for this request (canonical).",
    )
    conversation_history: list[ChatMessage] | None = Field(
        None,
        description="Deprecated alias for chat_history.",
    )

    @model_validator(mode="after")
    def validate_user_text_and_bootstrap(self):
        q = (self.query or "").strip()
        m = (self.message or "").strip()
        if not q and not m:
            raise ValueError("query or message is required")
        if q and m:
            raise ValueError("send query or message, not both")
        if len(q) < 1 and len(m) < 1:
            raise ValueError("query or message must be non-empty")

        sid = (self.session_id or "").strip()
        if sid and self.stakeholder_type is not None:
            raise ValueError(
                "stakeholder_type is only allowed when session_id is omitted (legacy bootstrap)"
            )
        return self

    def user_text(self) -> str:
        return (self.query or self.message or "").strip()


class ChatSuccessResponse(BaseModel):
    assistant_message: str
    citations: list[CitationItem] = Field(default_factory=list)
    session_id: str
    usage: UsageStats = Field(default_factory=lambda: UsageStats())
    request_id: str
    created_at: str


class ErrorBody(BaseModel):
    code: str
    message: str
