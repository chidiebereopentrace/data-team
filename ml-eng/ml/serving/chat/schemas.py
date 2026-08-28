from __future__ import annotations



from typing import Literal



from pydantic import BaseModel, ConfigDict, Field, model_validator



from ml.rag.api_schemas import ACFSignal, ArtifactItem, CitationItem, UsageStats, UserProfile
from ml.rag.session_store import session_ttl_seconds



CategoryType = Literal["Government", "NGOs", "Agribusinesses", "Farmers"]





class SessionCreateRequest(BaseModel):

    category: CategoryType





class SessionCreateResponse(BaseModel):

    session_id: str

    created_at: str

    category: CategoryType





class ChatMessage(BaseModel):

    role: str = Field(..., description="user or assistant")

    content: str = Field(..., min_length=1)





class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str | None = Field(None, description="User message (alias for query)")

    query: str | None = Field(None, description="User message (alias for message)")

    session_id: str | None = None

    user_id: str | None = Field(
        None,
        description="Optional product user id for Langfuse analytics (client-supplied until auth).",
    )

    user_profile: UserProfile | None = Field(

        None,

        description="User profile: country, plan_type, category.",

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

        return self



    def user_text(self) -> str:

        return (self.query or self.message or "").strip()





class ChatSuccessResponse(BaseModel):

    assistant_message: str

    citations: list[CitationItem] = Field(default_factory=list)

    acf: ACFSignal = Field(
        ...,
        description="ADZA Confidence Framework (Path B) signal from cited evidence.",
    )

    session_id: str

    session_found: bool = Field(
        False,
        description=(
            "True when a prior server-side session blob was loaded for this request "
            "(Redis / in-memory). False for new sessions, expired/missing ids, or "
            "when chat_history is supplied (server session is not read)."
        ),
    )

    session_ttl_seconds: int = Field(
        default_factory=session_ttl_seconds,
        description="Configured server session TTL (RAG_SESSION_TTL_SECONDS, default 604800).",
    )

    usage: UsageStats = Field(default_factory=lambda: UsageStats())

    request_id: str

    created_at: str
    plan_type: str | None = Field(
        None,
        description="The plan tier that was applied to this request (echoed for debugging).",
    )
    langfuse_trace_id: str | None = Field(
        None,
        description="Langfuse trace id when tracing is enabled (for feedback).",
    )
    artifacts: list[ArtifactItem] = Field(
        default_factory=list,
        description=(
            "Downloadable exports (CSV, chart, DOCX, PDF). Populated only on "
            "POST /v1/chat/agribusinesses and POST /v1/chat/integrated."
        ),
    )





class SessionStatusResponse(BaseModel):
    """Response for GET /v1/sessions/{session_id}."""

    session_id: str
    alive: bool
    category: str | None = None
    turn_count: int = 0
    has_summary: bool = False


class ErrorBody(BaseModel):

    code: str

    message: str


