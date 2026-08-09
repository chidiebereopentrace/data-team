"""
Cheap heuristic gate for greetings and out-of-scope / non-substantive queries.

Short-circuits full RAG so social messages do not burn retrieval + generation cost,
and so prior session memory cannot steer a "hi" into an unrelated agronomy thread.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from ml.rag.chatbot.answer_language import (
    detect_answer_language,
    is_english_answer_lang,
    language_instruction,
)
from ml.rag.chatbot.assistant_identity import META_ANSWER_FOOTER
from ml.rag.chatbot.assistant_identity import is_meta_query
from ml.rag.chatbot.product_knowledge import is_product_query

SocialKind = Literal["greeting", "out_of_scope"]

# Whole-message / short social greetings and courtesy (EN + high-frequency FR/SW/Pidgin/PT/Hausa).
_GREETING_RE = re.compile(
    r"^\s*(?:"
    r"hi+|hello|hey+|howdy|yo|"
    r"good\s*(?:morning|afternoon|evening|day|night)|"
    r"greetings?|"
    r"thanks?(?:\s+you)?|thank\s*you|thx|ty|"
    r"bye+|goodbye|good\s*bye|see\s*ya|see\s*you(?:\s+later)?|"
    r"bonjour|bonsoir|salut|merci(?:\s+beaucoup)?|au\s*revoir|"
    r"jambo|habari(?:\s+yako)?|asante(?:\s+sana)?|karibu|"
    r"sannu|nagode|"
    r"olá|ola|obrigad[oa]|"
    r"how\s+far|abeg|"
    r"whats?\s*up|wass?up|sup"
    r")[\s!.?]*\s*$",
    re.IGNORECASE,
)

# Clear off-topic intents (jokes, coding help, celebrity, sports scores, etc.).
_OFF_TOPIC_RE = re.compile(
    r"(?:"
    r"\btell\s+me\s+a\s+joke\b|"
    r"\bwrite\s+(?:me\s+)?(?:some\s+)?(?:python|javascript|code|sql)\b|"
    r"\bhelp\s+me\s+(?:code|debug|program)\b|"
    r"\bwho\s+won\s+(?:the\s+)?(?:match|game|election)\b|"
    r"\b(?:football|soccer|nba|premier\s+league)\s+score\b|"
    r"\bcelebrity\b|\bhollywood\b|\bgossip\b|"
    r"\bplay\s+(?:a\s+)?game\b|"
    r"\bhoroscope\b|"
    r"\bweather\s+in\s+(?:tokyo|london|new\s+york|paris)\b"
    r")",
    re.IGNORECASE,
)

# Light agronomy / OpenTrace substance cues — if present, never treat as out-of-scope.
_AG_SUBSTANCE_RE = re.compile(
    r"(?:"
    r"\b(?:maize|corn|wheat|rice|sorghum|millet|cassava|yam|cocoa|coffee|cotton|"
    r"tea|palm\s*oil|soy(?:bean)?s?|groundnut|livestock|cattle|poultry|fish(?:eries)?|"
    r"fertiliz(?:er|ation)|irrigation|drought|rainfall|yield|harvest|crop|farm(?:er|ing)?|"
    r"agricultur(?:e|al)|agronom(?:y|ic)|food\s*security|climate|soil|seed|"
    r"subsidy|export|import|commodity|market\s*price|smallholder|"
    r"kenya|ghana|nigeria|ethiopia|tanzania|uganda|senegal|malawi|zambia|"
    r"rwanda|cameroon|ivory\s*coast|côte\s*d['']ivoire|sahel|"
    r"fao|cgiar|opentrace|ask\s*adza|ofia|bigquery|qdrant)\b"
    r")",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_GREETING_ANSWER = (
    "Hello — I am Ask ADZA, OpenTrace Africa's agricultural advisory assistant. "
    "Ask me about crops, markets, climate, policy, or other agricultural topics across Africa, "
    "and I will ground answers in OpenTrace evidence where available."
)

_OUT_OF_SCOPE_ANSWER = (
    "I am Ask ADZA, focused on African agricultural intelligence. "
    "That request is outside what I can help with here. "
    "Please ask about farming, food systems, markets, climate, or related policy and data in Africa."
)

_SOCIAL_SYSTEM = (
    "You are Ask ADZA, OpenTrace Africa's agricultural advisory assistant. "
    "Reply briefly and warmly. Do not invent agronomy facts. "
    "Do not continue any prior conversation topic. "
    "Invite the user to ask about African agriculture, markets, climate, or policy."
)


def is_greeting_query(query: str) -> bool:
    """Return True for short social greetings / courtesy messages."""
    if not query or not query.strip():
        return False
    return bool(_GREETING_RE.match(query.strip()))


def _facet_list(decomposition: dict[str, Any] | None, key: str) -> list[Any]:
    if not isinstance(decomposition, dict):
        return []
    raw = decomposition.get(key)
    return raw if isinstance(raw, list) else []


def _facets_empty(decomposition: dict[str, Any] | None) -> bool:
    return not (
        _facet_list(decomposition, "entities")
        or _facet_list(decomposition, "geography")
        or _facet_list(decomposition, "domains")
    )


def _token_count(query: str) -> int:
    return len(_TOKEN_RE.findall(query))


def is_out_of_scope_query(query: str, decomposition: dict[str, Any] | None = None) -> bool:
    """
    Conservative out-of-scope / non-substantive gate.

    Requires empty decompose facets, no ag substance cues, and either a very short
    non-greeting message or a clear off-topic pattern. Real agronomy questions must
    never match.
    """
    q = (query or "").strip()
    if not q:
        return False
    if is_greeting_query(q) or is_meta_query(q) or is_product_query(q, decomposition):
        return False
    if not _facets_empty(decomposition):
        return False
    if _AG_SUBSTANCE_RE.search(q):
        return False
    if _OFF_TOPIC_RE.search(q):
        return True
    # Bare acknowledgements / filler already covered by greeting; other ultra-short
    # non-ag text with empty facets (e.g. "lol", "hmm", "test") skip full RAG.
    if _token_count(q) <= 3 and not re.search(r"[?？]", q):
        return True
    return False


def classify_social_query(
    query: str, decomposition: dict[str, Any] | None = None
) -> SocialKind | None:
    """Return greeting | out_of_scope | None (greeting wins)."""
    if is_greeting_query(query):
        return "greeting"
    if is_out_of_scope_query(query, decomposition):
        return "out_of_scope"
    return None


def _with_footer(text: str) -> str:
    if not text:
        return text
    if "opentrace.africa" in text.lower():
        return text
    return text.rstrip() + META_ANSWER_FOOTER


def static_social_answer(kind: SocialKind) -> str:
    if kind == "greeting":
        return _with_footer(_GREETING_ANSWER)
    return _with_footer(_OUT_OF_SCOPE_ANSWER)


def generate_social_answer(
    kind: SocialKind,
    query: str,
    *,
    answer_lang: str | None = None,
) -> str:
    """
    Produce a social-gate answer with no chat-memory injection.

    English: static copy. Non-English: short LLM reply with language mirror only.
    """
    lang = answer_lang or detect_answer_language(query)
    if is_english_answer_lang(lang):
        return static_social_answer(kind)

    from ml.rag.chatbot.generator import _call_llama  # local import avoids circular deps

    intent = (
        "Greet the user briefly and invite an agriculture question."
        if kind == "greeting"
        else "Politely say this is out of scope and invite an African agriculture question."
    )
    system = _SOCIAL_SYSTEM + "\n\n" + language_instruction(lang)
    user = f"{intent}\n\nUser message: {query}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = _call_llama(messages, purpose="generate_social")
    fallback = _GREETING_ANSWER if kind == "greeting" else _OUT_OF_SCOPE_ANSWER
    answer = raw.strip() if raw else fallback
    return _with_footer(answer)


__all__ = [
    "SocialKind",
    "is_greeting_query",
    "is_out_of_scope_query",
    "classify_social_query",
    "static_social_answer",
    "generate_social_answer",
]
