"""English enrich shadow for African / local-language Ask ADZA queries.

Used for decompose, BQ bind, and a second vector retrieve probe.
Generation and answer_lang always stay on the original user text.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Final

from ml.rag.chatbot.answer_language import detect_answer_language, is_english_answer_lang
from ml.rag.llm_chat import llm_chat_complete, llm_model_id

logger = logging.getLogger(__name__)

# Named local languages that get an English enrich shadow when the flag is on.
ENRICHABLE_LANGS: Final[frozenset[str]] = frozenset(
    {
        "ig",
        "yo",
        "ha",
        "sw",
        "fr",
        "pcm",
        "pt",
        "tw",
        "efi",
        "zu",
        "xh",
        "so",
        "wo",
        "rw",
        "ar",
        "am",
    }
)

_CACHE_TTL_S = 3600.0
_CACHE_MAX = 256
_enrich_cache: dict[str, tuple[float, str]] = {}

_SYSTEM = (
    "You rewrite African agricultural questions into one clear English sentence "
    "for warehouse lookup and document retrieval. "
    "Preserve crop, country/place, years/time span, and measure intent. "
    "Expand well-known local crop names only when confident "
    "(e.g. akpu → cassava, mahindi → maize). "
    "Do not invent years, places, or crops that are not in the input. "
    "Output only the English sentence — no quotes, no preamble, no JSON."
)


@dataclass(frozen=True)
class LanguageEnrichment:
    answer_lang: str
    needs_enrich: bool
    english_query: str
    enrich_source: str  # none | llm | cache


def lang_enrich_enabled() -> bool:
    return os.environ.get("RAG_LANG_ENRICH", "").strip().lower() in ("1", "true", "yes", "on")


def needs_language_enrich(answer_lang: str) -> bool:
    if not lang_enrich_enabled():
        return False
    tag = (answer_lang or "").strip().lower()
    if is_english_answer_lang(tag) or tag in ("mixed", "unknown", ""):
        return False
    return tag in ENRICHABLE_LANGS


def _enrich_model_id() -> str:
    for key in ("RAG_LANG_ENRICH_MODEL_ID", "RAG_SUMMARY_MODEL_ID"):
        mid = (os.environ.get(key) or "").strip()
        if mid:
            return mid
    return llm_model_id()


def _normalize_cache_key(query: str, answer_lang: str) -> str:
    text = unicodedata.normalize("NFC", (query or "").strip())
    digest = hashlib.sha256(f"{answer_lang}\n{text}".encode("utf-8")).hexdigest()
    return digest


def _cache_get(key: str) -> str | None:
    hit = _enrich_cache.get(key)
    if not hit:
        return None
    ts, value = hit
    if time.monotonic() - ts > _CACHE_TTL_S:
        _enrich_cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: str) -> None:
    if len(_enrich_cache) >= _CACHE_MAX:
        # Drop oldest ~25% by timestamp.
        ordered = sorted(_enrich_cache.items(), key=lambda kv: kv[1][0])
        for old_key, _ in ordered[: max(1, _CACHE_MAX // 4)]:
            _enrich_cache.pop(old_key, None)
    _enrich_cache[key] = (time.monotonic(), value)


def clear_language_enrich_cache() -> None:
    _enrich_cache.clear()


def _clean_english(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    # Drop common wrapper noise.
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip().strip('"').strip("'").strip()
    # Single line / first paragraph only.
    text = re.split(r"[\r\n]+", text, maxsplit=1)[0].strip()
    return text[:800]


def enrich_query_to_english(
    query: str,
    *,
    answer_lang: str | None = None,
) -> LanguageEnrichment:
    """
    Build an English analytical paraphrase for bind/retrieve.

    Never replaces the original for answer_lang or generation.
    """
    q = (query or "").strip()
    lang = (answer_lang or "").strip().lower() or detect_answer_language(q)
    if not q or not needs_language_enrich(lang):
        return LanguageEnrichment(
            answer_lang=lang or "en",
            needs_enrich=False,
            english_query="",
            enrich_source="none",
        )

    cache_key = _normalize_cache_key(q, lang)
    cached = _cache_get(cache_key)
    if cached is not None:
        return LanguageEnrichment(
            answer_lang=lang,
            needs_enrich=True,
            english_query=cached,
            enrich_source="cache",
        )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Language code: {lang}\n"
                f"Original question:\n{q}\n\n"
                "English analytical paraphrase:"
            ),
        },
    ]
    try:
        raw = llm_chat_complete(
            messages,
            model=_enrich_model_id(),
            max_tokens=180,
            temperature=0.0,
            purpose="lang.enrich",
        )
    except Exception:
        logger.warning("language_enrich LLM failed", exc_info=True)
        raw = ""

    english = _clean_english(raw)
    if not english:
        return LanguageEnrichment(
            answer_lang=lang,
            needs_enrich=True,
            english_query="",
            enrich_source="none",
        )

    _cache_set(cache_key, english)
    return LanguageEnrichment(
        answer_lang=lang,
        needs_enrich=True,
        english_query=english,
        enrich_source="llm",
    )


__all__ = [
    "ENRICHABLE_LANGS",
    "LanguageEnrichment",
    "clear_language_enrich_cache",
    "enrich_query_to_english",
    "lang_enrich_enabled",
    "needs_language_enrich",
]
