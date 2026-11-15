"""Hybrid bilingual translation for BQ + Qdrant vector retrieval.

Forward translation: African language → English (for BQ SQL warehouse only)
Back-translation: English answer → User's original language (before response)
Uses the same LLM model as generator via llm_model_id().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Literal

from ml.rag.llm_chat import llm_chat_complete, llm_model_id

logger = logging.getLogger(__name__)

_SYSTEM_TRANSLATE_TO_ENGLISH = (
    "Translate this agricultural question from {lang} into clear English for warehouse lookup. "
    "Preserve crop, country/place, years/time span, and measure intent. "
    "Expand well-known local names only when confident (e.g., akpu → cassava, mahindi → maize). "
    "Do not invent years, places, or crops. Output ONLY the English sentence — no quotes, preamble, JSON."
)

_SYSTEM_TRANSLATE_TO_LANG = (
    "Translate this English agricultural answer into {lang} naturally and accurately. "
    "Preserve all technical details, numbers, units, and proper nouns. "
    "Output ONLY the translated text — no quotes, preamble, JSON."
)


@dataclass(frozen=True)
class BilingualResponse:
    """Full response with both English query for BQ and original language."""
    bq_query: str  # English translation for SQL warehouse
    original_query: str  # User's original African language (for multi-lingual vector embeddings)
    
    def __init__(self, bq_query: str, original_query: str):
        object.__setattr__(self, "bq_query", bq_query)
        object.__setattr__(self, "original_query", original_query)


def translate_to_english(
    text: str,
    source_lang: str,
) -> str:
    """Forward translation: African language → English for BQ retrieval only.
    
    For multi-lingual vector embeddings, the original query is used for Qdrant search.
    """
    try:
        messages = [
            {"role": "system", "content": _SYSTEM_TRANSLATE_TO_ENGLISH.format(lang=source_lang)},
            {"role": "user", "content": f"Original question:\n{text}\n\nEnglish translation:"},
        ]
        raw = llm_chat_complete(
            messages,
            model=llm_model_id(),  # Same as generator
            max_tokens=200,
            temperature=0.0,
            purpose="bq.translation.forward",
        )
        return _clean_text(raw)
    except Exception:
        logger.warning("Forward translation to English failed", exc_info=True)
        return text  # Fallback to original if translation fails


def translate_to_user_language(
    english_text: str,
    target_lang: str,
    max_tokens: int = 500,
) -> str:
    """Back-translation: English → User's language for response.
    
    Only run when the answer is non-empty and translation was attempted.
    """
    if not english_text or len(english_text.strip()) < 20:
        return english_text  # Skip trivial translations
    
    try:
        messages = [
            {"role": "system", "content": _SYSTEM_TRANSLATE_TO_LANG.format(lang=target_lang)},
            {"role": "user", "content": f"English answer:\n{english_text}\n\nTranslation:"},
        ]
        raw = llm_chat_complete(
            messages,
            model=llm_model_id(),  # Same as generator
            max_tokens=max_tokens,
            temperature=0.2,
            purpose="bq.translation.backward",
        )
        return _clean_text(raw)
    except Exception:
        logger.warning(f"Back-translation to {target_lang} failed", exc_info=True)
        return english_text  # Fallback to English if translation fails


def _clean_text(raw: str) -> str:
    """Remove markdown artifacts and normalize whitespace."""
    text = (raw or "").strip()
    if not text:
        return ""
    text = text.strip('"').strip("'").strip("`")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines)


def create_bilingual_response(
    original_query: str,
    answer_lang: str,
    bq_query_variant: bool = True,  # Whether to translate for BQ
) -> tuple[str, str]:
    """
    Create bilingual query pair for hybrid retrieval.
    
    Returns: (bq_query_for_sql, original_query_for_vectors)
    
    For multi-lingual vector embeddings:
    - bq_query = English translation (BQ works best in English)
    - original_query = User's original language (Qdrant uses multi-lingual embeddings)
    """
    if answer_lang and answer_lang.strip().lower() not in ("en", "unknown", "mixed"):
        bq_query = translate_to_english(original_query, source_lang=answer_lang)
        return (bq_query, original_query)
    
    # English or uncertain language: use original for both
    return (original_query, original_query)
