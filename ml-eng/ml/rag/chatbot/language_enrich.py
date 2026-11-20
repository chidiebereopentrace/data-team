"""English enrich shadow for African / local-language Ask ADZA queries.

Used for decompose, BQ bind, and a second vector retrieve probe.
Generation and answer_lang always stay on the original user text.

Enrich is on by default; set RAG_LANG_ENRICH=0 to disable.
Trigger uses model-first language detection (CommonLingua).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, replace
from typing import Final

from ml.rag.chatbot.answer_language import detect_language_with_confidence
from ml.rag.chatbot.geo_iso3 import iso3_to_canonical_name, name_to_iso3
from ml.rag.llm_chat import llm_chat_complete, llm_model_id

logger = logging.getLogger(__name__)

# Named local languages that normally get an English enrich shadow.
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

# Skip enrich only when English is detected at least this confidently.
ENRICH_EN_CONFIDENCE_MIN: Final[float] = 0.6

_CACHE_TTL_S = 3600.0
_CACHE_MAX = 256
_enrich_cache: dict[str, tuple[float, LanguageEnrichment]] = {}
# Subject anchors are terminology, not sentence-specific translations. Keep
# ambiguous words (e.g. Igbo "ala" alone) out of these high-precision patterns.
_SOIL_ANCHORS = {
    "en": r"\bsoil\b",
    "fr": r"\bsols?\b|\bfertilite\b",
    "ig": r"\baja\b(?=.*\b(?:ala\s*ugbo|alaugbo|ugbo|ubi)\b)",
    "sw": r"\budongo\b",
}
_IGBO_CURRENT_PRICE = re.compile(
    r"(?:kedu\s+)?ego\s+ole\s+(?:ka\s+)?(?:a\s*|ha\s+)?na[-\s]?ele\s+"
    r"(?P<product>\w+)\s+na\s+(?P<country>.+?)\s+ugbu\s*a\s*\??",
)
_IGBO_PRICE_CROPS = {"osikapa": "rice"}

_SYSTEM = (
    "You rewrite African agricultural questions into one clear English sentence "
    "for warehouse lookup and document retrieval. "
    "Preserve every requested indicator, comparison, outcome, and causal relationship, "
    "as well as the crops, places, and time windows. "
    "Expand well-known local crop names only when confident "
    "(e.g. akpu → cassava, mahindi → maize, osikapa → rice). "
    "Keep the original subject and task. You may expand relevant supporting context, "
    "but do not replace the question with a different crop, policy or indicator. "
    "Do not change explicit years or places, or present suggested crops as an exhaustive requested list. "
    "Output only the English sentence — no quotes, no preamble, no JSON."
)


@dataclass(frozen=True)
class LanguageEnrichment:
    answer_lang: str
    needs_enrich: bool
    english_query: str
    enrich_source: str  # none | llm | cache | lexicon | anchored_original
    confidence: float = 0.0
    subjects: tuple[str, ...] = ()
    subject_guard_applied: bool = False


def lang_enrich_enabled() -> bool:
    """Enrich on by default; only explicit off disables."""
    raw = os.environ.get("RAG_LANG_ENRICH", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def needs_language_enrich(
    answer_lang: str,
    query: str = "",
    *,
    confidence: float | None = None,
) -> bool:
    """
    True when an English enrich shadow should run.

    Non-English, mixed, and unknown always enrich (when enabled).
    English skips only at high model confidence.
    """
    del query  # reserved for future; detection already happened upstream
    if not lang_enrich_enabled():
        return False
    tag = (answer_lang or "").strip().lower()
    if tag in ("", "unknown", "mixed"):
        return True
    if tag != "en":
        return True
    if confidence is None:
        return False
    return float(confidence) < ENRICH_EN_CONFIDENCE_MIN


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


def _cache_get(key: str) -> LanguageEnrichment | None:
    hit = _enrich_cache.get(key)
    if not hit:
        return None
    ts, value = hit
    if time.monotonic() - ts > _CACHE_TTL_S:
        _enrich_cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: LanguageEnrichment) -> None:
    if len(_enrich_cache) >= _CACHE_MAX:
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
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip().strip('"').strip("'").strip()
    text = re.split(r"[\r\n]+", text, maxsplit=1)[0].strip()
    return text[:800]


def enrich_query_to_english(
    query: str,
    *,
    answer_lang: str | None = None,
    confidence: float | None = None,
) -> LanguageEnrichment:
    """
    Build an English analytical paraphrase for bind/retrieve.

    Never replaces the original for answer_lang or generation.
    """
    q = (query or "").strip()
    if not q:
        return LanguageEnrichment(
            answer_lang="en",
            needs_enrich=False,
            english_query="",
            enrich_source="none",
            confidence=1.0,
        )

    if answer_lang is not None and confidence is not None:
        lang = (answer_lang or "").strip().lower() or "en"
        conf = float(confidence)
    else:
        lang, conf = detect_language_with_confidence(q)

    if not needs_language_enrich(lang, q, confidence=conf):
        return LanguageEnrichment(
            answer_lang=lang or "en",
            needs_enrich=False,
            english_query="",
            enrich_source="none",
            confidence=conf,
        )

    if lang == "ig":
        folded = "".join(c for c in unicodedata.normalize("NFD", q.casefold()) if not unicodedata.combining(c))
        match = _IGBO_CURRENT_PRICE.fullmatch(folded)
        if match and (crop := _IGBO_PRICE_CROPS.get(match["product"])):
            if country_id := name_to_iso3(match["country"]):
                country = iso3_to_canonical_name(country_id)
                return LanguageEnrichment(lang, True, f"What is the current market price of {crop} in {country}?", "lexicon", conf)

    cache_key = _normalize_cache_key(q, lang)
    cached = _cache_get(cache_key)
    if cached is not None:
        return replace(cached, enrich_source="cache", confidence=conf)

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
    folded = "".join(c for c in unicodedata.normalize("NFD", q.casefold()) if not unicodedata.combining(c))
    subjects = ("soil",) if re.search(_SOIL_ANCHORS.get(lang, r"(?!)"), folded) else ()
    if not english and not subjects:
        return LanguageEnrichment(
            answer_lang=lang,
            needs_enrich=True,
            english_query="",
            enrich_source="none",
            confidence=conf,
        )

    guarded = bool(subjects and not re.search(r"\bsoil\b|\bfertility\b|\bfarmland\b", english, re.I))
    if guarded:
        # Retain the complete original question and a known bilingual anchor.
        # An expansion that replaces its subject cannot become the request.
        english = q + " | Explicit subject: " + ", ".join(subjects)
    result = LanguageEnrichment(
        answer_lang=lang,
        needs_enrich=True,
        english_query=english,
        enrich_source="anchored_original" if guarded else "llm",
        confidence=conf,
        subjects=subjects,
        subject_guard_applied=guarded,
    )
    _cache_set(cache_key, result)
    return result


__all__ = [
    "ENRICHABLE_LANGS",
    "ENRICH_EN_CONFIDENCE_MIN",
    "LanguageEnrichment",
    "clear_language_enrich_cache",
    "enrich_query_to_english",
    "lang_enrich_enabled",
    "needs_language_enrich",
]
