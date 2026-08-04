"""
Detect user query language and produce generation instructions that mirror it.

Language affects answer prompts only — never translate the query before E5 retrieve.
Soft tags: en | non_en | ar | am | mixed. Generation uses a language-agnostic mirror
instruction (Igbo, Yoruba, Twi, Efik, Swahili, French, Hausa, Portuguese, Arabic,
Amharic, Nigerian Pidgin, code-mix) without requiring a tag per language.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Final

AnswerLang = str  # en | non_en | ar | am | mixed

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_AMHARIC_RE = re.compile(r"[\u1200-\u137F]")
_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ']+")

# High-precision markers → non_en (not separate generation tags).
_NON_EN_MARKERS: Final[frozenset[str]] = frozenset(
    {
        # Swahili
        "habari",
        "asante",
        "tafadhali",
        "nini",
        "nani",
        "wapi",
        "mazao",
        "kilimo",
        "chakula",
        "maendeleo",
        "jina",
        "lako",
        "wewe",
        "mahindi",
        "mvua",
        "ukame",
        # French
        "bonjour",
        "merci",
        "comment",
        "pourquoi",
        "récolte",
        "recolte",
        "sécheresse",
        "secheresse",
        "rendement",
        "maïs",
        "données",
        "donnees",
        "alimentaire",
        "pouvez",
        # Hausa
        "sannu",
        "yaya",
        "noma",
        "hatsi",
        "masara",
        "shinkafa",
        "menene",
        "yaushe",
        # Portuguese
        "olá",
        "ola",
        "obrigado",
        "obrigada",
        "você",
        "voce",
        "colheita",
        "produção",
        "producao",
        "também",
        "tambem",
        # Nigerian Pidgin (strong only)
        "wetin",
        "dey",
        "abi",
        "una",
        "wahala",
        "abeg",
        "sabi",
        "comot",
        "pikin",
        "oyibo",
        # Igbo
        "kedu",
        "biko",
        "ndewo",
        "dalụ",
        "dalu",
        "ịmeela",
        "imeela",
        # Yoruba (ascii folds)
        "bawo",
        "jowo",
        "ekabo",
        "pele",
        # Twi
        "medaase",
        "medaasepa",
        "woho",
        # Efik
        "mbọk",
        "mbok",
    }
)

_PCM_PHRASE_RE = re.compile(
    r"\b(?:wetin\s+be|who\s+you\s+be|wetin\s+your\s+name|how\s+far|abeg\s+|una\s+dey|e\s+dey)\b",
    re.IGNORECASE,
)
_FR_PHRASE_RE = re.compile(
    r"\b(?:qui\s+(?:es[- ]tu|êtes[- ]vous|etes[- ]vous|est[- ]ce)|"
    r"comment\s+(?:t['’]?appelles[- ]tu|vous\s+appelez)|"
    r"quel\s+est\s+ton\s+nom|qu['’]est[- ]ce\s+que\s+tu\s+fais|"
    r"bonjour|pourquoi|récolte|secheresse|sécheresse)\b",
    re.IGNORECASE,
)
_SW_PHRASE_RE = re.compile(
    r"\b(?:wewe\s+ni\s+nani|jina\s+lako\s+nani|una[- ]?fanya\s+nini|"
    r"wewe\s+unafanya\s+nini|habari\s+yako|kilimo|mazao|asante)\b",
    re.IGNORECASE,
)
_IG_PHRASE_RE = re.compile(
    r"\b(?:kedu|biko|ndewo|onye\s+ị\s+bụ|onye\s+i\s+bu|gịnị\s+ka|gini\s+ka)\b",
    re.IGNORECASE,
)
_YO_PHRASE_RE = re.compile(
    r"\b(?:bawo|jowo|ṣe\s+é|se\s+e|ta\s+ni\s+ẹ|ta\s+ni\s+e)\b",
    re.IGNORECASE,
)
_TWI_PHRASE_RE = re.compile(
    r"\b(?:medaase|wo\s+ho\s+te\s+s[ɛe]n|yɛ\s+fr[ɛe]|wo\s+din\s+de)\b",
    re.IGNORECASE,
)

_EN_FUNCTION: Final[frozenset[str]] = frozenset(
    {
        "the",
        "is",
        "are",
        "what",
        "how",
        "why",
        "show",
        "trend",
        "yield",
        "please",
        "can",
        "you",
        "about",
        "from",
        "with",
        "have",
        "this",
        "that",
        "which",
        "when",
        "where",
    }
)

_EN_INSTRUCTION = (
    "Write in active voice and plain business English. Avoid academic hedges "
    "('relatively', 'somewhat', 'arguably') unless the context explicitly supports the qualification."
)

_MIRROR_INSTRUCTION = (
    "Answer in the same language as the user question. This includes African and regional "
    "languages such as Igbo, Yoruba, Twi, Efik, Swahili, French, Hausa, Portuguese, Arabic, "
    "Amharic, and Nigerian Pidgin, as well as code-mixing when that is how they wrote. "
    "Keep citation footnote numbers [N], numbers, and proper nouns faithful. "
    "Do not switch to English unless the user wrote in English. "
    "Avoid academic hedges unless the context explicitly supports the qualification."
)

# Deterministic insufficient-context copy (6 canned languages).
_INSUFFICIENT_EN = (
    "I don't have enough reliable information to answer that confidently right now. "
    "My internal knowledge base didn't return a strong match and supplemental web "
    "search wasn't available for this query. Could you try rephrasing, narrowing the "
    "country or time range, or asking a related question I can ground in available "
    "sources?"
)

_INSUFFICIENT: Final[dict[str, str]] = {
    "en": _INSUFFICIENT_EN,
    "sw": (
        "Sina taarifa za kutosha za kuaminika kujibu swali lako kwa uhakika sasa hivi. "
        "Hifadhidata yangu ya ndani haikupata mechi thabiti na utafutaji wa wavuti wa ziada "
        "haukuwa unapatikana kwa swali hili. Je, unaweza kujaribu kuandika upya, kupunguza "
        "nchi au kipindi cha muda, au kuuliza swali linalohusiana ambalo naweza kulisimamia "
        "kwa vyanzo vilivyopo?"
    ),
    "fr": (
        "Je n'ai pas assez d'informations fiables pour répondre avec confiance pour le moment. "
        "Ma base de connaissances interne n'a pas trouvé de correspondance solide et la "
        "recherche web complémentaire n'était pas disponible pour cette question. Pouvez-vous "
        "reformuler, préciser le pays ou la période, ou poser une question connexe que je "
        "peux ancrer dans les sources disponibles ?"
    ),
    "pcm": (
        "I no get enough correct information wey I fit use answer dis question with confidence "
        "right now. My internal knowledge base no return strong match and web search no dey "
        "available for dis query. Abeg try rephrase am, narrow di country or time range, or "
        "ask related question wey I fit ground for di sources wey dey."
    ),
    "ar": (
        "ليس لدي معلومات موثوقة كافية للإجابة بثقة الآن. "
        "قاعدة معرفتي الداخلية لم تُرجع تطابقًا قويًا ولم تتوفر بحث ويب إضافي لهذا الاستعلام. "
        "هل يمكنك إعادة الصياغة أو تضييق البلد أو الفترة الزمنية أو طرح سؤال ذي صلة يمكنني "
        "تثبيته في المصادر المتاحة؟"
    ),
    # Amharic via unicode escapes (avoid encoding corruption in source edits).
    "am": (
        "\u12a0\u1201\u1295 \u1260\u12a5\u122d\u130d\u1320\u1295\u1290\u1275 "
        "\u1208\u1218\u1218\u1208\u1235 \u1260\u1242 \u12a0\u1235\u1270\u121b\u121b\u129d "
        "\u1218\u1228\u1303 \u12e8\u1208\u12dd\u121d\u1362 "
        "\u12e8\u12cd\u1235\u1325 \u12a5\u12cd\u1240\u1275 \u124b\u1274 "
        "\u1320\u1295\u12ab\u122b \u1270\u1218\u1233\u1233\u12ed\u1290\u1275 "
        "\u12a0\u120b\u1218\u1323\u121d \u12a5\u1293 \u1208\u12da\u1205 "
        "\u1325\u12eb\u1244 \u1270\u1328\u121b\u122a \u12e8\u12f5\u122d "
        "\u134d\u1208\u130b \u12a0\u120d\u1270\u1308\u1298\u121d\u1362 "
        "\u12a5\u1263\u12ad\u12ce \u12a5\u1295\u12f0\u1308\u1293 \u12ed\u133b\u1349\u1363 "
        "\u12a0\u1308\u122d \u12c8\u12ed\u121d \u130a\u12dc \u12ed\u1308\u12f5\u1261\u1363 "
        "\u12c8\u12ed\u121d \u1260\u121a\u1308\u1299 \u121d\u1295\u132e\u127d "
        "\u120b\u12ed \u120d\u1218\u1230\u122d\u1275 \u12e8\u121d\u127d\u120d "
        "\u1270\u12db\u121b\u1305 \u1325\u12eb\u1244 \u12ed\u1320\u12ed\u1241\u1362"
    ),
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _ascii_fold(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in norm if not unicodedata.combining(c))


def _accent_density(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    return non_ascii / len(letters)


def _marker_hits(tokens: list[str]) -> int:
    folded = {_ascii_fold(t) for t in tokens}
    folded |= set(tokens)
    return sum(1 for t in folded if t in _NON_EN_MARKERS or _ascii_fold(t) in _NON_EN_MARKERS)


def detect_answer_language(query: str) -> AnswerLang:
    """
    Soft language tag for routing (static English skip / mirror vs English).

    Returns one of: en, non_en, ar, am, mixed.
    """
    text = (query or "").strip()
    if not text:
        return "en"

    if _AMHARIC_RE.search(text):
        return "am"
    if _ARABIC_RE.search(text):
        return "ar"

    strong_phrase = bool(
        _PCM_PHRASE_RE.search(text)
        or _FR_PHRASE_RE.search(text)
        or _SW_PHRASE_RE.search(text)
        or _IG_PHRASE_RE.search(text)
        or _YO_PHRASE_RE.search(text)
        or _TWI_PHRASE_RE.search(text)
    )

    tokens = _tokenize(text)
    hits = _marker_hits(tokens)
    accents = _accent_density(text)
    en_hits = sum(1 for t in tokens if t in _EN_FUNCTION)

    non_en_signal = strong_phrase or hits >= 2 or accents >= 0.08

    if not non_en_signal:
        return "en"

    if en_hits >= 2 and (hits >= 1 or strong_phrase):
        return "mixed"

    return "non_en"


def detect_canned_insufficient_lang(query: str) -> str:
    """
    Specialty tag for deterministic insufficient copy: en|sw|fr|pcm|ar|am.

    Other local languages map to en (canned fallback).
    """
    text = (query or "").strip()
    if not text:
        return "en"
    if _AMHARIC_RE.search(text):
        return "am"
    if _ARABIC_RE.search(text):
        return "ar"
    if _PCM_PHRASE_RE.search(text):
        return "pcm"
    if _FR_PHRASE_RE.search(text):
        return "fr"
    if _SW_PHRASE_RE.search(text):
        return "sw"

    tokens = _tokenize(text)
    sw = sum(1 for t in tokens if t in {"habari", "asante", "kilimo", "mazao", "wewe", "jina", "tafadhali"})
    fr = sum(
        1
        for t in tokens
        if t in {"bonjour", "merci", "comment", "pourquoi", "récolte", "recolte", "sécheresse", "secheresse", "rendement"}
    )
    pcm = sum(1 for t in tokens if t in {"wetin", "dey", "abi", "una", "wahala", "abeg", "sabi"})
    if pcm >= 2:
        return "pcm"
    if sw >= 2:
        return "sw"
    if fr >= 2:
        return "fr"
    return "en"


def is_english_answer_lang(lang: str) -> bool:
    return (lang or "en").strip().lower() == "en"


def language_instruction(lang: str) -> str:
    """System-prompt addendum: English business prose vs agnostic mirror-user language."""
    tag = (lang or "en").strip().lower()
    if tag == "en":
        return _EN_INSTRUCTION
    if tag == "mixed":
        return (
            "The user is code-mixing languages. Mirror their mix; do not force pure English. "
            + _MIRROR_INSTRUCTION
        )
    return _MIRROR_INSTRUCTION


def insufficient_context_answer(lang: str = "", *, query: str | None = None) -> str:
    """
    Deterministic insufficient-context message.

    Prefer ``query=`` so specialty canned languages (sw/fr/pcm/ar/am) can be inferred.
    Soft tags ``non_en`` / ``mixed`` fall back to English canned.
    """
    if query is not None:
        tag = detect_canned_insufficient_lang(query)
    else:
        tag = (lang or "en").strip().lower()
        if tag in ("non_en", "mixed", ""):
            tag = "en"
    return _INSUFFICIENT.get(tag) or _INSUFFICIENT_EN


__all__ = [
    "detect_answer_language",
    "detect_canned_insufficient_lang",
    "is_english_answer_lang",
    "language_instruction",
    "insufficient_context_answer",
]
