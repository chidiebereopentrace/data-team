"""
Detect user query language and produce generation instructions that mirror it.

Language affects answer prompts only — never translate the query before E5 retrieve.
Named tags: en | sw | fr | pcm | ar | am | ig | yo | ha | pt | tw | efi | zu | xh | so | wo | rw | mixed | unknown.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Final

AnswerLang = str

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_AMHARIC_RE = re.compile(r"[\u1200-\u137F]")
_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ']+")

# Display names for client-facing language-help responses.
SUPPORTED_ANSWER_LANGUAGES: Final[tuple[tuple[str, str], ...]] = (
    ("en", "English"),
    ("sw", "Swahili"),
    ("fr", "French"),
    ("pcm", "Nigerian Pidgin"),
    ("ar", "Arabic"),
    ("am", "Amharic"),
    ("ig", "Igbo"),
    ("yo", "Yoruba"),
    ("ha", "Hausa"),
    ("pt", "Portuguese"),
    ("tw", "Twi"),
    ("efi", "Efik"),
    ("zu", "Zulu"),
    ("xh", "Xhosa"),
    ("so", "Somali"),
    ("wo", "Wolof"),
    ("rw", "Kinyarwanda"),
    ("mixed", "Code-mixed (user's mix)"),
)

_LANG_DISPLAY: Final[dict[str, str]] = {code: name for code, name in SUPPORTED_ANSWER_LANGUAGES}

# Marker tokens → named language codes (high precision).
_MARKER_LANG: Final[dict[str, str]] = {
    # Swahili
    "habari": "sw",
    "asante": "sw",
    "tafadhali": "sw",
    "nini": "sw",
    "nani": "sw",
    "wapi": "sw",
    "mazao": "sw",
    "kilimo": "sw",
    "chakula": "sw",
    "maendeleo": "sw",
    "jina": "sw",
    "lako": "sw",
    "wewe": "sw",
    "mahindi": "sw",
    "mvua": "sw",
    "ukame": "sw",
    # French
    "bonjour": "fr",
    "merci": "fr",
    "comment": "fr",
    "pourquoi": "fr",
    "récolte": "fr",
    "recolte": "fr",
    "sécheresse": "fr",
    "secheresse": "fr",
    "rendement": "fr",
    "maïs": "fr",
    "données": "fr",
    "donnees": "fr",
    "alimentaire": "fr",
    "pouvez": "fr",
    # Hausa
    "sannu": "ha",
    "yaya": "ha",
    "noma": "ha",
    "hatsi": "ha",
    "masara": "ha",
    "shinkafa": "ha",
    "menene": "ha",
    "yaushe": "ha",
    # Portuguese
    "olá": "pt",
    "ola": "pt",
    "obrigado": "pt",
    "obrigada": "pt",
    "você": "pt",
    "voce": "pt",
    "colheita": "pt",
    "produção": "pt",
    "producao": "pt",
    "também": "pt",
    "tambem": "pt",
    # Nigerian Pidgin
    "wetin": "pcm",
    "dey": "pcm",
    "abi": "pcm",
    "una": "pcm",
    "wahala": "pcm",
    "abeg": "pcm",
    "sabi": "pcm",
    "comot": "pcm",
    "pikin": "pcm",
    "oyibo": "pcm",
    # Igbo
    "kedu": "ig",
    "biko": "ig",
    "ndewo": "ig",
    "dalụ": "ig",
    "dalu": "ig",
    "ịmeela": "ig",
    "imeela": "ig",
    "obodo": "ig",
    "ako": "ig",
    # Yoruba
    "bawo": "yo",
    "jowo": "yo",
    "ekabo": "yo",
    "pele": "yo",
    # Twi
    "medaase": "tw",
    "medaasepa": "tw",
    "woho": "tw",
    # Efik
    "mbọk": "efi",
    "mbok": "efi",
    # Zulu
    "sawubona": "zu",
    "ngiyabonga": "zu",
    "unjani": "zu",
    "yebo": "zu",
    "ukulima": "zu",
    "ukudla": "zu",
    # Xhosa
    "molo": "xh",
    "enkosi": "xh",
    "kunjani": "xh",
    "ukutya": "xh",
    # Somali
    "salaan": "so",
    "mahadsanid": "so",
    "sidee": "so",
    "beeraha": "so",
    "cunto": "so",
    # Wolof
    "salaamalekum": "wo",
    "jërëjëf": "wo",
    "jerejef": "wo",
    # Kinyarwanda
    "muraho": "rw",
    "murakoze": "rw",
    "amakuru": "rw",
    "ubuhinzi": "rw",
    "ibiribwa": "rw",
}

_NON_EN_MARKERS: Final[frozenset[str]] = frozenset(_MARKER_LANG.keys())

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
    r"\b(?:kedu|biko|ndewo|onye\s+ị\s+bụ|onye\s+i\s+bu|gịnị\s+ka|gini\s+ka|"
    r"obodo|kacha)\b",
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
_HA_PHRASE_RE = re.compile(
    r"\b(?:sannu|menene|yaushe|ina\s+aiki)\b",
    re.IGNORECASE,
)
_PT_PHRASE_RE = re.compile(
    r"\b(?:olá|ola|obrigad[oa]|você|voce|colheita|produção|producao)\b",
    re.IGNORECASE,
)
_ZU_PHRASE_RE = re.compile(
    r"\b(?:sawubona|ngiyabonga|unjani|ukulima|ukudla)\b",
    re.IGNORECASE,
)
_XH_PHRASE_RE = re.compile(
    r"\b(?:molo|enkosi|kunjani|ukutya)\b",
    re.IGNORECASE,
)
_SO_PHRASE_RE = re.compile(
    r"\b(?:salaan|mahadsanid|sidee|beeraha|cunto)\b",
    re.IGNORECASE,
)
_WO_PHRASE_RE = re.compile(
    r"\b(?:salaamalekum|j[eë]r[eë]j[eë]f|jerejef|nanga\s+def)\b",
    re.IGNORECASE,
)
_RW_PHRASE_RE = re.compile(
    r"\b(?:muraho|murakoze|amakuru|ubuhinzi|ibiribwa)\b",
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
    "Amharic, Zulu, Xhosa, Somali, Wolof, Kinyarwanda, and Nigerian Pidgin, as well as "
    "code-mixing when that is how they wrote. "
    "Keep citation footnote numbers [N], numbers, and proper nouns faithful. "
    "Do not switch to English unless the user wrote in English. "
    "Avoid academic hedges unless the context explicitly supports the qualification."
)

_MIRROR_INSTRUCTION_NO_INLINE = (
    "Answer in the same language as the user question. This includes African and regional "
    "languages such as Igbo, Yoruba, Twi, Efik, Swahili, French, Hausa, Portuguese, Arabic, "
    "Amharic, Zulu, Xhosa, Somali, Wolof, Kinyarwanda, and Nigerian Pidgin, as well as "
    "code-mixing when that is how they wrote. "
    "Keep numbers and proper nouns faithful; do not insert [N] or [Source N] footnote markers. "
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
    "am": (
        "\u12a0\u1201\u1295 \u1260\u12a5\u122d\u130d\u1320\u1295\u1290\u1275 "
        "\u1208\u1218\u1218\u1208\u1235 \u1260\u1242 \u12a0\u1235\u1270\u121b\u121b\u129d "
        "\u1218\u1228\u1303 \u12e8\u1208\u12dd\u121d\u1362 "
        "\u12e8\u12cd\u1235\u1325 \u12a5\u12cd\u1240\u1275 \u124b\u1274 "
        "\u1320\u1295\u12ab\u122b \u1270\u1218\u1233\u1233\u12ed\u1290\u1275 "
        "\u12a0\u120b\u1218\u1323\u121d \u12a5\u1293 \u1208\u12da\u1205 "
        "\u1325\u12eb\u1244 \u1270\u1328\u121b\u122a \u12e8\u12f5\u122d "
        "\u134d\u1208\u130b \u12a0\u120d\u1270\u1308\u1298\u121d\u1362 "
        "\u12a5\u1263\u12ab\u12ce \u12a5\u1295\u12f0\u1308\u1293 \u12ed\u133b\u1349\u1363 "
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


def _phrase_lang(text: str) -> str | None:
    if _PCM_PHRASE_RE.search(text):
        return "pcm"
    if _FR_PHRASE_RE.search(text):
        return "fr"
    if _SW_PHRASE_RE.search(text):
        return "sw"
    if _IG_PHRASE_RE.search(text):
        return "ig"
    if _YO_PHRASE_RE.search(text):
        return "yo"
    if _TWI_PHRASE_RE.search(text):
        return "tw"
    if _HA_PHRASE_RE.search(text):
        return "ha"
    if _PT_PHRASE_RE.search(text):
        return "pt"
    if _ZU_PHRASE_RE.search(text):
        return "zu"
    if _XH_PHRASE_RE.search(text):
        return "xh"
    if _SO_PHRASE_RE.search(text):
        return "so"
    if _WO_PHRASE_RE.search(text):
        return "wo"
    if _RW_PHRASE_RE.search(text):
        return "rw"
    return None


def _marker_lang_votes(tokens: list[str]) -> dict[str, int]:
    votes: dict[str, int] = {}
    for t in tokens:
        folded = _ascii_fold(t)
        for key in (t, folded):
            code = _MARKER_LANG.get(key)
            if code:
                votes[code] = votes.get(code, 0) + 1
                break
    return votes


def detect_answer_language(query: str) -> AnswerLang:
    """
    Named language tag for routing and generation.

    Returns one of: en, sw, fr, pcm, ar, am, ig, yo, ha, pt, tw, efi, zu, xh, so, wo, rw, mixed, unknown.
    """
    text = (query or "").strip()
    if not text:
        return "en"

    if _AMHARIC_RE.search(text):
        return "am"
    if _ARABIC_RE.search(text):
        return "ar"

    phrase = _phrase_lang(text)
    tokens = _tokenize(text)
    votes = _marker_lang_votes(tokens)
    hit_count = sum(votes.values())
    accents = _accent_density(text)
    en_hits = sum(1 for t in tokens if t in _EN_FUNCTION)

    named: str | None = phrase
    if votes:
        top_code = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if named is None or votes.get(named, 0) < votes[top_code]:
            named = top_code
        elif named in votes and votes[named] == votes[top_code]:
            named = named  # phrase wins ties

    non_en_signal = named is not None or hit_count >= 2 or accents >= 0.08

    if not non_en_signal:
        # Default Latin / English agronomy questions to English.
        return "en"

    if en_hits >= 2 and (hit_count >= 1 or phrase is not None):
        return "mixed"

    if named:
        return named

    # Non-English signal (e.g. accents) without a named language → ask client.
    return "unknown"


def detect_canned_insufficient_lang(query: str) -> str:
    """
    Specialty tag for deterministic insufficient copy: en|sw|fr|pcm|ar|am.

    Other local languages map to en (canned fallback).
    """
    text = (query or "").strip()
    if not text:
        return "en"
    tag = detect_answer_language(text)
    if tag in _INSUFFICIENT:
        return tag
    if tag in ("mixed", "unknown", "ig", "yo", "ha", "pt", "tw", "efi", "zu", "xh", "so", "wo", "rw", "non_en"):
        return "en"
    return "en"


def is_english_answer_lang(lang: str) -> bool:
    return (lang or "en").strip().lower() == "en"


def language_instruction(lang: str, *, inline_citations: bool = False) -> str:
    """System-prompt addendum: English business prose vs named-language mirror."""
    mirror = _MIRROR_INSTRUCTION if inline_citations else _MIRROR_INSTRUCTION_NO_INLINE
    tag = (lang or "en").strip().lower()
    if tag == "en":
        return _EN_INSTRUCTION
    if tag == "mixed":
        return (
            "The user is code-mixing languages. Mirror their mix; do not force pure English. "
            + mirror
        )
    if tag == "unknown":
        return _EN_INSTRUCTION
    display = _LANG_DISPLAY.get(tag, tag)
    return (
        f"Answer in {display} (language code: {tag}). "
        + mirror
    )


def language_unclear_answer() -> str:
    """Client message when query language cannot be named definitively."""
    lines = [
        "I could not definitively detect the language of your question.",
        "Please rephrase in one of the languages Ask ADZA supports:",
    ]
    for code, name in SUPPORTED_ANSWER_LANGUAGES:
        if code == "mixed":
            continue
        lines.append(f"- {name} ({code})")
    lines.append(
        "You may also write in a natural code-mix of these languages."
    )
    return "\n".join(lines)


def insufficient_context_answer(lang: str = "", *, query: str | None = None) -> str:
    """
    Deterministic insufficient-context message.

    Prefer ``query=`` so specialty canned languages (sw/fr/pcm/ar/am) can be inferred.
    Soft tags ``non_en`` / ``mixed`` / named langs without canned copy fall back to English.
    """
    if query is not None:
        tag = detect_canned_insufficient_lang(query)
    else:
        tag = (lang or "en").strip().lower()
        if tag in ("non_en", "mixed", "unknown", ""):
            tag = "en"
        elif tag not in _INSUFFICIENT:
            tag = "en"
    return _INSUFFICIENT.get(tag) or _INSUFFICIENT_EN


__all__ = [
    "SUPPORTED_ANSWER_LANGUAGES",
    "detect_answer_language",
    "detect_canned_insufficient_lang",
    "is_english_answer_lang",
    "language_instruction",
    "language_unclear_answer",
    "insufficient_context_answer",
]
