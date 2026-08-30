"""
Cheap heuristic gate for greetings and out-of-scope / non-substantive queries.

Short-circuits full RAG so social messages do not burn retrieval + generation cost,
and so prior session memory cannot steer a "hi" into an unrelated agronomy thread.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from ml.rag.chatbot.answer_language import (
    detect_answer_language,
    is_english_answer_lang,
    language_instruction,
)
from ml.rag.chatbot.assistant_identity import META_ANSWER_FOOTER
from ml.rag.chatbot.assistant_identity import is_meta_query
from ml.rag.chatbot.product_knowledge import is_help_query, is_product_query

SocialKind = Literal["greeting", "out_of_scope"]
SocialSubKind = Literal["greeting", "thanks", "bye"]
EarlyRoute = Literal["meta", "product", "help", "greeting", "out_of_scope"]

_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]+",
    flags=re.UNICODE,
)

# Per-language whole-message social phrases (greetings, thanks, bye).
_SOCIAL_PHRASES: dict[str, tuple[tuple[str, SocialSubKind], ...]] = {
    "en": (
        ("hi", "greeting"),
        ("hello", "greeting"),
        ("hey", "greeting"),
        ("howdy", "greeting"),
        ("yo", "greeting"),
        ("good morning", "greeting"),
        ("good afternoon", "greeting"),
        ("good evening", "greeting"),
        ("good day", "greeting"),
        ("good night", "greeting"),
        ("greetings", "greeting"),
        ("thanks", "thanks"),
        ("thank you", "thanks"),
        ("thx", "thanks"),
        ("ty", "thanks"),
        ("bye", "bye"),
        ("goodbye", "bye"),
        ("good bye", "bye"),
        ("see you", "bye"),
        ("see you later", "bye"),
        ("see ya", "bye"),
        ("what's up", "greeting"),
        ("whats up", "greeting"),
        ("wassup", "greeting"),
        ("sup", "greeting"),
        ("how are you", "greeting"),
        ("how are you doing", "greeting"),
    ),
    "fr": (
        ("bonjour", "greeting"),
        ("bonsoir", "greeting"),
        ("salut", "greeting"),
        ("merci", "thanks"),
        ("merci beaucoup", "thanks"),
        ("au revoir", "bye"),
    ),
    "sw": (
        ("jambo", "greeting"),
        ("habari", "greeting"),
        ("habari yako", "greeting"),
        ("asante", "thanks"),
        ("asante sana", "thanks"),
        ("karibu", "greeting"),
        ("kwaheri", "bye"),
    ),
    "ha": (
        ("sannu", "greeting"),
        ("nagode", "thanks"),
        ("ina kwana", "greeting"),
        ("sai anjima", "bye"),
    ),
    "pt": (
        ("olá", "greeting"),
        ("ola", "greeting"),
        ("obrigado", "thanks"),
        ("obrigada", "thanks"),
        ("bom dia", "greeting"),
        ("tchau", "bye"),
    ),
    "pcm": (
        ("how far", "greeting"),
        ("abeg", "greeting"),
        ("thank you", "thanks"),
        ("bye", "bye"),
    ),
    "ig": (
        ("kedu", "greeting"),
        ("ndewo", "greeting"),
        ("dalu", "thanks"),
        ("dalụ", "thanks"),
        ("biko", "greeting"),
    ),
    "yo": (
        ("bawo", "greeting"),
        ("ekaaro", "greeting"),
        ("ese", "thanks"),
        ("odabo", "bye"),
    ),
    "tw": (
        ("medaase", "thanks"),
        ("medaase pa", "thanks"),
        ("mema wo akye", "greeting"),
    ),
    "efi": (
        ("mbok", "greeting"),
        ("mbọk", "greeting"),
    ),
    "zu": (
        ("sawubona", "greeting"),
        ("unjani", "greeting"),
        ("ngiyabonga", "thanks"),
        ("sala kahle", "bye"),
    ),
    "xh": (
        ("molo", "greeting"),
        ("kunjani", "greeting"),
        ("enkosi", "thanks"),
    ),
    "so": (
        ("salaan", "greeting"),
        ("subax wanaagsan", "greeting"),
        ("mahadsanid", "thanks"),
    ),
    "wo": (
        ("salaamalekum", "greeting"),
        ("jerejef", "thanks"),
        ("jërëjëf", "thanks"),
    ),
    "rw": (
        ("muraho", "greeting"),
        ("amakuru", "greeting"),
        ("murakoze", "thanks"),
    ),
    "ar": (
        ("مرحبا", "greeting"),
        ("السلام عليكم", "greeting"),
        ("شكرا", "thanks"),
        ("مع السلامة", "bye"),
    ),
    "am": (
        ("ሰላም", "greeting"),
    ),
}

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
    r"\bweather\s+in\s+(?:tokyo|london|new\s+york|paris)\b|"
    r"\braconte\s+(?:moi\s+)?(?:une\s+)?blague\b|"
    r"\bniambie\s+joke\b|"
    r"\bcont(?:e|a)\s+(?:me\s+)?(?:uma\s+)?piada\b"
    r")",
    re.IGNORECASE,
)

# Light agronomy / OpenTrace substance cues — if present, never treat as social/OOS.
_AG_SUBSTANCE_RE = re.compile(
    r"(?:"
    r"\b(?:maize|corn|wheat|rice|sorghum|millet|cassava|yam|cocoa|coffee|cotton|"
    r"tea|palm\s*oil|soy(?:bean)?s?|groundnut|livestock|cattle|poultry|fish(?:eries)?|"
    r"fertiliz(?:er|ation)|irrigation|drought|rainfall|yield|harvest|crop|farm(?:er|ing)?|"
    r"agricultur(?:e|al)|agronom(?:y|ic)|food\s*security|climate|soil|seed|"
    r"subsidy|export|import|commodity|market\s*price|smallholder|"
    r"kenya|ghana|nigeria|ethiopia|tanzania|uganda|senegal|malawi|zambia|"
    r"rwanda|cameroon|ivory\s*coast|côte\s*d['']ivoire|"
    r"sahel|maghreb|horn\s+of\s+africa|ecowas|sadc|eac|igad|comesa|cemac|"
    r"west\s+africa|east\s+africa|southern\s+africa|central\s+africa|"
    r"lake\s+chad|liptako|miombo|sudan(?:ian)?\s+zone|guinean\s+zone|"
    r"fao|cgiar|opentrace|ask\s*adza|ofia|bigquery|qdrant)\b"
    r")",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_GREETING_ANSWER = (
    "Hello — I am Ask ADZA, OpenTrace Africa's natural-language interface. "
    "OpenTrace builds Africa's agricultural intelligence layer — turning fragmented "
    "data into decision intelligence. Ask about crops, markets, climate, policy, or "
    "related topics across Africa, and I will ground answers in OpenTrace evidence "
    "where available."
)

_OUT_OF_SCOPE_ANSWER = (
    "I am Ask ADZA, OpenTrace Africa's natural-language interface for African "
    "agricultural intelligence. That request is outside what I can help with here. "
    "Please ask about farming, food systems, markets, climate, or related policy and data in Africa."
)

_SOCIAL_SYSTEM = (
    "You are Ask ADZA, OpenTrace Africa's natural-language interface. "
    "OpenTrace builds Africa's agricultural intelligence layer. "
    "Reply briefly and warmly. Do not invent agronomy facts. "
    "Do not continue any prior conversation topic. "
    "Invite the user to ask about African agriculture, markets, climate, or policy."
)

_SOCIAL_MAX_TOKENS = 100

SOCIAL_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "greeting": _GREETING_ANSWER,
        "thanks": (
            "You're welcome. I am Ask ADZA — ask about African agriculture, markets, "
            "climate, or policy whenever you are ready."
        ),
        "bye": (
            "Goodbye. I am Ask ADZA — return anytime with questions about African "
            "agriculture, markets, climate, or policy."
        ),
        "out_of_scope": _OUT_OF_SCOPE_ANSWER,
    },
    "fr": {
        "greeting": (
            "Bonjour — je suis Ask ADZA, l'interface en langage naturel d'OpenTrace Africa. "
            "Posez vos questions sur l'agriculture, les marchés, le climat ou les politiques en Afrique."
        ),
        "thanks": "Je vous en prie. Je suis Ask ADZA — posez vos questions agricoles africaines quand vous voulez.",
        "bye": "Au revoir. Je suis Ask ADZA — revenez quand vous voulez pour l'agriculture africaine.",
        "out_of_scope": (
            "Je suis Ask ADZA pour l'intelligence agricole africaine. "
            "Cette demande est hors de mon périmètre — posez une question agricole africaine."
        ),
    },
    "sw": {
        "greeting": (
            "Habari — mimi ni Ask ADZA, kiolesura cha lugha ya asili cha OpenTrace Africa. "
            "Uliza kuhusu kilimo, masoko, hali ya hewa, au sera nchini Afrika."
        ),
        "thanks": "Karibu. Mimi ni Ask ADZA — uliza kuhusu kilimo cha Afrika utakapokuwa tayari.",
        "bye": "Kwaheri. Mimi ni Ask ADZA — rudi wakati wowote kwa maswali ya kilimo cha Afrika.",
        "out_of_scope": (
            "Mimi ni Ask ADZA kwa akili ya kilimo cha Afrika. "
            "Ombi hili liko nje ya uwezo wangu — uliza kuhusu kilimo cha Afrika."
        ),
    },
    "ha": {
        "greeting": (
            "Sannu — ni Ask ADZA, hanyar sadarwa ta OpenTrace Africa. "
            "Tambayi game da noma, kasuwa, yanayi, ko manufofi a Afirka."
        ),
        "thanks": "Maraba. Ni Ask ADZA — tambayi game da noma a Afirka a duk lokaci.",
        "bye": "Sai anjima. Ni Ask ADZA — dawo don tambayoyi game da noma a Afirka.",
        "out_of_scope": (
            "Ni Ask ADZA don bayanan noma na Afirka. "
            "Wannan ba na aikinmu ba — tambayi game da noma a Afirka."
        ),
    },
    "pcm": {
        "greeting": (
            "How far — I be Ask ADZA, OpenTrace Africa natural-language interface. "
            "Ask about agriculture, market, climate, or policy for Africa."
        ),
        "thanks": "You welcome. I be Ask ADZA — ask about African agriculture anytime.",
        "bye": "Bye. I be Ask ADZA — come back anytime for African agriculture questions.",
        "out_of_scope": (
            "I be Ask ADZA for African agricultural intelligence. "
            "That one no dey my scope — ask about African agriculture."
        ),
    },
    "pt": {
        "greeting": (
            "Olá — sou o Ask ADZA, a interface em linguagem natural da OpenTrace Africa. "
            "Pergunte sobre agricultura, mercados, clima ou políticas na África."
        ),
        "thanks": "De nada. Sou o Ask ADZA — pergunte sobre agricultura africana quando quiser.",
        "bye": "Tchau. Sou o Ask ADZA — volte quando quiser com perguntas sobre agricultura africana.",
        "out_of_scope": (
            "Sou o Ask ADZA para inteligência agrícola africana. "
            "Esse pedido está fora do meu escopo — pergunte sobre agricultura africana."
        ),
    },
    "ig": {
        "greeting": (
            "Ndewo — abụ m Ask ADZA, interface nke OpenTrace Africa. "
            "Jụọ gbasara ọrụ ugbo, ahia, ihu igwe, ma ọ bụ iwu na Afrịka."
        ),
        "thanks": "Ndo. Abụ m Ask ADZA — jụọ gbasara ọrụ ugbo Afrịka mgbe ọ bụla.",
        "bye": "Ka ọ dị. Abụ m Ask ADZA — laghachi maka ajụjụ ọrụ ugbo Afrịka.",
        "out_of_scope": (
            "Abụ m Ask ADZA maka ọgwụgwọ ọrụ ugbo Afrịka. "
            "Arịrịọ a adịghị n'ime oke m — jụọ gbasara ọrụ ugbo Afrịka."
        ),
    },
    "yo": {
        "greeting": (
            "Bawo — emi ni Ask ADZA, interface OpenTrace Africa. "
            "Beere nipa oko, ọja, oju-ọjọ, tabi eto ni Afirika."
        ),
        "thanks": "E se. Emi ni Ask ADZA — beere nipa oko Afirika nigbakugba.",
        "bye": "O dabo. Emi ni Ask ADZA — pada fun awọn ibeere oko Afirika.",
        "out_of_scope": (
            "Emi ni Ask ADZA fun imọ oko Afirika. "
            "Ibeere yii ko wa ni aaye mi — beere nipa oko Afirika."
        ),
    },
    "zu": {
        "greeting": (
            "Sawubona — ngingu-Ask ADZA, isixhumi esixhumanisa u-OpenTrace Africa. "
            "Buza ngolimo, imakethe, isimo sezulu, noma amathelo e-Afrika."
        ),
        "thanks": "Wamukelekile. Ngingu-Ask ADZA — buza ngolimo lwe-Afrika noma nini.",
        "bye": "Sala kahle. Ngingu-Ask ADZA — buya noma nini ngemibuzo yolimo lwe-Afrika.",
        "out_of_scope": (
            "Ngingu-Ask ADZA wobuhlakani bolimo lwe-Afrika. "
            "Lokho akukho emkhakheni wami — buza ngolimo lwe-Afrika."
        ),
    },
    "xh": {
        "greeting": (
            "Molo — ndingu-Ask ADZA, indlela yolwimi lwe-OpenTrace Africa. "
            "Buza malunga nezolimo, imarike, isimo sezulu, okanye imigaqo e-Afrika."
        ),
        "thanks": "Wamkelekile. Ndingu-Ask ADZA — buza malunga nezolimo zase-Afrika nanini na.",
        "bye": "Sala kakuhle. Ndingu-Ask ADZA — buya ngemibuzo yezolimo zase-Afrika.",
        "out_of_scope": (
            "Ndingu-Ask ADZA yobulumko bezolimo zase-Afrika. "
            "Le mceli ayikho kum — buza malunga nezolimo zase-Afrika."
        ),
    },
    "rw": {
        "greeting": (
            "Muraho — ndi Ask ADZA, uburyo bwo guhuza OpenTrace Africa. "
            "Baza ku by'ubuhinzi, isoko, ikirere, cyangwa amategeko mu Afrika."
        ),
        "thanks": "Murakoze. Ndi Ask ADZA — baza ku buhinzi bwa Afrika igihe cyose.",
        "bye": "Murabeho. Ndi Ask ADZA — garuka igihe cyose ku bibazo by'ubuhinzi bwa Afrika.",
        "out_of_scope": (
            "Ndi Ask ADZA ku bw'ubumenyi bw'ubuhinzi bwa Afrika. "
            "Iki gisaba ntikiri mu gaciro kacu — baza ku buhinzi bwa Afrika."
        ),
    },
}


def _normalize_social_text(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "").strip())
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"[\s!.?,:;]+", " ", text).strip()
    if not text:
        return ""
    if text.isascii():
        return text.casefold()
    return text


def _matches_phrase(normalized: str, phrase: str) -> bool:
    target = _normalize_social_text(phrase)
    if not target or not normalized:
        return False
    if normalized == target:
        return True
    if normalized.startswith(target + " "):
        return True
    # Mild elongation for Latin greetings: hiii → hi, mercii → merci
    if target.isascii() and normalized.startswith(target):
        tail = normalized[len(target) :]
        return bool(re.fullmatch(r"[aeiouh]+", tail))
    if target.isascii() and len(target) >= 2:
        stem = target.rstrip("aeiou")
        if stem and normalized.startswith(stem):
            tail = normalized[len(stem) :]
            return bool(re.fullmatch(r"[aeiouh]+", tail))
    return False


def classify_social_subkind(query: str) -> SocialSubKind | None:
    """Return greeting/thanks/bye when query is a pure social courtesy message."""
    q = (query or "").strip()
    if not q or _AG_SUBSTANCE_RE.search(q):
        return None
    normalized = _normalize_social_text(q)
    for phrases in _SOCIAL_PHRASES.values():
        for phrase, subkind in phrases:
            if _matches_phrase(normalized, phrase):
                return subkind
    return None


def is_greeting_query(query: str) -> bool:
    """Return True for short social greetings / courtesy messages (all supported langs)."""
    return classify_social_subkind(query) is not None


def early_non_rag_route(raw_query: str) -> EarlyRoute | None:
    """
    Classify raw user text before memory enrichment or decompose LLM.

    Priority: meta → product → greeting → out_of_scope → None (full RAG).
    """
    q = (raw_query or "").strip()
    if not q:
        return None
    if is_meta_query(q):
        return "meta"
    if is_help_query(q):
        return "help"
    if is_product_query(q, {}):
        return "product"
    if is_greeting_query(q):
        return "greeting"
    if is_out_of_scope_query(q, None):
        return "out_of_scope"
    return None


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
    if is_greeting_query(q) or is_meta_query(q) or is_help_query(q) or is_product_query(q, decomposition):
        return False
    if not _facets_empty(decomposition):
        return False
    if _AG_SUBSTANCE_RE.search(q):
        return False
    if _OFF_TOPIC_RE.search(q):
        return True
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


def _template_key(kind: SocialKind, query: str) -> str:
    if kind == "out_of_scope":
        return "out_of_scope"
    subkind = classify_social_subkind(query)
    return subkind or "greeting"


def static_social_answer(kind: SocialKind, *, query: str = "", answer_lang: str | None = None) -> str:
    lang = (answer_lang or detect_answer_language(query)).strip().lower()
    key = _template_key(kind, query)
    by_lang = SOCIAL_TEMPLATES.get(lang) or {}
    text = by_lang.get(key) or by_lang.get("greeting")
    if text:
        return _with_footer(text)
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

    Prefer static per-language templates; tiny LLM only when no template exists.
    """
    lang = (answer_lang or detect_answer_language(query)).strip().lower()
    key = _template_key(kind, query)
    by_lang = SOCIAL_TEMPLATES.get(lang) or {}
    template = by_lang.get(key) or (by_lang.get("greeting") if kind == "greeting" else by_lang.get("out_of_scope"))
    if template:
        return _with_footer(template)
    if is_english_answer_lang(lang):
        return static_social_answer(kind, query=query, answer_lang=lang)

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
    raw = _call_llama(messages, purpose="generate_social", max_tokens=_SOCIAL_MAX_TOKENS)
    fallback = _GREETING_ANSWER if kind == "greeting" else _OUT_OF_SCOPE_ANSWER
    answer = raw.strip() if raw else fallback
    return _with_footer(answer)


__all__ = [
    "EarlyRoute",
    "SocialKind",
    "SocialSubKind",
    "SOCIAL_TEMPLATES",
    "early_non_rag_route",
    "is_greeting_query",
    "is_help_query",
    "is_out_of_scope_query",
    "classify_social_query",
    "classify_social_subkind",
    "static_social_answer",
    "generate_social_answer",
]
