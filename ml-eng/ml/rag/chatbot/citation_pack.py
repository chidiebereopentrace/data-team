"""Citation marker cleanup and the source ids an answer actually cites."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

_SOURCE_REF_RE = re.compile(
    r"\[Source\s+(\d+)\]"
    r"|\[(?!\s*Source\s)(\d+)\]"
    r"|\bSource\s+(\d+)\b",
    re.IGNORECASE,
)
_VERBOSE_INLINE_REF_RE = re.compile(r"\[Source\s+(\d+)\s*\|[^\]]*\]", re.IGNORECASE)
_SOURCE_MULTI_REF_RE = re.compile(r"\[Sources?\s+([\d,\sand]+)\]", re.IGNORECASE)
_PLAIN_MULTI_REF_RE = re.compile(r"\[(?!\s*Source\s)(\d+(?:\s*[,–-]\s*\d+)+)\]")
_PIPE_LABEL_REF_RE = re.compile(
    r"\[(?:Wikipedia|News|Academic|Policy/Public|Policy|Public|Web|Structured\s+data)\s*\|[^\]]*\]",
    re.IGNORECASE,
)
_NON_PLAIN_BRACKET_RE = re.compile(r"\[(?!\d+\])[^\]]*\]")

_EMPTY_PAREN_RE = re.compile(r"\(\s*[,.;:]*\s*\)")
_SOCIAL_URL_HOSTS = ("linkedin.com", "twitter.com", "x.com", "facebook.com", "fb.com")
_KNOWN_ANALYTICAL_HEADINGS = (
    "Key Findings",
    "Regional & Country Picture",
    "Production, Trade & Markets",
    "Drivers & Context",
    "Data Notes",
    "Executive summary",
    "Regional overview",
)
_MODEL_SOURCES_APPENDIX_RE = re.compile(
    r"\n+(?:Sources|References|Bibliography)\s*:?\s*\n[\s\S]*\Z",
    re.IGNORECASE,
)


def _expand_id_bracket_content(inner: str) -> str:
    """Expand comma/range id lists into sequential plain [N] markers."""
    text = (inner or "").strip()
    if not text:
        return ""
    range_m = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", text)
    if range_m:
        start, end = int(range_m.group(1)), int(range_m.group(2))
        if start <= end and end <= 30 and (end - start) <= 20:
            return "".join(f"[{i}]" for i in range(start, end + 1))
        return ""
    ids = re.findall(r"\d+", text)
    return "".join(f"[{i}]" for i in ids)


def _strip_non_plain_brackets(text: str) -> str:
    """Remove bracket labels that are not plain numeric footnotes."""
    return _NON_PLAIN_BRACKET_RE.sub("", text or "")


def _collapse_duplicate_citation_markers(text: str) -> str:
    """Collapse immediate duplicate footnote markers such as [3][3] or ([3]) [3]."""
    if not text:
        return text
    out = text
    out = re.sub(r"\(\[(\d+)\]\)\s*\[\1\]", r"[\1]", out)
    out = re.sub(r"\[(\d+)\]\s*\(\[\1\]\)", r"[\1]", out)
    prev = None
    while prev != out:
        prev = out
        out = re.sub(r"\[(\d+)\]\s*\[\1\]", r"[\1]", out)
    return out


def _strip_empty_citation_parens(text: str) -> str:
    """Remove parenthetical wrappers left empty after citation stripping, e.g. ()."""
    if not text:
        return text
    prev = None
    out = text
    while prev != out:
        prev = out
        out = _EMPTY_PAREN_RE.sub("", out)
    return out


def _cleanup_citation_spacing(text: str) -> str:
    """Tidy spacing after marker removal without collapsing newlines."""
    text = _strip_empty_citation_parens(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;])", r"\1", text)
    text = re.sub(r"\s+\.", ".", text)
    return text.strip()


def _looks_like_body_prose(line: str) -> bool:
    """True when a citation line looks like dumped body text rather than a one-liner."""
    if not line:
        return True
    if "\n" in line:
        return True
    if len(line) > 300:
        return True
    stripped = line.strip()
    return stripped.startswith("In ,") or stripped.startswith(", ")


def _normalize_markdown_headings(text: str) -> str:
    """Ensure ATX headings start on their own line and split known headings from body."""
    if not text:
        return text
    out = re.sub(r"(\S)\s+(#{2,3}\s)", r"\1\n\n\2", text)
    for heading in _KNOWN_ANALYTICAL_HEADINGS:
        pattern = rf"(#{{2,3}}\s+{re.escape(heading)})\s+(\S.+)"
        out = re.sub(pattern, r"\1\n\n\2", out, flags=re.IGNORECASE)
    return out


def _is_social_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(s in host for s in _SOCIAL_URL_HOSTS)


def _pick_best_http_url(candidates: list[str]) -> str | None:
    http = [u.strip() for u in candidates if str(u or "").strip().startswith("http")]
    if not http:
        return None
    non_social = [u for u in http if not _is_social_url(u)]
    return non_social[0] if non_social else http[0]


def _strip_model_sources_appendix(text: str) -> str:
    """Remove model-written Sources/References blocks that poison footnote extraction."""
    if not text:
        return text
    return _MODEL_SOURCES_APPENDIX_RE.sub("", text).strip()


def _answer_for_citation_extraction(answer: str) -> str:
    """Prose-only text used to detect which footnote numbers the model actually cited."""
    return _normalize_inline_citations(_strip_model_sources_appendix(answer))


def _normalize_inline_citations(text: str) -> str:
    """Collapse verbose model citations to Wikipedia-style footnote numbers [N]."""
    if not text:
        return text
    text = _VERBOSE_INLINE_REF_RE.sub(lambda m: f"[{m.group(1)}]", text)
    text = _SOURCE_MULTI_REF_RE.sub(lambda m: _expand_id_bracket_content(m.group(1)), text)
    text = re.sub(r"\[Source\s+(\d+)\]", r"[\1]", text, flags=re.IGNORECASE)
    text = _PLAIN_MULTI_REF_RE.sub(lambda m: _expand_id_bracket_content(m.group(1)), text)
    text = _PIPE_LABEL_REF_RE.sub("", text)
    text = re.sub(r"(?<!\[)\bSource\s+(\d+)\b(?!\])", r"[\1]", text, flags=re.IGNORECASE)
    text = _collapse_duplicate_citation_markers(text)
    text = _strip_non_plain_brackets(text)
    return text


def _strip_invalid_citation_markers(answer: str, source_registry: list[Any]) -> str:
    """Remove [N] / [Source N] footnotes that reference missing registry IDs."""
    if not answer:
        return answer
    text = _normalize_inline_citations(answer)
    if not source_registry:
        return _cleanup_citation_spacing(_strip_non_plain_brackets(text))
    valid = {ref.source_id for ref in source_registry}

    def _keep_bracket(m: re.Match[str]) -> str:
        try:
            n = int(m.group(1))
        except (TypeError, ValueError):
            return ""
        return m.group(0) if n in valid else ""

    text = re.sub(r"\[(?!\s*Source\s)(\d+)\]", _keep_bracket, text)
    text = re.sub(r"\[Source\s+(\d+)\]", _keep_bracket, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\[)\bSource\s+(\d+)\b(?!\])", _keep_bracket, text, flags=re.IGNORECASE)
    text = _strip_non_plain_brackets(text)
    return _cleanup_citation_spacing(text)


def _strip_all_inline_citation_markers(answer: str) -> str:
    """Remove every [N] / [Source N] marker when inline footnotes are disabled."""
    if not answer:
        return answer
    text = _normalize_inline_citations(answer)
    text = re.sub(r"\[(?!\s*Source\s)\d+\]", "", text)
    text = re.sub(r"\[Source\s+\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\[)\bSource\s+\d+\b(?!\])", "", text, flags=re.IGNORECASE)
    text = _strip_non_plain_brackets(text)
    return _cleanup_citation_spacing(text)


def extract_referenced_source_ids(answer: str) -> set[int]:
    """Return source IDs cited inline in answer prose ([N], [Source N], or Source N)."""
    normalized = _answer_for_citation_extraction(answer)
    ids: set[int] = set()
    for m in _SOURCE_REF_RE.finditer(normalized):
        for g in m.groups():
            if g:
                ids.add(int(g))
                break
    return ids

