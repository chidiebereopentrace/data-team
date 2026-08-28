"""Split markdown-ish report bodies into blocks and styled inline runs."""
from __future__ import annotations

import re
from xml.sax.saxutils import escape

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_BULLET_RE = re.compile(r"^[-*•]\s+")
_ORDERED_RE = re.compile(r"^\d+[.)]\s+")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")


def iter_inline_runs(text: str) -> list[tuple[str, str]]:
    """Return (kind, chunk) where kind is text, bold, or italic."""
    if not text:
        return []
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in _BOLD_RE.finditer(text):
        if m.start() > pos:
            parts.extend(_italic_runs(text[pos:m.start()]))
        parts.append(("bold", m.group(1)))
        pos = m.end()
    if pos < len(text):
        parts.extend(_italic_runs(text[pos:]))
    return parts or [("text", text)]


def _italic_runs(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in _ITALIC_RE.finditer(text):
        if m.start() > pos:
            parts.append(("text", text[pos:m.start()]))
        parts.append(("italic", m.group(1)))
        pos = m.end()
    if pos < len(text):
        parts.append(("text", text[pos:]))
    return parts or [("text", text)]


def iter_markdown_blocks(text: str) -> list[tuple[str, str]]:
    """Return (kind, content) where kind is paragraph, bullet, ordered, or heading."""
    blocks: list[tuple[str, str]] = []
    para_lines: list[str] = []

    def flush_para() -> None:
        if para_lines:
            blocks.append(("paragraph", " ".join(para_lines).strip()))
            para_lines.clear()

    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            flush_para()
            continue
        heading = _HEADING_RE.match(stripped)
        if heading:
            flush_para()
            blocks.append(("heading", heading.group(2).strip()))
            continue
        if _BULLET_RE.match(stripped):
            flush_para()
            blocks.append(("bullet", _BULLET_RE.sub("", stripped)))
            continue
        if _ORDERED_RE.match(stripped):
            flush_para()
            blocks.append(("ordered", _ORDERED_RE.sub("", stripped)))
            continue
        para_lines.append(stripped)
    flush_para()
    return blocks


def to_reportlab_html(text: str) -> str:
    """Escape text and wrap bold/italic runs for ReportLab Paragraphs."""
    bits: list[str] = []
    for kind, chunk in iter_inline_runs(text):
        escaped = escape(chunk)
        if kind == "bold":
            bits.append(f"<b>{escaped}</b>")
        elif kind == "italic":
            bits.append(f"<i>{escaped}</i>")
        else:
            bits.append(escaped)
    return "".join(bits)


__all__ = ["iter_inline_runs", "iter_markdown_blocks", "to_reportlab_html"]
