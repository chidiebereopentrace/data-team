"""
Unsourced-figure detection (ML-058, Sprint 2 Week 2).

Internal Test 2 found the system stating specific statistics that could not be
traced to any cited source. ``_SLIM_BASE_PROMPT`` already says "Never invent
statistics" -- but that is an instruction, not an enforcement. This module is
the enforcement half: after generation, check whether the numbers in the answer
actually appear in the evidence that was cited.

Design stance -- deliberately conservative, mirroring
``looks_like_degenerate_repetition``:

* Flag, never delete. Removing figures from otherwise-correct prose risks
  mangling good answers; a caveat plus an ACF cap is the honest signal.
* Skip number shapes that are not factual claims (years, list markers,
  section numbers, percentages already attached to a cited figure).
* A false positive on every answer would be worse than the bug, so matching
  is generous: digit-normalised substring matching against the full cited
  text, tolerant of thousands separators and decimal noise.

This cannot catch a hallucinated figure that happens to appear somewhere in a
cited chunk, and it does not verify that a number was used in the right
*context*. It catches the blatant case -- a figure with no basis in the
evidence at all.
"""

from __future__ import annotations

import re
from typing import Any

# Numbers we never treat as sourceable factual claims.
_YEAR_RE = re.compile(r"^(1[89]\d{2}|20\d{2}|21\d{2})$")

# A year-range number followed by a unit is a quantity, not a year:
# "1850 tonnes" and "2000 hectares" are figures; "in 2024," is not.
_UNIT_AFTER_RE = re.compile(
    r"^\s*(?:"
    r"t|kg|g|mt|ha|km|km2|m2|l|ml|"
    r"tonnes?|tons?|metric\s+tonnes?|kilograms?|grams?|"
    r"hectares?|acres?|litres?|liters?|bags?|sacks?|"
    r"usd|ksh|ngn|dollars?|shillings?|naira|"
    r"people|persons?|households?|farmers?|smallholders?|"
    r"units?|heads?|birds?|animals?"
    r")\b",
    re.IGNORECASE,
)

# A candidate figure: integers, decimals, thousands-separated, optional percent.
_NUMBER_RE = re.compile(
    r"(?<![\w.])"           # not mid-word / mid-version
    r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"\s*(%|percent|per\s?cent)?",
    re.IGNORECASE,
)

# Markdown list / heading markers: "1. ", "2) ", "## 3"
_LIST_MARKER_RE = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s*)?\d{1,2}[.)]\s")

# Inline citation markers [1] must never be read as figures.
_CITATION_MARKER_RE = re.compile(r"\[\d+\]")

# Ordinals and small counts carry little factual risk on their own.
_SMALL_NUMBER_CEILING = 10

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_UNSUPPORTED_LIMIT = (
    "The cited evidence does not support a numerical claim for this question."
)


def _normalise(value: str) -> str:
    """Strip thousands separators and a trailing .0 so 1,234 == 1234 == 1234.0."""
    out = value.replace(",", "").strip()
    if out.endswith(".0"):
        out = out[:-2]
    return out


def _evidence_text(cited_items: list[Any]) -> str:
    """Flatten cited evidence (SourceRef or raw dict) into one searchable blob."""
    parts: list[str] = []
    for ref in cited_items or []:
        item = getattr(ref, "item", None)
        if not isinstance(item, dict):
            item = ref if isinstance(ref, dict) else None
        if not isinstance(item, dict):
            continue
        for key in ("text", "content", "title", "snippet", "summary", "citation_line"):
            val = item.get(key)
            if isinstance(val, str) and val:
                parts.append(val)
        # Structured warehouse rows carry figures in arbitrary fields.
        for val in item.values():
            if isinstance(val, (int, float)):
                parts.append(str(val))
            elif isinstance(val, dict):
                for inner in val.values():
                    if isinstance(inner, (int, float, str)):
                        parts.append(str(inner))
        line = getattr(ref, "citation_line", None)
        if isinstance(line, str) and line:
            parts.append(line)
    return _normalise(" ".join(parts))


def _candidate_figures(answer: str) -> list[str]:
    """Pull figures from the answer that represent factual claims."""
    if not answer:
        return []
    text = _CITATION_MARKER_RE.sub(" ", answer)
    text = _LIST_MARKER_RE.sub(" ", text)

    found: list[str] = []
    seen: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(1)
        norm = _normalise(raw)
        if not norm or norm in seen:
            continue
        # A year-shaped number is only a year when no unit follows it --
        # otherwise "1850 tonnes" would be silently exempt from checking.
        if _YEAR_RE.match(norm):
            trailing = text[match.end(1) :]
            if match.group(2) or _UNIT_AFTER_RE.match(trailing):
                pass  # quantity, keep checking it
            else:
                continue
        try:
            as_float = float(norm)
        except ValueError:
            continue
        # Bare small integers ("three of the 5 regions") are rarely the
        # fabricated-statistic failure mode and drive false positives.
        if as_float <= _SMALL_NUMBER_CEILING and "." not in norm:
            continue
        seen.add(norm)
        found.append(norm)
    return found


def unsourced_figures(answer: str, cited_items: list[Any]) -> list[str]:
    """
    Return figures stated in ``answer`` that do not appear in cited evidence.

    Empty list means everything checked is traceable, or there was nothing
    worth checking.
    """
    figures = _candidate_figures(answer)
    if not figures:
        return []
    evidence = _evidence_text(cited_items)
    if not evidence:
        # No usable evidence text: do not claim every figure is invented.
        # The no-citations path already forces ACF to no_evidence.
        return []
    return [f for f in figures if f not in evidence]


def strip_unsourced_figures(answer: str, unsourced: list[str]) -> str:
    """Drop sentences that state a figure the cited evidence does not contain."""
    if not answer or not unsourced:
        return answer
    flagged = {_normalise(fig) for fig in unsourced}
    kept: list[str] = []
    for sentence in _SENTENCE_RE.split(answer.strip()):
        figures = {_normalise(fig) for fig in _candidate_figures(sentence)}
        if figures & flagged:
            continue
        if sentence.strip():
            kept.append(sentence.strip())
    if not kept:
        return _UNSUPPORTED_LIMIT
    return " ".join(kept)


def append_unsourced_caveat(answer: str, unsourced: list[str]) -> str:
    """Compatibility name: unsupported figures are removed, not disclaimed."""
    return strip_unsourced_figures(answer, unsourced)
