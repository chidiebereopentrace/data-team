"""
Document-level bibliographic metadata for research PDFs (citation / attribution).

Sidecar (highest priority):
  - ``{pdf}.meta.json`` next to the PDF
  - ``research_bibliography.json`` in the input folder (keyed by PDF filename or stem)

Heuristic fallback scans the first pages for title, authors, journal, year, and DOI.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.rag.text_processors.preprocess.unstructured_fast import ParsedElement

_DOI_RE = re.compile(
    r"(?:doi[:\s]*|(?:dx\.)?doi\.org/|https?://(?:dx\.)?doi\.org/)"
    r"(10\.\d{4,9}/[^\s\]>\"\'\),]+)",
    re.IGNORECASE,
)
_YEAR_PAREN_RE = re.compile(r"\((19|20)\d{2}\)")
_YEAR_TOKEN_RE = re.compile(r"\b(19|20)\d{2}\b")
_ISSN_RE = re.compile(r"\bISSN[:\s]+[\d\-Xx]+", re.IGNORECASE)
_AFFILIATION_START_RE = re.compile(
    r"\s*\d*\s*(?:School|Department|Institute|Institut|Centre|Center|University|Université|Faculty|College)\b",
    re.IGNORECASE,
)
_AUTHOR_LINE_RE = re.compile(
    r"^[A-ZÀ-ÖØ-Ý][\w\s\.\-'’]+?\d+\*?(?:,\s*[A-ZÀ-ÖØ-Ý][\w\s\.\-'’]+?\d+\*?){1,}",
)
_JOURNAL_HEADER_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9 &\-]{4,80}?)\s+(?:NS|Vol\.?|Volume|n[o°]\.?)\s*\d+.*\((19|20)\d{2}\)",
    re.IGNORECASE,
)
_VOLUME_PAGES_RE = re.compile(
    r"(?:No\.?\s*\d+[:\s]*(\d+)\s*[-–]\s*(\d+)|\((19|20)\d{2}\)\s*[:\s]*(\d+)\s*[-–]\s*(\d+))",
    re.IGNORECASE,
)

_BIBLIO_KEYS = (
    "article_title",
    "authors",
    "publication_year",
    "journal",
    "doi",
    "volume",
    "issue",
    "pages",
)


@dataclass
class BibliographicMetadata:
    article_title: str = ""
    authors: str = ""
    publication_year: str = ""
    journal: str = ""
    doi: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    metadata_source: str = ""  # sidecar | heuristic | sidecar+heuristic

    def to_metadata_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key in _BIBLIO_KEYS:
            val = getattr(self, key, "")
            if isinstance(val, str) and val.strip():
                out[key] = val.strip()
        if self.metadata_source:
            out["bibliography_source"] = self.metadata_source
        return out

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, source: str = "") -> BibliographicMetadata:
        def _s(key: str) -> str:
            raw = data.get(key)
            if raw is None:
                return ""
            return str(raw).strip()

        year = _s("publication_year") or _s("year")
        if year.isdigit() and len(year) == 4:
            year_str = year
        else:
            year_str = year[:4] if re.match(r"^\d{4}", year) else year
        return cls(
            article_title=_s("article_title") or _s("title"),
            authors=_s("authors") or _s("author"),
            publication_year=year_str,
            journal=_s("journal") or _s("venue"),
            doi=_s("doi"),
            volume=_s("volume"),
            issue=_s("issue"),
            pages=_s("pages"),
            metadata_source=source,
        )


def _merge_biblio(base: BibliographicMetadata, override: BibliographicMetadata) -> BibliographicMetadata:
    data = asdict(base)
    for key in _BIBLIO_KEYS:
        new_val = getattr(override, key, "")
        if isinstance(new_val, str) and new_val.strip():
            data[key] = new_val.strip()
    sources = [s for s in (base.metadata_source, override.metadata_source) if s]
    data["metadata_source"] = "+".join(dict.fromkeys(sources)) if sources else ""
    return BibliographicMetadata(**data)


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_sidecar_metadata(pdf_path: Path, *, input_dir: Path | None = None) -> BibliographicMetadata | None:
    candidates: list[Path] = [
        pdf_path.with_suffix(pdf_path.suffix + ".meta.json"),
        pdf_path.with_suffix(".meta.json"),
    ]
    search_roots = [pdf_path.parent]
    if input_dir is not None:
        search_roots.append(input_dir)
    for root in search_roots:
        candidates.append(root / "research_bibliography.json")
        candidates.append(root / "bibliography.json")

    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        raw = _load_json(path)
        if raw is None:
            continue
        if path.name in ("research_bibliography.json", "bibliography.json") and isinstance(raw, dict):
            for key in (pdf_path.name, pdf_path.stem, str(pdf_path)):
                entry = raw.get(key)
                if isinstance(entry, dict):
                    return BibliographicMetadata.from_mapping(entry, source="sidecar")
            continue
        if isinstance(raw, dict):
            return BibliographicMetadata.from_mapping(raw, source="sidecar")
    return None


def _first_page_text(full_text: str, *, limit: int = 4500) -> str:
    return (full_text or "")[:limit]


def _extract_doi(text: str) -> str:
    for match in _DOI_RE.finditer(text):
        doi = match.group(1).rstrip(".,;)")
        if doi.startswith("10."):
            return doi
    return ""


def _extract_year(text: str, doi: str = "") -> str:
    if doi:
        m = re.search(r"/(19|20)\d{2}/", doi)
        if m:
            return m.group(0).strip("/")
    for match in _YEAR_PAREN_RE.finditer(text[:6000]):
        year = match.group(0).strip("()")
        if 1950 <= int(year) <= 2035:
            return year
    for match in _JOURNAL_HEADER_RE.finditer(text[:6000]):
        return match.group(2)
    for match in _YEAR_TOKEN_RE.finditer(text[:2500]):
        year = match.group(0)
        if 1950 <= int(year) <= 2035:
            return year
    return ""


def _extract_journal(text: str, elements: list[ParsedElement]) -> str:
    for el in elements[:25]:
        title = (el.text or "").strip()
        m = _JOURNAL_HEADER_RE.match(title)
        if m:
            return m.group(1).strip()
        if _ISSN_RE.search(title) and len(title) < 180:
            return title.split("ISSN")[0].strip(" -–|,")
    for line in text.splitlines()[:40]:
        stripped = line.strip()
        m = _JOURNAL_HEADER_RE.match(stripped)
        if m:
            return m.group(1).strip()
        if _ISSN_RE.search(stripped) and 10 < len(stripped) < 180:
            return stripped.split("ISSN")[0].strip(" -–|,")
    return ""


def _looks_like_title(line: str) -> bool:
    s = line.strip()
    if len(s) < 25 or len(s) > 350:
        return False
    if _AUTHOR_LINE_RE.match(s):
        return False
    if re.match(r"^\d+$", s):
        return False
    if re.search(r"\b(?:abstract|introduction|résumé|issn|doi|copyright)\b", s, re.I):
        return False
    letters = sum(1 for c in s if c.isalpha())
    if letters < 15:
        return False
    upper = sum(1 for c in s if c.isupper())
    if upper >= max(8, int(letters * 0.55)):
        return True
    words = s.split()
    if len(words) >= 4 and sum(1 for w in words if w[:1].isupper()) >= len(words) * 0.6:
        return True
    return False


def _extract_title(text: str, elements: list[ParsedElement]) -> str:
    for el in elements[:20]:
        kind = (el.element_type or "").lower()
        if kind in ("title", "header"):
            candidate = (el.text or "").strip()
            if _looks_like_title(candidate):
                return candidate
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text[:3500]) if p.strip()]
    best = ""
    best_score = 0
    for para in paragraphs:
        for line in para.splitlines():
            line = line.strip()
            if not _looks_like_title(line):
                continue
            score = len(line) + (20 if line.isupper() else 0)
            if score > best_score:
                best = line
                best_score = score
    return best


def _clean_authors(line: str) -> str:
    s = re.sub(r"\s+", " ", line.strip())
    m = _AFFILIATION_START_RE.search(s)
    if m:
        s = s[: m.start()]
    s = re.sub(r"\d+\*?", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r",\s*,", ", ", s).strip(" ,;")
    return s[:500]


def _extract_authors(text: str, elements: list[ParsedElement]) -> str:
    for el in elements[:25]:
        candidate = (el.text or "").strip()
        if _AUTHOR_LINE_RE.match(candidate):
            return _clean_authors(candidate)
    lines = [ln.strip() for ln in text.splitlines()[:35] if ln.strip()]
    for i, line in enumerate(lines):
        if _AUTHOR_LINE_RE.match(line):
            return _clean_authors(line)
        if i + 1 < len(lines) and _looks_like_title(line) and _AUTHOR_LINE_RE.match(lines[i + 1]):
            return _clean_authors(lines[i + 1])
    return ""


def _extract_pages(text: str) -> str:
    m = _VOLUME_PAGES_RE.search(text[:5000])
    if not m:
        return ""
    if m.group(1) and m.group(2):
        return f"{m.group(1)}-{m.group(2)}"
    if m.group(4) and m.group(5):
        return f"{m.group(4)}-{m.group(5)}"
    return ""


def extract_bibliographic_metadata(
    full_text: str,
    elements: list[ParsedElement],
) -> BibliographicMetadata:
    head = _first_page_text(full_text)
    doi = _extract_doi(head) or _extract_doi(full_text[:12000])
    year = _extract_year(head, doi=doi)
    journal = _extract_journal(head, elements)
    title = _extract_title(head, elements)
    authors = _extract_authors(head, elements)
    pages = _extract_pages(head)
    return BibliographicMetadata(
        article_title=title,
        authors=authors,
        publication_year=year,
        journal=journal,
        doi=doi,
        pages=pages,
        metadata_source="heuristic",
    )


def resolve_bibliographic_metadata(
    pdf_path: Path,
    full_text: str,
    elements: list[ParsedElement],
    *,
    input_dir: Path | None = None,
) -> dict[str, str]:
    sidecar = load_sidecar_metadata(pdf_path, input_dir=input_dir)
    heuristic = extract_bibliographic_metadata(full_text, elements)
    if sidecar is None:
        return heuristic.to_metadata_dict()
    merged = _merge_biblio(heuristic, sidecar)
    if sidecar.metadata_source:
        merged.metadata_source = sidecar.metadata_source + (
            "+heuristic" if heuristic.metadata_source else ""
        )
    return merged.to_metadata_dict()


def format_academic_citation(meta: dict[str, Any]) -> str:
    """Compact citation string for the Sources block (not inline footnotes)."""
    authors = str(meta.get("authors") or "").strip()
    year = str(meta.get("publication_year") or meta.get("year") or "").strip()
    title = str(meta.get("article_title") or meta.get("title") or "").strip()
    journal = str(meta.get("journal") or "").strip()
    doi = str(meta.get("doi") or "").strip()

    if authors:
        lead = f"{authors} ({year})" if year else authors
        parts = [lead]
        if title:
            parts.append(title)
    elif title:
        parts = [f"{title} ({year})" if year else title]
    else:
        lead = journal or (f"DOI {doi}" if doi else "Document")
        parts = [f"{lead} ({year})" if year else lead]

    if journal and journal not in parts[0]:
        parts.append(journal)
    if doi:
        parts.append(f"DOI {doi}")
    return ". ".join(parts)
