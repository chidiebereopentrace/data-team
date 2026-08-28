"""
Convert news DOCX files into the same shape as scraped .txt (YAML front matter + body).
"""
from __future__ import annotations

from pathlib import Path

from ml.rag.text_processors.preprocess.docx_io import read_docx_text


def docx_to_news_txt_content(path: Path) -> str:
    """Return synthetic .txt content with minimal YAML front matter + body."""
    title = path.stem.replace("_", " ").strip()
    body = read_docx_text(path).strip()
    lines = [
        "---",
        f"title: {title!r}",
        "body_source: docx",
        f"source_file: {path}",
        "---",
        "",
        body,
    ]
    return "\n".join(lines)
