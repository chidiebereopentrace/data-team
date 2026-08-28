#!/usr/bin/env python3
"""Generate OpenTrace RAG pipeline architecture PDF with Mermaid diagram PNGs."""
from __future__ import annotations

import argparse

from architecture_pdf_builder import build_architecture_pdf
from rag_architecture_content import DOCS_DIR, OUTPUT_PDF
from render_architecture_diagrams import render_diagrams


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG pipeline architecture PDF.")
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Build PDF from existing PNGs in ml/rag/docs/diagrams/png/ (skip mmdc).",
    )
    args = parser.parse_args()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_render:
        render_diagrams()

    build_architecture_pdf(OUTPUT_PDF)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
