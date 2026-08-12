"""Build OpenTrace RAG pipeline architecture PDF with embedded diagram PNGs."""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from rag_architecture_content import (
    BQ_ENRICH_BULLETS,
    BQ_RETRIEVAL_BULLETS,
    CORPUS_TABLE,
    DIAGRAM_SECTIONS,
    DIAGRAMS_PNG_DIR,
    ENV_VARS,
    INGEST_ARCHITECTURE_BULLETS,
    INGEST_ERD_ENTITIES,
    INGEST_ERD_RELATIONSHIPS,
    INFRA_ERD_ENTITIES,
    INFRA_ERD_RELATIONSHIPS,
    LLM_MATRIX,
    NODE_INVENTORY,
    NODE_SPECS,
    OBSERVABILITY_BULLETS,
    PURPOSE_BULLETS,
    RAG_STATE_FIELDS,
    ROUTING_TABLE,
    RUNTIME_ERD_ENTITIES,
    RUNTIME_ERD_RELATIONSHIPS,
    VECTOR_RETRIEVAL_BULLETS,
)

GREEN = colors.HexColor("#2E7D32")
LIGHT_GREEN = colors.HexColor("#E8F5E9")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ArchTitle",
            parent=base["Heading1"],
            fontSize=22,
            textColor=GREEN,
            spaceAfter=12,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "ArchSubtitle",
            parent=base["Normal"],
            fontSize=12,
            textColor=colors.HexColor("#444444"),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "h1": ParagraphStyle("ArchH1", parent=base["Heading1"], fontSize=16, textColor=GREEN, spaceBefore=14, spaceAfter=8),
        "h2": ParagraphStyle("ArchH2", parent=base["Heading2"], fontSize=13, textColor=GREEN, spaceBefore=10, spaceAfter=6),
        "h3": ParagraphStyle("ArchH3", parent=base["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("ArchBody", parent=base["Normal"], fontSize=10, leading=14),
        "bullet": ParagraphStyle("ArchBullet", parent=base["Normal"], fontSize=10, leading=13, leftIndent=14, bulletIndent=0),
    }


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, col_widths: list[float] | None = None) -> Table:
    data = [list(headers)] + [[str(c) for c in row] for row in rows]
    tbl = Table(data, repeatRows=1, colWidths=col_widths)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GREEN),
                ("TEXTCOLOR", (0, 0), (-1, 0), GREEN),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
            ]
        )
    )
    return tbl


def _bullets(st: dict[str, ParagraphStyle], items: Sequence[str]) -> list[Any]:
    return [Paragraph(f"• {_escape(item)}", st["bullet"]) for item in items]


def _diagram_image(png_path: Path, *, max_width: float, max_height: float, landscape_page: bool = False) -> Image:
    img = Image(str(png_path))
    iw, ih = img.imageWidth, img.imageHeight
    if iw <= 0 or ih <= 0:
        return img
    scale = min(max_width / iw, max_height / ih)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    img.hAlign = "CENTER"
    return img


def _add_erd_tables(st: dict[str, ParagraphStyle], story: list[Any], entities: Sequence[tuple[str, Sequence[Sequence[str]]]], relationships: Sequence[Sequence[str]]) -> None:
    for entity_name, fields in entities:
        story.append(Paragraph(_escape(entity_name), st["h3"]))
        story.append(_table(("Field", "Type", "Key"), fields))
        story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Relationships", st["h3"]))
    story.append(_table(("From", "Cardinality", "To", "Label"), relationships))
    story.append(Spacer(1, 0.15 * inch))


def _add_node_section(st: dict[str, ParagraphStyle], story: list[Any], spec: dict[str, Any]) -> None:
    story.append(Paragraph(f"Node: {_escape(spec['name'])}", st["h2"]))
    story.append(Paragraph(_escape(spec["purpose"]), st["body"]))
    story.append(
        _table(
            ("Attribute", "Value"),
            [
                ("Module(s)", spec["module"]),
                ("Reads state", spec["reads"]),
                ("Writes state", spec["writes"]),
                ("LLM", spec["llm"]),
                ("External I/O", spec["external"]),
                ("Failure behavior", spec["failure"]),
            ],
        )
    )
    story.append(Paragraph("Sub-steps", st["h3"]))
    story.extend(_bullets(st, spec["substeps"]))
    story.append(Spacer(1, 0.12 * inch))


def build_architecture_pdf(output_path: Path, *, png_dir: Path = DIAGRAMS_PNG_DIR) -> None:
    st = _styles()
    story: list[Any] = []

    when = date.today().isoformat()
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("OpenTrace RAG Pipeline Architecture", st["title"]))
    story.append(Paragraph("Full LangGraph pipeline, data flows, and node reference", st["subtitle"]))
    story.append(Paragraph(f"Version 1.0  |  Generated {when}", st["subtitle"]))
    story.append(PageBreak())

    story.append(Paragraph("Purpose and design principles", st["h1"]))
    story.extend(_bullets(st, PURPOSE_BULLETS))
    story.append(Spacer(1, 0.2 * inch))

    for stem, title, caption in DIAGRAM_SECTIONS:
        png = png_dir / f"{stem}.png"
        if not png.exists():
            raise FileNotFoundError(f"Missing diagram PNG: {png}. Run render_architecture_diagrams.py first.")

        story.append(Paragraph(title, st["h1"]))
        story.append(Paragraph(_escape(caption), st["body"]))
        story.append(Spacer(1, 0.1 * inch))

        use_landscape = stem == "runtime_graph"
        page_w = landscape(A4)[0] if use_landscape else A4[0]
        page_h = landscape(A4)[1] if use_landscape else A4[1]
        max_w = page_w - 1.5 * inch
        max_h = page_h - 2.5 * inch
        story.append(_diagram_image(png, max_width=max_w, max_height=max_h, landscape_page=use_landscape))
        story.append(PageBreak())

    story.append(Paragraph("Node inventory", st["h1"]))
    story.append(_table(("Node", "Type", "Description"), NODE_INVENTORY))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Conditional routing table", st["h1"]))
    story.append(_table(("Stage", "Condition", "Target node", "Notes"), ROUTING_TABLE))
    story.append(PageBreak())

    story.append(Paragraph("Runtime domain — entity tables", st["h1"]))
    _add_erd_tables(st, story, RUNTIME_ERD_ENTITIES, RUNTIME_ERD_RELATIONSHIPS)
    story.append(PageBreak())

    story.append(Paragraph("Ingest and vector store — entity tables", st["h1"]))
    _add_erd_tables(st, story, INGEST_ERD_ENTITIES, INGEST_ERD_RELATIONSHIPS)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Infrastructure — entity tables", st["h1"]))
    _add_erd_tables(st, story, INFRA_ERD_ENTITIES, INFRA_ERD_RELATIONSHIPS)
    story.append(PageBreak())

    story.append(Paragraph("BQ enrich and ACF metadata", st["h1"]))
    story.extend(_bullets(st, BQ_ENRICH_BULLETS))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Node reference", st["h1"]))
    story.append(Paragraph("Detailed behavior for each LangGraph node.", st["body"]))
    for spec in NODE_SPECS:
        _add_node_section(st, story, spec)
    story.append(PageBreak())

    story.append(Paragraph("RAGGraphState field catalog", st["h1"]))
    story.append(_table(("Field", "Type", "Description"), RAG_STATE_FIELDS))
    story.append(PageBreak())

    story.append(Paragraph("Retrieval subsystems", st["h1"]))
    story.append(Paragraph("Vector retrieval", st["h2"]))
    story.extend(_bullets(st, VECTOR_RETRIEVAL_BULLETS))
    story.append(Paragraph("Corpus router (six collections)", st["h2"]))
    story.append(_table(("Key", "Qdrant collection", "Role", "Key payload indexes"), CORPUS_TABLE))
    story.append(Paragraph("BigQuery retrieval", st["h2"]))
    story.extend(_bullets(st, BQ_RETRIEVAL_BULLETS))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Ingest architecture (offline)", st["h1"]))
    story.extend(_bullets(st, INGEST_ARCHITECTURE_BULLETS))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("LLM usage matrix", st["h1"]))
    story.append(_table(("Step", "Module", "Backend", "Notes"), LLM_MATRIX))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Observability and debugging", st["h1"]))
    story.extend(_bullets(st, OBSERVABILITY_BULLETS))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Environment variables (subset)", st["h1"]))
    story.append(_table(("Variable", "Purpose"), ENV_VARS))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Related documentation", st["h1"]))
    story.extend(
        _bullets(
            st,
            [
                "OpenTrace-RAG-Pipeline-Architecture.docx — Word version with ASCII diagrams",
                "OpenTrace-RAG-API-Documentation.docx — HTTP /query reference",
                "OpenTrace-Chatbot-API-v1-Documentation.docx — public v1 chat + artifacts",
                "ml/rag/docs/API.md — markdown API reference",
                "ml/rag/ARCHITECTURE.md — package architecture (markdown)",
                "Regenerate PDF: python scripts/generate_rag_architecture_pdf.py from ml-eng/",
            ],
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="OpenTrace RAG Pipeline Architecture",
        author="OpenTrace",
    )
    doc.build(story)


__all__ = ["build_architecture_pdf"]
