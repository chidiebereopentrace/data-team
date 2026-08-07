"""Build printable PDF reports."""
from __future__ import annotations

import io
from typing import Any


def build_pdf(
    *,
    title: str,
    sections: list[dict[str, str]],
    table_rows: list[dict[str, Any]] | None = None,
    chart_png: bytes | None = None,
    citations: list[dict[str, Any]] | None = None,
    acf_summary: str | None = None,
    filename: str = "report.pdf",
) -> tuple[bytes, str]:
    # Lazy imports — avoids loading reportlab at Railway startup when no export runs.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "OTTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=14,
        textColor=colors.HexColor("#2E7D32"),
    )
    body_style = ParagraphStyle("OTBody", parent=styles["Normal"], fontSize=10, leading=14)
    story: list[Any] = []

    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2 * inch))

    for sec in sections:
        heading = (sec.get("heading") or "").strip()
        body = (sec.get("body") or "").strip()
        if heading:
            story.append(Paragraph(heading, styles["Heading2"]))
        if body:
            for para in body.split("\n\n"):
                story.append(Paragraph(para.strip().replace("\n", "<br/>"), body_style))
                story.append(Spacer(1, 0.1 * inch))

    if chart_png:
        story.append(Paragraph("Chart", styles["Heading2"]))
        story.append(Image(io.BytesIO(chart_png), width=6 * inch, height=3.5 * inch))
        story.append(Spacer(1, 0.15 * inch))

    if table_rows:
        cols = list({k for row in table_rows for k in row})
        data = [cols] + [[str(row.get(c, "")) for c in cols] for row in table_rows[:30]]
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F5E9")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(Paragraph("Data table", styles["Heading2"]))
        story.append(tbl)
        story.append(Spacer(1, 0.15 * inch))

    if citations:
        story.append(Paragraph("Sources", styles["Heading2"]))
        for c in citations:
            cid = c.get("id", "")
            text = str(c.get("text", "")).replace("&", "&amp;").replace("<", "&lt;")
            story.append(Paragraph(f"[{cid}] {text}", body_style))

    if acf_summary:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("Confidence summary", styles["Heading2"]))
        story.append(Paragraph(acf_summary.replace("\n", "<br/>"), body_style))

    doc.build(story)
    name = filename if filename.endswith(".pdf") else f"{filename}.pdf"
    return buf.getvalue(), name


__all__ = ["build_pdf"]
