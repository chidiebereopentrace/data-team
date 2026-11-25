"""Write the weekly report as one HTML page and a matching PDF."""

from __future__ import annotations

import csv
import html
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

Bar = tuple[str, int]
COUNTRY_LIMIT = 12
ATTEMPTED = frozenset({"LOADED", "PARTIAL", "FAILED"})
SUMMARY_FIELDS = (
    ("Sources fetched", "sources_fetched"),
    ("Sources failed", "sources_failed"),
    ("Rows added", "rows_loaded"),
    ("Rows quarantined", "rows_quarantined"),
    ("Rows outside Africa", "rows_outside_africa"),
    ("Sources current", "sources_current"),
    ("Sources skipped", "sources_skipped"),
)
STATUSES = ("LOADED", "PARTIAL", "FAILED", "CURRENT", "SKIPPED")
ERROR_CATEGORIES = ("NONE", "MANIFEST", "SCHEMA", "AUTH", "NETWORK", "DATA_QUALITY", "PUBLISH")


def write_report(
    reports: Path,
    *,
    history: list[dict[str, object]] | None = None,
    publish_error: str = "",
    uploaded: bool = False,
) -> tuple[Path, Path, Path]:
    """Write the summary page, the PDF, and the filterable jobs page."""
    summary = json.loads((reports / "run.json").read_text(encoding="utf-8"))
    sources = _read_csv(reports / "ingest_sources.csv")
    coverage = _read_csv(reports / "ingest_coverage.csv")
    runs = _runs(summary, history or [])
    classes = _grouped(coverage, "indicator_class")
    countries = _grouped(coverage, "country")[:COUNTRY_LIMIT]
    note = _publish_note(publish_error, uploaded)
    html_path = reports / "ingest_report.html"
    pdf_path = reports / "ingest_report.pdf"
    jobs_path = reports / "ingest_jobs.html"
    html_path.write_text(
        _html(summary, sources, runs, classes, countries, note),
        encoding="utf-8",
    )
    _pdf(pdf_path, summary, sources, runs, classes, countries, note)
    jobs_path.write_text(_jobs_html(summary, sources, coverage), encoding="utf-8")
    return html_path, pdf_path, jobs_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if "country_name" in row or "country_iso3" in row:
            name = (row.get("country_name") or "").strip()
            iso3 = (row.get("country_iso3") or "").strip()
            row["country"] = f"{name} ({iso3})" if name and iso3 else name or iso3
    return rows


def _count(value: object) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def _runs(summary: dict[str, object], history: list[dict[str, object]]) -> list[Bar]:
    points: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for item in history:
        run_id = str(item.get("run_id") or "")
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        points.append((str(item.get("started_at") or ""), run_id, _count(item.get("rows_loaded"))))
    current = str(summary.get("run_id") or "")
    if current and current not in seen:
        points.append((str(summary.get("started_at") or ""), current, _count(summary.get("rows_loaded"))))
    points.sort(key=lambda item: (item[0], item[1]))
    return [(f"{started[:10]} {run_id}".strip(), rows) for started, run_id, rows in points]


def _grouped(rows: list[dict[str, str]], key: str) -> list[Bar]:
    totals: dict[str, int] = {}
    for row in rows:
        label = (row.get(key) or "").strip() or "(blank)"
        totals[label] = totals.get(label, 0) + _count(row.get("rows_loaded"))
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


def _publish_note(publish_error: str, uploaded: bool) -> str:
    if publish_error:
        return f"BigQuery load failed: {publish_error}"
    if uploaded:
        return "BigQuery load finished."
    return ""


def _html(
    summary: dict[str, object],
    sources: list[dict[str, str]],
    runs: list[Bar],
    classes: list[Bar],
    countries: list[Bar],
    note: str,
) -> str:
    status = str(summary.get("status") or "")
    title = f"Ingest {summary.get('run_id') or ''}"
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(title)}</title></head>",
        "<body style=\"margin:0;padding:24px;background:#f6f8fa;color:#1f2328;"
        "font-family:Segoe UI,Helvetica,Arial,sans-serif;\">",
        "<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" "
        "style=\"max-width:720px;margin:0 auto;background:#ffffff;border:1px solid #d0d7de;\">",
        "<tr><td style=\"padding:24px;\">",
        f"<p style=\"margin:0 0 8px;font-size:13px;color:{_status_color(status)};"
        f"font-weight:700;\">{html.escape(status)}</p>",
        f"<h1 style=\"margin:0 0 16px;font-size:22px;\">{html.escape(title)}</h1>",
        _metrics(summary),
        _chart("Rows added by run", runs, "No earlier runs."),
        _chart("Rows added by indicator", classes, "No prepared rows."),
        _chart("Rows added by country", countries, "No prepared rows."),
        _source_table(_attempted(sources)),
        "<p style=\"margin:16px 0 0;font-size:13px;color:#59636e;\">"
        "Coverage is attached as ingest_coverage.csv, one row per indicator class, country, and year."
        "</p>",
    ]
    if note:
        parts.append(f"<p style=\"margin:8px 0 0;font-size:13px;\">{html.escape(note)}</p>")
    parts.append("</td></tr></table></body></html>\n")
    return "".join(parts)


def _status_color(status: str) -> str:
    if status == "FINISHED":
        return "#1a7f37"
    if status == "PARTIAL":
        return "#9a6700"
    return "#cf222e"


def _metrics(summary: dict[str, object]) -> str:
    fields = [(label, summary.get(key, 0)) for label, key in SUMMARY_FIELDS]
    cells = []
    for label, value in fields:
        cells.append(
            "<td style=\"padding:8px 12px 8px 0;vertical-align:top;\">"
            f"<div style=\"font-size:12px;color:#59636e;\">{html.escape(label)}</div>"
            f"<div style=\"font-size:20px;font-weight:700;\">{html.escape(f'{_count(value):,}')}</div>"
            "</td>"
        )
    return "<table cellpadding=\"0\" cellspacing=\"0\"><tr>" + "".join(cells) + "</tr></table>"


def _chart(title: str, bars: list[Bar], empty: str) -> str:
    heading = f"<h2 style=\"margin:24px 0 8px;font-size:16px;\">{html.escape(title)}</h2>"
    if not bars:
        return heading + f"<p style=\"margin:0;font-size:14px;\">{html.escape(empty)}</p>"
    largest = max(value for _label, value in bars) or 1
    rows = []
    for label, value in bars:
        width = max(1, round(100 * value / largest)) if value else 0
        bar = (
            f"<td width=\"{width}%\" bgcolor=\"#1f6feb\" "
            "style=\"background:#1f6feb;height:14px;font-size:1px;\">&nbsp;</td>"
        )
        gap = f"<td width=\"{100 - width}%\"></td>" if width < 100 else ""
        rows.append(
            "<tr>"
            f"<td width=\"220\" style=\"padding:4px 8px 4px 0;font-size:13px;\">{html.escape(label)}</td>"
            "<td>"
            "<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\"><tr>"
            f"{bar}{gap}"
            "</tr></table></td>"
            f"<td width=\"70\" style=\"padding-left:8px;font-size:13px;\">{value:,}</td>"
            "</tr>"
        )
    return heading + "<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\">" + "".join(rows) + "</table>"


def _source_table(sources: list[dict[str, str]]) -> str:
    heading = "<h2 style=\"margin:24px 0 8px;font-size:16px;\">Sources fetched</h2>"
    if not sources:
        return heading + "<p style=\"margin:0;font-size:14px;\">No source was due this run.</p>"
    header = (
        "<tr>"
        + "".join(
            f"<th style=\"text-align:left;padding:6px 8px;border-bottom:1px solid #d0d7de;font-size:12px;\">{html.escape(name)}</th>"
            for name in ("Source", "Status", "Owner", "Loaded", "Quarantined", "Outside Africa")
        )
        + "</tr>"
    )
    body = []
    for row in sources:
        cells = (
            _label(row),
            row.get("status", ""),
            row.get("owner", ""),
            f"{_count(row.get('rows_loaded')):,}",
            f"{_count(row.get('rows_quarantined')):,}",
            f"{_count(row.get('rows_outside_africa')):,}",
        )
        body.append(
            "<tr>"
            + "".join(
                f"<td style=\"padding:6px 8px;border-bottom:1px solid #eef1f4;font-size:13px;\">{html.escape(cell)}</td>"
                for cell in cells
            )
            + "</tr>"
        )
        error = (row.get("error") or row.get("quarantine_reasons") or "").strip()
        if error:
            body.append(
                "<tr><td colspan=\"6\" style=\"padding:0 8px 8px;font-size:12px;color:#59636e;\">"
                f"{html.escape(error)}</td></tr>"
            )
    return heading + "<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\">" + header + "".join(body) + "</table>"


def _pdf(
    path: Path,
    summary: dict[str, object],
    sources: list[dict[str, str]],
    runs: list[Bar],
    classes: list[Bar],
    countries: list[Bar],
    note: str,
) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    family, unicode_font = _use_font(pdf)

    def text(value: object) -> str:
        raw = str(value)
        if unicode_font:
            return raw
        return raw.encode("latin-1", "replace").decode("latin-1")

    def heading(value: str) -> None:
        pdf.ln(4)
        pdf.set_font(family, "B", 13)
        pdf.cell(0, 8, text(value), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(family, "B", 16)
    pdf.cell(0, 9, text(f"{summary.get('status') or ''}  Ingest {summary.get('run_id') or ''}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(family, "", 11)
    for label, key in SUMMARY_FIELDS:
        pdf.cell(0, 6, text(f"{label}: {_count(summary.get(key)):,}"), new_x="LMARGIN", new_y="NEXT")
    _pdf_chart(pdf, family, text, "Rows added by run", runs, "No earlier runs.")
    _pdf_chart(pdf, family, text, "Rows added by indicator", classes, "No prepared rows.")
    _pdf_chart(pdf, family, text, "Rows added by country", countries, "No prepared rows.")
    heading("Sources fetched")
    pdf.set_font(family, "", 10)
    attempted = _attempted(sources)
    if not attempted:
        pdf.cell(0, 6, text("No source was due this run."), new_x="LMARGIN", new_y="NEXT")
    for row in attempted:
        line = (
            f"{_label(row)}  {row.get('status', '')}  owner={row.get('owner', '')}  "
            f"loaded={_count(row.get('rows_loaded')):,}  "
            f"quarantined={_count(row.get('rows_quarantined')):,}  "
            f"outside_africa={_count(row.get('rows_outside_africa')):,}"
        )
        pdf.multi_cell(0, 5, text(line), new_x="LMARGIN", new_y="NEXT")
        error = (row.get("error") or row.get("quarantine_reasons") or "").strip()
        if error:
            pdf.multi_cell(0, 5, text(error), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font(family, "", 10)
    pdf.multi_cell(
        0,
        5,
        text("Coverage is attached as ingest_coverage.csv, one row per indicator class, country, and year."),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    if note:
        pdf.multi_cell(0, 5, text(note), new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def _pdf_chart(
    pdf: FPDF,
    family: str,
    text: Callable[[object], str],
    title: str,
    bars: list[Bar],
    empty: str,
) -> None:
    pdf.ln(4)
    pdf.set_font(family, "B", 13)
    pdf.cell(0, 8, text(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(family, "", 10)
    if not bars:
        pdf.cell(0, 6, text(empty), new_x="LMARGIN", new_y="NEXT")
        return
    largest = max(value for _label, value in bars) or 1
    label_width = 70
    bar_width = 80
    for label, value in bars:
        if pdf.get_y() > 270:
            pdf.add_page()
        y = pdf.get_y()
        pdf.set_x(pdf.l_margin)
        pdf.cell(label_width, 6, text(_clip(label, 36)), new_x="RIGHT", new_y="TOP")
        width = bar_width * value / largest
        if width > 0:
            pdf.set_fill_color(31, 111, 235)
            pdf.rect(pdf.get_x(), y + 1, width, 4, "F")
        pdf.set_xy(pdf.l_margin + label_width + bar_width + 2, y)
        pdf.cell(25, 6, text(f"{value:,}"), new_x="LMARGIN", new_y="NEXT")


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _use_font(pdf: FPDF) -> tuple[str, bool]:
    pair = _font_pair()
    if pair is None:
        return "Helvetica", False
    regular, bold = pair
    pdf.add_font("Report", "", str(regular))
    pdf.add_font("Report", "B", str(bold))
    return "Report", True


def _font_pair() -> tuple[Path, Path] | None:
    pairs = (
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
        (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")),
    )
    for regular, bold in pairs:
        if regular.is_file():
            return regular, bold if bold.is_file() else regular
    return None


def letter_subject(summary: dict[str, object]) -> str:
    start, end = _period(summary)
    return f"Weekly Ingestion Report - {start} - {end}"


def letter_text(
    summary: dict[str, object],
    sources: list[dict[str, str]],
    *,
    publish_error: str = "",
    uploaded: bool = False,
) -> str:
    start, end = _period(summary)
    loaded, added, rate = _headline(summary, sources)
    attempted = _attempted(sources)
    lines = [
        "Weekly Data Ingestion Report",
        f"Period: {start} - {end}",
        "",
        "Executive summary",
        f"Sources fetched: {len(attempted)}",
        f"Sources loaded: {loaded}",
        f"Rows added: {added}",
        f"Success rate: {rate}",
        f"Sources current: {_count(summary.get('sources_current'))}",
        f"Sources skipped: {_count(summary.get('sources_skipped'))}",
        "",
        "Sources fetched",
        "Source\tStatus\tOwner\tRows loaded\tRows quarantined",
    ]
    if not attempted:
        lines.append("No source was due this run.")
    for row in attempted:
        lines.append(
            f"{_label(row)}\t{row.get('status', '')}\t{row.get('owner', '')}\t"
            f"{_count(row.get('rows_loaded'))}\t{_count(row.get('rows_quarantined'))}"
        )
    problems = _problems(sources)
    if problems:
        lines.extend(["", "Failed and partial sources", "Source\tOwner\tError"])
        for row in problems:
            lines.append(f"{_label(row)}\t{row.get('owner', '')}\t{row.get('error', '')}")
    lines.extend([
        "",
        "The detailed report is attached as ingest_jobs.html. It contains:",
        "- Filterable source cards",
        "- Rows by status and contract",
        "- The error on each failed or partial source",
        "",
        f"Report generated automatically at {summary.get('ended_at', '')}.",
    ])
    note = _publish_note(publish_error, uploaded)
    if note:
        lines.append(note)
    return "\n".join(lines) + "\n"


def letter_html(summary: dict[str, object], sources: list[dict[str, str]]) -> str:
    start, end = _period(summary)
    loaded, added, rate = _headline(summary, sources)
    attempted = _attempted(sources)
    figures = (
        ("Sources fetched", str(len(attempted))),
        ("Sources loaded", str(loaded)),
        ("Rows added", f"{added:,}"),
        ("Success rate", rate),
        ("Sources current", str(_count(summary.get("sources_current")))),
    )
    cells = "".join(
        "<td style=\"padding:12px 18px 12px 0;vertical-align:top;\">"
        f"<div style=\"font-size:28px;font-weight:700;color:#3498db;\">{html.escape(value)}</div>"
        f"<div style=\"font-size:13px;color:#59636e;\">{html.escape(label)}</div></td>"
        for label, value in figures
    )
    source_rows = "".join(
        "<tr>"
        + "".join(
            f"<td style=\"padding:6px 10px;border-bottom:1px solid #eef1f4;\">{html.escape(cell)}</td>"
            for cell in (
                _label(row),
                row.get("status", ""),
                row.get("owner", ""),
                f"{_count(row.get('rows_loaded')):,}",
                f"{_count(row.get('rows_quarantined')):,}",
            )
        )
        + "</tr>"
        for row in attempted
    ) or "<tr><td colspan=\"5\" style=\"padding:6px 10px;\">No source was due this run.</td></tr>"
    problems = _problems(sources)
    problem_block = ""
    if problems:
        body = "".join(
            "<tr>"
            + "".join(
                f"<td style=\"padding:6px 10px;border-bottom:1px solid #eef1f4;\">{html.escape(cell)}</td>"
                for cell in (_label(row), row.get("owner", ""), row.get("error", ""))
            )
            + "</tr>"
            for row in problems
        )
        problem_block = (
            "<h2 style=\"font-size:18px;\">Failed and partial sources</h2>"
            "<table cellpadding=\"0\" cellspacing=\"0\" style=\"border-collapse:collapse;width:100%;\">"
            "<tr>"
            + "".join(
                f"<th style=\"text-align:left;padding:6px 10px;border-bottom:1px solid #d0d7de;\">{html.escape(name)}</th>"
                for name in ("Source", "Owner", "Error")
            )
            + f"</tr>{body}</table>"
        )
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>Weekly Data Ingestion Report</title></head>"
        "<body style=\"margin:0;padding:24px;background:#f8f9fa;color:#1f2328;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;\">"
        "<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" "
        "style=\"max-width:720px;margin:0 auto;background:#ffffff;border-radius:10px;\">"
        "<tr><td style=\"padding:24px;\">"
        "<h1 style=\"margin:0 0 8px;font-size:22px;\">Weekly Data Ingestion Report</h1>"
        f"<p style=\"margin:0 0 20px;\">Period: {html.escape(start)} - {html.escape(end)}</p>"
        "<h2 style=\"font-size:18px;\">Executive summary</h2>"
        f"<table cellpadding=\"0\" cellspacing=\"0\"><tr>{cells}</tr></table>"
        "<h2 style=\"font-size:18px;\">Sources fetched</h2>"
        "<table cellpadding=\"0\" cellspacing=\"0\" style=\"border-collapse:collapse;width:100%;\">"
        "<tr>"
        + "".join(
            f"<th style=\"text-align:left;padding:6px 10px;border-bottom:1px solid #d0d7de;\">{html.escape(name)}</th>"
            for name in ("Source", "Status", "Owner", "Rows loaded", "Rows quarantined")
        )
        + f"</tr>{source_rows}</table>"
        f"{problem_block}"
        "<h2 style=\"font-size:18px;\">Attached HTML file</h2>"
        "<p>The detailed report is attached as ingest_jobs.html. It contains:</p>"
        "<ul>"
        "<li>Filterable source cards</li>"
        "<li>Rows by status and contract</li>"
        "<li>The error on each failed or partial source</li>"
        "</ul>"
        f"<p style=\"color:#59636e;\">Report generated automatically at {html.escape(str(summary.get('ended_at', '')))}.</p>"
        "</td></tr></table></body></html>\n"
    )


def _coverage_payload(coverage: list[dict[str, str]]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for row in coverage:
        payload.append({
            "indicator": (row.get("indicator_class") or "").strip() or "(blank)",
            "country": (row.get("country") or "").strip() or "(blank)",
            "year": (row.get("year") or "").strip() or "(blank)",
            "rows": _count(row.get("rows_loaded")),
        })
    return payload


def _coverage_options(payload: list[dict[str, object]], key: str, name: str) -> str:
    values = sorted({str(item[key]) for item in payload})
    return "".join(
        f'<label><input type="checkbox" data-coverage="{html.escape(name)}" value="{html.escape(value)}"> '
        f"{html.escape(value)}</label>"
        for value in values
    )


def _jobs_html(summary: dict[str, object], sources: list[dict[str, str]], coverage: list[dict[str, str]]) -> str:
    loaded, added, rate = _headline(summary, sources)
    counts = {status: sum(1 for row in sources if row.get("status") == status) for status in STATUSES}
    start, end = _period(summary)
    cards = "".join(_job_card(row) for row in sources)
    if not cards:
        cards = "<p>No sources.</p>"
    payload = _coverage_payload(coverage)
    data = json.dumps(payload).replace("<", "\\u003c")
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>Data Ingestion Engine Report</title>"
        "<style>"
        "body{margin:0;padding:24px;background:#f8f9fa;color:#1f2328;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}"
        "main{max-width:880px;margin:0 auto;}"
        ".banner{background:linear-gradient(90deg,#3498db,#1f6f9a);color:#fff;border-radius:10px;padding:16px 20px;}"
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:16px 0;}"
        ".kpi{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);padding:12px;}"
        ".kpi b{display:block;font-size:22px;color:#3498db;}"
        ".filters{background:#fff;border-radius:10px;padding:12px 16px;margin-bottom:16px;}"
        ".filters label{display:inline-block;margin:4px 10px 4px 0;padding:4px 8px;border-radius:999px;background:#eef6fc;}"
        ".job{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);"
        "border-left:4px solid #6c757d;padding:12px 16px;margin:0 0 12px;}"
        ".job[data-status=LOADED]{border-left-color:#28a745;background:#f4fbf6;}"
        ".job[data-status=FAILED]{border-left-color:#dc3545;background:#fff5f5;}"
        ".job[data-status=PARTIAL]{border-left-color:#ffc107;background:#fffaf0;}"
        ".job[data-status=CURRENT]{border-left-color:#3498db;}"
        ".tag{display:inline-block;margin-right:6px;padding:2px 8px;border-radius:999px;background:#eef1f4;font-size:12px;}"
        ".chart{background:#fff;border-radius:10px;padding:12px 16px;margin-bottom:16px;}"
        ".chart table{border-collapse:collapse;width:100%;}"
        ".chart th,.chart td{text-align:left;padding:6px 10px;border-bottom:1px solid #d0d7de;}"
        ".bar{display:grid;grid-template-columns:180px 1fr auto;gap:8px;align-items:center;margin:4px 0;}"
        ".track{background:#eef1f4;border-radius:6px;height:12px;}"
        ".fill{display:block;height:12px;background:#3498db;border-radius:6px;}"
        "pre{white-space:pre-wrap;background:#f6f8fa;padding:8px;border-radius:8px;}"
        "@media print {.filters{display:none;} .job details > *{display:block !important;}}"
        "</style></head><body><main>"
        "<div class=\"banner\"><h1 style=\"margin:0;font-size:22px;\">Data Ingestion Engine Report</h1>"
        f"<p style=\"margin:6px 0 0;\">{html.escape(str(summary.get('run_id', '')))}"
        f" · {html.escape(str(summary.get('started_at', '')))} – {html.escape(str(summary.get('ended_at', '')))}"
        f" · {html.escape(str(summary.get('status', '')))}</p>"
        f"<p style=\"margin:4px 0 0;\">Period: {html.escape(start)} - {html.escape(end)}</p></div>"
        "<section class=\"cards\">"
        + _kpi("Sources fetched", str(len(_attempted(sources))))
        + _kpi("Loaded", str(loaded))
        + _kpi("Partial", str(counts["PARTIAL"]))
        + _kpi("Failed", str(counts["FAILED"]))
        + _kpi("Success rate", rate)
        + _kpi("Rows added", f"{added:,}")
        + _kpi("Current", str(counts["CURRENT"]))
        + _kpi("Skipped", str(counts["SKIPPED"]))
        + "</section>"
        + _class_rows(sources, coverage)
        + "<section id=\"coverage\" class=\"chart\">"
        "<h2>Loaded rows</h2>"
        "<p id=\"coverage-rows\"></p>"
        "<p><strong>Indicator class</strong><br>"
        + _coverage_options(payload, "indicator", "class")
        + "</p><p><strong>Country</strong><br>"
        + _coverage_options(payload, "country", "country")
        + "</p><p><strong>Year</strong><br>"
        + _coverage_options(payload, "year", "year")
        + "</p><p><button type=\"button\" id=\"coverage-reset\">Reset coverage filters</button></p>"
        "<h3>Rows by country</h3><div id=\"by-country\"></div>"
        "<h3>Rows by year</h3><div id=\"by-year\"></div>"
        f'<script id="coverage-data" type="application/json">{data}</script>'
        "</section>"
        "<section class=\"filters\">"
        "<p id=\"shown\"></p>"
        + _filter_group("Status", "status", STATUSES, sources, "status")
        + _filter_group("Contract", "contract", ("csv", "xlsx", "xls", "xlsb", "ods", "json_api", "html_table"), sources, "contract_type")
        + _filter_group(
            "Error",
            "error",
            ERROR_CATEGORIES,
            sources,
            "error_category",
        )
        + "<p><button type=\"button\" class=\"reset\" id=\"reset\">Reset filters</button></p>"
        "<p>An empty filter group shows every source.</p>"
        "</section>"
        f"<section id=\"jobs\"><h2>{html.escape(start)}</h2>{cards}"
        "<p id=\"empty\" hidden>No sources match these filters.</p></section>"
        "<script>"
        "const groups=['status','contract','error'];"
        "function chosen(name){return [...document.querySelectorAll('input[data-filter=\"'+name+'\"]:checked')].map(el=>el.value);}"
        "function apply(){"
        "const pick=Object.fromEntries(groups.map(name=>[name,chosen(name)]));"
        "let shown=0;"
        "for (const card of document.querySelectorAll('.job')){"
        "const ok=groups.every(name=>!pick[name].length||pick[name].includes(card.dataset[name]));"
        "card.hidden=!ok; if(ok) shown+=1;}"
        "const total=document.querySelectorAll('.job').length;"
        "document.getElementById('shown').textContent='Showing '+shown+' source(s) of '+total+' total';"
        "document.getElementById('empty').hidden=shown!==0;}"
        "document.querySelector('.filters').addEventListener('change', apply);"
        "document.getElementById('reset').addEventListener('click',()=>{"
        "document.querySelectorAll('.filters input').forEach(el=>{el.checked=false;}); apply();});"
        "apply();"
        "const coverage=JSON.parse(document.getElementById('coverage-data').textContent);"
        "const coverageGroups=['class','country','year'];"
        "function coverageChosen(name){return [...document.querySelectorAll('input[data-coverage=\"'+name+'\"]:checked')].map(el=>el.value);}"
        "function coverageText(value){return String(value).replace(/[&<>\"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[ch]));}"
        "function coverageBars(rows,key){"
        "const totals=new Map();"
        "for (const row of rows){totals.set(row[key],(totals.get(row[key])||0)+row.rows);}"
        "const items=[...totals.entries()].sort((a,b)=>b[1]-a[1]||String(a[0]).localeCompare(String(b[0])));"
        "if(!items.length) return '<p>No prepared rows.</p>';"
        "const max=Math.max(...items.map(item=>item[1]),1);"
        "return items.map(([label,value])=>'<div class=\"bar\"><span>'+coverageText(label)+'</span><span class=\"track\"><span class=\"fill\" style=\"width:'+(100*value/max)+'%\"></span></span><span>'+value.toLocaleString()+'</span></div>').join('');}"
        "function applyCoverage(){"
        "const pick=Object.fromEntries(coverageGroups.map(name=>[name,coverageChosen(name)]));"
        "const rows=coverage.filter(row=>coverageGroups.every(name=>!pick[name].length||pick[name].includes(name==='class'?row.indicator:row[name])));"
        "const total=rows.reduce((sum,row)=>sum+row.rows,0);"
        "document.getElementById('coverage-rows').textContent='Loaded rows: '+total.toLocaleString();"
        "document.getElementById('by-country').innerHTML=coverageBars(rows,'country');"
        "document.getElementById('by-year').innerHTML=coverageBars(rows,'year');}"
        "document.getElementById('coverage').addEventListener('change', applyCoverage);"
        "document.getElementById('coverage-reset').addEventListener('click',()=>{"
        "document.querySelectorAll('#coverage input').forEach(el=>{el.checked=false;}); applyCoverage();});"
        "applyCoverage();"
        "</script></main></body></html>\n"
    )


def _kpi(label: str, value: str) -> str:
    return (
        "<div class=\"kpi\"><b>"
        + html.escape(value)
        + "</b><span>"
        + html.escape(label)
        + "</span></div>"
    )


def _filter_group(
    title: str,
    name: str,
    options: tuple[str, ...],
    sources: list[dict[str, str]],
    field: str,
) -> str:
    counts = {option: 0 for option in options}
    for row in sources:
        key = row.get(field) or ""
        if field == "error_category":
            key = _error_category(row)
        if key in counts:
            counts[key] += 1
    labels = "".join(
        f"<label><input type=\"checkbox\" data-filter=\"{html.escape(name)}\" value=\"{html.escape(option)}\"> "
        f"{html.escape(option)} ({counts[option]})</label>"
        for option in options
    )
    return f"<p><strong>{html.escape(title)}</strong><br>{labels}</p>"


def _job_card(row: dict[str, str]) -> str:
    category = _error_category(row)
    contract = row.get("contract_type") or ""
    error = (row.get("error") or "").strip()
    status = row.get("status", "")
    details = "\n".join([
        f"status: {status}",
        f"manifest: {row.get('manifest_file', '')}",
        f"contract: {contract}",
        f"failure_stage: {row.get('failure_stage', '')}",
        f"error_category: {category}",
        f"quarantine: {row.get('quarantine_reasons', '')}",
        f"last_loaded_at: {row.get('last_loaded_at', '')}",
        f"error: {error}",
    ])
    error_block = ""
    if error and status in {"FAILED", "PARTIAL"}:
        error_block = (
            "<p style=\"border-left:4px solid #dc3545;padding-left:8px;\">"
            + html.escape(error)
            + "</p>"
        )
    facts = [f"Owner {row.get('owner', '') or '(none)'}"]
    for value in (row.get("indicator_class", ""), row.get("table_name", "")):
        if value:
            facts.append(value)
    facts.append(f"Manifest {row.get('manifest_file', '') or '(unknown)'}")
    rows_line = ""
    if status in ATTEMPTED:
        rows_line = (
            f"<p>Rows loaded {_count(row.get('rows_loaded')):,} · "
            f"quarantined {_count(row.get('rows_quarantined')):,} · "
            f"outside Africa {_count(row.get('rows_outside_africa')):,}"
            f"{_quality(row)}</p>"
        )
    last = (row.get("last_loaded_at") or "").strip()
    last_line = f"<p>Last loaded {html.escape(_day(last))}</p>" if last else ""
    return (
        f"<article class=\"job\" data-status=\"{html.escape(status)}\" "
        f"data-contract=\"{html.escape(contract)}\" data-error=\"{html.escape(category)}\">"
        f"<h2 style=\"margin:0 0 6px;font-size:18px;\">{html.escape(_label(row))} "
        f"<span class=\"tag\">{html.escape(status)}</span></h2>"
        f"<p style=\"margin:0 0 8px;\">{html.escape(row.get('source_name', ''))}</p>"
        f"<p>{_tag(contract)}{_tag(category if category != 'NONE' else '')}</p>"
        f"<p>{html.escape(' · '.join(facts))}</p>"
        f"{rows_line}"
        f"{last_line}"
        f"{error_block}"
        f"<details><summary>Technical details</summary><pre>{html.escape(details)}</pre></details>"
        "</article>"
    )


def _tag(value: str) -> str:
    if not value:
        return ""
    return f"<span class=\"tag\">{html.escape(value)}</span>"


def _problems(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in sources if row.get("status") in {"FAILED", "PARTIAL"}]


def _attempted(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in sources if row.get("status") in ATTEMPTED]


def _label(row: dict[str, str]) -> str:
    """Name a source by its id, or by its manifest file when the whole file failed."""
    return (row.get("source_id") or "").strip() or (row.get("manifest_file") or "").strip() or "(unnamed)"


def _step(row: dict[str, str]) -> float | None:
    """Score one fetched source: 1 when its rows reached raw, 0 when it failed."""
    status = row.get("status") or ""
    if status in {"LOADED", "PARTIAL"}:
        return 1.0
    if status == "FAILED":
        return 0.0
    return None


def _rate(steps: list[float]) -> str:
    if not steps:
        return "n/a"
    return f"{100 * sum(steps) / len(steps):.1f}%"


def _headline(summary: dict[str, object], sources: list[dict[str, str]]) -> tuple[int, int, str]:
    loaded = sum(1 for row in sources if row.get("status") == "LOADED")
    steps = [step for row in sources if (step := _step(row)) is not None]
    return loaded, _count(summary.get("rows_loaded")), _rate(steps)


def _class_rows(sources: list[dict[str, str]], coverage: list[dict[str, str]]) -> str:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in sources:
        if _step(row) is None:
            continue
        name = (row.get("indicator_class") or "").strip() or "(blank)"
        groups.setdefault(name, []).append(row)
    loaded_rows: dict[str, int] = {}
    for row in coverage:
        name = (row.get("indicator_class") or "").strip() or "(blank)"
        loaded_rows[name] = loaded_rows.get(name, 0) + _count(row.get("rows_loaded"))
    body = []
    for name in sorted(groups):
        rows = groups[name]
        counts = {status: sum(1 for row in rows if row.get("status") == status) for status in ("LOADED", "PARTIAL", "FAILED")}
        steps = [step for row in rows if (step := _step(row)) is not None]
        body.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{counts['LOADED']}</td>"
            f"<td>{counts['PARTIAL']}</td>"
            f"<td>{counts['FAILED']}</td>"
            f"<td>{html.escape(_rate(steps))}</td>"
            f"<td>{loaded_rows.get(name, 0):,}</td>"
            "</tr>"
        )
    if not body:
        return ""
    heading = ("Class", "Loaded", "Partial", "Failed", "Success rate", "Rows loaded")
    head = "".join(f"<th>{html.escape(name)}</th>" for name in heading)
    return (
        "<section class=\"chart\"><h2>Success rate by indicator</h2>"
        "<table><tr>"
        + head
        + "</tr>"
        + "".join(body)
        + "</table></section>"
    )


def _quality(row: dict[str, str]) -> str:
    loaded = _count(row.get("rows_loaded"))
    total = loaded + _count(row.get("rows_quarantined"))
    if total == 0:
        return ""
    return f" · {100 * loaded / total:.0f}% of landed rows reached raw"


def _error_category(row: dict[str, str]) -> str:
    status = row.get("status") or ""
    if status == "PARTIAL":
        return "DATA_QUALITY"
    if status != "FAILED":
        return "NONE"
    stage = (row.get("failure_stage") or "").strip()
    if stage == "manifest":
        return "MANIFEST"
    if stage == "auth":
        return "AUTH"
    if stage == "fetch":
        return "NETWORK"
    if stage == "publish":
        return "PUBLISH"
    if _count(row.get("rows_quarantined")) or _count(row.get("rows_outside_africa")):
        return "DATA_QUALITY"
    return "SCHEMA"


def _period(summary: dict[str, object]) -> tuple[str, str]:
    return _day(summary.get("started_at")), _day(summary.get("ended_at"))


def _day(value: object) -> str:
    text = str(value or "")
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text[:10]
    return when.strftime("%d %b %Y")
