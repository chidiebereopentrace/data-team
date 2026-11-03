"""Email the weekly ingestion report."""

from __future__ import annotations

import csv
import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Protocol

from ingestion_engine.report_doc import letter_html, letter_subject, letter_text, write_report

ATTACHMENTS = ("ingest_coverage.csv", "ingest_sources.csv", "ingest_failures.csv")


class SmtpClient(Protocol):
    def __enter__(self) -> SmtpClient: ...

    def __exit__(self, *args: object) -> bool: ...

    def ehlo(self) -> object: ...

    def starttls(self) -> object: ...

    def login(self, user: str, password: str) -> object: ...

    def send_message(self, message: EmailMessage) -> object: ...


SmtpFactory = Callable[..., SmtpClient]


def send_report(
    out_dir: Path,
    *,
    publish_error: str = "",
    uploaded: bool = False,
    history: list[dict[str, object]] | None = None,
    smtp: SmtpFactory | None = None,
) -> None:
    """Send the run report. An empty recipient list sends nothing."""
    recipients = [item.strip() for item in os.environ.get("INGEST_REPORT_EMAILS", "").split(",") if item.strip()]
    if not recipients:
        return
    try:
        reports = out_dir / "reports"
        summary = json.loads((reports / "run.json").read_text(encoding="utf-8"))
        _summary_path, pdf_path, jobs_path = write_report(
            reports,
            history=history,
            publish_error=publish_error,
            uploaded=uploaded,
        )
        sources = _source_rows(reports)
        message = EmailMessage()
        message["Subject"] = letter_subject(summary)
        message["From"] = os.environ["INGEST_SMTP_FROM"]
        message["To"] = ", ".join(recipients)
        message.set_content(letter_text(summary, sources, publish_error=publish_error, uploaded=uploaded))
        message.add_alternative(letter_html(summary, sources), subtype="html")
        message.add_attachment(pdf_path.read_bytes(), maintype="application", subtype="pdf", filename=pdf_path.name)
        message.add_attachment(jobs_path.read_bytes(), maintype="text", subtype="html", filename=jobs_path.name)
        for name in ATTACHMENTS:
            path = reports / name
            if path.is_file() and path.stat().st_size:
                message.add_attachment(
                    path.read_bytes(),
                    maintype="text",
                    subtype="csv",
                    filename=name,
                )
        host = os.environ["INGEST_SMTP_HOST"]
        port = int(os.environ.get("INGEST_SMTP_PORT", "587"))
        user = os.environ.get("INGEST_SMTP_USER", "")
        password = os.environ.get("INGEST_SMTP_PASSWORD", "")
        client = (smtp or smtplib.SMTP)(host, port, timeout=30)
        with client:
            client.ehlo()
            if port != 25:
                client.starttls()
                client.ehlo()
            if user:
                client.login(user, password)
            client.send_message(message)
    except Exception as exc:
        print(f"report email failed\t{exc}", flush=True)


def _source_rows(reports: Path) -> list[dict[str, str]]:
    path = reports / "ingest_sources.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
