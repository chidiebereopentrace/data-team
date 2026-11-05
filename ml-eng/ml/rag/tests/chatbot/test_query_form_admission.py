"""Live warehouse gate. Queries are admitted only after BigQuery returns rows."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from google.cloud import bigquery

from ml.rag.chatbot.query_form_admission import admit_catalog, admitted_sql_form_ids
from ml.rag.chatbot.query_forms import QUERY_FORMS

_KEY = (
    Path(__file__).resolve().parents[5]
    / "data-eng"
    / "data"
    / "local"
    / "keys"
    / "opentrace-bq-key.json"
)


def _client() -> bigquery.Client:
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and _KEY.is_file():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_KEY)
    if not os.environ.get("BQ_PROJECT"):
        os.environ["BQ_PROJECT"] = "opentrace-prod-5ga4"
    project = os.environ["BQ_PROJECT"]
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds or not Path(creds).is_file():
        pytest.skip("BigQuery credentials are not available")
    return bigquery.Client(project=project)


def test_forms_are_admitted_only_when_the_warehouse_returns_rows() -> None:
    client = _client()
    results = admit_catalog(client)
    ok = [row for row in results if row.reason == "ok"]
    assert ok, "no compiled form returned rows from the warehouse"
    for row in results:
        if row.reason == "ok":
            assert row.admitted
            assert row.row_count >= 1
        elif row.reason == "report_only":
            assert row.admitted
        else:
            assert not row.admitted
    admitted = admitted_sql_form_ids(results)
    shapes = {form.shape for form in QUERY_FORMS if form.form_id in admitted}
    assert "point" in shapes
    assert "series" in shapes
