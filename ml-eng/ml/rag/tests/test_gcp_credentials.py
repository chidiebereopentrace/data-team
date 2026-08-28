"""Tests for GCP credentials bootstrap."""

from __future__ import annotations

import base64
import json
import os

import pytest

from ml.rag.gcp_credentials import bootstrap_gcp_credentials, credentials_ready_for_bq


def _sample_sa() -> dict:
    return {"type": "service_account", "project_id": "demo-project", "client_email": "x@y.iam.gserviceaccount.com"}


def test_bootstrap_decodes_base64_to_canonical_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    canonical = tmp_path / "gcp-sa-key.json"
    legacy = tmp_path / "gcp-sa.json"
    monkeypatch.setattr("ml.rag.gcp_credentials.CANONICAL_SA_PATH", canonical)
    monkeypatch.setattr("ml.rag.gcp_credentials.LEGACY_SA_PATH", legacy)

    sa = _sample_sa()
    b64 = base64.b64encode(json.dumps(sa).encode("utf-8")).decode("ascii")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64", b64)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(legacy))
    legacy.write_text("", encoding="utf-8")

    path = bootstrap_gcp_credentials()
    assert path == str(canonical)
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(canonical)
    assert canonical.is_file()
    assert json.loads(canonical.read_text(encoding="utf-8"))["project_id"] == "demo-project"
    assert not legacy.exists()


def test_bootstrap_invalid_base64_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("ml.rag.gcp_credentials.CANONICAL_SA_PATH", tmp_path / "key.json")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64", "not!!!valid!!!base64")
    with pytest.raises(RuntimeError, match="not valid base64"):
        bootstrap_gcp_credentials()


def test_bootstrap_invalid_json_after_decode_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("ml.rag.gcp_credentials.CANONICAL_SA_PATH", tmp_path / "key.json")
    b64 = base64.b64encode(b"{not-json").decode("ascii")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64", b64)
    with pytest.raises(RuntimeError, match="not valid JSON"):
        bootstrap_gcp_credentials()


def test_bootstrap_legacy_path_invalid_fails_when_flag_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    legacy = tmp_path / "gcp-sa.json"
    monkeypatch.setattr("ml.rag.gcp_credentials.LEGACY_SA_PATH", legacy)
    legacy.write_text("", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(legacy))
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64", raising=False)
    with pytest.raises(RuntimeError, match="not valid JSON"):
        bootstrap_gcp_credentials(fail_on_legacy_invalid=True)


def test_credentials_ready_for_bq_true_when_valid(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    sa = tmp_path / "sa.json"
    sa.write_text(json.dumps(_sample_sa()), encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa))
    assert credentials_ready_for_bq() is True


def test_load_rag_dotenv_bootstraps_base64(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    canonical = tmp_path / "gcp-sa-key.json"
    monkeypatch.setattr("ml.rag.gcp_credentials.CANONICAL_SA_PATH", canonical)
    monkeypatch.setattr("ml.rag.gcp_credentials.LEGACY_SA_PATH", tmp_path / "legacy.json")

    sa = _sample_sa()
    b64 = base64.b64encode(json.dumps(sa).encode("utf-8")).decode("ascii")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64", b64)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp-sa.json")
    monkeypatch.delenv("BQ_PROJECT", raising=False)

    from ml.rag.local_env import load_rag_dotenv

    load_rag_dotenv(tmp_path)
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(canonical)
    assert credentials_ready_for_bq() is True
