"""Target guard and credentials-path refusal."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.common.auth import AuthError, describe_auth, ensure_auth_safe
from ingestion.common.config import (
    REPO_ROOT,
    Settings,
    TargetGuardError,
    assert_target_allowed,
    credentials_path_inside_repo,
    get_settings,
    resolve_target,
)


def test_resolve_target_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_TARGET_DATASET", "landing_dev_alice")
    assert resolve_target(None) == "landing_dev_alice"
    assert resolve_target("landing_dev") == "landing_dev"


def test_landing_dev_ok() -> None:
    assert_target_allowed("landing_dev")
    assert_target_allowed("landing_dev_bob")
    settings = get_settings("landing_dev", load_dotenv_file=False)
    assert settings.target_dataset == "landing_dev"


def test_raw_dev_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_ALLOW_PROD", raising=False)
    monkeypatch.setenv("CI", "")
    assert_target_allowed("raw_dev")
    assert_target_allowed("raw_dev_bob")
    settings = get_settings("raw_dev", load_dotenv_file=False)
    assert settings.target_dataset == "raw_dev"
    with pytest.raises(TargetGuardError, match="not allowed"):
        assert_target_allowed("raw")


def test_landing_raises_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_ALLOW_PROD", raising=False)
    monkeypatch.setenv("CI", "")
    with pytest.raises(TargetGuardError, match="not allowed"):
        assert_target_allowed("landing")
    with pytest.raises(TargetGuardError):
        get_settings("landing", load_dotenv_file=False)


def test_landing_ok_only_with_ci_and_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_ALLOW_PROD", "1")
    monkeypatch.setenv("CI", "true")
    assert_target_allowed("landing")
    settings = get_settings("landing", load_dotenv_file=False)
    assert settings.target_dataset == "landing"


def test_credentials_inside_repo_detected(tmp_path: Path) -> None:
    inside = REPO_ROOT / "data-eng" / "fake-key.json"
    assert credentials_path_inside_repo(str(inside), REPO_ROOT) is True
    outside = tmp_path / "key.json"
    assert credentials_path_inside_repo(str(outside), REPO_ROOT) is False


def test_refuse_credentials_in_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(REPO_ROOT / "sa-secret.json"),
    )
    with pytest.raises(AuthError, match="inside the repo"):
        ensure_auth_safe()


def test_describe_auth_reports_key_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    key = key_dir / "ingest-dev.json"
    key.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
    monkeypatch.delenv("INGEST_IMPERSONATE_SA", raising=False)
    settings = Settings(
        project="opentrace-prod-5ga4",
        location="europe-west3",
        target_dataset="landing_dev",
        impersonate_sa=None,
        allow_prod=False,
        ci=False,
        offline_fixtures=True,
        repo_root=repo,
        package_root=repo,
        out_dir=repo / ".out",
    )
    info = describe_auth(settings, credentials=object())  # type: ignore[arg-type]
    assert info.kind == "key_file"
    assert info.principal == "ingest-dev.json"
    assert info.target == "landing_dev"
