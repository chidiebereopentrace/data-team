"""Settings from env/.env and target dataset guard."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEV_TARGET_RE = re.compile(r"^(?:landing_dev|raw_dev)(_[a-z0-9]+)?$")

# Project root (pyproject, tests/, .env): data-eng/ingestion/
# (code lives under package/ingestion/; parents[3] skips common → ingestion → package)
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PACKAGE_ROOT.parents[1]  # data-team/


class TargetGuardError(RuntimeError):
    """Raised when a non-dev target is requested without CI prod opt-in."""


@dataclass(frozen=True)
class Settings:
    project: str
    location: str
    target_dataset: str
    impersonate_sa: str | None
    allow_prod: bool
    ci: bool
    offline_fixtures: bool
    repo_root: Path
    package_root: Path
    out_dir: Path

    @property
    def is_dev_target(self) -> bool:
        return bool(DEV_TARGET_RE.match(self.target_dataset))


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_env(env_file: Path | None = None) -> None:
    path = env_file or (PACKAGE_ROOT / ".env")
    if path.is_file():
        load_dotenv(path, override=False)


def resolve_target(cli_target: str | None = None) -> str:
    if cli_target and cli_target.strip():
        return cli_target.strip()
    env_target = os.environ.get("INGEST_TARGET_DATASET", "").strip()
    if env_target:
        return env_target
    return "landing_dev"


def assert_target_allowed(target: str) -> None:
    """Enforce landing_dev* or raw_dev* unless INGEST_ALLOW_PROD=1 and CI=true."""
    if DEV_TARGET_RE.match(target):
        return
    allow_prod = _truthy(os.environ.get("INGEST_ALLOW_PROD"))
    ci = _truthy(os.environ.get("CI"))
    if allow_prod and ci:
        return
    raise TargetGuardError(
        f"Target dataset {target!r} is not allowed. "
        f"Local/dev runs must use landing_dev or raw_dev (or landing_dev_<user> / raw_dev_<user>). "
        f"Writing to {target!r} requires both INGEST_ALLOW_PROD=1 and CI=true. "
        f"(Current: INGEST_ALLOW_PROD={os.environ.get('INGEST_ALLOW_PROD')!r}, "
        f"CI={os.environ.get('CI')!r})"
    )


def get_settings(cli_target: str | None = None, *, load_dotenv_file: bool = True) -> Settings:
    if load_dotenv_file:
        load_env()
    target = resolve_target(cli_target)
    assert_target_allowed(target)
    out_dir = PACKAGE_ROOT / ".out"
    return Settings(
        project=os.environ.get("GCP_PROJECT", "opentrace-prod-5ga4"),
        location=os.environ.get("BQ_LOCATION", "europe-west3"),
        target_dataset=target,
        impersonate_sa=os.environ.get("INGEST_IMPERSONATE_SA") or None,
        allow_prod=_truthy(os.environ.get("INGEST_ALLOW_PROD")),
        ci=_truthy(os.environ.get("CI")),
        offline_fixtures=_truthy(os.environ.get("INGEST_OFFLINE_FIXTURES")),
        repo_root=REPO_ROOT,
        package_root=PACKAGE_ROOT,
        out_dir=out_dir,
    )


def credentials_path_inside_repo(cred_path: str | None, repo_root: Path | None = None) -> bool:
    if not cred_path:
        return False
    root = (repo_root or REPO_ROOT).resolve()
    try:
        path = Path(cred_path).expanduser().resolve()
    except OSError:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
