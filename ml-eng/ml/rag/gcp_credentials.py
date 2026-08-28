"""Bootstrap GCP service account credentials for production containers."""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CANONICAL_SA_PATH = Path("/tmp/gcp-sa-key.json")
LEGACY_SA_PATH = Path("/tmp/gcp-sa.json")

_BASE64_ENV = "GOOGLE_APPLICATION_CREDENTIALS_BASE64"
_CREDENTIALS_ENV = "GOOGLE_APPLICATION_CREDENTIALS"


def _validate_json_file(path: Path) -> None:
    with path.open(encoding="utf-8") as fh:
        json.load(fh)


def _decode_base64_to_file(raw_b64: str, dest: Path) -> None:
    try:
        payload = base64.b64decode(raw_b64, validate=True)
    except Exception as exc:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS_BASE64 is not valid base64. "
            "Re-encode with ml-eng/scripts/encode-gcp-key.sh — see deploy/PRODUCTION_ENV.md."
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    dest.chmod(0o600)
    try:
        _validate_json_file(dest)
    except json.JSONDecodeError as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Decoded GCP credentials are not valid JSON at {dest}. "
            "Check GOOGLE_APPLICATION_CREDENTIALS_BASE64 — see deploy/PRODUCTION_ENV.md."
        ) from exc


def _remove_legacy_path() -> None:
    if LEGACY_SA_PATH.is_file():
        try:
            LEGACY_SA_PATH.unlink()
        except OSError:
            logger.warning("Could not remove legacy credentials path %s", LEGACY_SA_PATH)


def bootstrap_gcp_credentials(*, fail_on_legacy_invalid: bool = False) -> str | None:
    """
    Ensure ``GOOGLE_APPLICATION_CREDENTIALS`` points at valid JSON.

    When ``GOOGLE_APPLICATION_CREDENTIALS_BASE64`` is set, always decode to
    ``CANONICAL_SA_PATH`` and override any stale platform env (e.g.
    ``/tmp/gcp-sa.json`` from legacy HF/serving bootstrap).

    Returns the credentials path when set/validated, else None.
    """
    raw_b64 = os.environ.get(_BASE64_ENV, "").strip()
    if raw_b64:
        _decode_base64_to_file(raw_b64, CANONICAL_SA_PATH)
        _remove_legacy_path()
        os.environ[_CREDENTIALS_ENV] = str(CANONICAL_SA_PATH)
        logger.info("GCP credentials bootstrapped at %s", CANONICAL_SA_PATH)
        return str(CANONICAL_SA_PATH)

    cred_path_raw = os.environ.get(_CREDENTIALS_ENV, "").strip()
    if not cred_path_raw:
        return None

    cred_path = Path(cred_path_raw)
    if not cred_path.is_file():
        if cred_path == LEGACY_SA_PATH or fail_on_legacy_invalid:
            raise RuntimeError(
                f"GOOGLE_APPLICATION_CREDENTIALS points to missing file: {cred_path}. "
                "Remove GOOGLE_APPLICATION_CREDENTIALS from Railway and set "
                "GOOGLE_APPLICATION_CREDENTIALS_BASE64 only — see deploy/PRODUCTION_ENV.md."
            )
        return None

    try:
        _validate_json_file(cred_path)
    except json.JSONDecodeError as exc:
        hint = (
            " Remove GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-sa.json from Railway and use "
            "GOOGLE_APPLICATION_CREDENTIALS_BASE64 — see deploy/PRODUCTION_ENV.md."
        )
        if cred_path == LEGACY_SA_PATH or fail_on_legacy_invalid:
            raise RuntimeError(
                f"GOOGLE_APPLICATION_CREDENTIALS is not valid JSON: {cred_path}.{hint}"
            ) from exc
        logger.warning("GCP credentials file invalid (ignored): %s", cred_path)
        return None

    if cred_path == LEGACY_SA_PATH:
        logger.warning(
            "Using legacy GCP path %s; prefer GOOGLE_APPLICATION_CREDENTIALS_BASE64 "
            "and remove GOOGLE_APPLICATION_CREDENTIALS from Railway.",
            LEGACY_SA_PATH,
        )
    return str(cred_path)


def credentials_ready_for_bq() -> bool:
    """True when ADC path is set and parses as JSON."""
    path = os.environ.get(_CREDENTIALS_ENV, "").strip()
    if not path:
        return False
    p = Path(path)
    if not p.is_file():
        return False
    try:
        _validate_json_file(p)
    except json.JSONDecodeError:
        return False
    return True
