"""CI guard: mart_dev_rag_tables.txt must match dbt mart_dev rag-relevant models."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
SYNC_SCRIPT = REPO_ROOT / "data-eng" / "data" / "local" / "scripts" / "sync_mart_dev_tables_list.py"


def test_mart_dev_tables_list_matches_dbt() -> None:
    assert SYNC_SCRIPT.is_file(), f"missing sync script: {SYNC_SCRIPT}"
    proc = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
