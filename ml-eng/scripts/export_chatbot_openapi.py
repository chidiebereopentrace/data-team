"""Export versioned OpenAPI spec for the public Chatbot API v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.serving.chat.app import app  # noqa: E402

DEFAULT_OUT = REPO_ROOT.parent / "docs" / "partners" / "openapi" / "chatbot-v1.json"


def export_openapi(out_path: Path | None = None) -> Path:
    destination = out_path or DEFAULT_OUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = app.openapi()
    schema["info"]["title"] = "Ask ADZA Enterprise Chatbot API"
    schema["info"]["description"] = (
        "Public v1 API for Ask ADZA enterprise partners. "
        "Authenticate with X-API-Key or Authorization: Bearer when enterprise auth is enabled."
    )
    destination.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    path = export_openapi(target)
    print(f"Wrote OpenAPI spec to {path}")
