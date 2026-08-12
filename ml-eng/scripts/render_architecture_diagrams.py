#!/usr/bin/env python3
"""Render Mermaid architecture diagrams to PNG via mermaid-cli (mmdc)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from rag_architecture_content import DIAGRAMS_DIR, DIAGRAMS_PNG_DIR, DIAGRAM_FILES

MMDC_INSTALL = (
    "mermaid-cli (mmdc) is not on PATH.\n"
    "Install Node.js, then run:\n"
    "  npm install -g @mermaid-js/mermaid-cli\n"
    "Or render once with npx:\n"
    "  npx @mermaid-js/mermaid-cli -i <file.mmd> -o <file.png>"
)

# Width/height overrides for large diagrams (pixels).
DIAGRAM_RENDER_OPTS: dict[str, list[str]] = {
    "runtime_graph.mmd": ["-w", "2800", "-H", "2000"],
    "runtime_erd.mmd": ["-w", "2400", "-H", "1800"],
    "infra_erd.mmd": ["-w", "2000", "-H", "1200"],
    "bq_retrieve_subflow.mmd": ["-w", "1800", "-H", "1400"],
}


def _find_mmdc() -> tuple[list[str], bool]:
    """Return (command prefix, use_shell) to invoke mermaid-cli."""
    if shutil.which("mmdc"):
        return ["mmdc"], False
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx:
        # Windows .cmd shims require shell=True for subprocess.run.
        use_shell = npx.lower().endswith(".cmd")
        return [npx, "-y", "@mermaid-js/mermaid-cli"], use_shell
    return [], False


def render_diagrams(*, force: bool = True) -> list[Path]:
    """Render all .mmd files to PNG. Returns list of output PNG paths."""
    mmdc_cmd, use_shell = _find_mmdc()
    if not mmdc_cmd:
        print(MMDC_INSTALL, file=sys.stderr)
        raise SystemExit(1)

    DIAGRAMS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for name in DIAGRAM_FILES:
        src = DIAGRAMS_DIR / name
        if not src.exists():
            print(f"Missing diagram source: {src}", file=sys.stderr)
            raise SystemExit(1)

        stem = src.stem
        dest = DIAGRAMS_PNG_DIR / f"{stem}.png"
        extra = DIAGRAM_RENDER_OPTS.get(name, ["-w", "1600", "-H", "1200"])

        cmd = [
            *mmdc_cmd,
            "-i",
            str(src),
            "-o",
            str(dest),
            "-b",
            "white",
            *extra,
        ]
        print(f"Rendering {name} -> {dest.name}")
        subprocess.run(cmd, check=True, shell=use_shell)
        outputs.append(dest)

    return outputs


def main() -> None:
    paths = render_diagrams()
    print(f"Rendered {len(paths)} diagram(s) to {DIAGRAMS_PNG_DIR}")


if __name__ == "__main__":
    main()
