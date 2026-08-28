"""Build OpenTrace_Mart_Complete_Guide.pdf from the markdown guide.

Renders Mermaid ERDs via Kroki, converts MD → HTML (mistune), prints PDF
with Chrome headless.

Usage:
  python data-eng/scripts/build_opentrace_mart_complete_guide_pdf.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

import mistune
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = ROOT / "docs" / "OpenTrace_Mart_Complete_Guide.md"
DEFAULT_PDF = ROOT / "docs" / "OpenTrace_Mart_Complete_Guide.pdf"

MERMAID_FENCE = re.compile(
    r"```mermaid\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

KROKI_URL = "https://kroki.io/mermaid/png"
MERMAID_INK_TMPL = "https://mermaid.ink/img/{payload}"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
]

PRINT_CSS = """
@page { size: A4; margin: 18mm 14mm; }
body {
  font-family: "Segoe UI", Calibri, Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #1a1a1a;
  max-width: 100%;
}
h1 { font-size: 18pt; margin-top: 0; color: #1F4E79; page-break-after: avoid; }
h2 { font-size: 14pt; margin-top: 1.4em; color: #1F4E79; page-break-after: avoid; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
h3 { font-size: 12pt; margin-top: 1.1em; color: #2E75B6; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4px 6px; vertical-align: top; text-align: left; }
th { background: #1F4E79; color: #fff; }
tr:nth-child(even) td { background: #f5f8fb; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f0f0f0; padding: 0 3px; }
pre { background: #f4f4f4; border: 1px solid #ddd; padding: 8px 10px; overflow-x: auto; font-size: 8.5pt; page-break-inside: avoid; }
pre code { background: transparent; padding: 0; }
img.mermaid-diagram {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 1em auto;
  page-break-inside: avoid;
}
a { color: #1F4E79; }
strong { color: #111; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
"""


def _find_chrome() -> Path:
    for path in CHROME_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Chrome/Chromium not found. Install Google Chrome or set CHROME_PATH."
    )


def _render_mermaid_kroki(source: str, out_png: Path, timeout: int = 60) -> None:
    resp = requests.post(
        KROKI_URL,
        data=source.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        timeout=timeout,
    )
    if resp.status_code == 200 and resp.content[:8] == b"\x89PNG\r\n\x1a\n":
        out_png.write_bytes(resp.content)
        return
    # Fallback: mermaid.ink (deflate + base64url)
    import base64
    import zlib

    compressed = zlib.compress(source.encode("utf-8"), 9)
    payload = base64.urlsafe_b64encode(compressed).decode("ascii")
    ink = requests.get(MERMAID_INK_TMPL.format(payload=payload), timeout=timeout)
    ink.raise_for_status()
    if ink.content[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(
            f"Mermaid render failed (Kroki {resp.status_code}, mermaid.ink not PNG)"
        )
    out_png.write_bytes(ink.content)


def _replace_mermaid_with_images(md: str, asset_dir: Path) -> tuple[str, int]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        source = match.group(1).strip()
        png = asset_dir / f"mermaid_{count:02d}.png"
        print(f"  Rendering mermaid diagram {count} -> {png.name}")
        _render_mermaid_kroki(source, png)
        # mistune will turn this into <img>; use relative path (same folder as HTML)
        return f"\n\n![ERD diagram {count}]({png.name})\n\n"

    replaced = MERMAID_FENCE.sub(_sub, md)
    return replaced, count


def _md_to_html_body(md: str) -> str:
    # mistune 2.x
    plugins = []
    for name in ("table", "strikethrough", "url", "task_lists"):
        try:
            plugins.append(mistune.PLUGINS[name])
        except (KeyError, TypeError, AttributeError):
            pass
    render = mistune.create_markdown(escape=False, plugins=plugins or None)
    html = render(md)
    # Tag mermaid images for CSS
    html = re.sub(
        r'<img src="(mermaid_\d+\.png)"([^>]*)>',
        r'<img class="mermaid-diagram" src="\1"\2>',
        html,
    )
    return html


def _wrap_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>{PRINT_CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def _chrome_print_pdf(chrome: Path, html_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    # file URI that works on Windows
    uri = html_path.resolve().as_uri()
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}",
        uri,
    ]
    print("  Running Chrome print-to-pdf…")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not pdf_path.is_file() or pdf_path.stat().st_size < 1000:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Chrome PDF failed (code {proc.returncode}): {err}")


def build(md_path: Path, pdf_path: Path) -> Path:
    if not md_path.is_file():
        raise FileNotFoundError(md_path)

    chrome = Path(__import__("os").environ["CHROME_PATH"]) if __import__("os").environ.get("CHROME_PATH") else _find_chrome()
    print(f"Source: {md_path}")
    print(f"Chrome: {chrome}")

    raw = md_path.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="opentrace_guide_pdf_") as tmp:
        tmp_dir = Path(tmp)
        print("Rendering Mermaid diagrams…")
        md_with_imgs, n_diagrams = _replace_mermaid_with_images(raw, tmp_dir)
        if n_diagrams == 0:
            print("WARNING: no mermaid blocks found")
        else:
            print(f"  {n_diagrams} diagram(s) rendered")

        body = _md_to_html_body(md_with_imgs)
        # Prefer absolute file URLs for images so Chrome always finds them
        for png in tmp_dir.glob("mermaid_*.png"):
            body = body.replace(
                f'src="{png.name}"',
                f'src="{png.resolve().as_uri()}"',
            )
            body = body.replace(
                f'src="{quote(png.name)}"',
                f'src="{png.resolve().as_uri()}"',
            )

        html_doc = _wrap_html("OpenTrace Mart Complete Guide", body)
        html_path = tmp_dir / "guide.html"
        html_path.write_text(html_doc, encoding="utf-8")

        _chrome_print_pdf(chrome, html_path, pdf_path)

    size_kb = pdf_path.stat().st_size / 1024
    print(f"Wrote {pdf_path} ({size_kb:.0f} KB, {n_diagrams} diagrams)")
    return pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()
    try:
        build(args.md, args.out)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
