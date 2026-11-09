#!/usr/bin/env bash
# Rebuild everything from input/ into build/ and dist/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build dist
[ -d node_modules ] || npm install --silent
python3 src/parse_indicators.py
python3 src/build_data.py > build/build_log.txt && head -4 build/build_log.txt
node src/geo.mjs
python3 src/build_xlsx.py
python3 src/assemble_html.py
echo "Done. Open dist/Continental_OTA_Dashboard.html in a browser."
echo "Note: Excel formulas calculate when the workbook is opened in Excel or LibreOffice."
