#!/usr/bin/env bash
# Smoke test for Railway Streamlit QA service.
# Usage: ./scripts/smoke_streamlit_railway.sh https://your-streamlit.up.railway.app

set -euo pipefail

BASE_URL="${1:-}"
if [ -z "$BASE_URL" ]; then
    echo "Usage: $0 <streamlit-base-url>" >&2
    echo "Example: $0 https://opentrace-rag-streamlit.up.railway.app" >&2
    exit 1
fi

BASE_URL="${BASE_URL%/}"
HEALTH_URL="${BASE_URL}/_stcore/health"

echo "=== Streamlit Railway smoke test ==="
echo "Base URL: ${BASE_URL}"
echo ""

echo "1. Health check: GET ${HEALTH_URL}"
BODY="$(curl -fsS --max-time 30 "${HEALTH_URL}")"
if [ "$BODY" != "ok" ]; then
    echo "   FAIL: expected body 'ok', got: ${BODY}" >&2
    exit 1
fi
echo "   OK (200, body=ok)"

echo ""
echo "2. App shell: GET ${BASE_URL}/"
STATUS="$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 30 "${BASE_URL}/")"
if [ "$STATUS" != "200" ]; then
    echo "   FAIL: expected HTTP 200, got ${STATUS}" >&2
    exit 1
fi
echo "   OK (HTTP ${STATUS})"

echo ""
echo "=== All checks passed ==="
echo "Open ${BASE_URL} and run preset queries (meta / product / full_rag)."
