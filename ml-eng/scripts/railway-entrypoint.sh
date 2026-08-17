#!/bin/bash
# Railway entrypoint: validate GCP SA JSON before starting the app.
# Canonical credential path: /tmp/gcp-sa-key.json (never leave a stale /tmp/gcp-sa.json).

set -euo pipefail

GCP_SA_PATH="/tmp/gcp-sa-key.json"
LEGACY_SA_PATH="/tmp/gcp-sa.json"

echo "=== Railway Entrypoint ==="

if [ -n "${GOOGLE_APPLICATION_CREDENTIALS_BASE64:-}" ]; then
    echo "Decoding GOOGLE_APPLICATION_CREDENTIALS_BASE64..."
    # Prefer printf over echo to avoid trailing-newline / flag issues with some shells.
    if ! printf '%s' "$GOOGLE_APPLICATION_CREDENTIALS_BASE64" | base64 -d > "$GCP_SA_PATH" 2>/dev/null; then
        echo "✗ Failed to base64-decode GOOGLE_APPLICATION_CREDENTIALS_BASE64"
        rm -f "$GCP_SA_PATH"
        exit 1
    fi
    chmod 600 "$GCP_SA_PATH"
    if ! python -c "import json,sys; json.load(open(sys.argv[1], encoding='utf-8'))" "$GCP_SA_PATH"; then
        echo "✗ Decoded GCP credentials are not valid JSON: $GCP_SA_PATH"
        rm -f "$GCP_SA_PATH"
        exit 1
    fi
    # Remove legacy empty/corrupt path that ADC may still point at from Railway env.
    rm -f "$LEGACY_SA_PATH"
    export GOOGLE_APPLICATION_CREDENTIALS="$GCP_SA_PATH"
    echo "✓ GCP credentials validated at $GCP_SA_PATH"
elif [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    CRED_PATH="$GOOGLE_APPLICATION_CREDENTIALS"
    if [ ! -f "$CRED_PATH" ]; then
        echo "✗ GOOGLE_APPLICATION_CREDENTIALS path does not exist: $CRED_PATH"
        exit 1
    fi
    if ! python -c "import json,sys; json.load(open(sys.argv[1], encoding='utf-8'))" "$CRED_PATH"; then
        echo "✗ GOOGLE_APPLICATION_CREDENTIALS is not valid JSON: $CRED_PATH"
        exit 1
    fi
    echo "✓ Existing GOOGLE_APPLICATION_CREDENTIALS validated: $CRED_PATH"
else
    echo "⚠ Neither GOOGLE_APPLICATION_CREDENTIALS_BASE64 nor GOOGLE_APPLICATION_CREDENTIALS set"
    if [ -n "${BQ_PROJECT:-}" ]; then
        echo "✗ BQ_PROJECT is set but GCP credentials are missing"
        exit 1
    fi
    echo "  BigQuery NL2SQL will fail until credentials are provided"
fi

if [ -z "${QDRANT_URL:-}" ]; then
    echo "✗ QDRANT_URL not set"
    exit 1
fi
echo "✓ QDRANT_URL configured"

if [ -z "${QDRANT_API_KEY:-}" ]; then
    echo "✗ QDRANT_API_KEY not set"
    exit 1
fi
echo "✓ QDRANT_API_KEY configured"

if [ -z "${BQ_PROJECT:-}" ]; then
    echo "⚠ BQ_PROJECT not set — BigQuery retrieval disabled"
fi

echo "==========================="
echo "Starting: $*"
exec "$@"
