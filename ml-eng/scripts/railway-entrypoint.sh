#!/bin/bash
# Railway entrypoint script for OpenTrace RAG API
# Handles GCP credentials from base64-encoded environment variable

set -e

echo "=== Railway Entrypoint ==="
echo "Starting OpenTrace RAG API..."

# Handle GCP credentials if provided as base64
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS_BASE64" ]; then
    echo "Decoding GCP service account credentials..."
    echo "$GOOGLE_APPLICATION_CREDENTIALS_BASE64" | base64 -d > /tmp/gcp-sa-key.json
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-sa-key.json
    echo "✓ GCP credentials configured"
else
    echo "⚠ GOOGLE_APPLICATION_CREDENTIALS_BASE64 not set"
    echo "  BigQuery retrieval will fail without credentials"
fi

# Verify required environment variables
echo "Checking required environment variables..."

if [ -z "$QDRANT_URL" ]; then
    echo "✗ QDRANT_URL not set"
    exit 1
fi
echo "✓ QDRANT_URL configured"

if [ -z "$QDRANT_API_KEY" ]; then
    echo "✗ QDRANT_API_KEY not set"
    exit 1
fi
echo "✓ QDRANT_API_KEY configured"

if [ -z "$HF_API_TOKEN" ]; then
    echo "⚠ HF_API_TOKEN not set - LLM generation will fail"
fi

if [ -z "$BQ_PROJECT" ]; then
    echo "⚠ BQ_PROJECT not set - BigQuery retrieval disabled"
fi

echo "==========================="
echo "Environment ready. Starting application..."
echo ""

# Execute the command passed to the script (from CMD in Dockerfile)
exec "$@"
