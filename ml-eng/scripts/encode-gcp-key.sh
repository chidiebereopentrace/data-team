#!/bin/bash
# Helper script to base64-encode GCP service account key for Railway
# Usage: ./encode-gcp-key.sh path/to/your-key.json

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-gcp-service-account-key.json>"
    echo ""
    echo "Example:"
    echo "  $0 ../config/keys/opentrace-bq-key.json"
    echo ""
    exit 1
fi

KEY_FILE="$1"

if [ ! -f "$KEY_FILE" ]; then
    echo "Error: File not found: $KEY_FILE"
    exit 1
fi

echo "Encoding GCP service account key..."
echo ""
echo "File: $KEY_FILE"
echo ""
echo "=== BASE64 ENCODED KEY (copy this to Railway) ==="
echo ""

# Encode the file and output
cat "$KEY_FILE" | base64 | tr -d '\n'

echo ""
echo ""
echo "=== END ==="
echo ""
echo "To use in Railway:"
echo "1. Copy the base64 string above"
echo "2. Go to Railway project → Variables"
echo "3. Add variable: GOOGLE_APPLICATION_CREDENTIALS_BASE64"
echo "4. Paste the base64 string as the value"
echo ""
