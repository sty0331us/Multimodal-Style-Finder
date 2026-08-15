#!/usr/bin/env bash
# Download the precomputed outfit embedding catalog.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/data/swift-style-embeddings.pkl}"
URL="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/95eJ0YJVtqTZhEd7RaUlew/processed-swift-style-with-embeddings.pkl"

mkdir -p "$(dirname "$DEST")"
echo "Downloading catalog embeddings to ${DEST}"
curl -L --fail --retry 3 -o "$DEST" "$URL"
echo "Saved $(wc -c < "$DEST") bytes"
