#!/usr/bin/env bash
# task3 klasöründe çalıştırın: bash run_tests.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFIER="$SCRIPT_DIR/classify_face_direction.py"

echo "=== Face Direction Classifier — Test ==="
for img in face-1.png face-2.png face-3.png; do
    result=$(python3 "$CLASSIFIER" "$SCRIPT_DIR/$img")
    echo "$img  →  $result"
done
echo "========================================"
