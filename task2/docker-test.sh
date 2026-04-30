#!/usr/bin/env bash
# task2 içinde: imajı derler ve pose-1/2/3 için Docker ile test eder
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
IMAGE="${IMAGE:-pose-arm-classifier}"

echo "==> docker build ..."
docker build -t "$IMAGE" .

echo ""
for f in pose-1.jpg pose-2.jpg pose-3.jpg; do
  printf '%s -> ' "$f"
  docker run --rm -v "$DIR:/data:ro" "$IMAGE" "/data/$f"
done

echo ""
echo "Docker Compose ile aynı test:"
echo "  docker compose build && docker compose run --rm classify /data/pose-1.jpg"
