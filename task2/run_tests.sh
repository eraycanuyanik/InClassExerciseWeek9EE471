#!/usr/bin/env bash
# Yerel test: task2 dizininde çalıştırın
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-task2}"
for img in pose-1.jpg pose-2.jpg pose-3.jpg; do
  printf '%s -> ' "$img"
  python3 classify_pose_arms.py "$img"
done
