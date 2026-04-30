#!/usr/bin/env python3
"""
CLI: tek argüman = giriş görsel yolu. stdout’a yalnızca: left | right | both | None

Örnek: python classify_pose_arms.py /path/to/pose-1.jpg
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

from arm_classifier import classify_image_path, load_landmarker


def main() -> None:
    if len(sys.argv) != 2:
        sys.stderr.write("Kullanim: classify_pose_arms.py PATH-TO-INPUT-IMAGE\n")
        sys.exit(2)

    path = sys.argv[1]

    lm = load_landmarker()
    try:
        label = classify_image_path(path, lm)
    finally:
        lm.close()

    print(label, flush=True)


if __name__ == "__main__":
    main()
