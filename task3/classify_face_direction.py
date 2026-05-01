#!/usr/bin/env python3
"""
CLI: tek argüman = giriş görsel yolu. stdout'a yalnızca: left | right | straight

Örnek: python classify_face_direction.py /path/to/face-1.png
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

from face_direction_classifier import classify_image_path, load_landmarker


def main() -> None:
    if len(sys.argv) != 2:
        sys.stderr.write("Kullanim: classify_face_direction.py PATH-TO-INPUT-IMAGE\n")
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
