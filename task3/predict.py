"""
Cog / Replicate uyumlu predict modülü — giriş görsel dosyası,
çıkış: "left", "right", "straight"
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from cog import BasePredictor, Input, Path

from face_direction_classifier import classify_image_path, load_landmarker


class Predict(BasePredictor):
    def setup(self) -> None:
        self._landmarker = load_landmarker()

    def predict(self, image: Path = Input(description="$PATH-TO-INPUT-IMAGE")) -> str:
        return classify_image_path(str(image), self._landmarker)
