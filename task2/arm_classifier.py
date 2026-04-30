"""
Mediapipe PoseLandmarker (IMAGE) ile statik görüntüden kol konumu sınıflandırması.
Çıktı: "left", "right", "both", "None" (string — tanı detect yok ya da güvenilir veri yok).
"""

from __future__ import annotations

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision

MODEL_NAME = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

VIS_THRESH = 0.45

# Mediapipe çözüm kataloğu (indeks doğruluğu için)
LM = mp.solutions.pose.PoseLandmark


def model_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME)


def ensure_model() -> None:
    p = model_path()
    if not os.path.isfile(p):
        urllib.request.urlretrieve(MODEL_URL, p)


def _ok(lm) -> bool:
    return float(lm.visibility or 0.0) >= VIS_THRESH


def left_arm_up(lms: list, nose_tol: float = 0.03) -> bool:
    lw, ln = LM.LEFT_WRIST, LM.NOSE
    if not (_ok(lms[lw.value]) and _ok(lms[ln.value])):
        return False
    # Kişinin anatomik sol bileği (Mediapipe sol); görüntüde y küçük → yukarı
    return float(lms[lw.value].y) + nose_tol < float(lms[ln.value].y)


def right_arm_up(lms: list, nose_tol: float = 0.03) -> bool:
    rw, ln = LM.RIGHT_WRIST, LM.NOSE
    if not (_ok(lms[rw.value]) and _ok(lms[ln.value])):
        return False
    # Kişinin anatomik sağ bileği (Mediapipe sağ)
    return float(lms[rw.value].y) + nose_tol < float(lms[ln.value].y)


def classify_pose_arms_rgb(rgb: np.ndarray, landmarker) -> str:
    """rgb: HWC RGB uint8"""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return "None"

    # İlk yüz için tanı kullan (Çoklu yüz sıklıkta değilse yeter.)
    lms = result.pose_landmarks[0]
    lu = left_arm_up(lms)
    ru = right_arm_up(lms)
    if lu and ru:
        return "both"
    if lu:
        return "left"
    if ru:
        return "right"
    return "None"


def classify_image_path(abs_path: str, landmarker) -> str:
    bgr = cv2.imread(abs_path)
    if bgr is None:
        raise FileNotFoundError(f"Görsel okunamadı: {abs_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return classify_pose_arms_rgb(rgb, landmarker)


def load_landmarker():
    """Tek seferlik kurulum için (CLI veya cog setup)."""
    ensure_model()
    opt = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path()),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return vision.PoseLandmarker.create_from_options(opt)
