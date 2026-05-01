"""
Mediapipe FaceLandmarker (IMAGE) ile statik görüntüden yüz bakış yönü sınıflandırması.
Çıktı: "left", "right", "straight"

Algoritma:
  - Burun ucu (landmark 1) ile sol/sağ göz dış köşelerinin (33, 263) x koordinatları kullanılır.
  - ratio = (nose_x - left_eye_x) / (right_eye_x - left_eye_x)
  - ratio < LEFT_THRESH  → "left"   (kişi kendi soluna bakıyor, izleyicinin soluna doğru)
  - ratio > RIGHT_THRESH → "right"
  - aksi hâlde          → "straight"
"""

from __future__ import annotations

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision

MODEL_NAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Yatay oran eşikleri (0..1 aralığında)
LEFT_THRESH = 0.44
RIGHT_THRESH = 0.56

# Kritik landmark indisleri (FaceLandmarker 478 nokta)
NOSE_TIP = 1          # burun ucu
LEFT_EYE_OUTER = 33   # sol göz dış köşe  (görüntünün solunda)
RIGHT_EYE_OUTER = 263 # sağ göz dış köşe  (görüntünün sağında)


def model_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME)


def ensure_model() -> None:
    p = model_path()
    if not os.path.isfile(p):
        urllib.request.urlretrieve(MODEL_URL, p)


def classify_face_direction_rgb(rgb: np.ndarray, landmarker) -> str:
    """rgb: HWC RGB uint8 görüntü → 'left' | 'right' | 'straight'"""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return "straight"  # yüz bulunamadıysa varsayılan

    lms = result.face_landmarks[0]

    nose_x = lms[NOSE_TIP].x
    left_eye_x = lms[LEFT_EYE_OUTER].x
    right_eye_x = lms[RIGHT_EYE_OUTER].x

    eye_span = right_eye_x - left_eye_x
    if eye_span < 1e-6:
        return "straight"

    ratio = (nose_x - left_eye_x) / eye_span

    if ratio < LEFT_THRESH:
        return "left"
    if ratio > RIGHT_THRESH:
        return "right"
    return "straight"


def classify_image_path(abs_path: str, landmarker) -> str:
    bgr = cv2.imread(abs_path)
    if bgr is None:
        raise FileNotFoundError(f"Görsel okunamadı: {abs_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return classify_face_direction_rgb(rgb, landmarker)


def load_landmarker():
    """Tek seferlik model yükleme (CLI veya cog setup için)."""
    ensure_model()
    opt = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path()),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.FaceLandmarker.create_from_options(opt)
