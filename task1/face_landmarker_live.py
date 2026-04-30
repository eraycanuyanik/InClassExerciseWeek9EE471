"""
Face Landmark — MediaPipe Tasks API (LIVE_STREAM)
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python

Webcam + sağ kenarda seçilmiş yüz bölgeleri için JOINT(idx) / X / Y / Z paneli ve konsola özet.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision

BaseOptions            = mp.tasks.BaseOptions
FaceLandmarker         = vision.FaceLandmarker
FaceLandmarkerOptions  = vision.FaceLandmarkerOptions
FaceLandmarkerResult   = vision.FaceLandmarkerResult

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

SHOW_PREVIEW     = True
PRINT_INTERVAL_S = 0.25
SIDE_PANEL_WIDTH = 360

# Face Mesh indeksleriyle bilinen birkaç “ana” nokta (tablo ve daire çizimi için)
FEATURE_GROUPS = (
    ("BURUN / ORTA", (
        ("Burun ucu", 4),
        ("Cene", 152),
        ("Alin", 10),
    )),
    ("GOZLER", (
        ("Sol goz ic", 33),
        ("Sag goz ic", 263),
        ("Sol goz dis", 133),
        ("Sag goz dis", 362),
    )),
    ("AGIZ", (
        ("Agiz sol", 61),
        ("Agiz sag", 291),
        ("Ust dudak", 13),
        ("Alt dudak", 14),
    )),
    ("YANAK / KULAK", (
        ("Sol yanak", 50),
        ("Sag yanak", 280),
        ("Sol kulak", 234),
        ("Sag kulak", 454),
    )),
)

FACEMESH_CONTOURS = mp.solutions.face_mesh.FACEMESH_CONTOURS

_lock: threading.Lock | None = None
_latest: FaceLandmarkerResult | None = None
_latest_ts_ms: int = -1


def ensure_model(path: str = MODEL_PATH) -> None:
    if os.path.isfile(path):
        return
    print(f"Model bulunamadı, indiriliyor:\n  {MODEL_URL}", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"Kaydedildi: {path}", file=sys.stderr)


def _on_result(
    result: FaceLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int,
) -> None:
    global _latest, _latest_ts_ms
    assert _lock is not None
    with _lock:
        _latest = result
        _latest_ts_ms = timestamp_ms


def draw_face_bgr(frame_bgr, result: FaceLandmarkerResult) -> None:
    if not result.face_landmarks:
        return
    h, w = frame_bgr.shape[:2]
    colors = ((0, 255, 180), (255, 180, 0), (200, 100, 255))

    for fi, lms in enumerate(result.face_landmarks):
        col = colors[fi % len(colors)]
        pts = [(int(v.x * w), int(v.y * h)) for v in lms]
        n = len(pts)

        for a, b in FACEMESH_CONTOURS:
            if a < n and b < n:
                cv2.line(frame_bgr, pts[a], pts[b], col, 1, cv2.LINE_AA)

        for _grp, rows in FEATURE_GROUPS:
            for _name, idx in rows:
                if idx < n:
                    px, py = pts[idx]
                    cv2.circle(frame_bgr, (px, py), 4, (0, 200, 255), -1, cv2.LINE_AA)
                    cv2.putText(
                        frame_bgr,
                        str(idx),
                        (px + 4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )


def top_blendshapes(
    categories: list | None,
    k: int = 6,
) -> list[tuple[str, float]]:
    if not categories:
        return []
    scored = [
        (c.category_name or "?", float(c.score or 0.0))
        for c in categories
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]


def render_landmarks_sidebar(
    panel_h: int,
    panel_w: int,
    result: FaceLandmarkerResult | None,
) -> np.ndarray:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    panel = np.full((panel_h, panel_w, 3), (42, 36, 32), dtype=np.uint8)
    WHITE  = (245, 240, 232)
    MUTED  = (130, 120, 150)
    ACCENT = (0, 212, 255)
    SPLIT  = (60, 55, 50)

    y = 18
    cv2.rectangle(panel, (0, 0), (panel_w, 42), (50, 44, 40), thickness=-1)
    cv2.putText(panel, "LANDMARKS", (10, 28), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    bx = panel_w - 56
    cv2.rectangle(panel, (bx, 11), (panel_w - 8, 33), SPLIT, thickness=-1)
    cv2.putText(panel, "LIVE", (bx + 5, 28), FONT, 0.38, WHITE, 1, cv2.LINE_AA)

    y = 52
    n_faces = len(result.face_landmarks) if result and result.face_landmarks else 0
    tight = panel_h < 560 or n_faces >= 2
    fz = 0.30 if tight else 0.35
    step = 12 if tight else 15

    def put(x0: int, yy: int, s: str, col=WHITE, th: int = 1, sc: float | None = None) -> None:
        cv2.putText(
            panel, s, (x0, yy), FONT, sc if sc is not None else fz,
            col, th, cv2.LINE_AA,
        )

    if not result or not result.face_landmarks:
        put(12, panel_h // 2, "Yuz bekleniyor...", MUTED)
        return panel

    col_j = 6
    col_x = 108
    col_y = 178
    col_z = 242

    clipped = False

    for fi, landmarks in enumerate(result.face_landmarks):
        if y > panel_h - 40:
            clipped = True
            break

        put(col_j, y, f"YUZ {fi + 1}", ACCENT, 2, fz + 0.05)
        y += step + 4

        put(col_j, y, "idx ad", MUTED)
        put(col_x, y, "X", MUTED)
        put(col_y, y, "Y", MUTED)
        put(col_z, y, "Z", MUTED)
        y += step
        cv2.line(panel, (6, y), (panel_w - 6, y), SPLIT, 1, cv2.LINE_AA)
        y += step

        for grp_title, rows in FEATURE_GROUPS:
            if y > panel_h - 18:
                clipped = True
                break
            put(col_j, y, grp_title, (100, 200, 120), 1, fz + 0.02)
            y += step
            for name, idx in rows:
                if y > panel_h - 12:
                    clipped = True
                    break
                if idx >= len(landmarks):
                    continue
                lm = landmarks[idx]
                put(col_j, y, f"{idx} {name[:10]}", WHITE)
                put(col_x, y, f"{lm.x:.3f}")
                put(col_y, y, f"{lm.y:.3f}")
                put(col_z, y, f"{lm.z:.3f}")
                y += step
            y += 2

        if clipped:
            break

        # Blend shape (en yuksek skorlar)
        if (
            result.face_blendshapes
            and fi < len(result.face_blendshapes)
            and result.face_blendshapes[fi]
        ):
            if y > panel_h - 80:
                clipped = True
            else:
                y += 4
                put(col_j, y, "BLEND (ust skorlar)", MUTED, 1, fz)
                y += step
                for bname, sc in top_blendshapes(result.face_blendshapes[fi], k=5):
                    if y > panel_h - 10:
                        clipped = True
                        break
                    short = (bname[:22] + "..") if len(bname) > 24 else bname
                    put(col_j, y, f"{short}", WHITE, 1, fz - 0.02)
                    put(col_x + 40, y, f"{sc:.2f}", (180, 200, 255))
                    y += step

        if fi < len(result.face_landmarks) - 1:
            y += 6
            cv2.line(panel, (6, y), (panel_w - 6, y), SPLIT, 1, cv2.LINE_AA)
            y += step + 2

        if y > panel_h - 40:
            break

    if clipped:
        put(8, panel_h - 8, "... (liste kesildi)", MUTED, 1, fz)
    return panel


def fmt_console(result: FaceLandmarkerResult | None) -> str:
    if not result or not result.face_landmarks:
        return "Yuz algilanmadi.\n"
    lines: list[str] = []
    for fi, lms in enumerate(result.face_landmarks):
        lines.append(f"=== Yuz {fi + 1} ===")
        for grp, rows in FEATURE_GROUPS:
            lines.append(f"  [{grp}]")
            for name, idx in rows:
                if idx < len(lms):
                    lm = lms[idx]
                    lines.append(
                        f"    {idx:3d} {name:<16}  x={lm.x:.4f} y={lm.y:.4f} z={lm.z:.4f}"
                    )
        if result.face_blendshapes and fi < len(result.face_blendshapes):
            lines.append("  [Blend top 5]")
            for bname, sc in top_blendshapes(result.face_blendshapes[fi], k=5):
                lines.append(f"    {bname:<30} {sc:.3f}")
        lines.append("")
    lines.append("q: cikis (onizleme aciksa)")
    return "\n".join(lines) + "\n"


def main() -> None:
    global _lock, _latest, _latest_ts_ms

    ensure_model()
    _lock = threading.Lock()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        result_callback=_on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Webcam acilamadi.")

    last_print = 0.0
    t0 = time.time()

    if SHOW_PREVIEW:
        print("Calisiyor… q = cikis.", file=sys.stderr)

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.time() - t0) * 1000)
                landmarker.detect_async(mp_image, ts_ms)

                with _lock:
                    result = _latest

                if result is not None:
                    if SHOW_PREVIEW:
                        draw_face_bgr(frame, result)
                    now = time.time()
                    if now - last_print >= PRINT_INTERVAL_S:
                        last_print = now
                        sys.stdout.write("\033[H\033[2J" + fmt_console(result))
                        sys.stdout.flush()

                if SHOW_PREVIEW:
                    side = render_landmarks_sidebar(
                        frame.shape[0], SIDE_PANEL_WIDTH, result,
                    )
                    cv2.imshow(
                        "Face (TASKS LIVE_STREAM)",
                        np.hstack([frame, side]),
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(0.005)

    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
