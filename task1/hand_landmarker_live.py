"""
Hand Landmark — MediaPipe Tasks API (LIVE_STREAM)
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

Konsola normalize (x,y,z) ve world (metre) konumları yazar.
OpenCV: sol webcam, sağda gruplanmış LANDMARKS (JOINT / X / Y / Z tablosu). Web UI yok.
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

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult  = vision.HandLandmarkerResult

# İlk çalıştırmada model indirilir (internet gerekir).
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# True: sadece iskelet + noktalar (web yok). False: yalnızca terminal çıktısı.
SHOW_PREVIEW     = True
PRINT_INTERVAL_S = 0.2

# Grup tablosuna göre fotoğrafındaki yapıyla eşlenen indeksler:
# WRIST → THUMB (CMC…) → parmak MCP/PIP/DIP/Tip.
TABLE_GROUPS = (
    ("WRIST", (("Wrist", 0),)),
    ("THUMB", (("CMC", 1), ("MCP", 2), ("IP", 3), ("Tip", 4))),
    ("INDEX", (("MCP", 5), ("PIP", 6), ("DIP", 7), ("Tip", 8))),
    ("MIDDLE", (("MCP", 9), ("PIP", 10), ("DIP", 11), ("Tip", 12))),
    ("RING", (("MCP", 13), ("PIP", 14), ("DIP", 15), ("Tip", 16))),
    ("PINKY", (("MCP", 17), ("PIP", 18), ("DIP", 19), ("Tip", 20))),
)

SIDE_PANEL_WIDTH = 340

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

LANDMARK_NAMES = (
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
)


def mp_handedness_to_tr(category_name: str | None) -> tuple[str, str]:
    """Ayna webcam (flip) ön izlemesinde etiketi kullanıcıya göre düzelt: MP Left↔Right swap."""
    if category_name == "Left":
        return "SAĞ EL", "MP Left"
    if category_name == "Right":
        return "SOL EL", "MP Right"
    return (category_name or "?"), "?"


def ensure_model(path: str = MODEL_PATH) -> None:
    if os.path.isfile(path):
        return
    print(f"Model bulunamadı, indiriliyor:\n  {MODEL_URL}", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"Kaydedildi: {path}", file=sys.stderr)


_lock: threading.Lock | None = None
_latest: HandLandmarkerResult | None = None
_latest_ts_ms: int = -1


def _on_result(
    result: HandLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int,
) -> None:
    global _latest, _latest_ts_ms
    assert _lock is not None
    with _lock:
        _latest = result
        _latest_ts_ms = timestamp_ms


def render_landmarks_sidebar(
    panel_h: int,
    panel_w: int,
    result: HandLandmarkerResult | None,
) -> np.ndarray:
    """Fotoğraftaki panele yakın — gruplanmış JOINT / X / Y / Z (normalize) tablosu."""
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    # Koyu arka plan + metin renkleri (BGR)
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
    n_hands = (
        len(result.hand_landmarks) if result and result.hand_landmarks else 0
    )
    tight = panel_h < 560 or n_hands >= 2
    fz = 0.32 if tight else 0.38
    step = 13 if tight else 16

    def put(x0: int, yy: int, s: str, col=WHITE, th: int = 1, sc: float | None = None) -> None:
        cv2.putText(
            panel, s, (x0, yy), FONT, sc if sc is not None else fz,
            col, th, cv2.LINE_AA,
        )

    if not result or not result.hand_landmarks:
        put(12, panel_h // 2, "El bekleniyor...", MUTED)
        return panel

    col_j = 10
    col_x = 108
    col_y = 170
    col_z = 234

    # Sütun başlıkları
    put(col_j, y, "JOINT", MUTED)
    put(col_x, y, "X", MUTED)
    put(col_y, y, "Y", MUTED)
    put(col_z, y, "Z", MUTED)
    y += step + 2
    cv2.line(panel, (8, y), (panel_w - 8, y), SPLIT, 1, cv2.LINE_AA)
    y += step

    clipped = False

    for hi, landmarks in enumerate(result.hand_landmarks):
        cat = (
            result.handedness[hi][0]
            if hi < len(result.handedness) and result.handedness[hi]
            else None
        )
        label_tr = "? EL"
        label_mp = ""
        if cat:
            label_tr, _ = mp_handedness_to_tr(cat.category_name)
            label_mp = cat.category_name or ""

        headline = (
            f"{label_tr}"
            + (f" [{label_mp}]" if label_mp else "")
        )

        put(col_j, y, headline, ACCENT, 2, fz + 0.06)
        y += step + 4

        for grp_title, joints in TABLE_GROUPS:
            put(col_j, y, grp_title, (120, 200, 100), 1, fz + 0.02)
            y += step
            for jname, ji in joints:
                if y > panel_h - 12:
                    clipped = True
                    break
                if ji >= len(landmarks):
                    continue
                lm = landmarks[ji]
                put(col_j, y, jname, WHITE)
                put(col_x, y, f"{lm.x:.3f}")
                put(col_y, y, f"{lm.y:.3f}")
                put(col_z, y, f"{lm.z:.3f}")
                y += step

            if clipped:
                break
            y += 2

        if clipped:
            break

        if hi < len(result.hand_landmarks) - 1:
            y += 6
            cv2.line(panel, (6, y), (panel_w - 6, y), SPLIT, 1, cv2.LINE_AA)
            y += step + 2

        if y > panel_h - 80:
            break

    if clipped:
        put(8, panel_h - 8, "... (liste kesildi)", MUTED, 1, fz)
    return panel


def draw_hands_bgr(frame_bgr, result: HandLandmarkerResult) -> None:
    if not result.hand_landmarks:
        return
    h, w = frame_bgr.shape[:2]
    for hi, hand_lms in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 180), 2, cv2.LINE_AA)
        for i, (px, py) in enumerate(pts):
            cv2.circle(frame_bgr, (px, py), 4, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame_bgr,
                str(i),
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Hangi el + bileğin normalize x,y,z — ekranda net okunsun diye bilek yakınında
        if hi < len(result.handedness) and result.handedness[hi]:
            cat  = result.handedness[hi][0]
            tr_lbl, mp_key = mp_handedness_to_tr(cat.category_name)
            sc   = cat.score if cat.score is not None else 0.0
            wm   = hand_lms[0]
            bx, by = pts[0]
            ox = min(max(6, bx - 140), w - 300)
            oy = max(54, by - 52)

            line1 = f"{tr_lbl} ({mp_key}, guven:{sc:.2f})"
            line2 = f"bilek  x={wm.x:.3f}  y={wm.y:.3f}  z={wm.z:.3f}"

            (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            bw = max(tw1, tw2) + 18
            cv2.rectangle(frame_bgr, (ox - 6, oy - th1 - 18), (ox + bw, oy + th2 + 8), (0, 0, 0), -1)
            cv2.rectangle(frame_bgr, (ox - 6, oy - th1 - 18), (ox + bw, oy + th2 + 8), (0, 220, 130), 1)
            cv2.putText(
                frame_bgr, line1, (ox, oy - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr, line2, (ox, oy + th2 + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 240), 1, cv2.LINE_AA,
            )


def wrist_xyz_lines(result: HandLandmarkerResult) -> list[str]:
    """Konsola üst blok: hangi el, bilek x/y/z özetleri."""
    out: list[str] = []
    if not result.hand_landmarks:
        return out
    out.append("[Özet — bilek normalize x,y,z; [0–1]: görünür alan]")
    detected: list[str] = []
    for hi, landmarks in enumerate(result.hand_landmarks):
        if hi >= len(result.handedness) or not result.handedness[hi]:
            continue
        cat       = result.handedness[hi][0]
        tr_lbl, _ = mp_handedness_to_tr(cat.category_name)
        sc        = cat.score if cat.score is not None else 0.0
        wm        = landmarks[0]
        detected.append(f"{tr_lbl} (~{cat.category_name}, guven:{sc:.2f})")
        out.append(
            f"  • El {hi + 1}: {tr_lbl}  →  bilek x={wm.x:.4f} y={wm.y:.4f} z={wm.z:.4f}"
        )
    if detected:
        out.insert(1, "[Algilanan eller] " + " | ".join(detected))
    out.append("")
    out.append(
        "(Sol/sağ etiketi, yatay aynalı görüntü için MP Left↔Right ile uyumlu ters yazılır;",
    )
    out.append(" konsoldaki Left/Right = ham Mediapipe sınıflandırması.)")
    out.append("")
    return out


def fmt_landmarks(result: HandLandmarkerResult) -> str:
    if not result.hand_landmarks:
        return "El algılanmadı.\n"
    lines: list[str] = list(wrist_xyz_lines(result))
    lines.append("--- Tüm noktalar (normalize nx,ny,nz | world wx,wy,wz m) ---")
    for hi, landmarks in enumerate(result.hand_landmarks):
        cat = result.handedness[hi][0]
        tr_lbl, mp_key = mp_handedness_to_tr(cat.category_name)
        score = cat.score if cat.score is not None else 0.0
        lines.append(
            f"== El {hi + 1}: {tr_lbl}  [{mp_key} / {cat.category_name}]  guven: {score:.2f} =="
        )

        wl = (
            result.hand_world_landmarks[hi]
            if result.hand_world_landmarks and hi < len(result.hand_world_landmarks)
            else None
        )

        lines.append(
            f"{'#':>3}  {'ad':<12}  {'nx':>7} {'ny':>7} {'nz':>7}  |  "
            f"{'wx_m':>8} {'wy_m':>8} {'wz_m':>8}"
        )
        for i, lm in enumerate(landmarks):
            name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"L{i}"
            row = (
                f"{i:3d}  {name:<12}  {lm.x:7.4f} {lm.y:7.4f} {lm.z:7.4f}  |  "
            )
            if wl and i < len(wl):
                wm = wl[i]
                row += f"{wm.x:8.4f} {wm.y:8.4f} {wm.z:8.4f}"
            else:
                row += f"{'—':>8} {'—':>8} {'—':>8}"
            lines.append(row)
        lines.append("")
    footer = "\nq: çık (önizleme açıksa)" if SHOW_PREVIEW else "\nCtrl+C ile çık"
    return "\n".join(lines) + footer + "\n"


def main() -> None:
    global _lock, _latest, _latest_ts_ms

    ensure_model()

    _lock = threading.Lock()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=_on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Webcam açılamadı.")

    last_print_at = 0.0

    if SHOW_PREVIEW:
        print("Çalışıyor… Konsola konumlar yazılır. Pencerede q — çıkış.", file=sys.stderr)
    else:
        print(
            "Çalışıyor… Sadece konsol (önizleme kapalı). Çıkış: Ctrl+C.",
            file=sys.stderr,
        )

    stream_start = time.time()

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)

                ts_ms = int((time.time() - stream_start) * 1000)
                landmarker.detect_async(mp_image, ts_ms)

                with _lock:
                    result = _latest

                if result is not None:
                    if SHOW_PREVIEW:
                        draw_hands_bgr(frame, result)

                    now = time.time()
                    if now - last_print_at >= PRINT_INTERVAL_S:
                        last_print_at = now
                        # ANSI home + clear screen (birçok terminalde düzgün güncellenir)
                        sys.stdout.write("\033[H\033[2J" + fmt_landmarks(result))
                        sys.stdout.flush()

                if SHOW_PREVIEW:
                    sidebar = render_landmarks_sidebar(
                        frame.shape[0],
                        SIDE_PANEL_WIDTH,
                        result,
                    )
                    display = np.hstack([frame, sidebar])
                    cv2.imshow(
                        "Hands (TASKS LIVE_STREAM)",
                        display,
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
