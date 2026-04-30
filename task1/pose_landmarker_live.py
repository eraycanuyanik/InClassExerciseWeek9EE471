"""
Pose Landmark — MediaPipe Tasks API (LIVE_STREAM)
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

İskelet + isteğe bağlı kişi segmentasyon maskesi (dokümanda olduğu gibi),
sağ panelde seçili eklemlerin normalize x,y,z bilgisi, kol kalkmış mı özeti.
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
PoseLandmarker        = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult  = vision.PoseLandmarkerResult

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model: lite (hızlı) | full | heavy (daha ağır)
POSE_MODEL_VARIANT = "pose_landmarker_lite"
MODEL_PATH = os.path.join(SCRIPT_DIR, f"{POSE_MODEL_VARIANT}.task")
MODEL_URL = (
    f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"{POSE_MODEL_VARIANT}/float16/1/{POSE_MODEL_VARIANT}.task"
)

SHOW_PREVIEW           = True
PRINT_INTERVAL_S       = 0.25
SIDE_PANEL_WIDTH       = 360
SEG_OVERLAY_WEIGHT     = 0.28
SEG_CORNER_RATIO       = 0.22

PL         = mp.solutions.pose.PoseLandmark
POSE_CONN  = mp.solutions.pose.POSE_CONNECTIONS
VIS_THRESH = 0.35


FEATURE_GROUPS = (
    ("YUZ", (
        ("Burun", PL.NOSE),
        ("Sol goz ic", PL.LEFT_EYE_INNER),
        ("Sol goz", PL.LEFT_EYE),
        ("Sol goz dis", PL.LEFT_EYE_OUTER),
        ("Sag goz ic", PL.RIGHT_EYE_INNER),
        ("Sag goz", PL.RIGHT_EYE),
        ("Sag goz dis", PL.RIGHT_EYE_OUTER),
        ("Sol kulak", PL.LEFT_EAR),
        ("Sag kulak", PL.RIGHT_EAR),
        ("Agiz sol", PL.MOUTH_LEFT),
        ("Agiz sag", PL.MOUTH_RIGHT),
    )),
    ("SOL OMZ / KOL", (
        ("Sol omuz", PL.LEFT_SHOULDER),
        ("Sol dirsek", PL.LEFT_ELBOW),
        ("Sol bilek", PL.LEFT_WRIST),
        ("Sol serce", PL.LEFT_PINKY),
        ("Sol isaret", PL.LEFT_INDEX),
        ("Sol basparmak", PL.LEFT_THUMB),
    )),
    ("SAG OMZ / KOL", (
        ("Sag omuz", PL.RIGHT_SHOULDER),
        ("Sag dirsek", PL.RIGHT_ELBOW),
        ("Sag bilek", PL.RIGHT_WRIST),
        ("Sag serce", PL.RIGHT_PINKY),
        ("Sag isaret", PL.RIGHT_INDEX),
        ("Sag basparmak", PL.RIGHT_THUMB),
    )),
    ("KALCA", (
        ("Sol kalca", PL.LEFT_HIP),
        ("Sag kalca", PL.RIGHT_HIP),
    )),
    ("SOL BACAK / AYAK", (
        ("Sol diz", PL.LEFT_KNEE),
        ("Sol ay.bilegi", PL.LEFT_ANKLE),
        ("Sol topuk", PL.LEFT_HEEL),
        ("Sol ay.parmagi", PL.LEFT_FOOT_INDEX),
    )),
    ("SAG BACAK / AYAK", (
        ("Sag diz", PL.RIGHT_KNEE),
        ("Sag ay.bilegi", PL.RIGHT_ANKLE),
        ("Sag topuk", PL.RIGHT_HEEL),
        ("Sag ay.parmagi", PL.RIGHT_FOOT_INDEX),
    )),
)


_lock: threading.Lock | None = None
_latest: PoseLandmarkerResult | None = None
_latest_ts_ms: int = -1


def ensure_model(path: str = MODEL_PATH) -> None:
    if os.path.isfile(path):
        return
    print(f"Model bulunamadı, indiriliyor:\n  {MODEL_URL}", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"Kaydedildi: {path}", file=sys.stderr)


def _on_result(
    result: PoseLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int,
) -> None:
    global _latest, _latest_ts_ms
    assert _lock is not None
    with _lock:
        _latest = result
        _latest_ts_ms = timestamp_ms


def _vis_ok(lm) -> bool:
    return (lm.visibility if lm.visibility is not None else 0.0) >= VIS_THRESH


def kol_ozeti(lms) -> list[str]:
    """Bilek görüntüde omuzdan yukarıysa kol kalkmış sayılır (y küçük = üst)."""
    msg: list[str] = []

    def line_sol() -> None:
        if not (_vis_ok(lms[PL.LEFT_WRIST.value]) and _vis_ok(lms[PL.LEFT_SHOULDER.value])):
            msg.append("SOL kol: veri zayif")
            return
        lw, ls = lms[PL.LEFT_WRIST.value], lms[PL.LEFT_SHOULDER.value]
        if lw.y < ls.y - 0.025:
            up = (
                lw.y < lms[PL.NOSE.value].y
                if _vis_ok(lms[PL.NOSE.value])
                else False
            )
            msg.append("SOL kol yukarı" + (" (baş ustü)" if up else ""))
        else:
            msg.append("SOL kol aşağı / nötr")

    def line_sag() -> None:
        if not (_vis_ok(lms[PL.RIGHT_WRIST.value]) and _vis_ok(lms[PL.RIGHT_SHOULDER.value])):
            msg.append("SAĞ kol: veri zayıf")
            return
        rw, rs = lms[PL.RIGHT_WRIST.value], lms[PL.RIGHT_SHOULDER.value]
        if rw.y < rs.y - 0.025:
            up = (
                rw.y < lms[PL.NOSE.value].y
                if _vis_ok(lms[PL.NOSE.value])
                else False
            )
            msg.append("SAĞ kol yukarı" + (" (baş üstü)" if up else ""))
        else:
            msg.append("SAĞ kol aşağı / nötr")

    line_sol()
    line_sag()
    return msg


def mask_to_float_h_w(seg_image: mp.Image, tgt_h: int, tgt_w: int) -> np.ndarray:
    raw = np.asarray(seg_image.numpy_view(), dtype=np.float32)
    if raw.ndim == 3:
        raw = raw.squeeze(-1)
    if raw.shape[0] != tgt_h or raw.shape[1] != tgt_w:
        raw = cv2.resize(raw, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
    return raw


def apply_segmentation_overlay(frame_bgr: np.ndarray, seg_image: mp.Image | None) -> None:
    if seg_image is None:
        return
    h, w = frame_bgr.shape[:2]
    seg = mask_to_float_h_w(seg_image, h, w)
    seg_e = seg[..., np.newaxis]
    tint = np.full_like(frame_bgr, (0, 220, 130), dtype=np.float32)
    blended = (
        frame_bgr.astype(np.float32) * (1.0 - SEG_OVERLAY_WEIGHT * seg_e)
        + tint * (SEG_OVERLAY_WEIGHT * seg_e)
    ).clip(0, 255).astype(np.uint8)
    frame_bgr[:] = blended

    inset_w = max(140, int(w * SEG_CORNER_RATIO))
    inset_h = int(inset_w * h / w)
    u8 = (seg * 255.0).clip(0, 255).astype(np.uint8)
    _, bw = cv2.threshold(u8, 128, 255, cv2.THRESH_BINARY)
    small = cv2.resize(bw, (inset_w, inset_h))
    small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    margin = 8
    y1 = h - inset_h - margin
    x1 = w - inset_w - margin
    frame_bgr[y1:y1 + inset_h, x1:x1 + inset_w] = (
        small_bgr.astype(np.float32) * 0.88
    ).clip(0, 255).astype(np.uint8)
    cv2.rectangle(
        frame_bgr,
        (x1 - 2, y1 - 2),
        (x1 + inset_w + 2, y1 + inset_h + 2),
        (0, 220, 130),
        2,
    )
    cv2.putText(
        frame_bgr,
        "Segmentasyon",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 230, 150),
        1,
        cv2.LINE_AA,
    )


def draw_pose_skeleton(
    frame_bgr: np.ndarray,
    pose_idx: int,
    lms,
) -> None:
    h, w = frame_bgr.shape[:2]
    pts = [
        (int((lm.x or 0) * w), int((lm.y or 0) * h))
        for lm in lms
    ]
    cols = [(0, 220, 255), (160, 120, 255)]
    col = cols[pose_idx % len(cols)]

    n_lm = len(lms)
    for a, b in POSE_CONN:
        if a < n_lm and b < n_lm and _vis_ok(lms[a]) and _vis_ok(lms[b]):
            cv2.line(frame_bgr, pts[a], pts[b], col, 2, cv2.LINE_AA)

    for i, (px, py) in enumerate(pts):
        if i < n_lm and _vis_ok(lms[i]):
            cv2.circle(frame_bgr, (px, py), 4, (0, 165, 255), -1, cv2.LINE_AA)


def apply_first_segmentation_masks(
    frame_bgr: np.ndarray,
    masks,
) -> None:
    """Tek kişiye yakın sahne: ilk geçerli maskeyi uygula (üst üste çift overlay olmasın)."""
    if not masks:
        return
    chosen = None
    for seg in masks:
        if seg is not None:
            chosen = seg
            break
    apply_segmentation_overlay(frame_bgr, chosen)


def banner_kol(frame_bgr: np.ndarray, lms) -> None:
    lines = kol_ozeti(lms)
    y0 = 12
    for i, t in enumerate(lines):
        yy = y0 + i * 22
        cv2.rectangle(frame_bgr, (4, yy - 16), (360, yy + 4), (0, 0, 0), -1)
        cv2.putText(
            frame_bgr,
            t,
            (10, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 250, 120),
            1,
            cv2.LINE_AA,
        )


def render_sidebar(panel_h: int, panel_w: int, result: PoseLandmarkerResult | None) -> np.ndarray:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    panel = np.full((panel_h, panel_w, 3), (42, 36, 32), dtype=np.uint8)
    WHITE = (245, 240, 232)
    MUTED = (130, 120, 150)
    ACCENT = (0, 212, 255)
    SPLIT = (60, 55, 50)

    y = 18
    cv2.rectangle(panel, (0, 0), (panel_w, 42), (50, 44, 40), thickness=-1)
    cv2.putText(panel, "LANDMARKS", (10, 28), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    bx = panel_w - 56
    cv2.rectangle(panel, (bx, 11), (panel_w - 8, 33), SPLIT, thickness=-1)
    cv2.putText(panel, "LIVE", (bx + 5, 28), FONT, 0.38, WHITE, 1, cv2.LINE_AA)

    y = 50
    n_poses = len(result.pose_landmarks) if result and result.pose_landmarks else 0
    tight = panel_h < 560 or n_poses >= 2
    fz = 0.28 if tight else 0.34
    step = 11 if tight else 14

    def put(
        x0: int,
        yy: int,
        s: str,
        col=WHITE,
        sc: float | None = None,
    ) -> None:
        cv2.putText(
            panel,
            s,
            (x0, yy),
            FONT,
            sc if sc is not None else fz,
            col,
            1,
            cv2.LINE_AA,
        )

    if not result or not result.pose_landmarks:
        put(12, panel_h // 2, "Poz bekleniyor...", MUTED)
        return panel

    col_j = 4
    col_x, col_y, col_z = 95, 168, 240
    clipped = False

    for pi, lms in enumerate(result.pose_landmarks):
        if y > panel_h - 50:
            clipped = True
            break

        seg_ok = bool(
            result.segmentation_masks
            and pi < len(result.segmentation_masks)
            and result.segmentation_masks[pi] is not None,
        )
        put(
            col_j,
            y,
            f"VUCUT {pi + 1}  seg={'+' if seg_ok else '-'}",
            ACCENT,
            fz + 0.04,
        )

        for t in kol_ozeti(lms):
            y += step
            if y > panel_h - 20:
                clipped = True
                break
            put(col_j + 2, y, t, (180, 220, 170), fz)

        if clipped:
            break

        y += step + 4
        put(col_j, y, "idx eklem", MUTED)
        put(col_x, y, "X", MUTED)
        put(col_y, y, "Y", MUTED)
        put(col_z, y, "Z", MUTED)
        y += step
        cv2.line(panel, (4, y), (panel_w - 4, y), SPLIT, 1, cv2.LINE_AA)
        y += step

        for gtitle, rows in FEATURE_GROUPS:
            if clipped:
                break
            put(col_j, y, gtitle, (100, 200, 120), fz + 0.02)
            y += step
            for tr_name, pidx in rows:
                if y > panel_h - 10:
                    clipped = True
                    break
                idx = pidx.value
                if idx >= len(lms):
                    continue
                lm = lms[idx]
                if lm.x is None or lm.y is None or lm.z is None:
                    continue
                vn = lm.visibility if lm.visibility is not None else 0.0
                put(col_j, y, f"{idx} {tr_name[:11]}", WHITE)
                put(col_x, y, f"{lm.x:.3f}")
                put(col_y, y, f"{lm.y:.3f}")
                put(col_z, y, f"{lm.z:.3f}")
                y += step
                if vn < VIS_THRESH:
                    put(col_j + 4, y, f"(viz {vn:.2f})", MUTED, fz - 0.05)
                    y += step
            y += 2

        if pi < len(result.pose_landmarks) - 1:
            y += 4
            cv2.line(panel, (4, y), (panel_w - 4, y), SPLIT, 1, cv2.LINE_AA)
            y += step + 2

        if clipped:
            break

    if clipped:
        put(6, panel_h - 10, "... (kesildi)", MUTED)
    return panel


def fmt_console(result: PoseLandmarkerResult | None) -> str:
    if not result or not result.pose_landmarks:
        return "Poz algilanmadi.\n"
    out: list[str] = []
    wl = result.pose_world_landmarks

    for pi, lms in enumerate(result.pose_landmarks):
        out.append(f"=== Vücut {pi + 1} ===")
        for t in kol_ozeti(lms):
            out.append(f"  {t}")
        out.append("")
        for gtitle, rows in FEATURE_GROUPS:
            out.append(f"  [{gtitle}]")
            for tr_name, pl_enum in rows:
                i = pl_enum.value
                if i >= len(lms):
                    continue
                lm = lms[i]
                line = (
                    f"    {i:2d} {tr_name:<16} nx={lm.x:.4f} ny={lm.y:.4f} nz={lm.z:.4f}"
                )
                if wl and pi < len(wl) and i < len(wl[pi]):
                    ww = wl[pi][i]
                    line += f"  |  wx={ww.x:.3f} wy={ww.y:.3f} wz={ww.z:.3f} m"
                out.append(line)
        out.append("")
    out.append("q: çıkış (önizleme)")
    return "\n".join(out) + "\n"


def main() -> None:
    global _lock, _latest, _latest_ts_ms

    ensure_model()
    _lock = threading.Lock()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,
        result_callback=_on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Webcam acilamadi.")

    t0 = time.time()
    last_print = 0.0

    if SHOW_PREVIEW:
        print("Calisiyor… q=cikis. Segmentasyon + iskelet.", file=sys.stderr)

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.time() - t0) * 1000)
                landmarker.detect_async(mp_img, ts_ms)

                with _lock:
                    result = _latest

                disp = frame.copy()
                masks = result.segmentation_masks if result else None

                if result and result.pose_landmarks:
                    apply_first_segmentation_masks(disp, masks)
                    for pi, lms in enumerate(result.pose_landmarks):
                        draw_pose_skeleton(disp, pi, lms)
                        if pi == 0:
                            banner_kol(disp, lms)

                if result is not None:
                    now = time.time()
                    if now - last_print >= PRINT_INTERVAL_S:
                        last_print = now
                        sys.stdout.write("\033[H\033[2J" + fmt_console(result))
                        sys.stdout.flush()

                if SHOW_PREVIEW:
                    pad = render_sidebar(disp.shape[0], SIDE_PANEL_WIDTH, result)
                    cv2.imshow(
                        "Pose (TASKS LIVE_STREAM)",
                        np.hstack([disp, pad]),
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
