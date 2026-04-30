import math
import time
import threading

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template

app = Flask(__name__)

# ── Mediapipe aliases ──────────────────────────────────────────────────────
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face           = mp.solutions.face_mesh
mp_hands_mod      = mp.solutions.hands
mp_pose_mod       = mp.solutions.pose

# ── Shared state ───────────────────────────────────────────────────────────
_lock           = threading.Lock()
_mode           = 1          # 1=face  2=hands  3=pose
_landmarks_data = {}
_output_frame   = None       # latest processed BGR frame

# ── Geometry helper ────────────────────────────────────────────────────────
def angle_between(a, b, c):
    """Angle (degrees) at joint b formed by the a-b-c triplet."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return round(math.degrees(math.acos(np.clip(cos_val, -1.0, 1.0))), 1)

# ── Static landmark metadata ───────────────────────────────────────────────
FACE_KEY_POINTS = {
    "Nose Tip":        4,
    "Left Eye":        33,
    "Right Eye":       263,
    "Mouth Left":      61,
    "Mouth Right":     291,
    "Chin":            152,
    "Left Ear":        234,
    "Right Ear":       454,
    "Forehead":        10,
    "Left Cheek":      50,
    "Right Cheek":     280,
}

HAND_LM_NAMES = [
    "Wrist",
    "Thumb CMC", "Thumb MCP", "Thumb IP",    "Thumb Tip",
    "Index MCP", "Index PIP", "Index DIP",   "Index Tip",
    "Middle MCP","Middle PIP","Middle DIP",  "Middle Tip",
    "Ring MCP",  "Ring PIP",  "Ring DIP",    "Ring Tip",
    "Pinky MCP", "Pinky PIP", "Pinky DIP",   "Pinky Tip",
]

PL = mp_pose_mod.PoseLandmark
POSE_LM_NAMES = [lm.name.replace("_", " ").title() for lm in PL]

POSE_ANGLES_DEF = {
    "Left Elbow":     (PL.LEFT_SHOULDER,  PL.LEFT_ELBOW,   PL.LEFT_WRIST),
    "Right Elbow":    (PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW,  PL.RIGHT_WRIST),
    "Left Shoulder":  (PL.LEFT_ELBOW,     PL.LEFT_SHOULDER,PL.LEFT_HIP),
    "Right Shoulder": (PL.RIGHT_ELBOW,    PL.RIGHT_SHOULDER,PL.RIGHT_HIP),
    "Left Hip":       (PL.LEFT_SHOULDER,  PL.LEFT_HIP,     PL.LEFT_KNEE),
    "Right Hip":      (PL.RIGHT_SHOULDER, PL.RIGHT_HIP,    PL.RIGHT_KNEE),
    "Left Knee":      (PL.LEFT_HIP,       PL.LEFT_KNEE,    PL.LEFT_ANKLE),
    "Right Knee":     (PL.RIGHT_HIP,      PL.RIGHT_KNEE,   PL.RIGHT_ANKLE),
}

# Joints where we draw the angle value directly on the video
POSE_ANGLE_ON_FRAME = {
    "Left Elbow":     PL.LEFT_ELBOW,
    "Right Elbow":    PL.RIGHT_ELBOW,
    "Left Shoulder":  PL.LEFT_SHOULDER,
    "Right Shoulder": PL.RIGHT_SHOULDER,
    "Left Knee":      PL.LEFT_KNEE,
    "Right Knee":     PL.RIGHT_KNEE,
    "Left Hip":       PL.LEFT_HIP,
    "Right Hip":      PL.RIGHT_HIP,
}

# ── Processing functions ───────────────────────────────────────────────────
def process_face(frame, detector):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    data    = {}

    if results.multi_face_landmarks:
        h, w = frame.shape[:2]
        for fi, face_lms in enumerate(results.multi_face_landmarks):
            mp_drawing.draw_landmarks(
                frame, face_lms, mp_face.FACEMESH_TESSELATION, None,
                mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                frame, face_lms, mp_face.FACEMESH_CONTOURS, None,
                mp_drawing_styles.get_default_face_mesh_contours_style())

            face_data = {}
            for name, li in FACE_KEY_POINTS.items():
                lm   = face_lms.landmark[li]
                cx   = int(lm.x * w)
                cy   = int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 230, 180), -1)
                cv2.putText(frame, name.split()[0], (cx + 6, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 180), 1, cv2.LINE_AA)
                face_data[name] = {
                    "x": round(lm.x, 3),
                    "y": round(lm.y, 3),
                    "z": round(lm.z, 3),
                }
            data[f"Face {fi + 1}"] = face_data

    with _lock:
        globals()["_landmarks_data"] = {"type": "face", "data": data}
    return frame


def process_hands(frame, detector):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    data    = {}

    if results.multi_hand_landmarks:
        h, w = frame.shape[:2]
        tip_indices = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}
        for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(
                frame, hand_lms, mp_hands_mod.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            label = hand_info.classification[0].label
            for fname, fi in tip_indices.items():
                lm = hand_lms.landmark[fi]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(frame, fname, (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 190, 0), 1, cv2.LINE_AA)

            data[f"{label} Hand"] = {
                HAND_LM_NAMES[i]: {
                    "x": round(lm.x, 3),
                    "y": round(lm.y, 3),
                    "z": round(lm.z, 3),
                }
                for i, lm in enumerate(hand_lms.landmark)
            }

    with _lock:
        globals()["_landmarks_data"] = {"type": "hands", "data": data}
    return frame


def process_pose(frame, detector):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    data    = {}

    if results.pose_landmarks:
        lms  = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        # ── 1. Segmentation mask ───────────────────────────────────────────
        if results.segmentation_mask is not None:
            seg = results.segmentation_mask          # float32, 0–1, shape (h, w)

            # Semi-transparent teal silhouette overlay on main video
            seg_f    = np.expand_dims(seg, axis=2)   # (h,w,1)
            tint     = np.full_like(frame, (0, 220, 130), dtype=np.float32)
            frame    = (
                frame.astype(np.float32) * (1 - 0.28 * seg_f)
                + tint * (0.28 * seg_f)
            ).clip(0, 255).astype(np.uint8)

            # Small B&W inset mask in the bottom-right corner
            inset_w  = max(160, w // 5)
            inset_h  = int(inset_w * h / w)
            seg_u8   = (seg * 255).astype(np.uint8)
            _, seg_bin = cv2.threshold(seg_u8, 128, 255, cv2.THRESH_BINARY)
            inset    = cv2.resize(seg_bin, (inset_w, inset_h))
            inset_bgr = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)

            margin   = 10
            y1       = h - inset_h - margin
            x1       = w - inset_w - margin
            # Darkened blend with existing pixels
            frame[y1:y1 + inset_h, x1:x1 + inset_w] = (
                inset_bgr.astype(np.float32) * 0.85
            ).clip(0, 255).astype(np.uint8)
            # Border + label
            cv2.rectangle(frame, (x1 - 2, y1 - 2),
                          (x1 + inset_w + 2, y1 + inset_h + 2), (0, 220, 130), 2)
            cv2.putText(frame, "Segmentation", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 130), 1, cv2.LINE_AA)

        # ── 2. Skeleton overlay ────────────────────────────────────────────
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose_mod.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style())

        # ── 3. Joint angles ────────────────────────────────────────────────
        angles = {
            name: angle_between(lms[a.value], lms[b.value], lms[c.value])
            for name, (a, b, c) in POSE_ANGLES_DEF.items()
        }

        for name, lm_enum in POSE_ANGLE_ON_FRAME.items():
            lm   = lms[lm_enum.value]
            cx   = int(lm.x * w)
            cy   = int(lm.y * h)
            text = f"{angles[name]}\u00b0"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(frame, (cx + 3, cy - th - 7), (cx + tw + 9, cy + 3), (0, 0, 0), -1)
            cv2.putText(frame, text, (cx + 6, cy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 180), 2, cv2.LINE_AA)

        lm_data = {
            POSE_LM_NAMES[i]: {
                "x":   round(lm.x, 3),
                "y":   round(lm.y, 3),
                "z":   round(lm.z, 3),
                "vis": round(lm.visibility, 2),
            }
            for i, lm in enumerate(lms)
        }
        data = {
            "angles":    {k: f"{v}\u00b0" for k, v in angles.items()},
            "landmarks": lm_data,
        }

    with _lock:
        globals()["_landmarks_data"] = {"type": "pose", "data": data}
    return frame

# ── Camera worker thread ───────────────────────────────────────────────────
def camera_loop():
    face_det = mp_face.FaceMesh(
        max_num_faces=2, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hand_det = mp_hands_mod.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose_det = mp_pose_mod.Pose(
        enable_segmentation=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)

            with _lock:
                mode = _mode

            if mode == 1:
                frame = process_face(frame, face_det)
            elif mode == 2:
                frame = process_hands(frame, hand_det)
            elif mode == 3:
                frame = process_pose(frame, pose_det)

            with _lock:
                globals()["_output_frame"] = frame.copy()
    finally:
        cap.release()
        face_det.close()
        hand_det.close()
        pose_det.close()

# ── MJPEG stream generator ─────────────────────────────────────────────────
def generate_frames():
    while True:
        frame = None
        with _lock:
            if _output_frame is not None:
                frame = _output_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )
        time.sleep(1 / 30)

# ── Flask routes ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/set_mode/<int:mode>", methods=["POST"])
def set_mode(mode):
    if mode not in (1, 2, 3):
        return jsonify({"ok": False}), 400
    with _lock:
        globals()["_mode"] = mode
    return jsonify({"ok": True, "mode": mode})


@app.route("/landmarks")
def landmarks():
    with _lock:
        return jsonify(_landmarks_data)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("\n  Server running →  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
