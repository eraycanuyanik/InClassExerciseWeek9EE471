import cv2
import mediapipe as mp

# --- Mediapipe setup ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def process_face(frame, detector):
    """Detect and draw face mesh landmarks on the frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
    return frame


def process_hands(frame, detector):
    """Detect and draw hand landmarks on the frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
            )
    return frame


def process_pose(frame, detector):
    """Detect and draw body pose landmarks on the frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    return frame


MODE_LABELS = {
    1: "Face Landmarks  [1]",
    2: "Hand Landmarks  [2]",
    3: "Pose Landmarks  [3]",
}


def draw_ui(frame, active_mode):
    """Overlay mode indicator and key hints onto the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for mode, label in MODE_LABELS.items():
        color = (0, 255, 100) if mode == active_mode else (160, 160, 160)
        x = 10 + (mode - 1) * (w // 3)
        cv2.putText(frame, label, (x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # Press Q hint
    cv2.putText(frame, "Q: quit", (w - 90, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check that it is connected and not in use.")

    active_mode = 1  # start with face detection

    with (
        mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_detector,
        mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hand_detector,
        mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose_detector,
    ):
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)  # mirror for natural feel

            if active_mode == 1:
                frame = process_face(frame, face_detector)
            elif active_mode == 2:
                frame = process_hands(frame, hand_detector)
            elif active_mode == 3:
                frame = process_pose(frame, pose_detector)

            frame = draw_ui(frame, active_mode)
            cv2.imshow("Mediapipe Landmark Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # Q or Esc to quit
                break
            elif key == ord("1"):
                active_mode = 1
            elif key == ord("2"):
                active_mode = 2
            elif key == ord("3"):
                active_mode = 3

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
