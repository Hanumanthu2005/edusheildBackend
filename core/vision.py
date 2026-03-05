import cv2
import base64
import numpy as np
import mediapipe as mp
from collections import deque
from ultralytics import YOLO

yolo_model = YOLO('yolov8s.pt')

mp_face_mesh   = mp.solutions.face_mesh
mp_face_detect = mp.solutions.face_detection

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

face_detector = mp_face_detect.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5,
)

# ── YOLO class IDs ────────────────────────────────────────────────────────────
PERSON_CLASS_ID    = 0
CELLPHONE_CLASS_ID = 67
BOOK_CLASS_ID      = 73

PHONE_CONF_THRESHOLD  = 0.40
BOOK_CONF_THRESHOLD   = 0.50
PERSON_CONF_THRESHOLD = 0.45

# ── MediaPipe landmark indices ────────────────────────────────────────────────
NOSE_TIP    = 1
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454
CHIN        = 152
FOREHEAD    = 10
LEFT_IRIS   = 468
LEFT_EYE_L  = 33
LEFT_EYE_R  = 133
RIGHT_IRIS  = 473
RIGHT_EYE_L = 362
RIGHT_EYE_R = 263

H_YAW_LOW    = 0.28
H_YAW_HIGH   = 0.72
V_PITCH_LOW  = 0.28
V_PITCH_HIGH = 0.70
IRIS_LOW     = 0.22
IRIS_HIGH    = 0.78

# ── Temporal smoothing ────────────────────────────────────────────────────────
# Phone: NO smoothing — immediate detection on first frame
# Multiple persons: majority vote 2/3 frames
GAZE_WINDOW    = 4
GAZE_MIN_HITS  = 3

FACE_WINDOW    = 3
FACE_MIN_HITS  = 2

PERSON_WINDOW    = 3
PERSON_MIN_HITS  = 2

_gaze_hits         = deque(maxlen=GAZE_WINDOW)
_no_face_hits      = deque(maxlen=FACE_WINDOW)
_multi_face_hits   = deque(maxlen=FACE_WINDOW)
_multi_person_hits = deque(maxlen=PERSON_WINDOW)

_frame_count  = 0
WARMUP_FRAMES = 15

# ── Drawing colours ───────────────────────────────────────────────────────────
COLOR_RED      = (0,   0,   255)
COLOR_GREEN    = (0,   200, 0  )
COLOR_ORANGE   = (0,   140, 255)
COLOR_YELLOW   = (0,   220, 220)
COLOR_CRITICAL = (0,   0,   255)


def _majority(dq, min_hits: int) -> bool:
    return len(dq) >= min_hits and sum(dq) >= min_hits


def _draw_label(frame, text, pos, colour=COLOR_RED):
    # Shadow for readability
    cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)


def _draw_alert_banner(frame, text, colour=COLOR_RED):
    """Full-width banner at top of frame for critical alerts."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), colour, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, text, (w // 2 - len(text) * 9, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def _encode_snapshot(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode("utf-8")


def encode_annotated_frame(frame) -> str:
    """Encode the annotated frame at display quality for streaming."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("utf-8")


def _gaze_check(lm):
    reasons = []

    nose_x     = lm[NOSE_TIP].x
    l_cheek_x  = lm[LEFT_CHEEK].x
    r_cheek_x  = lm[RIGHT_CHEEK].x
    face_width  = abs(r_cheek_x - l_cheek_x) + 1e-6
    h_ratio_raw = (nose_x - l_cheek_x) / face_width
    h_ratio     = max(0.0, min(1.0, h_ratio_raw))
    h_away      = h_ratio < H_YAW_LOW or h_ratio > H_YAW_HIGH

    nose_y  = lm[NOSE_TIP].y
    chin_y  = lm[CHIN].y
    fore_y  = lm[FOREHEAD].y
    face_h  = abs(chin_y - fore_y) + 1e-6
    v_ratio = (nose_y - fore_y) / face_h
    v_away  = v_ratio < V_PITCH_LOW or v_ratio > V_PITCH_HIGH

    iris_away = False
    try:
        iris_x     = lm[LEFT_IRIS].x
        eye_l_x    = lm[LEFT_EYE_L].x
        eye_r_x    = lm[LEFT_EYE_R].x
        eye_width  = abs(eye_r_x - eye_l_x) + 1e-6
        iris_ratio = (iris_x - eye_l_x) / eye_width
        iris_away  = iris_ratio < IRIS_LOW or iris_ratio > IRIS_HIGH
    except Exception:
        pass

    looking_away = h_away or v_away or iris_away

    if h_away:    reasons.append(f"head yaw={h_ratio:.2f}")
    if v_away:    reasons.append(f"head pitch={v_ratio:.2f}")
    if iris_away: reasons.append("iris off-centre")

    return looking_away, reasons


def analyze_frame(frame):
    """
    Processes a single BGR frame.
    Returns (annotated_frame, violations: list[dict])

    Changes vs previous version:
    - Phone detection is IMMEDIATE (no majority vote) — first detection triggers critical violation
    - Multiple PERSONS detected via YOLO person class (in addition to MediaPipe faces)
    - Annotated frame is returned with all bounding boxes drawn for streaming
    - Critical alert banners drawn on frame for phone and multiple persons
    """
    global _frame_count
    _frame_count += 1

    violations = []
    seen_types: set = set()

    def add_violation(v_type, severity, detail):
        if v_type not in seen_types:
            seen_types.add(v_type)
            violations.append({
                "type":     v_type,
                "severity": severity,
                "snapshot": _encode_snapshot(frame),
                "detail":   detail,
            })

    # ── 1. YOLO — phone, book, person count ──────────────────────────────────
    results = yolo_model.predict(frame, verbose=False, imgsz=640)

    phone_detected_this_frame = False
    yolo_person_count         = 0

    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == PERSON_CLASS_ID and conf >= PERSON_CONF_THRESHOLD:
                yolo_person_count += 1
                color = COLOR_RED if yolo_person_count > 1 else COLOR_GREEN
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'⚠ Extra Person {conf:.0%}' if yolo_person_count > 1 else f'Student {conf:.0%}'
                _draw_label(frame, label, (x1, max(y1-10, 10)), color)

            elif cls == CELLPHONE_CLASS_ID and conf >= PHONE_CONF_THRESHOLD:
                phone_detected_this_frame = True
                # Bold red box with thick border for phone
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), COLOR_CRITICAL, 4)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 180), 2)
                _draw_label(frame, f'🚫 PHONE {conf:.0%}', (x1, max(y1-12, 12)), COLOR_CRITICAL)

            elif cls == BOOK_CLASS_ID and conf >= BOOK_CONF_THRESHOLD:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ORANGE, 2)
                _draw_label(frame, f'Book {conf:.0%}', (x1, max(y1-10, 10)), COLOR_ORANGE)
                add_violation(
                    "book_detected", "major",
                    f"Unauthorized material detected ({conf:.0%} confidence)"
                )

    # ── PHONE: IMMEDIATE termination — no majority vote ───────────────────────
    # Any single frame with a confident phone detection fires the violation.
    if phone_detected_this_frame:
        _draw_alert_banner(frame, "⚠  PHONE DETECTED — EXAM WILL BE TERMINATED", COLOR_CRITICAL)
        add_violation(
            "phone_detected", "critical",
            f"Mobile phone detected — immediate termination triggered"
        )

    # ── YOLO multiple-persons check (majority vote 2/3) ───────────────────────
    _multi_person_hits.append(yolo_person_count > 1)
    if _majority(_multi_person_hits, PERSON_MIN_HITS):
        _draw_alert_banner(frame, f"⚠  MULTIPLE PERSONS DETECTED ({yolo_person_count})", COLOR_CRITICAL)
        add_violation(
            "multiple_faces", "critical",
            f"{yolo_person_count} persons detected via YOLO — only the student should be present"
        )

    # ── 2. MediaPipe face count ───────────────────────────────────────────────
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fd_r = face_detector.process(rgb)
    face_count = len(fd_r.detections) if fd_r.detections else 0

    if fd_r.detections:
        h_f, w_f = frame.shape[:2]
        for det in fd_r.detections:
            bb  = det.location_data.relative_bounding_box
            x1  = max(int(bb.xmin * w_f), 0)
            y1  = max(int(bb.ymin * h_f), 0)
            x2  = min(int((bb.xmin + bb.width)  * w_f), w_f)
            y2  = min(int((bb.ymin + bb.height) * h_f), h_f)
            sc  = det.score[0] if det.score else 0
            col = COLOR_RED if face_count > 1 else COLOR_GREEN
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            _draw_label(frame, f'Face {sc:.0%}', (x1, max(y1-8, 10)), col)

    _no_face_hits.append(face_count == 0)
    _multi_face_hits.append(face_count > 1)

    if _majority(_no_face_hits, FACE_MIN_HITS):
        add_violation("no_face", "major", "No face visible in frame")
        _draw_label(frame, "WARNING: No Face Detected", (30, 70), COLOR_RED)
    elif _majority(_multi_face_hits, FACE_MIN_HITS) and "multiple_faces" not in seen_types:
        # MediaPipe also catches multiple faces if YOLO missed it
        add_violation(
            "multiple_faces", "critical",
            f"{face_count} faces detected — only the student should be present"
        )
        _draw_label(frame, f"WARNING: {face_count} Faces Detected", (30, 70), COLOR_RED)

    # ── 3. Gaze ───────────────────────────────────────────────────────────────
    mesh_r = face_mesh.process(rgb)

    gaze_away_this_frame = False
    gaze_reasons         = []

    if _frame_count > WARMUP_FRAMES and mesh_r.multi_face_landmarks:
        lm = mesh_r.multi_face_landmarks[0].landmark
        gaze_away_this_frame, gaze_reasons = _gaze_check(lm)

        mp.solutions.drawing_utils.draw_landmarks(
            image                   = frame,
            landmark_list           = mesh_r.multi_face_landmarks[0],
            connections             = mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec   = None,
            connection_drawing_spec = mp.solutions.drawing_styles
                                        .get_default_face_mesh_tesselation_style(),
        )

    _gaze_hits.append(gaze_away_this_frame)
    if _majority(_gaze_hits, GAZE_MIN_HITS):
        add_violation(
            "looking_away", "major",
            "Student not looking at screen: " + (", ".join(gaze_reasons) or "gaze off-screen")
        )
        _draw_label(frame, "WARNING: Look at Screen", (30, 110), COLOR_YELLOW)

    # ── 4. Status overlay (bottom-left HUD) ───────────────────────────────────
    h, w = frame.shape[:2]
    hud_y = h - 10
    status_text = f"Faces: {face_count}  Persons: {yolo_person_count}  Frame: {_frame_count}"
    cv2.putText(frame, status_text, (10, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame, violations