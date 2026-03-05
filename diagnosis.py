"""
EduShield Vision Diagnostic
============================
Run this directly on the backend machine:
    python diagnose_vision.py

It opens your webcam, runs YOLO + MediaPipe on every frame in real time,
and prints exactly what is and isn't being detected.
Press Q to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

print("Loading models...")
try:
    yolo = YOLO("yolov8s.pt")
    print("  ✓ YOLOv8s loaded")
except Exception as e:
    print(f"  ✗ YOLO failed: {e}")
    print("  → Try: pip install ultralytics  or check yolov8s.pt exists")
    exit(1)

try:
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.4
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=3, refine_landmarks=True,
        min_detection_confidence=0.4, min_tracking_confidence=0.4
    )
    print("  ✓ MediaPipe loaded")
except Exception as e:
    print(f"  ✗ MediaPipe failed: {e}")
    exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ Cannot open webcam (index 0). Try index 1 or 2.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"  ✓ Webcam opened — resolution: {int(actual_w)}×{int(actual_h)}")
print("\nWindow opened. Press Q to quit.\n")

CELLPHONE_ID = 67
PERSON_ID    = 0
BOOK_ID      = 73

frame_n = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("✗ Failed to grab frame")
        break

    frame_n += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── YOLO ──────────────────────────────────────────────────────────────────
    results = yolo.predict(frame, verbose=False, imgsz=640)
    phone_confs = []
    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo.names[cls]
            colour = (200, 200, 200)

            if cls == CELLPHONE_ID:
                phone_confs.append(conf)
                colour = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
                cv2.putText(frame, f"PHONE {conf:.0%}", (x1, max(y1-8,10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
                print(f"[Frame {frame_n}] 📱 PHONE detected — conf={conf:.2f}  box=({x1},{y1},{x2},{y2})")

            elif cls == PERSON_ID and conf >= 0.40:
                colour = (0, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            elif cls == BOOK_ID and conf >= 0.35:
                colour = (0, 140, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"BOOK {conf:.0%}", (x1, max(y1-8,10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
                print(f"[Frame {frame_n}] 📖 BOOK detected — conf={conf:.2f}")

    # ── Face detection ────────────────────────────────────────────────────────
    fd_r = face_detector.process(rgb)
    face_count = len(fd_r.detections) if fd_r.detections else 0

    if fd_r.detections:
        h_f, w_f = frame.shape[:2]
        for det in fd_r.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(int(bb.xmin * w_f), 0)
            y1 = max(int(bb.ymin * h_f), 0)
            x2 = min(int((bb.xmin + bb.width)  * w_f), w_f)
            y2 = min(int((bb.ymin + bb.height) * h_f), h_f)
            sc = det.score[0] if det.score else 0
            col = (0,255,255) if face_count > 1 else (0,200,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, f"Face {sc:.0%}", (x1, max(y1-8,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    if face_count == 0 and frame_n % 10 == 0:
        print(f"[Frame {frame_n}] ⚠ No face detected")
    elif face_count > 1:
        print(f"[Frame {frame_n}] ⚠ MULTIPLE FACES: {face_count}")

    # ── Gaze ──────────────────────────────────────────────────────────────────
    mesh_r = face_mesh.process(rgb)
    if mesh_r.multi_face_landmarks:
        lm = mesh_r.multi_face_landmarks[0].landmark

        nose_x    = lm[1].x
        l_cheek_x = lm[234].x
        r_cheek_x = lm[454].x
        h_ratio   = (nose_x - l_cheek_x) / (abs(r_cheek_x - l_cheek_x) + 1e-6)

        nose_y  = lm[1].y
        chin_y  = lm[152].y
        fore_y  = lm[10].y
        v_ratio = (nose_y - fore_y) / (abs(chin_y - fore_y) + 1e-6)

        try:
            iris_x     = lm[468].x
            eye_l_x    = lm[33].x
            eye_r_x    = lm[133].x
            iris_ratio = (iris_x - eye_l_x) / (abs(eye_r_x - eye_l_x) + 1e-6)
        except Exception:
            iris_ratio = 0.5

        h_away    = h_ratio < 0.25 or h_ratio > 0.75
        v_away    = v_ratio < 0.30 or v_ratio > 0.68
        iris_away = iris_ratio < 0.20 or iris_ratio > 0.80

        # Print raw values every 30 frames so you can see what's normal for YOU
        if frame_n % 30 == 0:
            print(f"[Frame {frame_n}] 👁 Gaze ratios — "
                  f"H={h_ratio:.3f}({'AWAY' if h_away else 'ok'})  "
                  f"V={v_ratio:.3f}({'AWAY' if v_away else 'ok'})  "
                  f"iris={iris_ratio:.3f}({'AWAY' if iris_away else 'ok'})")

        if (h_away or v_away) and iris_away:
            cv2.putText(frame, "GAZE AWAY", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # HUD overlay
        cv2.putText(frame, f"H={h_ratio:.2f}  V={v_ratio:.2f}  iris={iris_ratio:.2f}",
                    (10, frame.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"Faces: {face_count}  |  Frame: {frame_n}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)

    cv2.imshow("EduShield Vision Diagnostic — press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDiagnostic complete.")
print("Share the printed output (H/V/iris ratios + any detections) so thresholds can be tuned.")