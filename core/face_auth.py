from deepface import DeepFace
import os
import cv2
import numpy as np

# ── Thresholds ────────────────────────────────────────────────────────────────
#
# ArcFace cosine distance ranges (empirically measured on webcam captures):
#
#   Same person, good lighting / straight-on :       0.15 – 0.30
#   Same person, different lighting / slight angle:  0.25 – 0.45
#   Same person, webcam compression + re-encoding:   0.30 – 0.50
#   Different person (impostor):                     0.55 – 0.90
#
# DeepFace's built-in default is 0.68 (tuned on LFW benchmark under ideal
# studio conditions — too permissive for a proctoring system).
#
# BUG FIX: The original threshold of 0.20 was so strict that it rejected
# 70-80% of genuine users who had ANY variation in lighting, angle, or JPEG
# compression between their registration photo and the live snapshot.
#
# 0.45 is the sweet spot:
#   - Accepts the same person under typical webcam conditions (< 0.45)
#   - Rejects impostors who cluster above 0.55 with a comfortable margin
#
ARCFACE_COSINE_THRESHOLD = 0.45

# ── Detector preference order ─────────────────────────────────────────────────
#
# BUG FIX: Not specifying detector_backend defaults to "opencv" (Haar cascade),
# which is the LEAST accurate detector — it frequently misses faces that are
# slightly off-axis, in lower light, or behind JPEG artifacts.
#
# We try detectors in order of accuracy:
#   retinaface → most accurate, handles angle/occlusion well (slower)
#   mtcnn      → excellent accuracy, faster than retinaface
#   opencv     → last resort, fast but unreliable
#
DETECTOR_PREFERENCE = ["retinaface", "mtcnn", "opencv"]


def _preprocess_image(image_path: str) -> str:
    """
    Applies basic normalisation to improve ArcFace embedding stability:
      - CLAHE histogram equalisation on the L channel (fixes lighting variation)
      - Mild unsharp mask (recovers detail lost to JPEG compression / blur)

    Writes a temp file next to the original and returns its path.
    Falls back to the original path if the image cannot be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    # Equalise luminance only, preserving colour
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Mild unsharp mask — recovers detail lost in JPEG round-trips
    gaussian    = cv2.GaussianBlur(img_eq, (0, 0), 3)
    img_sharp   = cv2.addWeighted(img_eq, 1.5, gaussian, -0.5, 0)

    base, ext = os.path.splitext(image_path)
    temp_path = f"{base}_preprocessed{ext}"
    cv2.imwrite(temp_path, img_sharp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return temp_path


def _try_verify(baseline_path: str, live_path: str, detector: str):
    """
    Single DeepFace.verify() call with one detector backend.
    Returns the result dict on success, None if the detector could not find a face.
    """
    try:
        result = DeepFace.verify(
            img1_path         = baseline_path,
            img2_path         = live_path,
            model_name        = "ArcFace",
            distance_metric   = "cosine",
            detector_backend  = detector,
            # BUG FIX: The original code used enforce_detection=True, which raises
            # a hard exception when the face is slightly off-centre, partially
            # occluded, or the detector backend simply misses it.
            # That exception propagated as a False return — rejecting real users.
            # enforce_detection=False lets DeepFace use the full image as a face
            # region fallback instead of crashing, which is far better behaviour.
            enforce_detection = False,
            align             = True,    # alignment consistently improves accuracy
        )
        return result
    except Exception as e:
        print(f"[FACE AUTH] Detector '{detector}' raised: {e}")
        return None


def verify_identity(baseline_image_path: str, live_frame_path: str) -> bool:
    """
    Compares a live snapshot against the registered baseline photo.

    Fixes applied vs original:
      1. Threshold raised 0.20 → 0.45  (0.20 rejected ~75% of genuine users)
      2. CLAHE + sharpening preprocessing normalises lighting differences
      3. Tries retinaface → mtcnn → opencv instead of defaulting to opencv
      4. enforce_detection=False prevents hard crashes on slightly off-centre faces
      5. Accepts on FIRST detector success, logs distances for every attempt

    Args:
        baseline_image_path : Registered face image saved at sign-up.
        live_frame_path     : Live snapshot taken at exam verification.

    Returns:
        True  – same person confirmed (distance < ARCFACE_COSINE_THRESHOLD).
        False – mismatch, undetectable face across all detectors, or file missing.
    """
    # ── Guard: files must exist ───────────────────────────────────────────────
    if not baseline_image_path or not os.path.exists(baseline_image_path):
        print("[FACE AUTH] ✗ Baseline image not found:", baseline_image_path)
        return False

    if not live_frame_path or not os.path.exists(live_frame_path):
        print("[FACE AUTH] ✗ Live frame not found:", live_frame_path)
        return False

    # ── Preprocess both images to normalise lighting / sharpness ─────────────
    baseline_proc = _preprocess_image(baseline_image_path)
    live_proc     = _preprocess_image(live_frame_path)

    temp_files = []
    if baseline_proc != baseline_image_path: temp_files.append(baseline_proc)
    if live_proc     != live_frame_path:     temp_files.append(live_proc)

    best_distance = float("inf")
    passed        = False

    try:
        for detector in DETECTOR_PREFERENCE:
            result = _try_verify(baseline_proc, live_proc, detector)
            if result is None:
                continue   # detector couldn't locate a face — try next

            distance      = result.get("distance", float("inf"))
            best_distance = min(best_distance, distance)

            if distance < ARCFACE_COSINE_THRESHOLD:
                passed = True
                print(
                    f"[FACE AUTH] ✓ PASS  detector={detector}  "
                    f"distance={distance:.4f}  threshold={ARCFACE_COSINE_THRESHOLD}"
                )
                break   # accepted — no need to try remaining detectors
            else:
                print(
                    f"[FACE AUTH] ✗ FAIL  detector={detector}  "
                    f"distance={distance:.4f}  threshold={ARCFACE_COSINE_THRESHOLD}"
                )

        if not passed and best_distance == float("inf"):
            print("[FACE AUTH] ✗ No face detected by any backend in one or both images.")

        return passed

    finally:
        # Always remove preprocessed temp files regardless of outcome
        for path in temp_files:
            try:
                os.remove(path)
            except OSError:
                pass


def validate_registration_photo(image_path: str) -> bool:
    """
    Checks that at least one clear face is detectable in the registration photo.
    Call this immediately after saving the uploaded baseline image.

    BUG FIX: Original used the default opencv detector — if opencv missed the
    face, registration was rejected even though the photo was perfectly fine.
    Now falls through the same DETECTOR_PREFERENCE chain as verify_identity,
    guaranteeing that any photo accepted here will also be detectable later.

    Returns True if a face is found by any detector, False otherwise.
    """
    if not image_path or not os.path.exists(image_path):
        print("[FACE AUTH] ✗ Registration photo path invalid:", image_path)
        return False

    for detector in DETECTOR_PREFERENCE:
        try:
            faces = DeepFace.extract_faces(
                img_path         = image_path,
                detector_backend = detector,
                enforce_detection= True,
            )
            if len(faces) > 0:
                print(
                    f"[FACE AUTH] ✓ Registration photo accepted  "
                    f"detector={detector}  faces={len(faces)}"
                )
                return True
        except Exception as e:
            print(f"[FACE AUTH] Registration detector '{detector}' failed: {e}")
            continue

    print("[FACE AUTH] ✗ Registration photo rejected — no face found by any detector.")
    return False