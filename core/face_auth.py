from deepface import DeepFace
import os

# ── Threshold ────────────────────────────────────────────────────────────────
# ArcFace cosine distance: 0.0 = identical, 1.0 = completely different.
# DeepFace's built-in default for ArcFace/cosine is 0.68 — far too permissive.
# 0.20 is strict enough to reject similar-looking strangers while still
# accepting the same person under different lighting / slight angle changes.
ARCFACE_COSINE_THRESHOLD = 0.20


def verify_identity(baseline_image_path: str, live_frame_path: str) -> bool:
    """
    Compares a live snapshot against the registered baseline photo.

    Uses ArcFace (more discriminative than VGG-Face) with cosine distance
    and a manually tightened threshold to prevent false-positive matches.

    Args:
        baseline_image_path : Path to the registered face image (saved at signup).
        live_frame_path     : Path to the live snapshot (taken at login).

    Returns:
        True  – faces belong to the same person AND distance < THRESHOLD.
        False – mismatch, detection failure, or missing files.
    """
    # ── Guard: files must exist ───────────────────────────────────────────────
    if not baseline_image_path or not os.path.exists(baseline_image_path):
        print("[FACE AUTH] ✗ Baseline image not found:", baseline_image_path)
        return False

    if not live_frame_path or not os.path.exists(live_frame_path):
        print("[FACE AUTH] ✗ Live frame not found:", live_frame_path)
        return False

    try:
        result = DeepFace.verify(
            img1_path        = baseline_image_path,
            img2_path        = live_frame_path,
            model_name       = "ArcFace",    # much more accurate than VGG-Face
            distance_metric  = "cosine",     # more reliable than euclidean
            enforce_detection= True,         # reject frames with no clear face
        )

        distance = result["distance"]
        # DeepFace's own verdict AND our stricter threshold must both pass
        passed = result["verified"] and distance < ARCFACE_COSINE_THRESHOLD

        print(
            f"[FACE AUTH] distance={distance:.4f}  "
            f"threshold={ARCFACE_COSINE_THRESHOLD}  "
            f"deepface_verified={result['verified']}  "
            f"→ {'✓ PASS' if passed else '✗ FAIL'}"
        )
        return passed

    except Exception as e:
        # DeepFace raises ValueError when no face is detectable in either image.
        print(f"[FACE AUTH ERROR] {e}")
        return False


def validate_registration_photo(image_path: str) -> bool:
    """
    Checks that at least one clear face is detectable in a registration photo.
    Call this right after saving the uploaded baseline image.

    Returns True if a face was found, False otherwise.
    """
    try:
        faces = DeepFace.extract_faces(
            img_path         = image_path,
            enforce_detection= True,
        )
        return len(faces) > 0
    except Exception as e:
        print(f"[FACE AUTH] Registration photo rejected — no face: {e}")
        return False