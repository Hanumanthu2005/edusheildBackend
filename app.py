"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         EduShield — Flask Backend                          ║
║                                                                              ║
║  This file is the main application entry point. It uses the Flask app       ║
║  factory pattern (create_app) and registers all API routes.                  ║
║                                                                              ║
║  ROUTE GROUPS:                                                               ║
║    🔐  Auth         — /api/register, /api/login, /api/verify_face           ║
║    📹  Video        — /api/video_feed  (MJPEG stream)                        ║
║    📚  Student      — /api/student/*                                         ║
║    📌  Violations   — /api/log_violation, /api/analyze_frame, /api/violations║
║    📊  Admin Exams  — /api/admin/exams/*                                     ║
║    📊  Admin Sess.  — /api/admin/sessions/*                                  ║
║    📊  Admin Results— /api/admin/results/*                                   ║
║    📊  Admin Users  — /api/admin/students/*                                  ║
║    🧠  RL Feedback  — /api/admin/feedback, /api/admin/rl/*                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import csv
import io
import time
import base64
from collections import Counter
from datetime import timedelta, datetime
from functools import wraps

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
    decode_token,
)
from sqlalchemy import func
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ── Internal modules ──────────────────────────────────────────────────────────
from config import Config
from models import (
    UserAnswer, db, User, ExamSession, ViolationLog,
    Exam, Question, Option, UserExam, FeedbackLog,
)
from core.vision import analyze_frame, encode_annotated_frame
from core.audio import AudioMonitor
from core.face_auth import verify_identity, validate_registration_photo
from core.rl_agent import rl_optimizer


# ==============================================================================
# 🗂️  MODULE-LEVEL STATE
# ==============================================================================

# Per-session cooldown tracker so the same violation type isn't logged to the
# database more than once every 10 seconds.
# Key format:  "session_id:violation_type"  →  last log timestamp (float)
last_db_log_time: dict[str, float] = {}


# ==============================================================================
# 🚨  TERMINATION THRESHOLDS
# ==============================================================================
# How many times a violation must be logged before the exam session is
# automatically terminated.  Lower number = stricter / faster termination.

TERMINATION_THRESHOLDS = {
    "phone_detected":  1,   # IMMEDIATE — first detection terminates
    "book_detected":   3,
    "multiple_faces":  2,
    "looking_away":    5,
    "no_face":         5,
    "tab_switch":      3,
    "fullscreen_exit": 3,
    "loud_noise":      5,
}

# Violations listed here bypass the 10-second cooldown and trigger instant
# session termination on the very first detection.
INSTANT_TERMINATE_VIOLATIONS = {"phone_detected"}


# ==============================================================================
# 🛠️  MODULE-LEVEL HELPERS
# ==============================================================================

def _check_and_terminate(session) -> tuple[bool, str | None]:
    """
    Query cumulative violation counts for a session and terminate it if any
    violation type has hit or exceeded its threshold.

    Args:
        session: An ExamSession ORM object (must be within an app context).

    Returns:
        (True, violation_type)  if session was terminated.
        (False, None)           if no threshold was breached.
    """
    counts = (
        db.session.query(
            ViolationLog.violation_type,
            func.count(ViolationLog.id).label("cnt"),
        )
        .filter(ViolationLog.session_id == session.id)
        .group_by(ViolationLog.violation_type)
        .all()
    )

    for v_type, cnt in counts:
        threshold = TERMINATION_THRESHOLDS.get(v_type)
        if threshold and cnt >= threshold:
            session.status   = "terminated"
            session.end_time = datetime.utcnow()
            db.session.commit()
            return True, v_type

    return False, None


# ==============================================================================
# 🏭  APP FACTORY
# ==============================================================================

def create_app() -> Flask:
    """
    Flask application factory.

    Initialises extensions (SQLAlchemy, JWT, CORS, Flask-Migrate), creates DB
    tables, starts the audio monitor, registers all routes, and returns the
    configured Flask app.
    """
    app = Flask(__name__)
    app.config.from_object(Config)

    # JWT configuration — replace secret in production via env var
    app.config["JWT_SECRET_KEY"]           = "super-secret-key"
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=5)

    # ── Extension setup ───────────────────────────────────────────────────────
    db.init_app(app)

    from flask_migrate import Migrate
    migrate = Migrate(app, db)   # noqa: F841  (used by Flask-Migrate CLI)

    # Allow requests from the React dev server and the deployed frontend
    CORS(
        app,
        resources={r"/api/*": {"origins": [
            "http://localhost:5173",
            "https://edusheild-frontend.vercel.app",
        ]}},
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        supports_credentials=True,
    )
    JWTManager(app)

    # ── Database & upload folder initialisation ───────────────────────────────
    with app.app_context():
        db.create_all()
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # ── Background audio monitor ──────────────────────────────────────────────
    # Listens for loud noises (amplitude > 800) in a background thread.
    audio_monitor = AudioMonitor(threshold=800)
    audio_monitor.start()


    # ==========================================================================
    # 🔧  INNER HELPERS  (share app / db context with route functions)
    # ==========================================================================

    def _decode_and_save_image(data_uri: str, dest_path: str) -> bool:
        """
        Decode a base64 data-URI (or raw base64 string) and write the bytes
        to *dest_path*, creating any missing parent directories.

        Returns True on success, False if decoding or writing fails.
        """
        try:
            # Strip the "data:image/jpeg;base64," prefix if present
            if "," in data_uri:
                _, encoded = data_uri.split(",", 1)
            else:
                encoded = data_uri

            img_bytes = base64.b64decode(encoded)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            with open(dest_path, "wb") as fh:
                fh.write(img_bytes)
            return True

        except Exception as e:
            print(f"[IMAGE SAVE ERROR] {e}")
            return False


    def _log_violation_to_db(
        session,
        v_type: str,
        severity: str,
        detail: str,
        snapshot: str | None = None,
    ) -> bool:
        """
        Persist a single ViolationLog record, subject to a 10-second per-key
        rate-limit (bypassed for INSTANT_TERMINATE_VIOLATIONS).

        Side effects:
            • Increments session.total_warnings.
            • Updates the in-memory last_db_log_time cache.
            • Does NOT call db.session.commit() — caller is responsible.

        Returns True if the violation was actually logged, False if rate-limited.
        """
        now = time.time()
        key = f"{session.id}:{v_type}"

        # Rate-limit: skip if the same violation was logged within the last 10 s
        # (except for instant-terminate violations which must always be logged)
        if v_type not in INSTANT_TERMINATE_VIOLATIONS:
            if now - last_db_log_time.get(key, 0) < 10:
                return False

        db.session.add(ViolationLog(
            session_id     = session.id,
            violation_type = v_type,
            severity       = severity,
            detail         = detail,
            snapshot_b64   = snapshot,
        ))
        session.total_warnings = (session.total_warnings or 0) + 1
        last_db_log_time[key]  = now
        return True


    def _grade(score: float) -> str:
        """Map a numeric percentage score to a letter grade."""
        if score >= 90: return "A+"
        if score >= 80: return "A"
        if score >= 70: return "B"
        if score >= 60: return "C"
        if score >= 50: return "D"
        return "F"


    # ==========================================================================
    # 🔐  ROLE-BASED ACCESS DECORATOR
    # ==========================================================================

    def admin_required(fn):
        """
        Route decorator that restricts access to users whose role == 'admin'.
        Must be placed *after* @app.route but *before* any other decorators.

        Returns 403 if the authenticated user is not an admin.
        """
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            user = User.query.get(int(get_jwt_identity()))
            if not user or user.role != "admin":
                return jsonify({"error": "Admin access required"}), 403
            return fn(*args, **kwargs)
        return wrapper


    # ==========================================================================
    # 🔐  AUTH ROUTES
    # ==========================================================================

    @app.route("/api/register", methods=["POST"])
    def register():
        """
        Register a new student account.

        Expects JSON:
            {
                "username":                   "alice",
                "password":                   "s3cr3t",
                "register_snapshot_base64":   "<data-uri>"
            }

        Steps:
            1. Validate that username / password / snapshot are present.
            2. Check username is not already taken.
            3. Decode & save the face snapshot to UPLOAD_FOLDER.
            4. Run face-detection validation on the saved image.
            5. Hash the password and persist the new User record.

        Returns 201 on success; 400/409 on validation failures.
        """
        data     = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "")
        snapshot = data.get("register_snapshot_base64", "")

        # ── Input validation ──────────────────────────────────────────────────
        if not username:
            return jsonify({"error": "Username is required"}), 400
        if not password:
            return jsonify({"error": "Password is required"}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username already exists"}), 400
        if not snapshot or len(snapshot) <= 100:
            return jsonify({"error": "Face snapshot required"}), 400

        # ── Save baseline face photo ──────────────────────────────────────────
        filename = secure_filename(f"{username}_baseline.jpg")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        if not _decode_and_save_image(snapshot, filepath):
            return jsonify({"error": "Failed to process image"}), 400

        # ── Face-detection check: reject if no face is visible ────────────────
        if not validate_registration_photo(filepath):
            os.remove(filepath)
            return jsonify({
                "error": (
                    "No face detected in your photo. "
                    "Please ensure good lighting and look directly at the camera."
                )
            }), 400

        # ── Persist new user ──────────────────────────────────────────────────
        db.session.add(User(
            username             = username,
            password_hash        = generate_password_hash(password),
            registered_face_path = filepath,
            role                 = "student",
        ))
        db.session.commit()
        return jsonify({"message": "User registered successfully"}), 201


    @app.route("/api/login", methods=["POST"])
    def login():
        """
        Authenticate a user and return a JWT access token.

        Expects JSON:
            {
                "username":             "alice",
                "password":             "s3cr3t",
                "live_snapshot_base64": "<data-uri>"   // students only
            }

        Flow:
            • Admins: password-only check, no face verification required.
            • Students: password check + ArcFace identity verification against
              the stored baseline photo.
            • A new ExamSession is created (or the existing in-progress one is
              reused) for student logins.

        Returns 200 with { token, role, session_id } on success.
        Returns 401 on bad credentials or failed face verification.
        """
        data     = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "")
        snapshot = data.get("live_snapshot_base64", "")

        # ── Credential check ──────────────────────────────────────────────────
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid credentials"}), 401

        # ── Admin shortcut: skip face verification ────────────────────────────
        if user.role == "admin":
            token = create_access_token(
                identity=str(user.id),
                additional_claims={"role": user.role},
            )
            return jsonify({
                "message":    "Admin login successful",
                "token":      token,
                "role":       user.role,
                "session_id": None,
            }), 200

        # ── Student: live face verification ───────────────────────────────────
        if not snapshot or len(snapshot) <= 100:
            return jsonify({"error": "Live face scan required"}), 400

        temp_path = os.path.join(
            app.config["UPLOAD_FOLDER"], f"temp_login_{username}.jpg"
        )
        try:
            if not _decode_and_save_image(snapshot, temp_path):
                return jsonify({"error": "Failed to process live snapshot"}), 400

            if not verify_identity(user.registered_face_path, temp_path):
                return jsonify({
                    "error": (
                        "Face verification failed. "
                        "Please ensure good lighting and look directly at the camera."
                    )
                }), 401
        finally:
            # Always clean up the temporary login image
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # ── Create or reuse an in-progress exam session ───────────────────────
        session = (
            ExamSession.query
            .filter_by(user_id=user.id, status="in_progress")
            .order_by(ExamSession.id.desc())
            .first()
        )
        if not session:
            session = ExamSession(user_id=user.id)
            db.session.add(session)
            db.session.commit()

        token = create_access_token(
            identity=str(user.id),
            additional_claims={"role": user.role},
        )
        return jsonify({
            "message":    "Login successful",
            "token":      token,
            "role":       user.role,
            "session_id": session.id,
        }), 200


    @app.route("/api/verify_face", methods=["POST"])
    @jwt_required()
    def verify_face():
        """
        Pre-exam face verification endpoint.

        Compares a live snapshot against the user's stored baseline photo using
        ArcFace and returns a pass/fail result before the exam begins.

        Expects JSON:
            { "live_snapshot_base64": "<data-uri or raw base64>" }

        Returns JSON (200):
            { "verified": true/false, "message": "..." }
        """
        user_id  = int(get_jwt_identity())
        data     = request.json or {}
        snapshot = data.get("live_snapshot_base64", "")

        # ── Guard clauses ─────────────────────────────────────────────────────
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        if not snapshot or len(snapshot) <= 100:
            return jsonify({"verified": False, "message": "No snapshot provided"}), 400
        if not user.registered_face_path or not os.path.exists(user.registered_face_path):
            return jsonify({"verified": False, "message": "No registered face on file"}), 400

        # Unique temp path prevents race conditions on concurrent requests
        temp_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"temp_examverify_{user_id}_{int(time.time())}.jpg",
        )
        try:
            if not _decode_and_save_image(snapshot, temp_path):
                return jsonify({"verified": False, "message": "Failed to process snapshot"}), 400

            verified = verify_identity(user.registered_face_path, temp_path)
            return jsonify({
                "verified": verified,
                "message":  (
                    "Identity confirmed" if verified else
                    "Face verification failed. Please ensure good lighting and look directly at the camera."
                ),
            }), 200

        except Exception as e:
            print(f"[VERIFY FACE ERROR] {e}")
            return jsonify({"verified": False, "message": "Verification error"}), 500

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


    # ==========================================================================
    # 📹  MJPEG VIDEO STREAM  — annotated backend webcam feed
    # ==========================================================================

    @app.route("/api/video_feed", methods=["GET"])
    def video_feed():
        """
        Stream an annotated MJPEG video feed from the server-side webcam.

        Authentication: JWT passed as a query parameter (?token=<jwt>) because
        browsers cannot set Authorization headers on <img src="..."> elements.

        The generator:
            1. Opens the default webcam (index 0) at 1280×720 @ 15 fps.
            2. On each frame, calls analyze_frame() which returns an annotated
               image and a list of detected violations.
            3. Violations are logged to the DB via _log_violation_to_db().
            4. Phone detection triggers immediate session termination.
            5. Yields each frame as a multipart/x-mixed-replace JPEG chunk.
            6. Stops when the session is terminated or the camera is lost.
        """
        # ── Token auth (query param) ──────────────────────────────────────────
        token = request.args.get("token", "")
        try:
            decoded = decode_token(token)
            user_id = int(decoded["sub"])
        except Exception:
            return jsonify({"error": "Invalid or missing token"}), 401

        def generate():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 15)

            # ── Camera unavailable: yield a single error frame then exit ──────
            if not cap.isOpened():
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    error_frame, "Camera not available", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                )
                _, buf = cv2.imencode(".jpg", error_frame)
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                return

            try:
                while True:
                    # ── Poll session status — stop streaming if terminated ─────
                    with app.app_context():
                        s = (
                            ExamSession.query
                            .filter_by(user_id=user_id)
                            .order_by(ExamSession.id.desc())
                            .first()
                        )
                        if s and s.status == "terminated":
                            break

                    # ── Capture & flip frame (mirror effect) ──────────────────
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)

                    # ── AI analysis: face pose, YOLO object detection, etc. ────
                    annotated, violations = analyze_frame(frame)

                    # ── Log any detected violations ───────────────────────────
                    if violations:
                        with app.app_context():
                            s = (
                                ExamSession.query
                                .filter_by(user_id=user_id, status="in_progress")
                                .order_by(ExamSession.id.desc())
                                .first()
                            )
                            if s:
                                needs_commit = False
                                for v in violations:
                                    logged = _log_violation_to_db(
                                        s,
                                        v["type"],
                                        v.get("severity", "major"),
                                        v.get("detail", ""),
                                        v.get("snapshot"),
                                    )
                                    if logged:
                                        needs_commit = True
                                        # Instant termination for high-risk violations
                                        if v["type"] in INSTANT_TERMINATE_VIOLATIONS:
                                            s.status   = "terminated"
                                            s.end_time = datetime.utcnow()
                                if needs_commit:
                                    db.session.commit()

                    # ── Encode and yield the annotated JPEG frame ─────────────
                    _, buf = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
                    time.sleep(1 / 15)   # Throttle to ~15 fps

            finally:
                cap.release()

        return Response(
            stream_with_context(generate()),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )


    # ==========================================================================
    # 📚  STUDENT ROUTES
    # ==========================================================================

    @app.route("/api/student/exams", methods=["GET"])
    @jwt_required()
    def get_student_exams():
        """
        Return a list of all available exams (id, title, description,
        duration, total_marks, created_at).  Any authenticated user can call
        this endpoint.
        """
        return jsonify([
            {
                "id":          e.id,
                "title":       e.title,
                "description": e.description,
                "duration":    e.duration_minutes,
                "total_marks": e.total_marks,
                "created_at":  e.created_at.isoformat() if e.created_at else None,
            }
            for e in Exam.query.all()
        ])


    @app.route("/api/student/exams/<int:exam_id>", methods=["GET"])
    @jwt_required()
    def get_student_exam_details(exam_id):
        """
        Return full details for a single exam including all questions and
        answer options (without revealing which option is correct).
        """
        exam = Exam.query.get_or_404(exam_id)
        return jsonify({
            "id":          exam.id,
            "title":       exam.title,
            "description": exam.description,
            "duration":    exam.duration_minutes,
            "total_marks": exam.total_marks,
            "questions": [
                {
                    "id":      q.id,
                    "text":    q.question_text,
                    "marks":   q.marks,
                    "type":    q.question_type,
                    "options": [
                        {"id": o.id, "text": o.option_text}
                        for o in q.options
                    ],
                }
                for q in exam.questions
            ],
        })


    @app.route("/api/student/exams/<int:exam_id>/start", methods=["POST"])
    @jwt_required()
    def start_exam(exam_id):
        """
        Create a UserExam record to mark the exam as 'in_progress' for this
        student, or return the existing one if the student already started.

        Returns 201 (new attempt) or 200 (resumed attempt).
        """
        user_id = int(get_jwt_identity())
        Exam.query.get_or_404(exam_id)   # 404 if exam doesn't exist

        # Reuse an in-progress attempt rather than creating duplicates
        existing = UserExam.query.filter_by(
            user_id=user_id, exam_id=exam_id, status="in_progress"
        ).first()
        if existing:
            return jsonify({
                "user_exam_id": existing.id,
                "message":      "Exam already in progress",
            }), 200

        ue = UserExam(user_id=user_id, exam_id=exam_id, status="in_progress")
        db.session.add(ue)
        db.session.commit()
        return jsonify({
            "user_exam_id": ue.id,
            "message":      "Exam started successfully",
        }), 201


    @app.route("/api/student/exams/<int:exam_id>/submit", methods=["POST"])
    @jwt_required()
    def submit_exam(exam_id):
        """
        Grade and finalise a student's exam attempt.

        Expects JSON:
            {
                "user_exam_id": 42,
                "answers": { "<question_id>": <selected_option_id>, ... }
            }

        For each answered question:
            • Fetches the correct option.
            • Awards full marks if the answer matches, zero otherwise.
            • Creates a UserAnswer record.

        Updates UserExam with final score (as 0-100 percentage) and sets
        status to 'completed'.

        Returns { score, total_marks, marks_obtained }.
        """
        user_id      = int(get_jwt_identity())
        data         = request.json
        answers      = data.get("answers", {})
        user_exam_id = data.get("user_exam_id")

        ue = UserExam.query.filter_by(
            id=user_exam_id, user_id=user_id, exam_id=exam_id
        ).first_or_404()

        total_score = total_marks = 0

        for qid_str, opt_id in answers.items():
            q = Question.query.get(int(qid_str))
            if not q:
                continue

            total_marks += q.marks
            correct = Option.query.filter_by(question_id=q.id, is_correct=True).first()
            awarded = q.marks if (correct and correct.id == opt_id) else 0
            total_score += awarded

            db.session.add(UserAnswer(
                user_exam_id       = ue.id,
                question_id        = q.id,
                selected_option_id = opt_id,
                marks_awarded      = awarded,
            ))

        # Persist final score as a percentage (0-100)
        ue.end_time = datetime.utcnow()
        ue.status   = "completed"
        ue.score    = (total_score / total_marks * 100) if total_marks else 0
        db.session.commit()

        return jsonify({
            "message":        "Exam submitted successfully",
            "score":          ue.score,
            "total_marks":    total_marks,
            "marks_obtained": total_score,
        }), 200


    @app.route("/api/student/exams/<int:exam_id>/terminate", methods=["POST"])
    @jwt_required()
    def terminate_exam_by_student(exam_id):
        """
        Terminate an exam session due to a browser-side event (e.g. tab switch,
        fullscreen exit detected client-side).

        Expects JSON:
            { "reason": "tab_switch" }

        Logs the reason as a ViolationLog, marks the ExamSession and UserExam
        as 'terminated'.
        """
        user_id = int(get_jwt_identity())
        data    = request.json or {}
        reason  = data.get("reason", "unknown")

        # ── Terminate the monitoring session ──────────────────────────────────
        s = (
            ExamSession.query
            .filter_by(user_id=user_id, status="in_progress")
            .order_by(ExamSession.id.desc())
            .first()
        )
        if s:
            db.session.add(ViolationLog(
                session_id     = s.id,
                violation_type = reason,
                severity       = "major",
                detail         = f"Exam terminated by browser-side event: {reason}",
            ))
            s.status   = "terminated"
            s.end_time = datetime.utcnow()
            db.session.commit()

        # ── Also mark the UserExam attempt as terminated ───────────────────────
        ue = UserExam.query.filter_by(
            user_id=user_id, exam_id=exam_id, status="in_progress"
        ).first()
        if ue:
            ue.status   = "terminated"
            ue.end_time = datetime.utcnow()
            db.session.commit()

        return jsonify({"message": "Exam terminated", "reason": reason}), 200


    @app.route("/api/student/results", methods=["GET"])
    @jwt_required()
    def get_student_results():
        """
        Return all completed exam attempts for the authenticated student,
        sorted newest first.

        Each result includes: exam name, code, score (as raw marks), pass/fail
        status, date, duration, and question count.
        """
        user_id = int(get_jwt_identity())
        results = []

        for attempt in UserExam.query.filter_by(
            user_id=user_id, status="completed"
        ).all():
            exam   = attempt.exam
            raw    = attempt.score
            date_s = attempt.end_time or attempt.start_time

            # Calculate actual duration or fall back to the exam's configured duration
            dur_s = (
                f"{int((attempt.end_time - attempt.start_time).total_seconds() / 60)} mins"
                if attempt.start_time and attempt.end_time
                else f"{exam.duration_minutes} mins"
            )

            results.append({
                "id":        attempt.id,
                "examName":  exam.title,
                "code":      f"EXAM{exam.id:03d}",
                "score":     round((raw / 100) * exam.total_marks),
                "total":     exam.total_marks,
                "status":    "Pass" if raw >= 50 else "Fail",
                "date":      date_s.strftime("%Y-%m-%d") if date_s else "N/A",
                "duration":  dur_s,
                "questions": len(exam.questions),
            })

        results.sort(key=lambda x: x["date"], reverse=True)
        return jsonify(results), 200


    # ==========================================================================
    # 📌  VIOLATION ROUTES
    # ==========================================================================

    @app.route("/api/log_violation", methods=["POST"])
    @jwt_required()
    def log_violation():
        """
        Log a browser-reported violation (e.g. tab switch, fullscreen exit)
        that was detected client-side rather than by the server's CV pipeline.

        Expects JSON:
            { "type": "tab_switch" }

        Logged with severity 'minor' and linked to the user's latest session.
        """
        data   = request.json
        v_type = data.get("type")
        uid    = int(get_jwt_identity())

        s = (
            ExamSession.query
            .filter_by(user_id=uid)
            .order_by(ExamSession.id.desc())
            .first()
        )
        if s:
            db.session.add(ViolationLog(
                session_id     = s.id,
                violation_type = v_type,
                severity       = "minor",
                detail         = f"Browser-reported: {v_type}",
            ))
            db.session.commit()

        return jsonify({"message": "Violation logged"}), 200


    @app.route("/api/analyze_frame", methods=["POST"])
    @jwt_required()
    def analyze_frame_route():
        """
        Analyse a single uploaded frame for violations (alternative to MJPEG
        stream — useful for clients that cannot use the /video_feed endpoint).

        Expects a multipart/form-data upload with field 'frame' (JPEG bytes).

        Processing pipeline:
            1. Decode the uploaded image.
            2. Run analyze_frame() (face pose + YOLO).
            3. Check audio monitor for any pending loud-noise flag.
            4. Persist violations to the DB.
            5. Terminate session immediately for INSTANT_TERMINATE_VIOLATIONS.

        Returns:
            {
                "violations":        ["phone_detected", ...],
                "instant_terminate": true/false,
                "terminate_reason":  "phone_detected" | null,
                "annotated_frame":   "<base64 JPEG>"
            }
        """
        user_id = int(get_jwt_identity())

        if "frame" not in request.files:
            return jsonify({"error": "No frame provided"}), 400

        img_array = np.frombuffer(request.files["frame"].read(), dtype=np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        # ── Run CV pipeline ───────────────────────────────────────────────────
        _annotated, violations = analyze_frame(frame)

        # ── Merge audio violations from the background monitor ────────────────
        audio_flag = audio_monitor.get_violation()
        if audio_flag:
            violations.append({
                "type":     audio_flag,
                "severity": "minor",
                "detail":   "Loud noise detected by audio monitor",
                "snapshot": None,
            })

        if not violations:
            return jsonify({"violations": [], "annotated_frame": None}), 200

        # ── Log violations and check for instant termination ──────────────────
        s = (
            ExamSession.query
            .filter_by(user_id=user_id, status="in_progress")
            .order_by(ExamSession.id.desc())
            .first()
        )

        instant_terminate = False
        terminate_reason  = None

        if s:
            for v in violations:
                v_type = v["type"]
                logged = _log_violation_to_db(
                    s,
                    v_type,
                    v.get("severity", "major"),
                    v.get("detail", ""),
                    v.get("snapshot"),
                )
                if logged and v_type in INSTANT_TERMINATE_VIOLATIONS:
                    instant_terminate = True
                    terminate_reason  = v_type
                    s.status   = "terminated"
                    s.end_time = datetime.utcnow()

            db.session.commit()

        return jsonify({
            "violations":        [v["type"] for v in violations],
            "instant_terminate": instant_terminate,
            "terminate_reason":  terminate_reason,
            "annotated_frame":   encode_annotated_frame(_annotated),
        }), 200


    @app.route("/api/violations/latest", methods=["GET"])
    @jwt_required()
    def get_latest_violations():
        """
        Poll endpoint used by the student frontend to check for recent
        violations and session termination status.

        Returns violations logged in the last 5 seconds (de-duplicated by
        type) plus cumulative counts for all violations in the session.

        Response:
            {
                "violations": ["looking_away"],      // recent, unique types
                "counts":     {"looking_away": 3},   // cumulative totals
                "terminated": false,
                "reason":     null
            }
        """
        user_id = int(get_jwt_identity())
        s = (
            ExamSession.query
            .filter_by(user_id=user_id)
            .order_by(ExamSession.id.desc())
            .first()
        )

        # ── No session found ──────────────────────────────────────────────────
        if not s:
            return jsonify({
                "violations": [], "counts": {},
                "terminated": False, "reason": None,
            }), 200

        # ── Session already terminated ────────────────────────────────────────
        if s.status == "terminated":
            return jsonify({
                "violations": [], "counts": {},
                "terminated": True, "reason": "session_terminated",
            }), 200

        # ── Recent violations (last 5 seconds), de-duplicated by type ─────────
        cutoff = datetime.utcnow() - timedelta(seconds=5)
        recent = (
            ViolationLog.query
            .filter(
                ViolationLog.session_id == s.id,
                ViolationLog.timestamp  >= cutoff,
            )
            .order_by(ViolationLog.timestamp.desc())
            .all()
        )
        seen, violations = set(), []
        for log in recent:
            if log.violation_type not in seen:
                seen.add(log.violation_type)
                violations.append(log.violation_type)

        # ── Cumulative counts across entire session ───────────────────────────
        count_rows = (
            db.session.query(
                ViolationLog.violation_type,
                func.count(ViolationLog.id).label("cnt"),
            )
            .filter(ViolationLog.session_id == s.id)
            .group_by(ViolationLog.violation_type)
            .all()
        )
        counts = {r.violation_type: r.cnt for r in count_rows}

        # ── Check thresholds and auto-terminate if needed ─────────────────────
        terminated, reason = _check_and_terminate(s)

        return jsonify({
            "violations": violations,
            "counts":     counts,
            "terminated": terminated,
            "reason":     reason,
        }), 200


    # ==========================================================================
    # 📊  ADMIN — EXAM MANAGEMENT
    # ==========================================================================

    @app.route("/api/admin/exams", methods=["POST"])
    @admin_required
    def create_exam():
        """Create a new exam.  Expects JSON with title, description, duration, total_marks."""
        data = request.json
        e = Exam(
            title            = data["title"],
            description      = data.get("description"),
            duration_minutes = data["duration"],
            total_marks      = data["total_marks"],
        )
        db.session.add(e)
        db.session.commit()
        return jsonify({"message": "Exam created", "exam_id": e.id})


    @app.route("/api/admin/exams")
    @admin_required
    def list_exams():
        """Return a lightweight list of all exams (id, title, duration, total_marks)."""
        return jsonify([
            {
                "id":          e.id,
                "title":       e.title,
                "duration":    e.duration_minutes,
                "total_marks": e.total_marks,
            }
            for e in Exam.query.all()
        ])


    @app.route("/api/admin/exams/<int:exam_id>", methods=["GET"])
    @admin_required
    def get_exam_details(exam_id):
        """
        Return full exam details including questions and options with
        is_correct flags (admin view — not safe to expose to students).
        """
        e = Exam.query.get_or_404(exam_id)
        return jsonify({
            "id":          e.id,
            "title":       e.title,
            "description": e.description,
            "duration":    e.duration_minutes,
            "total_marks": e.total_marks,
            "questions": [
                {
                    "id":      q.id,
                    "text":    q.question_text,
                    "marks":   q.marks,
                    "type":    q.question_type,
                    "options": [
                        {"id": o.id, "text": o.option_text, "is_correct": o.is_correct}
                        for o in q.options
                    ],
                }
                for q in e.questions
            ],
        })


    @app.route("/api/admin/exams/<int:exam_id>", methods=["PUT"])
    @admin_required
    def update_exam(exam_id):
        """Update exam metadata (title, description, duration, total_marks)."""
        e = Exam.query.get_or_404(exam_id)
        d = request.json
        e.title            = d.get("title",       e.title)
        e.description      = d.get("description", e.description)
        e.duration_minutes = d.get("duration",    e.duration_minutes)
        e.total_marks      = d.get("total_marks",  e.total_marks)
        db.session.commit()
        return jsonify({"message": "Exam updated successfully"})


    @app.route("/api/admin/exams/<int:exam_id>", methods=["DELETE"])
    @admin_required
    def delete_exam(exam_id):
        """Permanently delete an exam and all its questions/options (cascade)."""
        db.session.delete(Exam.query.get_or_404(exam_id))
        db.session.commit()
        return jsonify({"message": "Exam deleted successfully"})


    @app.route("/api/admin/exams/<int:exam_id>/questions", methods=["POST"])
    @admin_required
    def add_question(exam_id):
        """
        Add a question to an existing exam.

        Expects JSON:
            { "question_text": "...", "marks": 2, "type": "mcq" }
        """
        d = request.json
        q = Question(
            exam_id       = exam_id,
            question_text = d["question_text"],
            marks         = d.get("marks", 1),
            question_type = d.get("type", "mcq"),
        )
        db.session.add(q)
        db.session.commit()
        return jsonify({"message": "Question added", "question_id": q.id})


    @app.route("/api/admin/questions/<int:question_id>", methods=["DELETE"])
    @admin_required
    def delete_question(question_id):
        """Delete a question (and its options via cascade)."""
        db.session.delete(Question.query.get_or_404(question_id))
        db.session.commit()
        return jsonify({"message": "Question deleted"})


    @app.route("/api/admin/questions/<int:question_id>/options", methods=["POST"])
    @admin_required
    def add_option(question_id):
        """
        Add an answer option to a question.

        Expects JSON:
            { "option_text": "...", "is_correct": true/false }
        """
        d = request.json
        db.session.add(Option(
            question_id = question_id,
            option_text = d["option_text"],
            is_correct  = d.get("is_correct", False),
        ))
        db.session.commit()
        return jsonify({"message": "Option added"})


    # ==========================================================================
    # 📊  ADMIN — SESSION MANAGEMENT
    # ==========================================================================

    @app.route("/api/admin/sessions/active")
    @admin_required
    def active_sessions():
        """
        Return all currently in-progress exam sessions with student username,
        warning count, and start time.  Used by the admin live-monitoring view.
        """
        sessions = ExamSession.query.filter_by(status="in_progress").all()
        return jsonify([
            {
                "session_id": s.id,
                "student":    s.student.username,
                "warnings":   s.total_warnings,
                "started_at": s.start_time.isoformat(),
            }
            for s in sessions
        ])


    @app.route("/api/admin/sessions/<int:session_id>/terminate", methods=["POST"])
    @admin_required
    def terminate_session(session_id):
        """Manually terminate a session from the admin panel."""
        s          = ExamSession.query.get_or_404(session_id)
        s.status   = "terminated"
        s.end_time = datetime.utcnow()
        db.session.commit()
        return jsonify({"message": "Session terminated"})


    @app.route("/api/admin/sessions/<int:session_id>/violations", methods=["GET"])
    @admin_required
    def admin_session_violations(session_id):
        """
        Return all violation logs for a session in chronological order.

        Query params:
            include_snapshot=1  — include the base64 snapshot image in each record
        """
        include_snapshot = request.args.get("include_snapshot", "0") == "1"
        logs = (
            ViolationLog.query
            .filter_by(session_id=session_id)
            .order_by(ViolationLog.timestamp.asc())
            .all()
        )
        return jsonify([
            {
                "id":             log.id,
                "violation_type": log.violation_type,
                "severity":       log.severity,
                "detail":         log.detail,
                "timestamp":      log.timestamp.isoformat(),
                **({"snapshot": log.snapshot_b64} if include_snapshot else {}),
            }
            for log in logs
        ]), 200


    @app.route("/api/admin/sessions/<int:session_id>/detail", methods=["GET"])
    @admin_required
    def admin_session_detail(session_id):
        """
        Full detail view for a single monitoring session.

        Returns:
            • Session metadata (status, timing, warnings).
            • Violation summary dict  { violation_type: count }.
            • Full violation log list.
            • Linked exam attempt (score, marks, status) — matched by proximity
              of start_time since there is no direct foreign key.

        Query params:
            include_snapshot=1  — include snapshot_b64 in each violation record
        """
        include_snapshot = request.args.get("include_snapshot", "0") == "1"
        s = ExamSession.query.get_or_404(session_id)

        logs = (
            ViolationLog.query
            .filter_by(session_id=session_id)
            .order_by(ViolationLog.timestamp.asc())
            .all()
        )

        # ── Aggregate violation counts ─────────────────────────────────────────
        violation_summary = {}
        for log in logs:
            violation_summary[log.violation_type] = (
                violation_summary.get(log.violation_type, 0) + 1
            )

        # ── Session duration ──────────────────────────────────────────────────
        duration_secs = None
        if s.start_time and s.end_time:
            duration_secs = int((s.end_time - s.start_time).total_seconds())

        # ── Nearest UserExam attempt (heuristic match by start_time delta) ────
        exam_attempt = (
            UserExam.query
            .filter_by(user_id=s.user_id)
            .filter(UserExam.status.in_(["completed", "terminated"]))
            .order_by(
                func.abs(
                    func.strftime("%s", UserExam.start_time) -
                    func.strftime("%s", s.start_time)
                )
            )
            .first()
        )

        exam_data = None
        if exam_attempt:
            raw   = exam_attempt.score or 0
            total = exam_attempt.exam.total_marks or 100
            exam_data = {
                "id":             exam_attempt.id,
                "exam_title":     exam_attempt.exam.title,
                "score":          round(raw, 2),
                "marks_obtained": round((raw / 100) * total),
                "total_marks":    total,
                "status":         exam_attempt.status,
            }

        return jsonify({
            "id":                session_id,
            "user_id":           s.user_id,
            "student":           s.student.username,
            "status":            s.status,
            "start_time":        s.start_time.isoformat() if s.start_time else None,
            "end_time":          s.end_time.isoformat()   if s.end_time   else None,
            "duration_seconds":  duration_secs,
            "total_warnings":    s.total_warnings or 0,
            "violation_summary": violation_summary,
            "exam_attempt":      exam_data,
            "violations": [
                {
                    "id":             log.id,
                    "violation_type": log.violation_type,
                    "severity":       log.severity or "major",
                    "detail":         log.detail,
                    "timestamp":      log.timestamp.isoformat(),
                    **({"snapshot_b64": log.snapshot_b64} if include_snapshot else {}),
                }
                for log in logs
            ],
        }), 200


    # ==========================================================================
    # 📊  ADMIN — VIOLATION MANAGEMENT
    # ==========================================================================

    @app.route("/api/admin/violations", methods=["GET"])
    @admin_required
    def admin_all_violations():
        """
        Filtered, paginated (implicitly limited to 500) view of all violations
        across all sessions and students.

        Query params:
            session_id        – filter to one session
            student_id        – filter to one student
            type              – filter by violation_type string
            include_snapshot  – 1 to include base64 snapshot
        """
        session_id       = request.args.get("session_id",  type=int)
        student_id       = request.args.get("student_id",  type=int)
        v_type_filter    = request.args.get("type")
        include_snapshot = request.args.get("include_snapshot", "0") == "1"

        q = (
            ViolationLog.query
            .join(ExamSession, ViolationLog.session_id == ExamSession.id)
            .join(User, ExamSession.user_id == User.id)
        )

        if session_id:    q = q.filter(ViolationLog.session_id == session_id)
        if student_id:    q = q.filter(ExamSession.user_id == student_id)
        if v_type_filter: q = q.filter(ViolationLog.violation_type == v_type_filter)

        logs = q.order_by(ViolationLog.timestamp.desc()).limit(500).all()

        return jsonify([
            {
                "id":             log.id,
                "session_id":     log.session_id,
                "student":        log.session.student.username,
                "violation_type": log.violation_type,
                "severity":       log.severity,
                "detail":         log.detail,
                "timestamp":      log.timestamp.isoformat(),
                **({"snapshot": log.snapshot_b64} if include_snapshot else {}),
            }
            for log in logs
        ]), 200


    @app.route("/api/admin/violations/summary", methods=["GET"])
    @admin_required
    def admin_violation_summary():
        """
        Return aggregated violation counts grouped by (violation_type, severity).
        Useful for the admin dashboard summary cards / charts.
        """
        rows = (
            db.session.query(
                ViolationLog.violation_type,
                ViolationLog.severity,
                func.count(ViolationLog.id).label("count"),
            )
            .group_by(ViolationLog.violation_type, ViolationLog.severity)
            .all()
        )
        return jsonify([
            {
                "violation_type": r.violation_type,
                "severity":       r.severity,
                "count":          r.count,
            }
            for r in rows
        ]), 200


    # ==========================================================================
    # 📊  ADMIN — RESULTS
    # ==========================================================================

    @app.route("/api/admin/exams/<int:exam_id>/results")
    @admin_required
    def exam_results(exam_id):
        """Return a quick score list (student, score, status) for one exam."""
        return jsonify([
            {
                "student": a.user.username,
                "score":   a.score,
                "status":  a.status,
            }
            for a in UserExam.query.filter_by(exam_id=exam_id).all()
        ])


    @app.route("/api/admin/results/all", methods=["GET"])
    @admin_required
    def admin_all_results():
        """
        Paginated, filterable list of every exam attempt across all exams and
        students, plus an aggregate summary object for dashboard stat-cards.

        Query params:
            exam_id   – filter to a single exam
            student   – case-insensitive username substring search
            status    – 'Pass' | 'Fail' | 'terminated' | 'in_progress'
            page      – 1-based page number (default: 1)
            per_page  – rows per page (default: 200, max: 500)

        Response:
            {
                "results":     [ ... ],          // current page
                "summary":     { ... },          // aggregate stats (all matching)
                "total_count": 123,
                "page":        1,
                "per_page":    200
            }
        """
        exam_id_filter = request.args.get("exam_id",  type=int)
        student_filter = request.args.get("student",  "").strip().lower()
        status_filter  = request.args.get("status",   "").strip()
        page_num       = max(request.args.get("page",     1,   type=int), 1)
        per_page       = min(request.args.get("per_page", 200, type=int), 500)

        # ── Base query with joins ──────────────────────────────────────────────
        q = (
            db.session.query(UserExam)
            .join(User, UserExam.user_id == User.id)
            .join(Exam, UserExam.exam_id == Exam.id)
        )

        # ── Apply filters ─────────────────────────────────────────────────────
        if exam_id_filter:
            q = q.filter(UserExam.exam_id == exam_id_filter)
        if student_filter:
            q = q.filter(func.lower(User.username).contains(student_filter))
        if status_filter in ("Pass", "Fail"):
            q = q.filter(UserExam.status == "completed")
        elif status_filter == "terminated":
            q = q.filter(UserExam.status == "terminated")
        elif status_filter == "in_progress":
            q = q.filter(UserExam.status == "in_progress")

        total_count = q.count()
        attempts    = (
            q.order_by(UserExam.start_time.desc())
             .offset((page_num - 1) * per_page)
             .limit(per_page)
             .all()
        )

        # ── Pre-load violation counts per student (single query) ──────────────
        viol_rows = (
            db.session.query(
                ExamSession.user_id,
                func.count(ViolationLog.id).label("cnt"),
            )
            .join(ViolationLog, ViolationLog.session_id == ExamSession.id)
            .group_by(ExamSession.user_id)
            .all()
        )
        viol_map = {r.user_id: r.cnt for r in viol_rows}

        # ── Local helpers ─────────────────────────────────────────────────────
        def _status(attempt):
            if attempt.status == "terminated":  return "terminated"
            if attempt.status == "in_progress": return "in_progress"
            return "Pass" if (attempt.score or 0) >= 50 else "Fail"

        def _duration(attempt):
            if attempt.start_time and attempt.end_time:
                mins = int((attempt.end_time - attempt.start_time).total_seconds() / 60)
                return f"{mins} min{'s' if mins != 1 else ''}"
            return f"{attempt.exam.duration_minutes} mins"

        # ── Build result rows ─────────────────────────────────────────────────
        results = []
        for a in attempts:
            st = _status(a)
            # Secondary filter for Pass/Fail (can't be done cleanly in SQL)
            if status_filter in ("Pass", "Fail") and st != status_filter:
                continue

            raw_score      = a.score or 0
            total_marks    = a.exam.total_marks or 100
            marks_obtained = round((raw_score / 100) * total_marks)

            results.append({
                "id":             a.id,
                "student_id":     a.user_id,
                "student":        a.user.username,
                "exam_id":        a.exam_id,
                "exam_title":     a.exam.title,
                "score":          round(raw_score, 2),
                "marks_obtained": marks_obtained,
                "total_marks":    total_marks,
                "questions":      len(a.exam.questions),
                "status":         st,
                "date":           (a.end_time or a.start_time).strftime("%Y-%m-%d")
                                  if (a.end_time or a.start_time) else "N/A",
                "duration":       _duration(a),
                "violations":     viol_map.get(a.user_id, 0),
            })

        # ── Aggregate summary (computed over ALL matching rows, not just page) ─
        all_attempts = q.all()
        completed    = [a for a in all_attempts if a.status == "completed"]
        scores       = [a.score for a in completed if a.score is not None]
        pass_c       = sum(1 for a in completed if (a.score or 0) >= 50)
        fail_c       = sum(1 for a in completed if (a.score or 0) <  50)
        term_c       = sum(1 for a in all_attempts if a.status == "terminated")
        avg_s        = sum(scores) / len(scores) if scores else 0
        pass_r       = (pass_c / (pass_c + fail_c) * 100) if (pass_c + fail_c) else 0

        exam_counts = Counter(a.exam.title for a in all_attempts)
        top_exam    = exam_counts.most_common(1)[0][0] if exam_counts else None

        return jsonify({
            "results": results,
            "summary": {
                "total_attempts":   len(all_attempts),
                "avg_score":        round(avg_s, 2),
                "pass_count":       pass_c,
                "fail_count":       fail_c,
                "terminated_count": term_c,
                "pass_rate":        round(pass_r, 2),
                "top_exam":         top_exam,
                "highest_score":    round(max(scores), 2) if scores else None,
                "lowest_score":     round(min(scores), 2) if scores else None,
            },
            "total_count": total_count,
            "page":        page_num,
            "per_page":    per_page,
        }), 200


    @app.route("/api/admin/results/export", methods=["GET"])
    @admin_required
    def admin_export_results():
        """
        Export all completed/terminated exam attempts as a CSV file download.

        Supports the same exam_id, student, and status filters as
        admin_all_results.  Returns a 'text/csv' response with a
        Content-Disposition: attachment header so the browser triggers a
        file download.

        CSV columns:
            Student, Exam, Score (%), Marks Obtained, Total Marks,
            Status, Grade, Date, Duration (mins), Violations
        """
        exam_id_filter = request.args.get("exam_id",  type=int)
        student_filter = request.args.get("student",  "").strip().lower()
        status_filter  = request.args.get("status",   "").strip()

        q = (
            db.session.query(UserExam)
            .join(User, UserExam.user_id == User.id)
            .join(Exam, UserExam.exam_id == Exam.id)
            .filter(UserExam.status.in_(["completed", "terminated"]))
        )
        if exam_id_filter:
            q = q.filter(UserExam.exam_id == exam_id_filter)
        if student_filter:
            q = q.filter(func.lower(User.username).contains(student_filter))

        attempts = q.order_by(UserExam.start_time.desc()).all()

        # ── Pre-load violation counts for the CSV column ─────────────────────
        viol_rows = (
            db.session.query(
                ExamSession.user_id,
                func.count(ViolationLog.id).label("cnt"),
            )
            .join(ViolationLog, ViolationLog.session_id == ExamSession.id)
            .group_by(ExamSession.user_id)
            .all()
        )
        viol_map = {r.user_id: r.cnt for r in viol_rows}

        # ── Build CSV in memory ───────────────────────────────────────────────
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Student", "Exam", "Score (%)", "Marks Obtained", "Total Marks",
            "Status", "Grade", "Date", "Duration (mins)", "Violations",
        ])

        for a in attempts:
            raw       = a.score or 0
            total_m   = a.exam.total_marks or 100
            marks_obt = round((raw / 100) * total_m)
            status    = (
                "terminated" if a.status == "terminated"
                else ("Pass" if raw >= 50 else "Fail")
            )
            dur_mins  = (
                int((a.end_time - a.start_time).total_seconds() / 60)
                if a.start_time and a.end_time else 0
            )

            # Apply status filter (easier to do here than in SQL for Pass/Fail)
            if status_filter and status != status_filter:
                continue

            writer.writerow([
                a.user.username,
                a.exam.title,
                round(raw, 2),
                marks_obt,
                total_m,
                status,
                _grade(raw),
                (a.end_time or a.start_time).strftime("%Y-%m-%d")
                    if (a.end_time or a.start_time) else "",
                dur_mins,
                viol_map.get(a.user_id, 0),
            ])

        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=edushield_results.csv"
            },
        )


    # ==========================================================================
    # 📊  ADMIN — STUDENT PROFILES, SESSIONS & VIOLATIONS
    # ==========================================================================

    @app.route("/api/admin/students/<int:student_id>/profile", methods=["GET"])
    @admin_required
    def admin_student_profile(student_id):
        """Return basic profile info (id, username, role, has face photo) for a student."""
        user = User.query.get_or_404(student_id)
        return jsonify({
            "id":              user.id,
            "username":        user.username,
            "role":            user.role,
            "registered_face": bool(user.registered_face_path),
        }), 200


    @app.route("/api/admin/students/<int:student_id>/violations", methods=["GET"])
    @admin_required
    def admin_student_violations(student_id):
        """
        Return all violation logs for a student across all (or one specific)
        session.

        Query params:
            session_id        – restrict to a single session
            type              – filter by violation_type
            severity          – filter by severity
            include_snapshot  – 1 to include snapshot_b64
            limit             – max rows (default 500, max 2000)
        """
        session_id_filter = request.args.get("session_id", type=int)
        type_filter       = request.args.get("type",     "").strip()
        severity_filter   = request.args.get("severity", "").strip()
        include_snapshot  = request.args.get("include_snapshot", "0") == "1"
        limit             = min(request.args.get("limit", 500, type=int), 2000)

        # Collect all session IDs belonging to this student
        sessions    = ExamSession.query.filter_by(user_id=student_id).all()
        session_ids = [s.id for s in sessions]
        if not session_ids:
            return jsonify([]), 200

        q = ViolationLog.query.filter(ViolationLog.session_id.in_(session_ids))

        if session_id_filter and session_id_filter in session_ids:
            q = q.filter(ViolationLog.session_id == session_id_filter)
        if type_filter:
            q = q.filter(ViolationLog.violation_type == type_filter)
        if severity_filter:
            q = q.filter(ViolationLog.severity == severity_filter)

        logs = q.order_by(ViolationLog.timestamp.desc()).limit(limit).all()

        return jsonify([
            {
                "id":             log.id,
                "session_id":     log.session_id,
                "violation_type": log.violation_type,
                "severity":       log.severity or "major",
                "detail":         log.detail,
                "timestamp":      log.timestamp.isoformat(),
                **({"snapshot_b64": log.snapshot_b64} if include_snapshot else {}),
            }
            for log in logs
        ]), 200


    @app.route("/api/admin/students/<int:student_id>/sessions", methods=["GET"])
    @admin_required
    def admin_student_sessions(student_id):
        """
        Return all exam sessions for a student, each enriched with:
            • Violation summary dict  { violation_type: count }
            • Last 20 violation entries
            • Nearest linked exam attempt (score, marks, status)
            • Session duration in seconds
        """
        User.query.get_or_404(student_id)   # 404 if student doesn't exist

        sessions = (
            ExamSession.query
            .filter_by(user_id=student_id)
            .order_by(ExamSession.start_time.desc())
            .all()
        )

        result = []
        for s in sessions:
            # ── Per-session violation summary ─────────────────────────────────
            viol_rows = (
                db.session.query(
                    ViolationLog.violation_type,
                    func.count(ViolationLog.id).label("cnt"),
                )
                .filter(ViolationLog.session_id == s.id)
                .group_by(ViolationLog.violation_type)
                .all()
            )
            violation_summary = {r.violation_type: r.cnt for r in viol_rows}

            # ── Last 20 violations for the timeline preview ───────────────────
            recent_viols = (
                ViolationLog.query
                .filter_by(session_id=s.id)
                .order_by(ViolationLog.timestamp.desc())
                .limit(20)
                .all()
            )

            # ── Nearest UserExam attempt (heuristic by start_time delta) ──────
            exam_attempt = (
                UserExam.query
                .filter_by(user_id=student_id)
                .filter(UserExam.status.in_(["completed", "terminated"]))
                .order_by(
                    func.abs(
                        func.strftime("%s", UserExam.start_time) -
                        func.strftime("%s", s.start_time)
                    )
                )
                .first()
            )

            exam_attempt_data = None
            if exam_attempt:
                raw   = exam_attempt.score or 0
                total = exam_attempt.exam.total_marks or 100
                exam_attempt_data = {
                    "id":             exam_attempt.id,
                    "exam_id":        exam_attempt.exam_id,
                    "exam_title":     exam_attempt.exam.title,
                    "score":          round(raw, 2),
                    "marks_obtained": round((raw / 100) * total),
                    "total_marks":    total,
                    "status":         exam_attempt.status,
                }

            duration_secs = None
            if s.start_time and s.end_time:
                duration_secs = int((s.end_time - s.start_time).total_seconds())

            result.append({
                "id":                s.id,
                "user_id":           s.user_id,
                "status":            s.status,
                "start_time":        s.start_time.isoformat() if s.start_time else None,
                "end_time":          s.end_time.isoformat()   if s.end_time   else None,
                "duration_seconds":  duration_secs,
                "total_warnings":    s.total_warnings or 0,
                "violation_summary": violation_summary,
                "violations": [
                    {
                        "violation_type": v.violation_type,
                        "severity":       v.severity,
                        "timestamp":      v.timestamp.isoformat(),
                    }
                    for v in recent_viols
                ],
                "exam_score":   exam_attempt_data["score"] if exam_attempt_data else None,
                "exam_attempt": exam_attempt_data,
            })

        return jsonify(result), 200


    # ==========================================================================
    # 🧠  RL FEEDBACK ROUTES
    # ==========================================================================

    @app.route("/api/admin/feedback", methods=["POST"])
    @admin_required
    def submit_rl_feedback():
        """
        Allow admins to correct a wrong termination decision, feeding a reward
        signal back into the RL threshold optimiser.

        Request JSON:
            {
                "session_id":        123,
                "is_false_positive": true,      // true  = system was WRONG
                                                // false = system was CORRECT
                "note":              "Student was adjusting glasses",  // optional
                "violation_type":    "looking_away"                    // optional hint
            }

        RL logic:
            • is_false_positive=True  → reward = -1 (loosen thresholds)
            • is_false_positive=False → reward = +1 (reinforce thresholds)

        Side effects:
            • Persists a FeedbackLog record.
            • If false positive, restores session status to 'in_progress' so
              the student can resume (comment out if you prefer permanent termination).

        Response:
            {
                "success":            true,
                "message":            "...",
                "updated_thresholds": { "yolo": 0.55, ... },
                "action_taken":       "YOLO_THRESH_UP",
                "accuracy":           0.82,
                "epsilon":            0.1
            }
        """
        admin_id          = int(get_jwt_identity())
        data              = request.json or {}
        session_id        = data.get("session_id")
        is_false_positive = data.get("is_false_positive")
        note              = data.get("note", "")
        hint_vtype        = data.get("violation_type", "")

        if session_id is None or is_false_positive is None:
            return jsonify({"error": "session_id and is_false_positive are required"}), 400

        session = ExamSession.query.get(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

        # ── Prevent duplicate feedback on the same session ────────────────────
        existing = FeedbackLog.query.filter_by(session_id=session_id).first()
        if existing:
            return jsonify({"error": "Feedback already submitted for this session"}), 409

        # ── Determine primary violation type for RL context ───────────────────
        if hint_vtype:
            primary_violation = hint_vtype
        else:
            # Fall back to the most-frequent violation in this session
            row = (
                db.session.query(
                    ViolationLog.violation_type,
                    func.count(ViolationLog.id).label("cnt"),
                )
                .filter(ViolationLog.session_id == session_id)
                .group_by(ViolationLog.violation_type)
                .order_by(func.count(ViolationLog.id).desc())
                .first()
            )
            primary_violation = row.violation_type if row else "looking_away"

        # ── Send reward signal to the RL optimiser ────────────────────────────
        reward    = -1 if is_false_positive else +1
        rl_result = rl_optimizer.update_thresholds(primary_violation, reward)

        # ── Persist feedback log ──────────────────────────────────────────────
        db.session.add(FeedbackLog(
            session_id        = session_id,
            admin_id          = admin_id,
            is_false_positive = is_false_positive,
            note              = note,
            violation_type    = primary_violation,
            thresholds_after  = str(rl_result["thresholds"]),
        ))

        # ── Optionally restore session if flagged as a false positive ─────────
        if is_false_positive and session.status == "terminated":
            session.status   = "in_progress"
            session.end_time = None

        db.session.commit()

        return jsonify({
            "success":            True,
            "message":            (
                "✓ Marked as false positive. System has learned to be less sensitive."
                if is_false_positive else
                "✓ Confirmed as real cheating. Detection reinforced."
            ),
            "updated_thresholds": rl_result["thresholds"],
            "action_taken":       rl_result["action_taken"],
            "accuracy":           rl_result["accuracy"],
            "epsilon":            rl_result["epsilon"],
        }), 200


    @app.route("/api/admin/feedback/<int:session_id>", methods=["GET"])
    @admin_required
    def get_session_feedback(session_id):
        """
        Check whether feedback has already been submitted for a given session.

        Returns:
            { "has_feedback": false }
            or
            { "has_feedback": true, "is_false_positive": ..., "note": ..., "submitted_at": ... }
        """
        log = FeedbackLog.query.filter_by(session_id=session_id).first()
        if not log:
            return jsonify({"has_feedback": False}), 200

        return jsonify({
            "has_feedback":      True,
            "is_false_positive": log.is_false_positive,
            "note":              log.note,
            "submitted_at":      log.created_at.isoformat() if log.created_at else None,
        }), 200


    @app.route("/api/admin/rl/stats", methods=["GET"])
    @admin_required
    def rl_stats():
        """
        Return current RL model statistics including accuracy, current
        thresholds, epsilon (exploration rate), and learning history.
        """
        return jsonify(rl_optimizer.get_stats()), 200


    @app.route("/api/admin/rl/thresholds", methods=["GET"])
    @admin_required
    def rl_thresholds():
        """Return only the current detection threshold values from the RL model."""
        return jsonify(rl_optimizer.thresholds), 200


    @app.route("/api/admin/sessions/terminated", methods=["GET"])
    @admin_required
    def terminated_sessions():
        """
        Return recently terminated sessions that have NOT yet received admin
        feedback.  Used to populate the feedback review queue in the admin UI.

        Query params:
            limit  – max rows returned (default 50, max 200)

        Each row includes the primary violation type and total violation count
        so admins can quickly triage which terminations to review.
        """
        limit = min(request.args.get("limit", 50, type=int), 200)

        # Exclude sessions that already have a feedback entry
        feedbacked_ids = [f.session_id for f in FeedbackLog.query.all()]
        q = ExamSession.query.filter_by(status="terminated")
        if feedbacked_ids:
            q = q.filter(~ExamSession.id.in_(feedbacked_ids))

        sessions = q.order_by(ExamSession.end_time.desc()).limit(limit).all()

        result = []
        for s in sessions:
            # Most-frequent violation for the session summary row
            row = (
                db.session.query(
                    ViolationLog.violation_type,
                    func.count(ViolationLog.id).label("cnt"),
                )
                .filter(ViolationLog.session_id == s.id)
                .group_by(ViolationLog.violation_type)
                .order_by(func.count(ViolationLog.id).desc())
                .first()
            )
            primary_viol = row.violation_type if row else "unknown"
            total_viols  = (
                db.session.query(func.count(ViolationLog.id))
                .filter(ViolationLog.session_id == s.id)
                .scalar() or 0
            )

            result.append({
                "session_id":        s.id,
                "student":           s.student.username,
                "student_id":        s.user_id,
                "status":            s.status,
                "start_time":        s.start_time.isoformat() if s.start_time else None,
                "end_time":          s.end_time.isoformat()   if s.end_time   else None,
                "total_warnings":    s.total_warnings or 0,
                "total_violations":  total_viols,
                "primary_violation": primary_viol,
                "has_feedback":      False,
            })

        return jsonify(result), 200


    # ── End of route definitions ───────────────────────────────────────────────
    return app


# ==============================================================================
# 🚀  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)