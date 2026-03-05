import os
import cv2
import time
import base64
import numpy as np
from datetime import timedelta, datetime
from functools import wraps

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

from config import Config
from models import (
    UserAnswer, db, User, ExamSession, ViolationLog,
    Exam, Question, Option, UserExam,
)
from core.vision import analyze_frame, encode_annotated_frame
from core.audio import AudioMonitor
from core.face_auth import verify_identity, validate_registration_photo


# Per-session cooldown tracker: {  "session_id:violation_type" : last_log_time  }
last_db_log_time: dict[str, float] = {}


# ==========================================================
# 🚨 TERMINATION THRESHOLDS
# ==========================================================

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

# Violations that bypass cooldown and trigger instant termination
INSTANT_TERMINATE_VIOLATIONS = {"phone_detected"}


def _check_and_terminate(session):
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


# ==========================================================
# 🏭 APP FACTORY
# ==========================================================

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["JWT_SECRET_KEY"]           = "super-secret-key"
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=5)

    db.init_app(app)
    from flask_migrate import Migrate
    migrate = Migrate(app, db)

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

    with app.app_context():
        db.create_all()
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    audio_monitor = AudioMonitor(threshold=800)
    audio_monitor.start()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _decode_and_save_image(data_uri: str, dest_path: str) -> bool:
        try:
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

    def _log_violation_to_db(session, v_type, severity, detail, snapshot=None):
        """
        Log a violation. For INSTANT_TERMINATE violations, bypass the 10s cooldown.
        Returns True if actually logged, False if rate-limited.
        """
        now = time.time()
        key = f"{session.id}:{v_type}"

        # Phone and other instant-terminate violations bypass cooldown
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
        session.total_warnings      = (session.total_warnings or 0) + 1
        last_db_log_time[key]       = now
        return True

    # ==========================================================
    # 🔐 ROLE-BASED DECORATOR
    # ==========================================================

    def admin_required(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            user = User.query.get(int(get_jwt_identity()))
            if not user or user.role != "admin":
                return jsonify({"error": "Admin access required"}), 403
            return fn(*args, **kwargs)
        return wrapper

    # ==========================================================
    # 🔐 AUTH ROUTES
    # ==========================================================

    @app.route("/api/register", methods=["POST"])
    def register():
        data     = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "")
        snapshot = data.get("register_snapshot_base64", "")

        if not username:
            return jsonify({"error": "Username is required"}), 400
        if not password:
            return jsonify({"error": "Password is required"}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username already exists"}), 400
        if not snapshot or len(snapshot) <= 100:
            return jsonify({"error": "Face snapshot required"}), 400

        filename = secure_filename(f"{username}_baseline.jpg")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        if not _decode_and_save_image(snapshot, filepath):
            return jsonify({"error": "Failed to process image"}), 400

        if not validate_registration_photo(filepath):
            os.remove(filepath)
            return jsonify({
                "error": "No face detected in your photo. "
                         "Please ensure good lighting and look directly at the camera."
            }), 400

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
        data     = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "")
        snapshot = data.get("live_snapshot_base64", "")

        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid credentials"}), 401

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
                    "error": "Face verification failed. "
                             "Please ensure good lighting and look directly at the camera."
                }), 401
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

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

    # ==========================================================
    # 📹 MJPEG VIDEO STREAM  — annotated backend feed
    # ==========================================================

    @app.route("/api/video_feed", methods=["GET"])
    def video_feed():
        """
        MJPEG stream of the annotated webcam feed.

        Auth: token passed as query param ?token=<jwt>  (EventSource / <img> can't
        set headers, so we accept the token in the URL and validate it manually).

        The client replaces the plain <video> element with:
            <img src="http://localhost:5001/api/video_feed?token=<jwt>" />
        """
        token = request.args.get("token", "")
        try:
            decoded   = decode_token(token)
            user_id   = int(decoded["sub"])
        except Exception:
            return jsonify({"error": "Invalid or missing token"}), 401

        def generate():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 15)

            if not cap.isOpened():
                # Send a single error frame then stop
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Camera not available", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, buf = cv2.imencode(".jpg", error_frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       buf.tobytes() + b"\r\n")
                return

            try:
                while True:
                    with app.app_context():
                        # Stop streaming if session was terminated
                        s = (
                            ExamSession.query
                            .filter_by(user_id=user_id)
                            .order_by(ExamSession.id.desc())
                            .first()
                        )
                        if s and s.status == "terminated":
                            break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Mirror so student sees natural view
                    frame = cv2.flip(frame, 1)

                    annotated, violations = analyze_frame(frame)

                    # Log any violations discovered during streaming
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
                                        # Immediate termination for phone
                                        if v["type"] in INSTANT_TERMINATE_VIOLATIONS:
                                            s.status   = "terminated"
                                            s.end_time = datetime.utcnow()

                                if needs_commit:
                                    db.session.commit()

                    _, buf = cv2.imencode(".jpg", annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" +
                           buf.tobytes() + b"\r\n")

                    time.sleep(1 / 15)   # ~15 fps

            finally:
                cap.release()

        return Response(
            stream_with_context(generate()),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    # ==========================================================
    # 📚 STUDENT ROUTES
    # ==========================================================

    @app.route("/api/student/exams", methods=["GET"])
    @jwt_required()
    def get_student_exams():
        return jsonify([
            {
                "id": e.id, "title": e.title, "description": e.description,
                "duration": e.duration_minutes, "total_marks": e.total_marks,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in Exam.query.all()
        ])

    @app.route("/api/student/exams/<int:exam_id>", methods=["GET"])
    @jwt_required()
    def get_student_exam_details(exam_id):
        exam = Exam.query.get_or_404(exam_id)
        return jsonify({
            "id": exam.id, "title": exam.title,
            "description": exam.description,
            "duration": exam.duration_minutes,
            "total_marks": exam.total_marks,
            "questions": [
                {
                    "id": q.id, "text": q.question_text, "marks": q.marks,
                    "type": q.question_type,
                    "options": [{"id": o.id, "text": o.option_text} for o in q.options],
                }
                for q in exam.questions
            ],
        })

    @app.route("/api/student/exams/<int:exam_id>/start", methods=["POST"])
    @jwt_required()
    def start_exam(exam_id):
        user_id = int(get_jwt_identity())
        Exam.query.get_or_404(exam_id)
        existing = UserExam.query.filter_by(
            user_id=user_id, exam_id=exam_id, status="in_progress"
        ).first()
        if existing:
            return jsonify({"user_exam_id": existing.id,
                            "message": "Exam already in progress"}), 200
        ue = UserExam(user_id=user_id, exam_id=exam_id, status="in_progress")
        db.session.add(ue)
        db.session.commit()
        return jsonify({"user_exam_id": ue.id,
                        "message": "Exam started successfully"}), 201

    @app.route("/api/student/exams/<int:exam_id>/submit", methods=["POST"])
    @jwt_required()
    def submit_exam(exam_id):
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
            correct = Option.query.filter_by(
                question_id=q.id, is_correct=True
            ).first()
            awarded = q.marks if (correct and correct.id == opt_id) else 0
            total_score += awarded
            db.session.add(UserAnswer(
                user_exam_id=ue.id, question_id=q.id,
                selected_option_id=opt_id, marks_awarded=awarded,
            ))

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
        user_id = int(get_jwt_identity())
        data    = request.json or {}
        reason  = data.get("reason", "unknown")

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
        user_id = int(get_jwt_identity())
        results = []
        for attempt in UserExam.query.filter_by(
            user_id=user_id, status="completed"
        ).all():
            exam  = attempt.exam
            raw   = attempt.score
            date_s = attempt.end_time or attempt.start_time
            dur_s  = (
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

    # ==========================================================
    # 📌 VIOLATION ROUTES
    # ==========================================================

    @app.route("/api/log_violation", methods=["POST"])
    @jwt_required()
    def log_violation():
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
        user_id = int(get_jwt_identity())

        if "frame" not in request.files:
            return jsonify({"error": "No frame provided"}), 400

        img_array = np.frombuffer(request.files["frame"].read(), dtype=np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        _annotated, violations = analyze_frame(frame)

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

        response_payload = {
            "violations":       [v["type"] for v in violations],
            "instant_terminate": instant_terminate,
            "terminate_reason":  terminate_reason,
            "annotated_frame":  encode_annotated_frame(_annotated),
        }
        return jsonify(response_payload), 200

    @app.route("/api/violations/latest", methods=["GET"])
    @jwt_required()
    def get_latest_violations():
        user_id = int(get_jwt_identity())
        s = (
            ExamSession.query
            .filter_by(user_id=user_id)
            .order_by(ExamSession.id.desc())
            .first()
        )

        if not s:
            return jsonify({"violations": [], "counts": {},
                            "terminated": False, "reason": None}), 200
        if s.status == "terminated":
            return jsonify({"violations": [], "counts": {},
                            "terminated": True, "reason": "session_terminated"}), 200

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

        terminated, reason = _check_and_terminate(s)
        return jsonify({
            "violations": violations,
            "counts":     counts,
            "terminated": terminated,
            "reason":     reason,
        }), 200

    # ==========================================================
    # 📊 ADMIN ROUTES
    # ==========================================================

    @app.route("/api/admin/exams", methods=["POST"])
    @admin_required
    def create_exam():
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

    @app.route("/api/admin/exams/<int:exam_id>", methods=["GET"])
    @admin_required
    def get_exam_details(exam_id):
        e = Exam.query.get_or_404(exam_id)
        return jsonify({
            "id": e.id, "title": e.title, "description": e.description,
            "duration": e.duration_minutes, "total_marks": e.total_marks,
            "questions": [
                {
                    "id": q.id, "text": q.question_text,
                    "marks": q.marks, "type": q.question_type,
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
        db.session.delete(Exam.query.get_or_404(exam_id))
        db.session.commit()
        return jsonify({"message": "Exam deleted successfully"})

    @app.route("/api/admin/exams")
    @admin_required
    def list_exams():
        return jsonify([
            {"id": e.id, "title": e.title,
             "duration": e.duration_minutes, "total_marks": e.total_marks}
            for e in Exam.query.all()
        ])

    @app.route("/api/admin/exams/<int:exam_id>/questions", methods=["POST"])
    @admin_required
    def add_question(exam_id):
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
        db.session.delete(Question.query.get_or_404(question_id))
        db.session.commit()
        return jsonify({"message": "Question deleted"})

    @app.route("/api/admin/questions/<int:question_id>/options", methods=["POST"])
    @admin_required
    def add_option(question_id):
        d = request.json
        db.session.add(Option(
            question_id = question_id,
            option_text = d["option_text"],
            is_correct  = d.get("is_correct", False),
        ))
        db.session.commit()
        return jsonify({"message": "Option added"})

    @app.route("/api/admin/sessions/active")
    @admin_required
    def active_sessions():
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
        s          = ExamSession.query.get_or_404(session_id)
        s.status   = "terminated"
        s.end_time = datetime.utcnow()
        db.session.commit()
        return jsonify({"message": "Session terminated"})

    @app.route("/api/admin/exams/<int:exam_id>/results")
    @admin_required
    def exam_results(exam_id):
        return jsonify([
            {"student": a.user.username, "score": a.score, "status": a.status}
            for a in UserExam.query.filter_by(exam_id=exam_id).all()
        ])

    @app.route("/api/admin/violations", methods=["GET"])
    @admin_required
    def admin_all_violations():
        session_id       = request.args.get("session_id",  type=int)
        student_id       = request.args.get("student_id",  type=int)
        v_type_filter    = request.args.get("type")
        include_snapshot = request.args.get("include_snapshot", "0") == "1"

        q = ViolationLog.query.join(
            ExamSession, ViolationLog.session_id == ExamSession.id
        ).join(User, ExamSession.user_id == User.id)

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
            {"violation_type": r.violation_type,
             "severity": r.severity, "count": r.count}
            for r in rows
        ]), 200

    @app.route("/api/admin/sessions/<int:session_id>/violations", methods=["GET"])
    @admin_required
    def admin_session_violations(session_id):
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

    @app.route("/api/admin/students/<int:student_id>/violations", methods=["GET"])
    @admin_required
    def admin_student_violations(student_id):
        sessions    = ExamSession.query.filter_by(user_id=student_id).all()
        session_ids = [s.id for s in sessions]
        if not session_ids:
            return jsonify([]), 200

        logs = (
            ViolationLog.query
            .filter(ViolationLog.session_id.in_(session_ids))
            .order_by(ViolationLog.timestamp.desc())
            .all()
        )
        return jsonify([
            {
                "id":             log.id,
                "session_id":     log.session_id,
                "violation_type": log.violation_type,
                "severity":       log.severity,
                "detail":         log.detail,
                "timestamp":      log.timestamp.isoformat(),
            }
            for log in logs
        ]), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)