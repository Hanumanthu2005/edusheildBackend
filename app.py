import os
import cv2
import time
import base64
from datetime import timedelta, datetime
from functools import wraps

from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from models import (
    UserAnswer,
    db,
    User,
    ExamSession,
    ViolationLog,
    Exam,
    Question,
    Option,
    UserExam
)

from core.vision import analyze_frame
from core.audio import AudioMonitor
from core.rl_cnn import rl_optimizer
from core.face_auth import verify_identity


last_db_log_time = {}


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # JWT Configuration
    app.config["JWT_SECRET_KEY"] = "super-secret-key"
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=5)

    db.init_app(app)
    CORS(app)
    jwt = JWTManager(app)

    with app.app_context():
        db.create_all()

    audio_monitor = AudioMonitor(threshold=800)
    audio_monitor.start()

    # ==========================================================
    # 🔐 ROLE-BASED DECORATOR
    # ==========================================================

    def admin_required(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            user_id = get_jwt_identity()
            user = User.query.get(user_id)

            if not user or user.role != "admin":
                return jsonify({"error": "Admin access required"}), 403

            return fn(*args, **kwargs)
        return wrapper

    # ==========================================================
    # 🔐 AUTH ROUTES
    # ==========================================================

    @app.route("/api/register", methods=["POST"])
    def register():
        data = request.json

        username = data.get("username")
        password = data.get("password")
        snapshot = data.get("register_snapshot_base64")

        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username already exists"}), 400

        filepath = None
        if snapshot and len(snapshot) > 100:
            try:
                header, encoded = snapshot.split(",", 1)
                img_data = base64.b64decode(encoded)

                os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
                filename = secure_filename(f"{username}_baseline.jpg")
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                with open(filepath, "wb") as f:
                    f.write(img_data)

            except Exception:
                return jsonify({"error": "Failed to process image"}), 400
        else:
            return jsonify({"error": "Face snapshot required"}), 400

        hashed_pw = generate_password_hash(password)

        user = User(
            username=username,
            password_hash=hashed_pw,
            registered_face_path=filepath,
            role="student"
        )

        db.session.add(user)
        db.session.commit()

        return jsonify({"message": "User registered successfully"}), 201

    @app.route("/api/login", methods=["POST"])
    def login():
        data = request.json

        username = data.get("username")
        password = data.get("password")
        snapshot = data.get("live_snapshot_base64")

        user = User.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid credentials"}), 401

        # 🔐 ADMIN LOGIN (No face verification)
        if user.role == "admin":
            access_token = create_access_token(
                identity=str(user.id),  # convert to string
                additional_claims={"role": user.role}
            )

            return jsonify({
                "message": "Admin login successful",
                "token": access_token,
                "role": user.role
            }), 200

        # 👨‍🎓 STUDENT LOGIN (Face verification required)
        if not snapshot:
            return jsonify({"error": "Live face scan required"}), 400

        try:
            header, encoded = snapshot.split(",", 1)
            img_data = base64.b64decode(encoded)

            temp_path = f"temp_login_{username}.jpg"
            with open(temp_path, "wb") as f:
                f.write(img_data)

            verified = verify_identity(user.registered_face_path, temp_path)
            os.remove(temp_path)

            if not verified:
                return jsonify({"error": "Face verification failed"}), 401

        except Exception:
            return jsonify({"error": "Face verification error"}), 400

        # Create Exam Session only for students
        session = ExamSession(user_id=user.id)
        db.session.add(session)
        db.session.commit()

        access_token = create_access_token(
            identity=str(user.id),
            additional_claims={"role": user.role}
        )

        return jsonify({
            "message": "Login successful",
            "token": access_token,
            "role": user.role,
            "session_id": session.id
        }), 200
    
    # ──────────────────────────────────────────────────────────────────
    # Add this route inside create_app() in app.py
    # under the "📌 LOG BROWSER VIOLATIONS" section
    # ──────────────────────────────────────────────────────────────────

    @app.route("/api/violations/latest", methods=["GET"])
    @jwt_required()
    def get_latest_violations():
        """
        Returns violations logged in the last 5 seconds for the
        student's active exam session. The frontend polls this every 3 s
        to drive the live alert UI.
        """
        user_id = get_jwt_identity()

        active_session = (
            ExamSession.query
            .filter_by(user_id=user_id, status="in_progress")
            .order_by(ExamSession.id.desc())
            .first()
        )

        if not active_session:
            return jsonify({"violations": []}), 200

        cutoff = datetime.utcnow() - timedelta(seconds=5)

        recent_logs = (
            ViolationLog.query
            .filter(
                ViolationLog.session_id == active_session.id,
                ViolationLog.timestamp >= cutoff
            )
            .order_by(ViolationLog.timestamp.desc())
            .all()
        )

        # Return unique violation types seen in the last 5 s
        seen = set()
        violations = []
        for log in recent_logs:
            if log.violation_type not in seen:
                seen.add(log.violation_type)
                violations.append(log.violation_type)

        return jsonify({"violations": violations}), 200

    # ==========================================================
    # 🎥 VIDEO STREAM
    # ==========================================================

    def generate_frames():
        camera = cv2.VideoCapture(0)

        while True:
            success, frame = camera.read()
            if not success:
                break

            annotated_frame, violations = analyze_frame(frame)

            audio_flag = audio_monitor.get_violation()
            if audio_flag:
                violations.append(audio_flag)

            now = time.time()

            for v_type in violations:
                if now - last_db_log_time.get(v_type, 0) > 10:
                    try:
                        active_session = ExamSession.query.order_by(
                            ExamSession.id.desc()
                        ).first()

                        if active_session:
                            log = ViolationLog(
                                session_id=active_session.id,
                                violation_type=v_type,
                                severity="major"
                            )
                            db.session.add(log)
                            db.session.commit()

                    except Exception as e:
                        print("DB ERROR:", e)

                    last_db_log_time[v_type] = now

            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    

    # ==========================================================
    # 📚 STUDENT ROUTES
    # ==========================================================

    @app.route("/api/student/exams", methods=["GET"])
    @jwt_required()
    def get_student_exams():
        """Get all available exams for students"""
        exams = Exam.query.all()

        return jsonify([
            {
                "id": e.id,
                "title": e.title,
                "description": e.description,
                "duration": e.duration_minutes,
                "total_marks": e.total_marks,
                "created_at": e.created_at.isoformat() if e.created_at else None
            } for e in exams
        ])

    @app.route("/api/student/exams/<int:exam_id>", methods=["GET"])
    @jwt_required()
    def get_student_exam_details(exam_id):
        """Get detailed exam information for students (without correct answers)"""
        exam = Exam.query.get_or_404(exam_id)

        return jsonify({
            "id": exam.id,
            "title": exam.title,
            "description": exam.description,
            "duration": exam.duration_minutes,
            "total_marks": exam.total_marks,
            "questions": [
                {
                    "id": q.id,
                    "text": q.question_text,
                    "marks": q.marks,
                    "type": q.question_type,
                    "options": [
                        {
                            "id": o.id,
                            "text": o.option_text
                            # Note: is_correct is intentionally excluded for students
                        } for o in q.options
                    ]
                } for q in exam.questions
            ]
        })
    
    # Add this to your STUDENT ROUTES section in app.py

    @app.route("/api/student/exams/<int:exam_id>/start", methods=["POST"])
    @jwt_required()
    def start_exam(exam_id):
        """Start an exam attempt for a student"""
        user_id = get_jwt_identity()
        
        # Check if exam exists
        exam = Exam.query.get_or_404(exam_id)
        
        # Check if user already has an active attempt
        existing_attempt = UserExam.query.filter_by(
            user_id=user_id,
            exam_id=exam_id,
            status='in_progress'
        ).first()
        
        if existing_attempt:
            return jsonify({
                "user_exam_id": existing_attempt.id,
                "message": "Exam already in progress"
            }), 200
        
        # Create new exam attempt
        user_exam = UserExam(
            user_id=user_id,
            exam_id=exam_id,
            status='in_progress'
        )
        
        db.session.add(user_exam)
        db.session.commit()
        
        return jsonify({
            "user_exam_id": user_exam.id,
            "message": "Exam started successfully"
        }), 201

    @app.route("/api/student/exams/<int:exam_id>/submit", methods=["POST"])
    @jwt_required()
    def submit_exam(exam_id):
        """Submit exam answers"""
        user_id = get_jwt_identity()
        data = request.json
        
        answers = data.get("answers", {})  # {question_id: option_id}
        user_exam_id = data.get("user_exam_id")
        
        # Get the user exam attempt
        user_exam = UserExam.query.filter_by(
            id=user_exam_id,
            user_id=user_id,
            exam_id=exam_id
        ).first_or_404()
        
        # Save answers and calculate score
        total_score = 0
        total_marks = 0
        
        for question_id_str, option_id in answers.items():
            question_id = int(question_id_str)
            question = Question.query.get(question_id)
            
            if not question:
                continue
                
            total_marks += question.marks
            
            # Check if answer is correct
            correct_option = Option.query.filter_by(
                question_id=question_id,
                is_correct=True
            ).first()
            
            marks_awarded = 0
            if correct_option and correct_option.id == option_id:
                marks_awarded = question.marks
                total_score += marks_awarded
            
            # Save user answer
            user_answer = UserAnswer(
                user_exam_id=user_exam.id,
                question_id=question_id,
                selected_option_id=option_id,
                marks_awarded=marks_awarded
            )
            db.session.add(user_answer)
        
        # Update user exam
        user_exam.end_time = datetime.utcnow()
        user_exam.status = 'completed'
        user_exam.score = (total_score / total_marks * 100) if total_marks > 0 else 0
        
        db.session.commit()
        
        return jsonify({
            "message": "Exam submitted successfully",
            "score": user_exam.score,
            "total_marks": total_marks,
            "marks_obtained": total_score
        }), 200
    
    # ==========================================================
    # 📊 STUDENT RESULTS ROUTE
    # Add this inside the STUDENT ROUTES section of app.py
    # ==========================================================

    @app.route("/api/student/results", methods=["GET"])
    @jwt_required()
    def get_student_results():
        """Get all completed exam results for the logged-in student"""
        user_id = get_jwt_identity()

        completed_exams = UserExam.query.filter_by(
            user_id=user_id,
            status="completed"
        ).all()

        results = []
        for attempt in completed_exams:
            exam = attempt.exam
            total_questions = len(exam.questions)
            raw_score = attempt.score  # This is already a percentage (0–100)

            # Convert percentage score back to marks obtained
            marks_obtained = round((raw_score / 100) * exam.total_marks)

            # Determine pass/fail (pass threshold: 50%)
            status = "Pass" if raw_score >= 50 else "Fail"

            # Format date
            date_str = attempt.end_time.strftime("%Y-%m-%d") if attempt.end_time else (
                attempt.start_time.strftime("%Y-%m-%d") if attempt.start_time else "N/A"
            )

            # Calculate duration taken
            if attempt.start_time and attempt.end_time:
                duration_mins = int((attempt.end_time - attempt.start_time).total_seconds() / 60)
                duration_str = f"{duration_mins} mins"
            else:
                duration_str = f"{exam.duration_minutes} mins"

            results.append({
                "id": attempt.id,
                "examName": exam.title,
                "code": f"EXAM{exam.id:03d}",      # e.g. EXAM001, EXAM002
                "score": marks_obtained,
                "total": exam.total_marks,
                "status": status,
                "date": date_str,
                "duration": duration_str,
                "questions": total_questions
            })

        # Sort by date descending (most recent first)
        results.sort(key=lambda x: x["date"], reverse=True)

        return jsonify(results), 200

    # ==========================================================
    # 📊 ADMIN ROUTES
    # ==========================================================

    @app.route("/api/admin/exams", methods=["POST"])
    @admin_required
    def create_exam():
        data = request.json

        exam = Exam(
            title=data["title"],
            description=data.get("description"),
            duration_minutes=data["duration"],
            total_marks=data["total_marks"]
        )

        db.session.add(exam)
        db.session.commit()
        print(Exam.query.all())

        return jsonify({"message": "Exam created", "exam_id": exam.id})
    
    @app.route("/api/admin/exams/<int:exam_id>", methods=["PUT"])
    @admin_required
    def update_exam(exam_id):
        exam = Exam.query.get_or_404(exam_id)
        data = request.json

        exam.title = data.get("title", exam.title)
        exam.description = data.get("description", exam.description)
        exam.duration_minutes = data.get("duration", exam.duration_minutes)
        exam.total_marks = data.get("total_marks", exam.total_marks)

        db.session.commit()

        return jsonify({"message": "Exam updated successfully"})
    
    @app.route("/api/admin/exams/<int:exam_id>", methods=["DELETE"])
    @admin_required
    def delete_exam(exam_id):
        exam = Exam.query.get_or_404(exam_id)

        db.session.delete(exam)
        db.session.commit()

        return jsonify({"message": "Exam deleted successfully"})
    
    @app.route("/api/admin/exams/<int:exam_id>/questions", methods=["POST"])
    @admin_required
    def add_question(exam_id):
        data = request.json

        question = Question(
            exam_id=exam_id,
            question_text=data["question_text"],
            marks=data.get("marks", 1),
            question_type=data.get("type", "mcq")
        )

        db.session.add(question)
        db.session.commit()

        return jsonify({"message": "Question added", "question_id": question.id})
    
    @app.route("/api/admin/questions/<int:question_id>", methods=["DELETE"])
    @admin_required
    def delete_question(question_id):
        question = Question.query.get_or_404(question_id)

        db.session.delete(question)
        db.session.commit()

        return jsonify({"message": "Question deleted"})
    
    @app.route("/api/admin/questions/<int:question_id>/options", methods=["POST"])
    @admin_required
    def add_option(question_id):
        data = request.json

        option = Option(
            question_id=question_id,
            option_text=data["option_text"],
            is_correct=data.get("is_correct", False)
        )

        db.session.add(option)
        db.session.commit()

        return jsonify({"message": "Option added"})
    
    @app.route("/api/admin/exams/<int:exam_id>")
    @admin_required
    def get_exam_details(exam_id):
        exam = Exam.query.get_or_404(exam_id)

        return jsonify({
            "id": exam.id,
            "title": exam.title,
            "description": exam.description,
            "duration": exam.duration_minutes,
            "total_marks": exam.total_marks,
            "questions": [
                {
                    "id": q.id,
                    "text": q.question_text,
                    "marks": q.marks,
                    "type": q.question_type,
                    "options": [
                        {
                            "id": o.id,
                            "text": o.option_text,
                            "is_correct": o.is_correct
                        } for o in q.options
                    ]
                } for q in exam.questions
            ]
        })
    
    @app.route("/api/admin/sessions/<int:session_id>/terminate", methods=["POST"])
    @admin_required
    def terminate_session(session_id):
        session = ExamSession.query.get_or_404(session_id)

        session.status = "terminated"
        session.end_time = datetime.utcnow()

        db.session.commit()

        return jsonify({"message": "Session terminated"})

    @app.route("/api/admin/exams")
    @admin_required
    def list_exams():
        exams = Exam.query.all()

        return jsonify([
            {
                "id": e.id,
                "title": e.title,
                "duration": e.duration_minutes,
                "total_marks": e.total_marks
            } for e in exams
        ])

    @app.route("/api/admin/sessions/active")
    @admin_required
    def active_sessions():
        sessions = ExamSession.query.filter_by(status="in_progress").all()

        return jsonify([
            {
                "session_id": s.id,
                "student": s.student.username,
                "warnings": s.total_warnings,
                "started_at": s.start_time
            } for s in sessions
        ])

    @app.route("/api/admin/exams/<int:exam_id>/results")
    @admin_required
    def exam_results(exam_id):
        attempts = UserExam.query.filter_by(exam_id=exam_id).all()

        return jsonify([
            {
                "student": attempt.user.username,
                "score": attempt.score,
                "status": attempt.status
            } for attempt in attempts
        ])

    # ==========================================================
    # 📌 LOG BROWSER VIOLATIONS
    # ==========================================================

    @app.route("/api/log_violation", methods=["POST"])
    @jwt_required()
    def log_violation():
        data = request.json
        v_type = data.get("type")

        current_user = get_jwt_identity()

        active_session = ExamSession.query.filter_by(
            user_id=current_user
        ).order_by(ExamSession.id.desc()).first()

        if active_session:
            log = ViolationLog(
                session_id=active_session.id,
                violation_type=v_type,
                severity="minor"
            )
            db.session.add(log)
            db.session.commit()

        return jsonify({"message": "Violation logged"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True)