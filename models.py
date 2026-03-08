"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         EduShield — Database Models                        ║
║                                                                              ║
║  Defines all SQLAlchemy ORM models used by the EduShield proctoring system. ║
║                                                                              ║
║  MODEL SUMMARY:                                                              ║
║    👤  User          — student / admin accounts + registered face path       ║
║    📋  Exam          — exam metadata (title, duration, marks)                ║
║    ❓  Question      — MCQ questions belonging to an exam                    ║
║    🔘  Option        — answer choices for a question (one marked correct)    ║
║    🎓  ExamSession   — proctoring session per student login                  ║
║    🚨  ViolationLog  — individual cheating/anomaly events in a session       ║
║    📝  UserExam      — a student's attempt at a specific exam                ║
║    ✅  UserAnswer    — the option a student selected for each question        ║
║    🧠  FeedbackLog   — admin corrections fed back into the RL optimiser      ║
║                                                                              ║
║  RELATIONSHIPS AT A GLANCE:                                                  ║
║    User ──< ExamSession ──< ViolationLog                                     ║
║    User ──< UserExam    ──< UserAnswer                                       ║
║    Exam ──< Question    ──< Option                                           ║
║    Exam ──< UserExam                                                         ║
║    ExamSession ──── FeedbackLog  (one-to-one)                                ║
║                                                                              ║
║  AFTER ADDING NEW MODELS / COLUMNS run:                                      ║
║    flask db migrate -m "<description>"                                       ║
║    flask db upgrade                                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

# Shared db instance — imported by create_app() and initialised via db.init_app(app)
db = SQLAlchemy()


# ==============================================================================
# 👤  USER
# ==============================================================================

class User(db.Model):
    """
    Represents both students and admins.

    role:
        'student' — can take exams; requires face verification on login.
        'admin'   — manages exams and reviews violations; no face check.

    registered_face_path:
        Filesystem path to the baseline JPEG captured during registration.
        Used by ArcFace to verify identity at login and before each exam.
    """
    __tablename__ = "users"

    id                   = db.Column(db.Integer,     primary_key=True)
    username             = db.Column(db.String(80),  unique=True, nullable=False)
    password_hash        = db.Column(db.String(256), nullable=False)
    registered_face_path = db.Column(db.String(256), nullable=True)
    role                 = db.Column(db.String(20),  default="student")
    # 'student' | 'admin'

    # ── Relationships ──────────────────────────────────────────────────────────
    # user.exam_sessions  →  list[ExamSession]
    exam_sessions = db.relationship(
        "ExamSession",
        backref="student",
        lazy="select",
        cascade="all, delete-orphan",
    )

    # user.exams_taken  →  list[UserExam]
    exams_taken = db.relationship(
        "UserExam",
        backref="user",
        lazy="select",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


# ==============================================================================
# 📋  EXAM
# ==============================================================================

class Exam(db.Model):
    """
    Top-level exam definition created by an admin.

    total_marks is the denominator for grading; individual question marks
    should ideally sum to this value but the system scores as a percentage
    so a mismatch won't cause errors.
    """
    __tablename__ = "exams"

    id               = db.Column(db.Integer,     primary_key=True)
    title            = db.Column(db.String(200), nullable=False)
    description      = db.Column(db.Text,        nullable=True)
    duration_minutes = db.Column(db.Integer,     default=60)
    total_marks      = db.Column(db.Integer,     default=100)
    created_at       = db.Column(db.DateTime,    default=datetime.utcnow)

    # ── Relationships ──────────────────────────────────────────────────────────
    # exam.questions  →  list[Question]
    questions = db.relationship(
        "Question",
        backref="exam",
        lazy="select",
        cascade="all, delete-orphan",
    )

    # exam.attempts  →  list[UserExam]
    attempts = db.relationship(
        "UserExam",
        backref="exam",
        lazy="select",
    )

    def __repr__(self):
        return f"<Exam {self.id}: {self.title}>"


# ==============================================================================
# ❓  QUESTION
# ==============================================================================

class Question(db.Model):
    """
    A single question belonging to an Exam.

    question_type is informational for the frontend ('mcq' is the default);
    grading logic in app.py always compares selected_option_id against the
    Option row where is_correct=True.
    """
    __tablename__ = "questions"

    id            = db.Column(db.Integer,    primary_key=True)
    exam_id       = db.Column(db.Integer,    db.ForeignKey("exams.id"), nullable=False)
    question_text = db.Column(db.Text,       nullable=False)
    marks         = db.Column(db.Integer,    default=1)
    question_type = db.Column(db.String(20), default="mcq")
    # 'mcq' | 'true_false' | (extendable)

    # ── Relationships ──────────────────────────────────────────────────────────
    # question.options  →  list[Option]
    options = db.relationship(
        "Option",
        backref="question",
        lazy="select",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Question {self.id} (exam {self.exam_id})>"


# ==============================================================================
# 🔘  OPTION
# ==============================================================================

class Option(db.Model):
    """
    One answer choice for a Question.  Exactly one Option per Question should
    have is_correct=True; the grading logic in app.py awards full marks only
    when the student's selected_option_id matches that row.
    """
    __tablename__ = "options"

    id          = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey("questions.id"), nullable=False)
    option_text = db.Column(db.Text,    nullable=False)
    is_correct  = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<Option {self.id} correct={self.is_correct}>"


# ==============================================================================
# 🎓  EXAM SESSION  (proctoring session — one per student login block)
# ==============================================================================

class ExamSession(db.Model):
    """
    Tracks a proctoring monitoring block for a single student.

    One ExamSession is created (or reused) each time a student logs in.
    It is the parent for all ViolationLog records captured during that block.

    status lifecycle:
        'in_progress'  →  'terminated'  (auto: threshold breach or phone detected)
        'in_progress'  →  'terminated'  (manual: admin action or browser event)
        'in_progress'  →  'completed'   (normal end — not currently triggered
                                          automatically; kept for future use)

    total_warnings:
        Incremented by _log_violation_to_db() each time a new violation is
        persisted.  Used as a quick summary counter in the admin dashboard.

    NOTE: ExamSession and UserExam are separate by design — proctoring concerns
    (face/object detection) are decoupled from exam-taking concerns (answers/scores).
    They are linked only heuristically via start_time proximity in admin views.
    """
    __tablename__ = "exam_sessions"

    id             = db.Column(db.Integer,    primary_key=True)
    user_id        = db.Column(db.Integer,    db.ForeignKey("users.id"), nullable=False)
    start_time     = db.Column(db.DateTime,   default=datetime.utcnow)
    end_time       = db.Column(db.DateTime,   nullable=True)
    status         = db.Column(db.String(20), default="in_progress")
    # 'in_progress' | 'completed' | 'terminated'

    total_warnings = db.Column(db.Integer, default=0)
    # Incremented each time a violation is written by _log_violation_to_db()

    # ── Relationships ──────────────────────────────────────────────────────────
    # session.violations  →  list[ViolationLog]
    # session.student     →  User     (backref defined on User.exam_sessions)
    # session.feedback    →  FeedbackLog | None  (backref defined on FeedbackLog)
    violations = db.relationship(
        "ViolationLog",
        backref="session",
        lazy="select",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<ExamSession {self.id} user={self.user_id} status={self.status}>"


# ==============================================================================
# 🚨  VIOLATION LOG
# ==============================================================================

class ViolationLog(db.Model):
    """
    Records a single cheating / anomaly event detected during an ExamSession.

    violation_type values (must match TERMINATION_THRESHOLDS keys in app.py):
        phone_detected  — mobile phone visible in frame
        book_detected   — physical book/notes detected
        multiple_faces  — more than one face in frame
        looking_away    — student's gaze strays off-screen
        no_face         — student's face not visible
        tab_switch      — browser tab/window change (client-reported)
        fullscreen_exit — student left fullscreen mode (client-reported)
        loud_noise      — audio amplitude exceeded threshold

    severity:
        'critical' — reserved for future high-confidence detections
        'major'    — default for CV-detected violations
        'minor'    — browser-reported or audio violations

    snapshot_b64:
        Base64-encoded JPEG of the frame at the moment of detection.
        Stored as TEXT; excluded from list API responses by default.
        Fetch by passing ?include_snapshot=1 to the relevant admin endpoints.

    timestamp:
        Indexed to support efficient time-range queries in
        GET /api/violations/latest (last-5-seconds window).
    """
    __tablename__ = "violation_logs"

    id             = db.Column(db.Integer,    primary_key=True)
    session_id     = db.Column(db.Integer,    db.ForeignKey("exam_sessions.id"), nullable=False)
    violation_type = db.Column(db.String(50), nullable=False)
    severity       = db.Column(db.String(20), default="major")
    # 'critical' | 'major' | 'minor'

    detail         = db.Column(db.Text, nullable=True)
    # Human-readable description, e.g. "Mobile phone detected (confidence 78%)"

    snapshot_b64   = db.Column(db.Text, nullable=True)
    # Base64 JPEG — omit from list responses; fetch with ?include_snapshot=1

    timestamp      = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    # Indexed for efficient time-range queries

    def __repr__(self):
        return f"<ViolationLog {self.id} {self.violation_type} session={self.session_id}>"


# ==============================================================================
# 📝  USER EXAM  (a student's attempt at a specific exam)
# ==============================================================================

class UserExam(db.Model):
    """
    Records one student's attempt at one Exam.

    Multiple UserExam rows can exist for the same (user_id, exam_id) pair if
    a student retakes an exam (though the app currently prevents this by
    returning the existing in-progress attempt from /start).

    score:
        Stored as a 0–100 percentage, not raw marks.
        Raw marks = round((score / 100) * exam.total_marks).

    status lifecycle:
        'in_progress'  →  'completed'   (student submits via /submit)
        'in_progress'  →  'terminated'  (violation threshold or browser event)
    """
    __tablename__ = "user_exams"

    id         = db.Column(db.Integer,    primary_key=True)
    user_id    = db.Column(db.Integer,    db.ForeignKey("users.id"),  nullable=False)
    exam_id    = db.Column(db.Integer,    db.ForeignKey("exams.id"),  nullable=False)
    status     = db.Column(db.String(20), default="in_progress")
    # 'in_progress' | 'completed' | 'terminated'

    score      = db.Column(db.Float,    nullable=True)
    # Percentage 0–100; populated on completion via /submit

    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time   = db.Column(db.DateTime, nullable=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    # user_exam.answers  →  list[UserAnswer]
    # user_exam.user     →  User  (backref defined on User.exams_taken)
    # user_exam.exam     →  Exam  (backref defined on Exam.attempts)
    answers = db.relationship(
        "UserAnswer",
        backref="user_exam",
        lazy="select",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return (
            f"<UserExam {self.id} "
            f"user={self.user_id} exam={self.exam_id} status={self.status}>"
        )


# ==============================================================================
# ✅  USER ANSWER
# ==============================================================================

class UserAnswer(db.Model):
    """
    Records which Option a student chose for a specific Question in a UserExam.

    selected_option_id is nullable to support unanswered questions (e.g. if
    a student submits before answering every question).

    marks_awarded:
        Set by the grading logic in submit_exam():
            = question.marks  if selected_option matches the correct Option
            = 0               otherwise
    """
    __tablename__ = "user_answers"

    id                 = db.Column(db.Integer, primary_key=True)
    user_exam_id       = db.Column(db.Integer, db.ForeignKey("user_exams.id"),  nullable=False)
    question_id        = db.Column(db.Integer, db.ForeignKey("questions.id"),   nullable=False)
    selected_option_id = db.Column(db.Integer, db.ForeignKey("options.id"),     nullable=True)
    marks_awarded      = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f"<UserAnswer {self.id} q={self.question_id} marks={self.marks_awarded}>"


# ==============================================================================
# 🧠  FEEDBACK LOG  (admin corrections → RL optimiser input)
# ==============================================================================

class FeedbackLog(db.Model):
    """
    Stores admin feedback on terminated sessions, used as training signal for
    the RL threshold optimiser (core/rl_agent.py).

    is_false_positive:
        True  — system terminated the session incorrectly (student was innocent).
                → RL reward = -1  (loosen detection thresholds)
        False — termination was correct (student was genuinely cheating).
                → RL reward = +1  (reinforce detection thresholds)

    violation_type:
        The primary violation the admin is giving feedback on.  Defaults to
        the most-frequent violation in the session if not explicitly provided.

    thresholds_after:
        JSON-serialised snapshot of the RL model's thresholds after applying
        this feedback step.  Stored as TEXT for auditability.

    unique=True on session_id ensures only one feedback entry per session,
    preventing double-counting of reward signals.

    Migration:
        flask db migrate -m "add feedback_log"
        flask db upgrade
    """
    __tablename__ = "feedback_log"

    id                = db.Column(db.Integer, primary_key=True)
    session_id        = db.Column(
        db.Integer,
        db.ForeignKey("exam_sessions.id"),
        nullable=False,
        unique=True,   # one feedback entry per session
    )
    admin_id          = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    is_false_positive = db.Column(db.Boolean, nullable=False)
    note              = db.Column(db.Text,       nullable=True)
    violation_type    = db.Column(db.String(64), nullable=True)
    thresholds_after  = db.Column(db.Text,       nullable=True)
    # Serialised dict of RL thresholds after this feedback was applied
    created_at        = db.Column(db.DateTime,   default=datetime.utcnow)

    # ── Relationships ──────────────────────────────────────────────────────────
    # feedback_log.session  →  ExamSession
    # feedback_log.admin    →  User
    session = db.relationship(
        "ExamSession",
        backref=db.backref("feedback", uselist=False),
    )
    admin = db.relationship("User", foreign_keys=[admin_id])

    def __repr__(self):
        kind = "FALSE_POS" if self.is_false_positive else "CONFIRMED"
        return f"<FeedbackLog session={self.session_id} type={kind}>"