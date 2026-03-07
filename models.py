from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

"""
NEW MODEL — Add this class to your existing models.py
======================================================
Also run:  flask db migrate -m "add feedback_log"
           flask db upgrade
"""

# from datetime import datetime
# from .db import db   # adjust import to match your models.py pattern


class FeedbackLog(db.Model):
    __tablename__ = "feedback_log"

    id                = db.Column(db.Integer, primary_key=True)
    session_id        = db.Column(db.Integer, db.ForeignKey("exam_sessions.id"), nullable=False, unique=True)
    admin_id          = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    is_false_positive = db.Column(db.Boolean, nullable=False)
    note              = db.Column(db.Text,    nullable=True)
    violation_type    = db.Column(db.String(64), nullable=True)
    thresholds_after  = db.Column(db.Text, nullable=True)
    created_at        = db.Column(db.DateTime, default=datetime.utcnow)

    session = db.relationship("ExamSession", backref=db.backref("feedback", uselist=False))
    admin   = db.relationship("User", foreign_keys=[admin_id])

    def __repr__(self):
        kind = "FALSE_POS" if self.is_false_positive else "CONFIRMED"
        return f"<FeedbackLog session={self.session_id} type={kind}>"


# ==========================================================
# 👤 USER
# ==========================================================

class User(db.Model):
    __tablename__ = 'users'

    id                   = db.Column(db.Integer, primary_key=True)
    username             = db.Column(db.String(80),  unique=True, nullable=False)
    password_hash        = db.Column(db.String(256), nullable=False)
    registered_face_path = db.Column(db.String(256), nullable=True)
    role                 = db.Column(db.String(20),  default='student')

    # user.exam_sessions  →  list[ExamSession]
    # user.exams_taken    →  list[UserExam]
    exam_sessions = db.relationship(
        'ExamSession', backref='student', lazy='select',
        cascade='all, delete-orphan'
    )
    exams_taken = db.relationship(
        'UserExam', backref='user', lazy='select',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f'<User {self.username} ({self.role})>'


# ==========================================================
# 📋 EXAM + QUESTIONS + OPTIONS
# ==========================================================

class Exam(db.Model):
    __tablename__ = 'exams'

    id               = db.Column(db.Integer, primary_key=True)
    title            = db.Column(db.String(200), nullable=False)
    description      = db.Column(db.Text,        nullable=True)
    duration_minutes = db.Column(db.Integer,     default=60)
    total_marks      = db.Column(db.Integer,     default=100)
    created_at       = db.Column(db.DateTime,    default=datetime.utcnow)

    # exam.questions  →  list[Question]
    # exam.attempts   →  list[UserExam]
    questions = db.relationship(
        'Question', backref='exam', lazy='select',
        cascade='all, delete-orphan'
    )
    attempts = db.relationship(
        'UserExam', backref='exam', lazy='select'
    )

    def __repr__(self):
        return f'<Exam {self.id}: {self.title}>'


class Question(db.Model):
    __tablename__ = 'questions'

    id            = db.Column(db.Integer, primary_key=True)
    exam_id       = db.Column(db.Integer, db.ForeignKey('exams.id'), nullable=False)
    question_text = db.Column(db.Text,        nullable=False)
    marks         = db.Column(db.Integer,     default=1)
    question_type = db.Column(db.String(20),  default='mcq')

    # question.options  →  list[Option]
    options = db.relationship(
        'Option', backref='question', lazy='select',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f'<Question {self.id} (exam {self.exam_id})>'


class Option(db.Model):
    __tablename__ = 'options'

    id          = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)
    option_text = db.Column(db.Text,    nullable=False)
    is_correct  = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<Option {self.id} correct={self.is_correct}>'


# ==========================================================
# 🎓 EXAM SESSION  (one per student login / proctoring block)
# ==========================================================

class ExamSession(db.Model):
    __tablename__ = 'exam_sessions'

    id             = db.Column(db.Integer,    primary_key=True)
    user_id        = db.Column(db.Integer,    db.ForeignKey('users.id'), nullable=False)
    start_time     = db.Column(db.DateTime,   default=datetime.utcnow)
    end_time       = db.Column(db.DateTime,   nullable=True)
    status         = db.Column(db.String(20), default='in_progress')
    # 'in_progress' | 'completed' | 'terminated'

    total_warnings = db.Column(db.Integer,    default=0)
    # incremented each time a new unique violation type is logged from a frame

    # session.violations  →  list[ViolationLog]
    # session.student     →  User   (backref defined on User.exam_sessions)
    violations = db.relationship(
        'ViolationLog', backref='session', lazy='select',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f'<ExamSession {self.id} user={self.user_id} status={self.status}>'


# ==========================================================
# 🚨 VIOLATION LOG
# ==========================================================

class ViolationLog(db.Model):
    __tablename__ = 'violation_logs'

    id             = db.Column(db.Integer,    primary_key=True)
    session_id     = db.Column(db.Integer,    db.ForeignKey('exam_sessions.id'), nullable=False)

    violation_type = db.Column(db.String(50), nullable=False)
    # Normalised names — must match TERM_THRESHOLDS keys in app.py:
    #   phone_detected | book_detected | multiple_faces | looking_away
    #   no_face        | tab_switch    | fullscreen_exit | loud_noise

    severity = db.Column(db.String(20), default='major')
    # 'critical' | 'major' | 'minor'

    detail = db.Column(db.Text, nullable=True)
    # Human-readable description, e.g. "Mobile phone detected (confidence 78%)"

    snapshot_b64 = db.Column(db.Text, nullable=True)
    # Base64-encoded JPEG of the frame at the moment of detection.
    # Stored as TEXT; omit from list responses, fetch with ?include_snapshot=1

    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    # Indexed for efficient time-range queries in /api/violations/latest

    def __repr__(self):
        return f'<ViolationLog {self.id} {self.violation_type} session={self.session_id}>'


# ==========================================================
# 📝 USER EXAM ATTEMPT  (one per student per exam attempt)
# ==========================================================

class UserExam(db.Model):
    __tablename__ = 'user_exams'

    id         = db.Column(db.Integer,    primary_key=True)
    user_id    = db.Column(db.Integer,    db.ForeignKey('users.id'),  nullable=False)
    exam_id    = db.Column(db.Integer,    db.ForeignKey('exams.id'),  nullable=False)
    status     = db.Column(db.String(20), default='in_progress')
    # 'in_progress' | 'completed' | 'terminated'

    score      = db.Column(db.Float,    nullable=True)
    # Percentage 0–100; set on completion

    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time   = db.Column(db.DateTime, nullable=True)

    # user_exam.answers  →  list[UserAnswer]
    # user_exam.user     →  User   (backref defined on User.exams_taken)
    # user_exam.exam     →  Exam   (backref defined on Exam.attempts)
    answers = db.relationship(
        'UserAnswer', backref='user_exam', lazy='select',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f'<UserExam {self.id} user={self.user_id} exam={self.exam_id} status={self.status}>'


# ==========================================================
# ✅ USER ANSWER
# ==========================================================

class UserAnswer(db.Model):
    __tablename__ = 'user_answers'

    id                 = db.Column(db.Integer, primary_key=True)
    user_exam_id       = db.Column(db.Integer, db.ForeignKey('user_exams.id'),  nullable=False)
    question_id        = db.Column(db.Integer, db.ForeignKey('questions.id'),   nullable=False)
    selected_option_id = db.Column(db.Integer, db.ForeignKey('options.id'),     nullable=True)
    marks_awarded      = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f'<UserAnswer {self.id} q={self.question_id} marks={self.marks_awarded}>'