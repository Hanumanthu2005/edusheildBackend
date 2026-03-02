from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# ==========================================================
# USERS
# ==========================================================

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), default='student')  # student / admin
    registered_face_path = db.Column(db.String(256), nullable=True)

    # Relationships
    sessions = db.relationship('ExamSession', backref='student', lazy=True)
    user_exams = db.relationship('UserExam', backref='user', lazy=True)


# ==========================================================
# EXAMS
# ==========================================================

class Exam(db.Model):
    __tablename__ = 'exams'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    duration_minutes = db.Column(db.Integer, nullable=False)
    total_marks = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    questions = db.relationship('Question', backref='exam', cascade="all, delete", lazy=True)
    user_exams = db.relationship('UserExam', backref='exam', lazy=True)


# ==========================================================
# QUESTIONS
# ==========================================================

class Question(db.Model):
    __tablename__ = 'questions'

    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exams.id'), nullable=False)

    question_text = db.Column(db.Text, nullable=False)
    marks = db.Column(db.Integer, default=1)
    question_type = db.Column(db.String(20), default='mcq')  # mcq / descriptive

    # Relationships
    options = db.relationship('Option', backref='question', cascade="all, delete", lazy=True)
    answers = db.relationship('UserAnswer', backref='question', lazy=True)


# ==========================================================
# OPTIONS (for MCQ)
# ==========================================================

class Option(db.Model):
    __tablename__ = 'options'

    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)

    option_text = db.Column(db.String(255), nullable=False)
    is_correct = db.Column(db.Boolean, default=False)


# ==========================================================
# USER EXAM ATTEMPT (Bridge Table)
# ==========================================================

class UserExam(db.Model):
    __tablename__ = 'user_exams'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    exam_id = db.Column(db.Integer, db.ForeignKey('exams.id'), nullable=False)

    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default='in_progress')  # completed / terminated

    score = db.Column(db.Float, default=0.0)

    # Relationships
    answers = db.relationship('UserAnswer', backref='user_exam', cascade="all, delete", lazy=True)


# ==========================================================
# USER ANSWERS
# ==========================================================

class UserAnswer(db.Model):
    __tablename__ = 'user_answers'

    id = db.Column(db.Integer, primary_key=True)

    user_exam_id = db.Column(db.Integer, db.ForeignKey('user_exams.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)

    selected_option_id = db.Column(db.Integer, db.ForeignKey('options.id'), nullable=True)
    descriptive_answer = db.Column(db.Text, nullable=True)

    marks_awarded = db.Column(db.Float, default=0.0)


# ==========================================================
# EXAM SESSION (Proctoring)
# ==========================================================

class ExamSession(db.Model):
    __tablename__ = 'exam_sessions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)

    status = db.Column(db.String(20), default='in_progress')
    total_warnings = db.Column(db.Integer, default=0)

    violations = db.relationship('ViolationLog', backref='session', lazy=True)


# ==========================================================
# VIOLATIONS
# ==========================================================

class ViolationLog(db.Model):
    __tablename__ = 'violation_logs'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('exam_sessions.id'), nullable=False)

    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    violation_type = db.Column(db.String(50), nullable=False)
    severity = db.Column(db.String(20), nullable=False)

    snapshot_path = db.Column(db.String(256), nullable=True)