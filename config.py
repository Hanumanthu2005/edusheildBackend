import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Secret key for session management and CSRF protection
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'super-secret-cheatguardx-key'
    
    # SQLAlchemy Database URI (Using SQLite for rapid prototyping)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(BASE_DIR, 'cheatguardx.db')
    
    # Disable tracking modifications to save resources
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload paths (for storing registered face images)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads', 'faces')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload