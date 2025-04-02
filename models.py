from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# Import db from app to avoid circular imports
from app import db

class User(UserMixin, db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with results
    results = db.relationship('Result', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        """Generate hashed password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password matches."""
        return check_password_hash(self.password_hash, password)
        
    def __repr__(self):
        return f"<User {self.username}>"

class Result(db.Model):
    """Model for storing prediction results."""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    filename = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # ML, DL, QML, QNN
    model_name = db.Column(db.String(50), nullable=False)  # Specific model name
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    confusion_matrix_img = db.Column(db.Text, nullable=False)  # Base64 encoded image
    roc_curve_img = db.Column(db.Text, nullable=False)  # Base64 encoded image
    
    # Foreign key to link results to users
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    def __repr__(self):
        return f"<Result {self.session_id}>"

# Tables will be created in the app context in app.py