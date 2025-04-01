from datetime import datetime
from sqlalchemy.orm import DeclarativeBase
from flask_sqlalchemy import SQLAlchemy

# Define base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy
db = SQLAlchemy(model_class=Base)

# Import this after db initialization to avoid circular imports
from app import app
db.init_app(app)

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

    def __repr__(self):
        return f"<Result {self.session_id}>"

# Create tables
with app.app_context():
    db.create_all()