import os
import uuid
import logging
import io
import base64
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize SQLAlchemy
db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Define models
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

# Import ML and DL models
from ml_models import train_logistic_regression, train_decision_tree, train_random_forest, train_svm, train_knn
from dl_models import train_dense_network_1layer, train_dense_network_2layer, train_dense_network_3layer
from quantum_models import train_variational_classifier, train_quantum_neural_network
from utils import calculate_metrics, generate_confusion_matrix_plot, generate_roc_curve_plot

# Define the expected columns for the heart disease dataset
EXPECTED_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Dictionary mapping model types to their training functions
MODEL_FUNCTIONS = {
    'logistic_regression': train_logistic_regression,
    'decision_tree': train_decision_tree,
    'random_forest': train_random_forest,
    'svm': train_svm,
    'knn': train_knn,
    'dense_1layer': train_dense_network_1layer,
    'dense_2layer': train_dense_network_2layer,
    'dense_3layer': train_dense_network_3layer,
    'variational_classifier': train_variational_classifier,
    'quantum_neural_network': train_quantum_neural_network
}

# Model category mapping
MODEL_CATEGORIES = {
    'logistic_regression': 'ML',
    'decision_tree': 'ML',
    'random_forest': 'ML',
    'svm': 'ML',
    'knn': 'ML',
    'dense_1layer': 'DL',
    'dense_2layer': 'DL',
    'dense_3layer': 'DL',
    'variational_classifier': 'QML',
    'quantum_neural_network': 'QNN'
}

# Routes
@app.route('/')
def index():
    """Render the main page with the upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process the uploaded file and chosen model to make predictions."""
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Check if model was selected
    model_type = request.form.get('model_type')
    if not model_type or model_type not in MODEL_FUNCTIONS:
        flash('Invalid model selection')
        return redirect(request.url)
    
    # Read and validate the uploaded CSV file
    try:
        df = pd.read_csv(file)
        
        # Check if the dataframe has the expected columns
        if not all(col in df.columns for col in EXPECTED_COLUMNS):
            flash(f'Invalid CSV format. Expected columns: {", ".join(EXPECTED_COLUMNS)}')
            return redirect(request.url)
        
        # Extract features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Train the selected model and get predictions
        logger.info(f"Training {model_type} model...")
        train_func = MODEL_FUNCTIONS[model_type]
        model, y_pred, y_prob, X_test, y_test = train_func(X, y)
        
        # Calculate evaluation metrics
        logger.info("Calculating evaluation metrics...")
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Generate plots
        logger.info("Generating visualization plots...")
        confusion_matrix_img = generate_confusion_matrix_plot(y_test, y_pred)
        roc_curve_img = generate_roc_curve_plot(y_test, y_prob)
        
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Save results to the database
        logger.info("Saving results to database...")
        result = Result(
            session_id=session_id,
            filename=secure_filename(file.filename),
            model_type=MODEL_CATEGORIES[model_type],
            model_name=model_type,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            confusion_matrix_img=confusion_matrix_img,
            roc_curve_img=roc_curve_img
        )
        
        db.session.add(result)
        db.session.commit()
        
        # Redirect to result page
        logger.info(f"Prediction complete. Redirecting to results page with session ID: {session_id}")
        return redirect(url_for('show_result', session_id=session_id))
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        flash(f'Error processing file: {str(e)}')
        return redirect(request.url)

@app.route('/result/<session_id>')
def show_result(session_id):
    """Display the prediction results for the given session ID."""
    try:
        # Retrieve the result from the database
        result = Result.query.filter_by(session_id=session_id).first()
        
        if not result:
            flash('Result not found')
            return redirect(url_for('index'))
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        logger.error(f"Error displaying result: {str(e)}")
        flash(f'Error displaying result: {str(e)}')
        return redirect(url_for('index'))

@app.route('/export/<session_id>')
def export_result(session_id):
    """Export the results as CSV."""
    try:
        # Retrieve the result from the database
        result = Result.query.filter_by(session_id=session_id).first()
        
        if not result:
            flash('Result not found')
            return redirect(url_for('index'))
        
        # Create a DataFrame from the result
        result_df = pd.DataFrame({
            'Session ID': [result.session_id],
            'Timestamp': [result.timestamp],
            'Filename': [result.filename],
            'Model Type': [result.model_type],
            'Model Name': [result.model_name],
            'Accuracy': [result.accuracy],
            'Precision': [result.precision],
            'Recall': [result.recall],
            'F1 Score': [result.f1_score]
        })
        
        # Convert to CSV
        csv_data = result_df.to_csv(index=False)
        
        # Create a downloadable file
        buffer = io.StringIO()
        buffer.write(csv_data)
        buffer.seek(0)
        
        return send_file(
            io.BytesIO(buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'heart_disease_prediction_{session_id}.csv'
        )
    
    except Exception as e:
        logger.error(f"Error exporting result: {str(e)}")
        flash(f'Error exporting result: {str(e)}')
        return redirect(url_for('index'))

@app.route('/history')
def history():
    """Show the history of all prediction results."""
    try:
        # Retrieve all results from the database
        results = Result.query.order_by(Result.timestamp.desc()).all()
        return render_template('history.html', results=results)
    
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        flash(f'Error retrieving history: {str(e)}')
        return redirect(url_for('index'))

@app.route('/plot/<session_id>/<plot_type>')
def get_plot(session_id, plot_type):
    """Serve the stored plot images."""
    try:
        # Retrieve the result from the database
        result = Result.query.filter_by(session_id=session_id).first()
        
        if not result:
            return "Plot not found", 404
        
        if plot_type == 'confusion_matrix':
            return send_file(
                io.BytesIO(base64.b64decode(result.confusion_matrix_img)),
                mimetype='image/png'
            )
        elif plot_type == 'roc_curve':
            return send_file(
                io.BytesIO(base64.b64decode(result.roc_curve_img)),
                mimetype='image/png'
            )
        else:
            return "Invalid plot type", 400
    
    except Exception as e:
        logger.error(f"Error retrieving plot: {str(e)}")
        return "Error retrieving plot", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)