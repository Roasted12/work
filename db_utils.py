from app import db
from models import Result
from sqlalchemy import desc

def save_result(session_id, filename, model_type, model_name, accuracy, 
                precision, recall, f1_score, confusion_matrix_img, roc_curve_img):
    """Save prediction results to the database."""
    result = Result(
        session_id=session_id,
        filename=filename,
        model_type=model_type,
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        confusion_matrix_img=confusion_matrix_img,
        roc_curve_img=roc_curve_img
    )
    
    db.session.add(result)
    db.session.commit()
    
    return result.id

def get_all_results():
    """Retrieve all results from the database, ordered by timestamp (newest first)."""
    return Result.query.order_by(desc(Result.timestamp)).all()

def get_result_by_id(session_id):
    """Retrieve a specific result by its session ID."""
    return Result.query.filter_by(session_id=session_id).first()
