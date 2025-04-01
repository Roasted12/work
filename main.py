from app import app
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Heart Disease Prediction System...")
    app.run(host="0.0.0.0", port=5000, debug=True)