import os
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import numpy as np
import tensorflow as tf
import joblib
from werkzeug.exceptions import BadRequest
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Configuration
class Config:
    MODEL_PATH = 'crypto_prediction_model.h5'
    SCALER_PATH = 'scaler.pkl'
    SEQUENCE_LENGTH = 60
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

# Global variables for model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load the trained model and scaler with error handling."""
    global model, scaler
    
    try:
        if os.path.exists(Config.MODEL_PATH):
            model = tf.keras.models.load_model(Config.MODEL_PATH)
            logger.info(f"Model loaded successfully from {Config.MODEL_PATH}")
        else:
            logger.error(f"Model file not found: {Config.MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {Config.MODEL_PATH}")
            
        if os.path.exists(Config.SCALER_PATH):
            scaler = joblib.load(Config.SCALER_PATH)
            logger.info(f"Scaler loaded successfully from {Config.SCALER_PATH}")
        else:
            logger.error(f"Scaler file not found: {Config.SCALER_PATH}")
            raise FileNotFoundError(f"Scaler file not found: {Config.SCALER_PATH}")
            
    except Exception as e:
        logger.error(f"Error loading model or scaler: {str(e)}")
        raise

def validate_input_data(data_string):
    """Validate and parse input data string."""
    try:
        # Remove any extra whitespace and split by comma
        data_string = data_string.strip()
        if not data_string:
            raise ValueError("Input data cannot be empty")
        
        # Split by comma and convert to float
        data_list = [x.strip() for x in data_string.split(',')]
        
        # Remove empty strings
        data_list = [x for x in data_list if x]
        
        if len(data_list) != Config.SEQUENCE_LENGTH:
            raise ValueError(f"Expected {Config.SEQUENCE_LENGTH} values, got {len(data_list)}")
        
        # Convert to float and validate range
        data = []
        for i, value_str in enumerate(data_list):
            try:
                value = float(value_str)
                if value < 0:
                    raise ValueError(f"Price values must be positive. Invalid value at position {i+1}: {value}")
                if value > 1000000:  # Reasonable upper limit
                    raise ValueError(f"Price value seems too high. Invalid value at position {i+1}: {value}")
                data.append(value)
            except ValueError as e:
                if "could not convert" in str(e):
                    raise ValueError(f"Invalid number format at position {i+1}: '{value_str}'")
                raise
        
        return data
        
    except Exception as e:
        logger.error(f"Input validation error: {str(e)}")
        raise

def make_prediction(data):
    """Make prediction with error handling."""
    try:
        if model is None or scaler is None:
            raise RuntimeError("Model or scaler not loaded")
        
        # Scale and reshape the data
        data_array = np.array(data).reshape(-1, 1)
        scaled_data = scaler.transform(data_array)
        scaled_data = np.reshape(scaled_data, (1, len(scaled_data), 1))
        
        # Make prediction
        prediction = model.predict(scaled_data, verbose=0)
        
        # Inverse transform the prediction
        prediction_original = scaler.inverse_transform(prediction)
        
        return float(prediction_original[0][0])
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

@app.route('/')
def index():
    """Render the main page."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return render_template('error.html', error="Unable to load the page"), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get form data
        data_string = request.form.get('data', '').strip()
        
        if not data_string:
            flash('Please enter price data', 'error')
            return redirect(url_for('index'))
        
        # Validate input data
        try:
            data = validate_input_data(data_string)
        except ValueError as e:
            flash(f'Input validation error: {str(e)}', 'error')
            return redirect(url_for('index'))
        
        # Make prediction
        try:
            prediction = make_prediction(data)
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'error')
            return redirect(url_for('index'))
        
        # Format prediction for display
        prediction_formatted = round(prediction, 2)
        
        logger.info(f"Successful prediction: {prediction_formatted}")
        
        return render_template('result.html', 
                             prediction=prediction_formatted, 
                             prediction_data=[prediction_formatted])
        
    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        if 'prices' not in data:
            return jsonify({'error': 'Missing required field: prices'}), 400
        
        prices = data['prices']
        
        if not isinstance(prices, list):
            return jsonify({'error': 'Prices must be a list'}), 400
        
        if len(prices) != Config.SEQUENCE_LENGTH:
            return jsonify({'error': f'Expected {Config.SEQUENCE_LENGTH} prices, got {len(prices)}'}), 400
        
        # Validate all prices are numbers
        try:
            prices = [float(p) for p in prices]
        except (ValueError, TypeError):
            return jsonify({'error': 'All prices must be valid numbers'}), 400
        
        # Make prediction
        prediction = make_prediction(prices)
        
        return jsonify({
            'prediction': round(prediction, 2),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            return jsonify({'status': 'unhealthy', 'reason': 'Model or scaler not loaded'}), 503
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'unhealthy', 'reason': str(e)}), 503

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('error.html', error="Internal server error"), 500

@app.errorhandler(BadRequest)
def bad_request_error(error):
    """Handle 400 errors."""
    return render_template('error.html', error="Bad request"), 400

# Initialize the application
def create_app():
    """Application factory."""
    try:
        load_model_and_scaler()
        logger.info("Application initialized successfully")
        return app
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        load_model_and_scaler()
        app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        exit(1)
