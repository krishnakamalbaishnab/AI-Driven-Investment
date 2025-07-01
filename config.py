import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

class Config:
    """Base configuration class."""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # Model configuration
    MODEL_PATH = os.environ.get('MODEL_PATH') or str(BASE_DIR / 'crypto_prediction_model.h5')
    SCALER_PATH = os.environ.get('SCALER_PATH') or str(BASE_DIR / 'scaler.pkl')
    SEQUENCE_LENGTH = int(os.environ.get('SEQUENCE_LENGTH', 60))
    
    # Data configuration
    DATA_FILE = os.environ.get('DATA_FILE') or str(BASE_DIR / 'crypto_data.csv')
    BACKUP_DIR = os.environ.get('BACKUP_DIR') or str(BASE_DIR / 'backups')
    
    # API configuration
    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
    DEFAULT_CRYPTO = os.environ.get('DEFAULT_CRYPTO', 'bitcoin')
    DEFAULT_CURRENCY = os.environ.get('DEFAULT_CURRENCY', 'usd')
    DEFAULT_DAYS = int(os.environ.get('DEFAULT_DAYS', 365))
    
    # Request configuration
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))
    RETRY_DELAY = int(os.environ.get('RETRY_DELAY', 5))
    REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', 30))
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE') or str(BASE_DIR / 'app.log')
    
    # Model training configuration
    LSTM_UNITS = int(os.environ.get('LSTM_UNITS', 50))
    DROPOUT_RATE = float(os.environ.get('DROPOUT_RATE', 0.2))
    EPOCHS = int(os.environ.get('EPOCHS', 50))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
    VALIDATION_SPLIT = float(os.environ.get('VALIDATION_SPLIT', 0.2))
    PATIENCE = int(os.environ.get('PATIENCE', 10))
    MIN_DELTA = float(os.environ.get('MIN_DELTA', 0.001))
    
    # Security configuration
    MAX_PRICE_VALUE = float(os.environ.get('MAX_PRICE_VALUE', 1000000))
    MIN_PRICE_VALUE = float(os.environ.get('MIN_PRICE_VALUE', 0))


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    @classmethod
    def validate(cls):
        """Validate production configuration."""
        if not os.environ.get('SECRET_KEY'):
            raise ValueError("SECRET_KEY environment variable must be set in production")


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    SECRET_KEY = 'test-secret-key'


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 