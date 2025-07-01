import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import numpy as np
from data_preprocessing import preprocess_data
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class ModelConfig:
    DATA_FILE = 'crypto_data.csv'
    MODEL_PATH = 'crypto_prediction_model.h5'
    SCALER_PATH = 'scaler.pkl'
    SEQUENCE_LENGTH = 60
    LSTM_UNITS = 50
    DROPOUT_RATE = 0.2
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    PATIENCE = 10
    MIN_DELTA = 0.001

def build_model(input_shape):
    """Build and compile the LSTM model with improved architecture."""
    try:
        model = Sequential([
            LSTM(units=ModelConfig.LSTM_UNITS, 
                 return_sequences=True, 
                 input_shape=input_shape,
                 name='lstm_1'),
            Dropout(ModelConfig.DROPOUT_RATE, name='dropout_1'),
            
            LSTM(units=ModelConfig.LSTM_UNITS, 
                 return_sequences=True,
                 name='lstm_2'),
            Dropout(ModelConfig.DROPOUT_RATE, name='dropout_2'),
            
            LSTM(units=ModelConfig.LSTM_UNITS, 
                 return_sequences=False,
                 name='lstm_3'),
            Dropout(ModelConfig.DROPOUT_RATE, name='dropout_3'),
            
            Dense(units=25, activation='relu', name='dense_1'),
            Dropout(ModelConfig.DROPOUT_RATE, name='dropout_4'),
            Dense(units=1, name='output')
        ])
        
        # Compile with improved optimizer settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
        )
        
        logger.info("Model built successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error building model: {str(e)}")
        raise

def create_callbacks():
    """Create callbacks for training."""
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=ModelConfig.PATIENCE,
            min_delta=ModelConfig.MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            ModelConfig.MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    return callbacks

def plot_training_history(history):
    """Plot training history."""
    try:
        plt.figure(figsize=(15, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot training & validation MAE
        plt.subplot(1, 3, 2)
        plt.plot(history.history['mean_absolute_error'], label='Training MAE')
        plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot training & validation MAPE
        plt.subplot(1, 3, 3)
        plt.plot(history.history['mean_absolute_percentage_error'], label='Training MAPE')
        plt.plot(history.history['val_mean_absolute_percentage_error'], label='Validation MAPE')
        plt.title('Mean Absolute Percentage Error')
        plt.ylabel('MAPE')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Training history plot saved as 'training_history.png'")
        
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")

def evaluate_model(model, X_val, y_val, scaler):
    """Evaluate the trained model."""
    try:
        # Make predictions
        predictions = model.predict(X_val, verbose=0)
        
        # Calculate metrics using TensorFlow operations
        mse = tf.reduce_mean(tf.square(y_val - predictions)).numpy()
        mae = tf.reduce_mean(tf.abs(y_val - predictions)).numpy()
        mape = tf.reduce_mean(tf.abs((y_val - predictions) / y_val) * 100).numpy()
        
        # Convert back to original scale for interpretable metrics
        y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1))
        predictions_original = scaler.inverse_transform(predictions)
        
        rmse_original = np.sqrt(np.mean((y_val_original - predictions_original) ** 2))
        mae_original = np.mean(np.abs(y_val_original - predictions_original))
        
        logger.info("=== Model Evaluation Results ===")
        logger.info(f"MSE (scaled): {mse:.6f}")
        logger.info(f"MAE (scaled): {mae:.6f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info(f"RMSE (original scale): ${rmse_original:.2f}")
        logger.info(f"MAE (original scale): ${mae_original:.2f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'rmse_original': rmse_original,
            'mae_original': mae_original
        }
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def main():
    """Main training function."""
    try:
        logger.info("Starting model training process...")
        
        # Check if data file exists
        if not os.path.exists(ModelConfig.DATA_FILE):
            logger.error(f"Data file not found: {ModelConfig.DATA_FILE}")
            logger.info("Please run data_collection.py first to generate the data file.")
            return
        
        # Preprocess the data
        logger.info("Preprocessing data...")
        X_train, X_val, y_train, y_val, scaler = preprocess_data(
            ModelConfig.DATA_FILE, 
            seq_length=ModelConfig.SEQUENCE_LENGTH
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        # Build the model
        logger.info("Building model...")
        input_shape = (X_train.shape[1], 1)
        model = build_model(input_shape)
        
        # Print model summary
        model.summary()
        
        # Create callbacks
        callbacks = create_callbacks()
        
        # Train the model
        logger.info("Starting training...")
        history = model.fit(
            X_train, y_train,
            epochs=ModelConfig.EPOCHS,
            batch_size=ModelConfig.BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Load the best model (saved by ModelCheckpoint)
        model = tf.keras.models.load_model(ModelConfig.MODEL_PATH)
        
        # Evaluate the model
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, X_val, y_val, scaler)
        
        # Plot training history
        plot_training_history(history)
        
        # Save the scaler
        joblib.dump(scaler, ModelConfig.SCALER_PATH)
        logger.info(f"Scaler saved to {ModelConfig.SCALER_PATH}")
        
        logger.info("Model training completed successfully!")
        logger.info(f"Model saved to {ModelConfig.MODEL_PATH}")
        
        return model, scaler, metrics
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        main()
        
    except Exception as e:
        logger.error(f"Failed to complete training: {str(e)}")
        exit(1)
