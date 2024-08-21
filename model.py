import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import joblib
from data_preprocessing import preprocess_data

# Preprocess the data
X_train, X_val, y_train, y_val, scaler = preprocess_data('crypto_data.csv')

# Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and train the model
model = build_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Save the trained model
model.save('crypto_prediction_model.h5')

# Save the fitted scaler
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
