from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = tf.keras.models.load_model('crypto_prediction_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['data']
    data = [float(x) for x in data.split(',')]
    
    # Scale and reshape the data
    scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
    scaled_data = np.reshape(scaled_data, (1, len(scaled_data), 1))
    
    # Make a prediction
    prediction = model.predict(scaled_data)
    
    # Inverse transform the prediction
    prediction = scaler.inverse_transform(prediction).flatten().tolist()  # Convert to list
    
    # Pass the prediction data to the template
    return render_template('result.html', prediction=prediction[0], prediction_data=prediction)

if __name__ == '__main__':
    app.run(debug=True)
