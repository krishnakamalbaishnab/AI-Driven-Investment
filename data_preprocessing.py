import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(csv_file, seq_length=60):
    data = pd.read_csv(csv_file)
    
    # Normalize the price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['price'].values.reshape(-1, 1))

    # Convert data to sequences
    def create_sequences(data, seq_length):
        sequences = []
        labels = []
        for i in range(seq_length, len(data)):
            sequences.append(data[i-seq_length:i, 0])
            labels.append(data[i, 0])
        return np.array(sequences), np.array(labels)

    X, y = create_sequences(scaled_data, seq_length)
    
    # Split into training and validation sets
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    return X_train, X_val, y_train, y_val, scaler

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, scaler = preprocess_data('crypto_data.csv')
