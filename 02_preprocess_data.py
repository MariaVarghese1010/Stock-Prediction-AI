import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data/AAPL.csv")

# Sort by date (just in case)
df = df.sort_values("Date")

# Use only the 'Close' column for prediction
close_prices = df[["Close"]].values

# Normalize the 'Close' prices between 0 and 1
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(close_prices)

# Create sequences: past 10 days → next day
def create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Create dataset
window_size = 10
X, y = create_sequences(normalized_data, window_size)

# Split into train/test (80% train, 20% test)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Confirm the shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Save the preprocessed arrays for the model
np.savez("data/preprocessed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
