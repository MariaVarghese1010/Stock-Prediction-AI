import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib  # to save the model
import matplotlib.pyplot as plt

# Load preprocessed data
data = np.load("data/preprocessed_data.npz")  # We'll save this next step
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Flatten input for Linear Regression
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train model
model = LinearRegression()
model.fit(X_train_flat, y_train)

# Predict
y_pred = model.predict(X_test_flat)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")

# Save model
joblib.dump(model, "linear_regression_model.pkl")

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title("Linear Regression - Stock Price Prediction")
plt.xlabel("Time (Days)")
plt.ylabel("Normalized Price")
plt.legend()
plt.tight_layout()
plt.show()
