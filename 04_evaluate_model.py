import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load preprocessed test data
data = np.load("data/preprocessed_data.npz")
X_test = data['X_test']
y_test = data['y_test']

# Flatten input for the linear regression model
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Load the trained model
model = joblib.load("linear_regression_model.pkl")

# Predict
y_pred = model.predict(X_test_flat)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title("Linear Regression (Reloaded Model) - Predictions")
plt.xlabel("Time (Days)")
plt.ylabel("Normalized Price")
plt.legend()
plt.tight_layout()
plt.show()
