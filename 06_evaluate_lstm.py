import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load preprocessed data
data = np.load('data/preprocessed_data.npz')
X_test = data['X_test']
y_test = data['y_test']

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Define the same LSTM model structure
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model and load trained weights
model = LSTMModel().to(device)
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    y_pred = model(X_test)

# Move predictions and actual values back to CPU and numpy
y_pred = y_pred.cpu().numpy()
y_test = y_test.cpu().numpy()

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(y_test[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.title("LSTM Model - Stock Price Predictions")
plt.xlabel("Time (Days)")
plt.ylabel("Normalized Price")
plt.legend()
plt.tight_layout()
plt.savefig("plots/lstm_predictions.png")
plt.show()
