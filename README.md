# 📈 Stock Prediction AI

This project predicts future stock prices using historical data and machine learning models. It is part of a course assignment focused on applying ML to real-world datasets.

## 📦 Dataset

- Historical daily stock prices from [Kaggle: Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- Used Apple stock data (`AAPL.csv`) located in the `data/` folder

## 🛠 Project Structure

```bash
stock_prediction_project/
│
├── 01_load_data.py               # Loads stock data from local CSV
├── 02_preprocess_data.py         # Normalizes data & creates sliding windows
├── 03_train_model.py             # Trains Linear Regression model
├── 04_evaluate_model.py          # Evaluates and plots Linear Regression results
├── 05_train_lstm.py              # Trains LSTM model
├── 06_evaluate_lstm.py           # Evaluates and plots LSTM results
│
├── data/
│   ├── AAPL.csv                  # Raw stock data
│   └── preprocessed_data.npz     # Saved numpy arrays for training/testing
│
├── plots/
│   ├── lstm_training_loss.png    # LSTM training loss curve
│   ├── lstm_predictions.png      # LSTM predicted vs actual plot
│
├── linear_regression_model.pkl   # Saved Linear Regression model
├── lstm_model.pth                 # Saved LSTM model
└── README.md

## 🔍 How It Works

- Past 10 days of prices → predict next day's price  
- Data is normalized between 0 and 1 using `MinMaxScaler`  
- Linear Regression used as a baseline model
- LSTM model implemented for time-series prediction
- Comparison of model performance included

## 📊 Results

Linear Regression:
- **Test MSE**: ~0.000083  
- **Test MAE**: ~0.0051  

LSTM Model:
- Test MSE: Much lower, smoother fitting curve
- Captures overall price trends better than Linear Regression

Plots of predictions vs actual stock prices are included in the plots/ folder.


## 🧠 Key Learnings

- Linear Regression performs reasonably but struggles with complex, sequential stock patterns.
- LSTM better captures temporal dependencies, providing smoother and more accurate future predictions.

## 📚 Requirements

- Python 3.8+
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `torch`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 👩‍💻 Contributors
**Maria Varghese** – Data processing, model development  
**Isabel Lara** – Report writing, research, documentation

---
