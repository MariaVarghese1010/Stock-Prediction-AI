# 📈 Stock Prediction AI

This project predicts future stock prices using historical data and machine learning models. It is part of a course assignment focused on applying ML to real-world datasets.

## 📦 Dataset

- Historical daily stock prices from [Kaggle: Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- Used Apple stock data (`AAPL.csv`) located in the `data/` folder

## 🛠 Project Structure

```bash
stock_prediction_project/
│
├── 01_load_data.py         # Loads stock data from local CSV
├── 02_preprocess_data.py   # Normalizes data & creates sliding windows
├── 03_train_model.py       # Trains linear regression model
├── 04_evaluate_model.py    # (optional) Reserved for future evaluation steps
├── data/
│   ├── AAPL.csv            # Raw stock data
│   └── preprocessed_data.npz  # Saved numpy arrays for training
├── linear_regression_model.pkl  # Trained model file
└── README.md

## 🔍 How It Works

- Past 10 days of prices → predict next day's price  
- Data is normalized between 0 and 1 using `MinMaxScaler`  
- Trained with **Linear Regression** as a baseline model

## 📊 Results

- **Test MSE**: ~0.000083  
- **Test MAE**: ~0.0051  
- Prediction vs Actual plot is included in the project

## 🧠 Next Steps

- Implement LSTM for time-series prediction (coming soon)
- Add ablation studies and hyperparameter tuning
- Enhance report with model comparisons and training loss visualizations

## 📚 Requirements

- Python 3.8+
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 👩‍💻 Contributors
**Maria Varghese** – Data processing, model development  
**Isabel Lara** – Report writing, research, documentation

---
