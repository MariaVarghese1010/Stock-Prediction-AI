import yfinance as yf
import pandas as pd

# Choose a stock — let's use Apple (AAPL) to start
ticker = "AAPL"

# Download daily stock data from 2015 to end of 2023
df = yf.download(ticker, start="2015-01-01", end="2023-12-31")

# Keep only the columns you need
df = df[["Open", "High", "Low", "Close", "Volume"]]

# Preview the first few rows
print(df.head())
