import pandas as pd

# Load the stock data (make sure the path is correct)
df = pd.read_csv("data/AAPL.csv")

# Show the first few rows
print(df.head())
print(df.columns)
print(df.describe())