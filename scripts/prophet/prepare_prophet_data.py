import pandas as pd
import os

# File paths
INPUT_FILE = 'data/raw/stock_yfinance_data.csv'
OUTPUT_FILE = 'data/processed/prophet_ready.csv'

print("Loading stock data...")
df = pd.read_csv(INPUT_FILE)
df['Date'] = pd.to_datetime(df['Date'])

# Rename columns for Prophet
df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df = df.sort_values('ds')

# Save formatted data
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f" Prophet-ready data saved to {OUTPUT_FILE}")
