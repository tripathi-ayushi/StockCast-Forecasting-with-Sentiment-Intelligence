import pandas as pd
import os

# File paths
INPUT_FILE = 'data/processed/merged_sentiment_stock.csv'
OUTPUT_FILE = 'data/processed/lgbm_ready.csv'

print("Loading merged sentiment + price data...")
df = pd.read_csv(INPUT_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Create lag features (1-day lag of Close_norm and Sentiment_norm)
print("Creating lag and rolling features...")
df['lag_close_1'] = df['Close_norm'].shift(1)
df['lag_sentiment_1'] = df['Sentiment_norm'].shift(1)

# Rolling statistics
df['roll_mean_3'] = df['Close_norm'].rolling(window=3).mean()
df['roll_std_3'] = df['Close_norm'].rolling(window=3).std()

# Target: predict next day's Close_norm
df['target'] = df['Close_norm'].shift(-1)

# Drop rows with NaNs from shifting/rolling
df = df.dropna().reset_index(drop=True)

# Save processed dataset
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f" LightGBM-ready dataset saved to {OUTPUT_FILE}")
