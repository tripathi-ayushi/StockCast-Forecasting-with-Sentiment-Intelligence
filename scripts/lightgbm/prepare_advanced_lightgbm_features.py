import pandas as pd
import os

# Paths
INPUT_FILE = 'data/processed/merged_sentiment_stock_avg.csv'
OUTPUT_FILE = 'data/processed/lgbm_advanced_ready.csv'

print("Loading data with sentiment averages...")
df = pd.read_csv(INPUT_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Lag features for price
for i in range(1, 4):
    df[f'lag_close_{i}'] = df['Close_norm'].shift(i)

# Lag features for sentiment
for i in range(1, 4):
    df[f'lag_sentiment_{i}'] = df['Sentiment_norm'].shift(i)

# Rolling stats
df['roll_mean_3'] = df['Close_norm'].rolling(window=3).mean()
df['roll_std_3'] = df['Close_norm'].rolling(window=3).std()

# Target = next day's close price
df['target'] = df['Close_norm'].shift(-1)

# Drop rows with NaNs
df = df.dropna().reset_index(drop=True)

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f" Advanced features saved to {OUTPUT_FILE}")
