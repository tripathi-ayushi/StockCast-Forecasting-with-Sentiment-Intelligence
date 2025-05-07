import pandas as pd
import os

INPUT_FILE = 'data/processed/merged_sentiment_stock.csv'
OUTPUT_FILE = 'data/processed/merged_sentiment_stock_avg.csv'

print("Loading merged sentiment + price data...")
df = pd.read_csv(INPUT_FILE)
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date').reset_index(drop=True)

# Rolling averages of sentiment
print("Creating sentiment averages...")
df['sentiment_avg_3'] = df['Sentiment_norm'].rolling(window=3).mean()
df['sentiment_avg_5'] = df['Sentiment_norm'].rolling(window=5).mean()

# Drop initial rows with NaNs
df = df.dropna().reset_index(drop=True)

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f" Saved with sentiment averages → {OUTPUT_FILE}")
