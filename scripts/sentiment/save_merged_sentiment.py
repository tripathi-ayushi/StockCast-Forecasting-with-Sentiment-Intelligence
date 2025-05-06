import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

SENTIMENT_FILE = 'data/processed/roberta_sentiment_10k.csv'
STOCK_FILE = 'data/raw/stock_yfinance_data.csv'
OUTPUT_FILE = 'data/processed/merged_sentiment_stock.csv'

print("Loading sentiment data...")
sentiment_df = pd.read_csv(SENTIMENT_FILE)
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date

print("Aggregating daily RoBERTa positive sentiment...")
daily_sentiment = sentiment_df.groupby('Date')['roberta_positive'].mean().reset_index()
daily_sentiment.columns = ['Date', 'roberta_positive_avg']

print("Loading stock data...")
stock_df = pd.read_csv(STOCK_FILE)
stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
stock_df = stock_df[['Date', 'Close']]  # Keep only needed columns

print("Merging datasets on Date...")
merged = pd.merge(stock_df, daily_sentiment, on='Date', how='inner')

print("Normalizing columns for modeling...")
scaler = MinMaxScaler()
merged[['Close_norm', 'Sentiment_norm']] = scaler.fit_transform(
    merged[['Close', 'roberta_positive_avg']]
)

# Save output
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
merged.to_csv(OUTPUT_FILE, index=False)
print(f" Saved merged and normalized data to {OUTPUT_FILE}")
