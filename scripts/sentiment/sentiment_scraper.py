import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Load tweets
INPUT_FILE = 'data/raw/stock_tweets.csv'
OUTPUT_FILE = 'data/processed/vader_sentiment_daily.csv'

df = pd.read_csv(INPUT_FILE)
df['Date'] = pd.to_datetime(df['Date']).dt.date  # Truncate to date only

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Apply VADER to each tweet
df['compound'] = df['Tweet'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

# Daily aggregation
daily_sentiment = df.groupby('Date')['compound'].mean().reset_index()
daily_sentiment.columns = ['Date', 'VADER_Compound_Mean']

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
daily_sentiment.to_csv(OUTPUT_FILE, index=False)

print(f"Saved daily VADER sentiment to {OUTPUT_FILE}")