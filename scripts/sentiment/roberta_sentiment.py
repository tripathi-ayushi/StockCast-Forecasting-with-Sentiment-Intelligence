import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
import os

# Setup
INPUT_FILE = 'data/raw/stock_tweets.csv'
OUTPUT_FILE = 'data/processed/roberta_sentiment_10k.csv'
MAX_TWEETS = 10000

print("Loading tweet data...")
df = pd.read_csv(INPUT_FILE).dropna(subset=['Tweet'])
df = df.head(MAX_TWEETS).copy()
df['Tweet'] = df['Tweet'].astype(str)

print(f"Using {len(df)} tweets for RoBERTa sentiment analysis")

# Load RoBERTa model from Hugging Face
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Label mapping
labels = ['negative', 'neutral', 'positive']

# Sentiment function
def get_roberta_sentiment(text):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded)
    scores = output.logits.detach().numpy()[0]
    scores = softmax(scores)
    return scores

# Analyze with tqdm progress bar
print("Scoring tweets...")
sentiments = []
for tweet in tqdm(df['Tweet'], desc="Analyzing"):
    try:
        scores = get_roberta_sentiment(tweet)
        sentiments.append(scores)
    except Exception as e:
        sentiments.append([None, None, None])
        print(f"Error processing tweet: {e}")

# Attach scores
scores_df = pd.DataFrame(sentiments, columns=[f"roberta_{l}" for l in labels])
result_df = pd.concat([df[['Date', 'Tweet']], scores_df], axis=1)

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
result_df.to_csv(OUTPUT_FILE, index=False)
print(f" Saved RoBERTa sentiment scores to {OUTPUT_FILE}")
