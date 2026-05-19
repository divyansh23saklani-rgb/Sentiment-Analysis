# realtime_scraper.py

import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import os
from tqdm import tqdm
from datetime import datetime
from joblib import load

# Optional: sentiment model and vectorizer (if exists)
try:
    model = load('model/sentiment_model.pkl')
    vectorizer = load('model/vectorizer.pkl')
    use_sentiment = True
except:
    print("⚠️ Sentiment model not found. Skipping sentiment analysis.")
    use_sentiment = False

def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|RT\s", "", text)
    return text.strip()

def predict_sentiment(texts):
    vectors = vectorizer.transform(texts)
    return model.predict(vectors)

def scrape_tweets(keyword, limit=100):
    tweets = []
    print(f"🔍 Scraping tweets for keyword: '{keyword}' (limit: {limit})...")
    for tweet in tqdm(sntwitter.TwitterSearchScraper(keyword).get_items()):
        if len(tweets) >= limit:
            break
        tweets.append([
            tweet.date.strftime("%Y-%m-%d %H:%M:%S"),
            clean_text(tweet.content),
            tweet.user.username
        ])
    
    df = pd.DataFrame(tweets, columns=['Date', 'Tweet', 'User'])

    if use_sentiment:
        print("💡 Predicting sentiment...")
        df['Sentiment'] = predict_sentiment(df['Tweet'])

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape and optionally analyze tweets by keyword")
    parser.add_argument("--keyword", required=True, help="Search keyword for tweets")
    parser.add_argument("--limit", type=int, default=50, help="Number of tweets to fetch")
    parser.add_argument("--output", default="data/scraped_tweets.csv", help="CSV output path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = scrape_tweets(args.keyword, args.limit)
    df.to_csv(args.output, index=False)
    print(f"✅ Data saved to {args.output}")
    print(df.head())
