# keyword_sentiment_tracker.py

import snscrape.modules.twitter as sntwitter
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import argparse
import os
import sys

# Load pretrained sentiment model
model = load('model/sentiment_model.pkl')

def fetch_tweets(keyword, limit):
    """
    Fetch tweets using snscrape matching the keyword.
    """
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()):
            if i >= limit:
                break
            tweets.append(tweet.content)
    except Exception as e:
        print(f"[❌] Error fetching tweets: {e}")
    return tweets

def analyze_sentiments(tweets):
    """
    Predict sentiment for each tweet.
    """
    return [model.predict([t])[0] for t in tweets]

def visualize(df, keyword):
    """
    Plot sentiment distribution.
    """
    color_map = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=[color_map.get(sent, 'blue') for sent in sentiment_counts.index])
    plt.title(f"Sentiment on '{keyword}'")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Tweets")
    plt.tight_layout()
    plt.show()

def keyword_tracker(keyword, limit=100, export_path=None):
    """
    Main function to perform sentiment tracking on tweets matching a keyword.
    """
    print(f"[🔍] Searching Twitter for '{keyword}'...")

    tweets = fetch_tweets(keyword, limit)
    if not tweets:
        print("⚠️ No tweets found or error occurred.")
        return

    sentiments = analyze_sentiments(tweets)
    df = pd.DataFrame({'Tweet': tweets, 'Sentiment': sentiments})

    if export_path:
        df.to_csv(export_path, index=False)
        print(f"[✅] Results saved to {export_path}")

    visualize(df, keyword)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track sentiment of a keyword on Twitter")
    parser.add_argument("--keyword", required=True, help="Keyword to search for")
    parser.add_argument("--limit", type=int, default=100, help="Number of tweets to fetch")
    parser.add_argument("--export", help="CSV file path to save results")

    args = parser.parse_args()

    keyword_tracker(args.keyword, args.limit, args.export)
