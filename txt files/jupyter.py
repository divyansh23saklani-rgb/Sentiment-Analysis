import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from tqdm import tqdm

# Ensure required NLTK data is downloaded
nltk.download('vader_lexicon')

# Set display options
pd.set_option('display.max_columns', None)

# Load the dataset
DATA_PATH = 'Reviews.csv'  # Ensure this file is in the same directory
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Please download it from Kaggle and place it in the current directory.")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# Drop duplicates and missing values
df.drop_duplicates(subset=['Text'], inplace=True)
df.dropna(subset=['Text'], inplace=True)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis
tqdm.pandas(desc="Applying VADER Sentiment Analysis")
df['VADER_Score'] = df['Text'].progress_apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify sentiment based on VADER score
def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['VADER_Sentiment'] = df['VADER_Score'].apply(classify_sentiment)

# Display sentiment distribution
sentiment_counts = df['VADER_Sentiment'].value_counts()
print("\nVADER Sentiment Distribution:")
print(sentiment_counts)

# Plot sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='VADER_Sentiment', order=['Positive', 'Neutral', 'Negative'], palette='viridis')
plt.title('VADER Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('vader_sentiment_distribution.png')
plt.close()

# Initialize RoBERTa sentiment analysis pipeline
print("\nLoading RoBERTa sentiment analysis model...")
roberta_pipeline = pipeline("sentiment-analysis")

# Apply RoBERTa sentiment analysis to a sample of the data
SAMPLE_SIZE = 1000  # Adjust as needed
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
tqdm.pandas(desc="Applying RoBERTa Sentiment Analysis")
df_sample['RoBERTa_Result'] = df_sample['Text'].progress_apply(lambda x: roberta_pipeline(x)[0])
df_sample['RoBERTa_Label'] = df_sample['RoBERTa_Result'].apply(lambda x: x['label'])
df_sample['RoBERTa_Score'] = df_sample['RoBERTa_Result'].apply(lambda x: x['score'])

# Display RoBERTa sentiment distribution
roberta_counts = df_sample['RoBERTa_Label'].value_counts()
print("\nRoBERTa Sentiment Distribution:")
print(roberta_counts)

# Plot RoBERTa sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df_sample, x='RoBERTa_Label', order=roberta_counts.index, palette='magma')
plt.title('RoBERTa Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('roberta_sentiment_distribution.png')
plt.close()

# Save the processed DataFrame to a CSV file
OUTPUT_CSV = 'processed_reviews.csv'
df_sample.to_csv(OUTPUT_CSV, index=False)
print(f"\nProcessed data saved to {OUTPUT_CSV}.")
