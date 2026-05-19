import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Download necessary resources
nltk.download('vader_lexicon')

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize DistilBERT
distilbert_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    # VADER analysis
    vader_score = vader_analyzer.polarity_scores(text)['compound']
    vader_sentiment = (
        "Positive" if vader_score >= 0.05 else
        "Negative" if vader_score <= -0.05 else
        "Neutral"
    )

    # DistilBERT analysis
    distilbert_result = distilbert_analyzer(text)[0]
    distilbert_label = distilbert_result['label']
    distilbert_score = distilbert_result['score']

    print(f"\nInput: {text}")
    print(f"VADER Sentiment: {vader_sentiment} (Score: {vader_score:.4f})")
    print(f"DistilBERT Sentiment: {distilbert_label} (Confidence: {distilbert_score:.4f})")

if __name__ == "__main__":
    while True:
        text = input("Enter text (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        analyze_sentiment(text)
