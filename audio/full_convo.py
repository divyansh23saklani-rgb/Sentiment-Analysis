import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, logging

# Suppress unimportant warnings from transformers
logging.set_verbosity_error()

# Download required NLTK resources
# nltk.download('punkt')
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize HuggingFace DistilBERT sentiment pipeline
print("⚙️ Loading DistilBERT sentiment model...")
try:
    distilbert_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"❌ Failed to load DistilBERT model: {e}")
    exit(1)

# Function to clean and split the conversation
def split_into_phrases(conversation):
    # Basic sentence splitting using regex (., !, ? followed by space or end of string)
    sentences = re.split(r'(?<=[.!?]) +', conversation)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


# Analyze sentiment for a phrase
def analyze_phrase(phrase):
    # VADER
    vader_scores = vader_analyzer.polarity_scores(phrase)
    vader_sentiment = (
        "Positive" if vader_scores['compound'] >= 0.05 else
        "Negative" if vader_scores['compound'] <= -0.05 else
        "Neutral"
    )

    # DistilBERT
    distilbert_result = distilbert_analyzer(phrase)[0]
    distilbert_label = distilbert_result['label']
    distilbert_score = distilbert_result['score']

    return {
        "text": phrase,
        "vader_sentiment": vader_sentiment,
        "vader_compound": vader_scores['compound'],
        "distilbert_label": distilbert_label,
        "distilbert_score": distilbert_score
    }

# Main processing
def process_conversation(conversation):
    phrases = split_into_phrases(conversation)
    print(f"\n📄 Total Phrases Detected: {len(phrases)}\n")
    
    results = []
    for idx, phrase in enumerate(phrases, start=1):
        analysis = analyze_phrase(phrase)
        results.append(analysis)

        print(f"🔹 Phrase {idx}: {analysis['text']}")
        print(f"   - VADER Sentiment: {analysis['vader_sentiment']} (Compound: {analysis['vader_compound']:.4f})")
        print(f"   - DistilBERT Sentiment: {analysis['distilbert_label']} (Confidence: {analysis['distilbert_score']:.4f})\n")

    return results

# CLI Entry
if __name__ == "__main__":
    print("\n💬 Conversation Sentiment Analyzer (VADER + DistilBERT)\n")
    try:
        conversation = input("Paste your full conversation text here:\n> ")
        if conversation.strip() == "":
            print("⚠️ Empty conversation entered. Exiting.")
        else:
            process_conversation(conversation)
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
