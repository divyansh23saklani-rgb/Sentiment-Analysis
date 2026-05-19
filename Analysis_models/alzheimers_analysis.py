import nltk
import dateparser
from dateparser.search import search_dates
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, logging

# Suppress unimportant warnings from transformers
logging.set_verbosity_error()

# Download necessary resources
nltk.download('vader_lexicon')

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize DistilBERT
print("⚙️ Loading DistilBERT sentiment model...")
try:
    distilbert_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"❌ Failed to load DistilBERT model: {e}")
    exit(1)

# Initialize NER pipeline for location extraction
print("⚙️ Loading NER model for location detection...")
try:
    ner_pipeline = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple"
    )
except Exception as e:
    print(f"❌ Failed to load NER model: {e}")
    exit(1)

# Alzheimer's-specific keywords for importance and monitoring
importance_keywords = [
    # General importance
    "urgent", "critical", "important", "must", "need", "required",
    # Medical and care
    "medication", "medicine", "doctor", "appointment", "therapy", "forgot", "remember",
    "confused", "lost", "disoriented",
    # Daily routine
    "eat", "food", "meal", "bath", "shower", "sleep", "walk", "exercise",
    # Safety concerns
    "fell", "fall", "hurt", "pain", "emergency", "help",
    # Emotional states
    "scared", "afraid", "anxious", "happy", "sad", "angry", "frustrated",
    # Original keywords
    "wish", "dream", "plan", "goal", "essential", "necessary", "vital", "priority"
]

# Daily activities to monitor
adl_keywords = {
    "medication": ["pill", "medicine", "prescription", "drug", "dose", "tablet"],
    "meals": ["breakfast", "lunch", "dinner", "eat", "food", "snack", "hungry"],
    "hygiene": ["bath", "shower", "wash", "clean", "toilet", "bathroom"],
    "sleep": ["sleep", "bed", "nap", "tired", "rest", "awake", "insomnia"],
    "mobility": ["walk", "fell", "fall", "trip", "exercise", "move", "sit", "stand"],
    "social": ["visit", "friend", "family", "talk", "called", "phone", "conversation"]
}

def map_distilbert_to_compound(label, score):
    """Convert DistilBERT's binary output to a compound-like score between -1 and 1"""
    if label == "POSITIVE":
        return score  # Score is already between 0 and 1
    else:  # NEGATIVE
        return -score  # Convert to negative value between -1 and 0

def detect_adls(text):
    """Detect activities of daily living in the text"""
    text_lower = text.lower()
    detected_adls = []
    
    for category, keywords in adl_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_adls.append(category)
    
    return detected_adls

def detect_time_of_day(text):
    """Detect time of day mentions"""
    text_lower = text.lower()
    times = {
        "morning": ["morning", "breakfast", "wake", "dawn", "early", "am"],
        "afternoon": ["afternoon", "lunch", "noon", "midday"],
        "evening": ["evening", "dinner", "supper", "sunset", "dusk", "pm"],
        "night": ["night", "bed", "sleep", "midnight", "late"]
    }
    
    detected_times = []
    for time_period, keywords in times.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_times.append(time_period)
    
    return detected_times if detected_times else ["no specific time detected"]

def get_event_age(text):
    # Use dateparser search_dates to find any dates
    results = dateparser.search.search_dates(text, settings={'PREFER_DATES_FROM': 'past'})
    if results:
        # Pick the first detected date
        parsed_date = results[0][1]
        today = datetime.now()
        diff = today - parsed_date
        days_ago = diff.days
        if days_ago == 0:
            return "📅 Happened Today"
        elif days_ago == 1:
            return "📅 Happened Yesterday"
        elif days_ago < 7:
            return f"📅 Happened {days_ago} days ago"
        elif days_ago < 30:
            return f"📅 Happened {days_ago // 7} week(s) ago"
        elif days_ago < 365:
            return f"📅 Happened {days_ago // 30} month(s) ago"
        else:
            return f"📅 Happened {days_ago // 365} year(s) ago"
    else:
        return "📅 No clear event date detected"

def get_event_location(text):
    entities = ner_pipeline(text)
    locations = [ent['word'] for ent in entities if ent['entity_group'] == 'LOC']
    if locations:
        return "🌍 Location Detected: " + ", ".join(set(locations))
    else:
        return "🌍 No clear location detected"
    
    
def get_importance_score(text):
    text_lower = text.lower()
    
    # Define tiers of importance with different weights
    high_importance = ["urgent", "critical", "emergency", "help", "fell", "fall", "hurt", "pain"]
    medium_importance = ["important", "must", "need", "required", "medication", "medicine", "doctor", "appointment"]
    low_importance = ["wish", "plan", "eat", "food", "bath", "shower", "sleep", "walk"]
    
    # Count weighted occurrences
    high_count = sum(1 for keyword in high_importance if keyword in text_lower)
    medium_count = sum(1 for keyword in medium_importance if keyword in text_lower)
    low_count = sum(1 for keyword in low_importance if keyword in text_lower)
    
    # Calculate weighted score (high=3x, medium=2x, low=1x)
    weighted_score = (high_count * 3 + medium_count * 2 + low_count) / 10
    
    # Ensure the score is between 0 and 1
    normalized_score = min(max(weighted_score, 0), 1.0)
    
    return round(normalized_score, 2)


def get_importance_label(score):
    if score >= 0.6:  # Changed from 0.8
        return "🔥 Very Important"
    elif score >= 0.3:  # Changed from 0.4
        return "⭐ Important"
    else:
        return "✅ Normal Routine"
    

def detect_potential_concerns(text):
    """Detect potential concerns or issues that might need caregiver attention"""
    text_lower = text.lower()
    
    concern_keywords = {
        "memory_issue": ["forgot", "can't remember", "don't remember", "confused", "where am i", "what day"],
        "safety_risk": ["fell", "fall", "hurt", "injury", "lost", "wander", "fire", "stove", "burn"],
        "emotional_distress": ["scared", "afraid", "anxious", "sad", "crying", "depressed", "upset"],
        "medication_issue": ["missed", "forgot pill", "forgot medication", "extra pill", "double dose"]
    }
    
    concerns = []
    for concern, keywords in concern_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            concerns.append(concern)
    
    return concerns

def analyze_sentiment(text):
    # VADER analysis
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_compound = vader_scores['compound']
    vader_sentiment = (
        "Positive" if vader_compound >= 0.05 else
        "Negative" if vader_compound <= -0.05 else
        "Neutral"
    )

    # DistilBERT analysis
    distilbert_result = distilbert_analyzer(text)[0]
    distilbert_label = distilbert_result['label']
    distilbert_score = distilbert_result['score']
    
    # Map DistilBERT to compound-like score
    distilbert_compound = map_distilbert_to_compound(distilbert_label, distilbert_score)
    
    # Combined sentiment score (average of both)
    combined_score = (vader_compound + distilbert_compound) / 2
    combined_sentiment = (
        "Positive" if combined_score >= 0.05 else
        "Negative" if combined_score <= -0.05 else
        "Neutral"
    )

    # Other analyses
    importance = get_importance_score(text)
    event_age = get_event_age(text)
    event_location = get_event_location(text)
    detected_adls = detect_adls(text)
    time_of_day = detect_time_of_day(text)
    potential_concerns = detect_potential_concerns(text)

    print(f"\n📝 Input Sentence: {text}")
    print("\n🔹 Sentiment Analysis:")
    print(f"   VADER: {vader_sentiment} (score: {vader_compound:.4f})")
    print(f"   DistilBERT: {distilbert_label} (confidence: {distilbert_score:.4f}, mapped score: {distilbert_compound:.4f})")
    print(f"   Combined Sentiment: {combined_sentiment} (score: {combined_score:.4f})")
    
    print("\n🔹 Alzheimer's Monitoring:")
    print(f"   Importance: {get_importance_label(importance)} (score: {importance:.2f})")
    print(f"   Activities Detected: {', '.join(detected_adls) if detected_adls else 'None'}")
    print(f"   Time of Day: {', '.join(time_of_day)}")
    
    if potential_concerns:
        print("\n⚠️ Potential Concerns Detected:")
        for concern in potential_concerns:
            formatted_concern = concern.replace('_', ' ').title()
            print(f"   - {formatted_concern}")
    
    print(f"\n🔹 Event Context:")
    print(f"   {event_age}")
    print(f"   {event_location}")

if __name__ == "__main__":
    print("\n💬 Alzheimer's Patient Monitoring - Sentiment & Activity Analysis\n")
    while True:
        try:
            user_input = input("Enter a patient statement (or type 'exit' to quit):\n> ")
            if user_input.strip().lower() == "exit":
                print("👋 Exiting...")
                break
            if user_input.strip() == "":
                print("⚠️ Please enter a non-empty statement.")
                continue
            analyze_sentiment(user_input)
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break