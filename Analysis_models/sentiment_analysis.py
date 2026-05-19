import nltk
import dateparser
from dateparser.search import search_dates
from datetime import datetime
import os
import sqlite3
import json
import hashlib
import emoji
import re
from functools import lru_cache
from langdetect import detect, LangDetectException
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from transformers import pipeline, logging
from gensim.summarization import summarize as gensim_summarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Suppress unimportant warnings from transformers
logging.set_verbosity_error()

# Download necessary resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".sentiment_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize database
DB_PATH = os.path.join(CACHE_DIR, "sentiment_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        text TEXT,
        vader_compound REAL,
        vader_sentiment TEXT,
        distilbert_label TEXT,
        distilbert_score REAL,
        importance_score REAL,
        importance_label TEXT,
        event_age TEXT,
        event_location TEXT,
        language TEXT,
        topics TEXT
    )
    ''')
    conn.commit()
    conn.close()

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()



# Cache model loading with function decorator
@lru_cache(maxsize=2)
def load_sentiment_model():
    print("⚙️ Loading DistilBERT sentiment model...")
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )
    except Exception as e:
        print(f"❌ Failed to load DistilBERT model: {e}")
        return None

@lru_cache(maxsize=2)
def load_ner_model():
    print("⚙️ Loading NER model for location detection...")
    try:
        return pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
    except Exception as e:
        print(f"❌ Failed to load NER model: {e}")
        return None
    
    
    
    
    
# Load models with caching
distilbert_analyzer = load_sentiment_model()
ner_pipeline = load_ner_model()

# Enhanced list of important keywords with categorization
important_keywords = {
    "urgency": [
        "urgent", "critical", "immediately", "asap", "emergency", "pressing", 
        "deadline", "crucial", "vital", "priority"
    ],
    "necessity": [
        "must", "need", "required", "essential", "necessary", "mandatory",
        "non-negotiable", "imperative", "required", "obligatory"
    ],
    "aspiration": [
        "wish", "dream", "plan", "goal", "intention", "objective", "ambition",
        "aspiration", "resolve", "dedication", "commitment", "focus", "vision",
        "strategy", "purpose", "mission"
    ],
    "significance": [
        "important", "meaningful", "significant", "major", "substantial",
        "considerable", "notable", "remarkable", "momentous", "pivotal",
        "top priority", "life goal", "milestone"
    ]
}



all_important_keywords = [keyword for category in important_keywords.values() for keyword in category]

def get_keyword_category(word):
    """Return the category of an important keyword"""
    for category, keywords in important_keywords.items():
        if word in keywords:
            return category
    return None

def detect_language(text):
    """Detect the language of the input text with error handling"""
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English if detection fails

def get_event_age(text):
    """Extract and calculate the age of events mentioned in the text"""
    try:
        # Use dateparser search_dates to find any dates
        results = dateparser.search.search_dates(text, settings={'PREFER_DATES_FROM': 'past'})
        if results:
            # Sort dates by earliest first
            sorted_dates = sorted(results, key=lambda x: x[1])
            parsed_date = sorted_dates[0][1]  # Use the earliest date
            today = datetime.now()
            diff = today - parsed_date
            days_ago = diff.days
            
            if days_ago < 0:
                return f"📅 Future event: {abs(days_ago)} days from now", parsed_date
            elif days_ago == 0:
                return "📅 Happened Today", parsed_date
            elif days_ago == 1:
                return "📅 Happened Yesterday", parsed_date
            elif days_ago < 7:
                return f"📅 Happened {days_ago} days ago", parsed_date
            elif days_ago < 30:
                return f"📅 Happened {days_ago // 7} week(s) ago", parsed_date
            elif days_ago < 365:
                return f"📅 Happened {days_ago // 30} month(s) ago", parsed_date
            else:
                return f"📅 Happened {days_ago // 365} year(s) ago", parsed_date
        else:
            return "📅 No clear event date detected", None
    except Exception as e:
        print(f"Date parsing error: {e}")
        return "📅 Error detecting event date", None
    
    
    
def get_event_location(text):
    """Extract locations from text with confidence scores"""
    if ner_pipeline is None:
        return "🌍 Location detection unavailable", []
    
    try:
        entities = ner_pipeline(text)
        locations = []
        
        # Group identical locations and calculate confidence
        location_confidence = {}
        for ent in entities:
            if ent['entity_group'] == 'LOC':
                loc = ent['word']
                score = ent['score']
                if loc in location_confidence:
                    # Average the confidence if we've seen this location before
                    location_confidence[loc] = (location_confidence[loc] + score) / 2
                else:
                    location_confidence[loc] = score
        
        # Convert to a list of (location, confidence) tuples and sort by confidence
        locations = [(loc, score) for loc, score in location_confidence.items()]
        locations.sort(key=lambda x: x[1], reverse=True)
        
        if locations:
            location_strs = [f"{loc} ({conf:.2f})" for loc, conf in locations]
            return "🌍 Locations detected: " + ", ".join(location_strs), locations
        else:
            return "🌍 No clear location detected", []
    except Exception as e:
        print(f"Location detection error: {e}")
        return "🌍 Error detecting locations", []

def get_importance_score(text):
    """Calculate importance score with stopword filtering"""
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    text_lower = text.lower()
    words = text_lower.split()
    
    # Filter out stopwords
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = " ".join(filtered_words)
    
    # Count occurrences of important keywords
    keyword_matches = {}
    for category, keywords in important_keywords.items():
        category_count = sum(1 for keyword in keywords if keyword in filtered_text)
        if category_count > 0:
            keyword_matches[category] = category_count
    
    total_matches = sum(keyword_matches.values())
    
    # Calculate normalized score
    text_length = max(len(filtered_words), 1)  # Avoid division by zero
    normalized_score = min(total_matches / max(text_length / 10, 1), 1.0)
    
    return round(normalized_score, 2), keyword_matches



def get_importance_label(score):
    """Convert importance score to a descriptive label"""
    if score >= 0.8:
        return "🔥 Very Important"
    elif score >= 0.4:
        return "⭐ Important"
    elif score >= 0.2:
        return "📌 Somewhat Important"
    else:
        return "✅ Standard Priority"

def analyze_emoji_sentiment(text):
    """Analyze sentiment conveyed by emojis in the text"""
    emoji_list = emoji.emoji_list(text)
    
    if not emoji_list:
        return "No emojis detected", 0
    
    # Simple emoji sentiment dictionary (could be expanded)
    positive_emojis = {"😊", "😁", "😄", "😃", "😀", "🙂", "😍", "🥰", "❤️", "👍", "✅", "🎉", "🎊"}
    negative_emojis = {"😢", "😭", "😞", "😔", "😟", "😠", "😡", "👎", "❌", "💔", "😒", "😣"}
    
    emoji_count = len(emoji_list)
    positive_count = sum(1 for e in emoji_list if e['emoji'] in positive_emojis)
    negative_count = sum(1 for e in emoji_list if e['emoji'] in negative_emojis)
    
    # Calculate net sentiment
    if emoji_count == 0:
        emoji_sentiment = 0
    else:
        emoji_sentiment = (positive_count - negative_count) / emoji_count
    
    # Return label and score
    if emoji_sentiment > 0.3:
        return "Positive emoji sentiment", emoji_sentiment
    elif emoji_sentiment < -0.3:
        return "Negative emoji sentiment", emoji_sentiment
    else:
        return "Neutral emoji sentiment", emoji_sentiment
    
    



def extract_topics(text, num_topics=2, num_words=3):
    """Extract main topics from text using LDA"""
    try:
        # Create a vectorizer
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        
        # If text is too short, return empty
        if len(text.split()) < 10:
            return ["Text too short for topic extraction"]
        
        # Create document-term matrix
        dtm = vectorizer.fit_transform([text])
        
        # Create and fit LDA model
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(dtm)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get topics
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[:-num_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(", ".join(top_words))
        
        return topics
    except Exception as e:
        print(f"Topic extraction error: {e}")
        return ["Topic extraction failed"]

def summarize_text(text, ratio=0.3):
    """Summarize longer text inputs"""
    try:
        if len(text.split()) > 100:  # Only summarize longer texts
            summary = gensim_summarize(text, ratio=ratio)
            return summary if summary else text  # Return original if summarization fails
        return text
    except Exception:
        return text  # Return original text if summarization fails
    
    
    
    
    
    
    
def save_analysis(analysis_data):
    """Save analysis results to SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO analysis_results 
        (timestamp, text, vader_compound, vader_sentiment, distilbert_label, 
        distilbert_score, importance_score, importance_label, event_age, 
        event_location, language, topics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            analysis_data['text'],
            analysis_data['vader_compound'],
            analysis_data['vader_sentiment'],
            analysis_data['distilbert_label'],
            analysis_data['distilbert_score'],
            analysis_data['importance_score'],
            analysis_data['importance_label'],
            analysis_data['event_age'],
            analysis_data['event_location'],
            analysis_data['language'],
            json.dumps(analysis_data['topics'])
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False

def compare_analyses(text1, text2):
    """Compare sentiment analysis between two texts"""
    # Analyze both texts
    analysis1 = analyze_sentiment(text1, print_results=False)
    analysis2 = analyze_sentiment(text2, print_results=False)
    
    # Calculate differences
    vader_diff = analysis2['vader_compound'] - analysis1['vader_compound']
    importance_diff = analysis2['importance_score'] - analysis1['importance_score']
    
    print("\n🔄 Sentiment Comparison:")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"VADER sentiment change: {vader_diff:.4f} ({analysis1['vader_sentiment']} → {analysis2['vader_sentiment']})")
    print(f"Importance score change: {importance_diff:.2f} ({analysis1['importance_label']} → {analysis2['importance_label']})")
    
    if vader_diff > 0.25:
        print("📈 Significant positive sentiment shift")
    elif vader_diff < -0.25:
        print("📉 Significant negative sentiment shift")
    else:
        print("↔️ Similar sentiment levels")
    
    return {
        'vader_diff': vader_diff,
        'importance_diff': importance_diff,
        'analysis1': analysis1,
        'analysis2': analysis2
    }
    
    
    
    
    
    
    
    
    
    
def analyze_sentiment(text, print_results=True):
    """Main sentiment analysis function with multiple metrics"""
    # Detect language
    language = detect_language(text)
    
    # Process text based on language
    if language != 'en':
        print(f"⚠️ Detected non-English text ({language}). Results may be less accurate.")
    
    # Get date information
    event_age, event_date = get_event_age(text)
    
    # Get location information
    event_location, location_data = get_event_location(text)
    
    # Get importance information
    importance_score, keyword_matches = get_importance_score(text)
    importance_label = get_importance_label(importance_score)
    
    # VADER sentiment analysis
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_sentiment = (
        "Positive" if vader_scores['compound'] >= 0.05 else
        "Negative" if vader_scores['compound'] <= -0.05 else
        "Neutral"
    )
    
    # DistilBERT sentiment analysis
    distilbert_label = "Neutral"
    distilbert_score = 0.5
    
    if distilbert_analyzer:
        try:
            distilbert_result = distilbert_analyzer(text)[0]
            distilbert_label = distilbert_result['label']
            distilbert_score = distilbert_result['score']
        except Exception as e:
            print(f"⚠️ DistilBERT analysis error: {e}")
    
    # Emoji sentiment analysis
    emoji_sentiment_label, emoji_sentiment_score = analyze_emoji_sentiment(text)
    
    # Topic extraction
    topics = extract_topics(text)
    
    # Create analysis data dictionary
    analysis_data = {
        'text': text,
        'vader_compound': vader_scores['compound'],
        'vader_sentiment': vader_sentiment,
        'distilbert_label': distilbert_label,
        'distilbert_score': distilbert_score,
        'importance_score': importance_score,
        'importance_label': importance_label,
        'event_age': event_age,
        'event_location': event_location,
        'language': language,
        'emoji_sentiment': emoji_sentiment_label,
        'emoji_score': emoji_sentiment_score,
        'topics': topics,
        'keyword_matches': keyword_matches
    }
    
    # Save to database
    save_analysis(analysis_data)
    
    if print_results:
        print(f"\n📝 Input Sentence: {text}")
        
        # Summarize if text is long
        if len(text.split()) > 100:
            summary = summarize_text(text)
            print(f"📋 Summary: {summary}")
        
        print(f"🔤 Detected Language: {language}")
        print("🔹 VADER Sentiment:")
        print(f"   Compound Score: {vader_scores['compound']:.4f}")
        print(f"   Sentiment: {vader_sentiment}")
        print("🔹 DistilBERT Sentiment:")
        print(f"   Label: {distilbert_label}")
        print(f"   Confidence Score: {distilbert_score:.4f}")
        print("🔹 Emoji Sentiment:")
        print(f"   {emoji_sentiment_label} ({emoji_sentiment_score:.2f})")
        print("🔹 Importance Score:")
        print(f"   {importance_score} - {importance_label}")
        
        if keyword_matches:
            print("   Keyword Categories:")
            for category, count in keyword_matches.items():
                print(f"      - {category.capitalize()}: {count}")
        
        print("🔹 Event Memory (Date Info):")
        print(f"   {event_age}")
        print("🔹 Event Location:")
        print(f"   {event_location}")
        print("🔹 Topics:")
        for i, topic in enumerate(topics):
            print(f"   Topic {i+1}: {topic}")
    
    return analysis_data

def print_help():
    """Print help information for the tool"""
    print("\n📘 Available Commands:")
    print("  help         - Show this help message")
    print("  exit         - Exit the program")
    print("  history      - Show recent analysis history")
    print("  compare      - Compare two sentences (you'll be prompted for input)")
    print("  [any text]   - Analyze the sentiment of the provided text")

def show_history(limit=5):
    """Show recent analysis history from the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT timestamp, text, vader_sentiment, importance_label 
        FROM analysis_results 
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print("📭 No analysis history found")
            return
        
        print("\n📜 Recent Analysis History:")
        for i, (timestamp, text, sentiment, importance) in enumerate(results):
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Truncate text if too long
            truncated_text = text[:50] + "..." if len(text) > 50 else text
            
            print(f"{i+1}. [{formatted_time}] - {truncated_text}")
            print(f"   Sentiment: {sentiment}, Importance: {importance}")
        
    except Exception as e:
        print(f"Error retrieving history: {e}")

if __name__ == "__main__":
    print("\n💬 Enhanced Sentiment Analysis Tool v2.0\n")
    print("Features: VADER + DistilBERT + Event Memory + Location + Importance + Emoji + Topics + Comparison")
    print("Type 'help' for available commands")
    
    # Initialize database
    init_db()
    
    previous_text = None
    
    while True:
        try:
            user_input = input("\n> ")
            
            if user_input.strip().lower() == "exit":
                print("👋 Exiting...")
                break
                
            elif user_input.strip().lower() == "help":
                print_help()
                
            elif user_input.strip().lower() == "history":
                show_history()
                
            elif user_input.strip().lower() == "compare":
                print("Enter first text:")
                text1 = input("> ")
                print("Enter second text:")
                text2 = input("> ")
                
                if text1.strip() and text2.strip():
                    compare_analyses(text1, text2)
                else:
                    print("⚠️ Both inputs must be non-empty")
                
            elif user_input.strip() == "":
                print("⚠️ Please enter a non-empty sentence.")
                
            else:
                analyze_sentiment(user_input)
                previous_text = user_input
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")