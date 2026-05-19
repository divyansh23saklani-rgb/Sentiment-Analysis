import nltk
import dateparser
import csv
import os
import datetime as dt
from dateparser.search import search_dates
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, logging

# Suppress unimportant warnings from transformers
logging.set_verbosity_error()

# Download necessary resources
nltk.download('vader_lexicon', quiet=True)

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

# Emotional and memory significance keywords
emotional_keywords = [
    # Grief and loss
    "lost", "lose", "died", "death", "passed away", "miss", "grief", "mourn", "crying", "cry", "tears",
    # Family relationships
    "child", "children", "grandchild", "grandchildren", "son", "daughter", "husband", "wife", "mother", "father",
    # Traumatic events
    "war", "accident", "disaster", "tragedy", "trauma", "catastrophe", "emergency", "attack",
    # Strong emotions
    "devastated", "heartbroken", "depressed", "anxious", "afraid", "terrified", "panic", "love", "hate",
    # Memory-related
    "remember", "memories", "forget", "recall", "reminisce", "nostalgia", "flashback", "traumatic memory"
]

# Standard importance keywords
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

# Family relationship terms
family_terms = [
    "grandchild", "grandchildren", "grandson", "granddaughter", "child", "children",
    "son", "daughter", "husband", "wife", "spouse", "partner", "brother", "sister",
    "mother", "father", "parent", "aunt", "uncle", "cousin", "niece", "nephew",
    "family", "relative", "loved one"
]

# Traumatic event terms
traumatic_events = [
    "war", "battle", "bombing", "attack", "violence", "assault", "accident", "crash",
    "disaster", "fire", "flood", "earthquake", "hurricane", "tornado", "trauma", 
    "abuse", "assault", "shooting", "explosion", "terrorist", "death", "funeral",
    "hospital", "emergency", "ICU", "catastrophe", "tragedy"
]

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
            return "Today"
        elif days_ago == 1:
            return "Yesterday"
        elif days_ago < 7:
            return f"{days_ago} days ago"
        elif days_ago < 30:
            return f"{days_ago // 7} week(s) ago"
        elif days_ago < 365:
            return f"{days_ago // 30} month(s) ago"
        else:
            return f"{days_ago // 365} year(s) ago"
    else:
        return "No clear date"

def get_raw_date(text):
    """Extract the raw date if available"""
    results = dateparser.search.search_dates(text, settings={'PREFER_DATES_FROM': 'past'})
    if results:
        parsed_date = results[0][1]
        return parsed_date.strftime('%Y-%m-%d')
    return ""

def get_event_location(text):
    entities = ner_pipeline(text)
    locations = [ent['word'] for ent in entities if ent['entity_group'] == 'LOC']
    if locations:
        return ", ".join(set(locations))
    else:
        return "No location detected"

def detect_emotional_significance(text):
    """Detect emotionally significant content"""
    text_lower = text.lower()
    
    # Check for emotional keywords
    emotion_score = sum(1 for keyword in emotional_keywords if keyword in text_lower)
    
    # Increase importance if family members are mentioned
    family_mentioned = any(term in text_lower for term in family_terms)
    
    # Increase importance if traumatic events are mentioned
    trauma_mentioned = any(event in text_lower for event in traumatic_events)
    
    # Check for grief indicators
    grief_indicators = ["miss", "lost", "died", "passed away", "cry", "tears", "sad"]
    grief_mentioned = any(indicator in text_lower for indicator in grief_indicators)
    
    # Calculate emotional significance score
    emotional_score = emotion_score * 0.2  # Base score from emotional keywords
    
    if family_mentioned:
        emotional_score += 0.3
    
    if trauma_mentioned:
        emotional_score += 0.3
    
    if grief_mentioned:
        emotional_score += 0.3
    
    # Cap the score at 1.0
    emotional_score = min(emotional_score, 1.0)
    
    return round(emotional_score, 2)

def get_importance_score(text):
    """Calculate importance score based on keywords and emotional significance"""
    text_lower = text.lower()
    
    # Standard importance from keywords
    basic_importance = sum(1 for keyword in importance_keywords if keyword in text_lower)
    basic_score = min(basic_importance / 5, 1.0)
    
    # Get emotional significance score
    emotional_score = detect_emotional_significance(text)
    
    # Combined score with higher weight for emotional content
    combined_score = max(basic_score, emotional_score)
    
    return round(combined_score, 2)

def get_importance_label(score):
    if score >= 0.8:
        return "Very Important"
    elif score >= 0.4:
        return "Important"
    else:
        return "Normal Routine"

def detect_potential_concerns(text):
    """Detect potential concerns or issues that might need caregiver attention"""
    text_lower = text.lower()
    
    concern_keywords = {
        "memory_issue": ["forgot", "can't remember", "don't remember", "confused", "where am i", "what day"],
        "safety_risk": ["fell", "fall", "hurt", "injury", "lost", "wander", "fire", "stove", "burn"],
        "emotional_distress": ["scared", "afraid", "anxious", "sad", "crying", "depressed", "upset", "miss", "grief"],
        "medication_issue": ["missed", "forgot pill", "forgot medication", "extra pill", "double dose"],
        "trauma_trigger": ["war", "death", "lost", "tragedy", "accident", "memory", "flashback", "nightmare"]
    }
    
    concerns = []
    for concern, keywords in concern_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            concerns.append(concern)
    
    return concerns

def analyze_family_relationships(text):
    """Analyze mentions of family relationships"""
    text_lower = text.lower()
    
    relations = []
    for term in family_terms:
        if term in text_lower:
            relations.append(term)
    
    return relations if relations else None

def analyze_grief_indicators(text):
    """Analyze text for indicators of grief or loss"""
    text_lower = text.lower()
    
    grief_indicators = {
        "active_grief": ["cry", "crying", "tears", "sob", "weep", "miss", "hurt"],
        "loss_mention": ["lost", "died", "passed away", "gone", "no more", "death"],
        "memory_reflection": ["remember", "memories", "used to", "would always", "think about"]
    }
    
    indicators = []
    for category, keywords in grief_indicators.items():
        if any(keyword in text_lower for keyword in keywords):
            indicators.append(category)
    
    return indicators if indicators else None

def analyze_trauma_indicators(text):
    """Analyze text for indicators of trauma"""
    text_lower = text.lower()
    
    trauma_indicators = {
        "war_related": ["war", "fight", "battle", "bomb", "attack", "soldier", "military"],
        "disaster_related": ["fire", "flood", "earthquake", "hurricane", "tornado", "disaster"],
        "accident_related": ["crash", "accident", "hit", "collision", "fell", "injury"],
        "violence_related": ["attack", "assault", "hit", "beat", "shoot", "kill", "wound"]
    }
    
    indicators = []
    for category, keywords in trauma_indicators.items():
        if any(keyword in text_lower for keyword in keywords):
            indicators.append(category)
    
    return indicators if indicators else None

def get_severity_indicators(text):
    """Get severity indicators from various analyses"""
    importance = get_importance_score(text)
    grief_indicators = analyze_grief_indicators(text)
    trauma_indicators = analyze_trauma_indicators(text)
    potential_concerns = detect_potential_concerns(text)
    
    severity = []
    
    if importance >= 0.8:
        severity.append("high_emotional_significance")
    
    if grief_indicators:
        severity.append("grief_present")
    
    if trauma_indicators:
        severity.append("trauma_mentioned")
    
    if "emotional_distress" in potential_concerns:
        severity.append("emotional_distress")
    
    if "trauma_trigger" in potential_concerns:
        severity.append("trauma_trigger")
        
    return severity if severity else ["normal"]

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
    importance_label = get_importance_label(importance)
    event_age = get_event_age(text)
    raw_date = get_raw_date(text)
    event_location = get_event_location(text)
    detected_adls = detect_adls(text)
    time_of_day = detect_time_of_day(text)
    potential_concerns = detect_potential_concerns(text)
    family_mentions = analyze_family_relationships(text)
    grief_indicators = analyze_grief_indicators(text)
    trauma_indicators = analyze_trauma_indicators(text)
    severity_indicators = get_severity_indicators(text)

    # Format results for both printing and CSV
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_text": text,
        "sentiment_label": combined_sentiment,
        "sentiment_score": round(combined_score, 4),
        "vader_score": round(vader_compound, 4),
        "distilbert_score": round(distilbert_compound, 4),
        "importance_label": importance_label,
        "importance_score": round(importance, 2),
        "event_age": event_age,
        "event_date": raw_date,
        "location_indicators": event_location,
        "routine_references": ", ".join(detected_adls) if detected_adls else "None",
        "time_indicators": ", ".join(time_of_day),
        "memory_references": ", ".join(family_mentions) if family_mentions else "None",
        "grief_indicators": ", ".join(grief_indicators) if grief_indicators else "None",
        "trauma_indicators": ", ".join(trauma_indicators) if trauma_indicators else "None",
        "severity_indicators": ", ".join(severity_indicators),
        "potential_concerns": ", ".join(potential_concerns) if potential_concerns else "None"
    }
    
    # Print formatted results
    print(f"\n📝 Input Statement: {text}")
    
    print("\n🔹 Sentiment Analysis:")
    print(f"   VADER: {vader_sentiment} (score: {vader_compound:.4f})")
    print(f"   DistilBERT: {distilbert_label} (confidence: {distilbert_score:.4f}, mapped score: {distilbert_compound:.4f})")
    print(f"   Combined Sentiment: {combined_sentiment} (score: {combined_score:.4f})")
    
    print("\n🔹 Patient Monitoring:")
    print(f"   Emotional Significance: {importance_label} (score: {importance:.2f})")
    
    if family_mentions:
        print(f"   Family Relationships Mentioned: {', '.join(family_mentions)}")
    
    if grief_indicators:
        print("   Grief Indicators: " + ", ".join(grief_indicators).replace("_", " ").title())
    
    if trauma_indicators:
        print("   Trauma Indicators: " + ", ".join(trauma_indicators).replace("_", " ").title())
    
    if detected_adls:
        print(f"   Activities Detected: {', '.join(detected_adls)}")
    
    print(f"   Time of Day Mentioned: {', '.join(time_of_day)}")
    
    if potential_concerns:
        print("\n⚠️ Potential Care Concerns:")
        for concern in potential_concerns:
            formatted_concern = concern.replace('_', ' ').title()
            print(f"   - {formatted_concern}")
            
            # Special recommendations for emotional distress
            if concern == "emotional_distress":
                print("     ► Recommended Action: Provide emotional support, validate feelings about loss, consider reminiscence therapy")
            
            # Special recommendations for trauma triggers
            if concern == "trauma_trigger":
                print("     ► Recommended Action: Gently redirect conversation, validate feelings, ensure calm environment, consider consulting mental health professional")
    
    print("\n🔹 Memory & Context Analysis:")
    print(f"   Event Age: {event_age}")
    print(f"   Location: {event_location}")
    
    # Add Alzheimer's-specific recommendations
    print("\n🔹 Care Recommendations:")
    if importance >= 0.8:
        print("   ► High Emotional Significance: Monitor for signs of distress related to this memory")
        print("   ► Consider: Reminiscence therapy with positive memories of the mentioned family member")
        print("   ► Consider: Journaling activity about positive memories")
    
    if grief_indicators:
        print("   ► Grief Support: Validate feelings and provide emotional support")
        print("   ► Consider: Scheduled discussion of memories in controlled therapeutic setting")
    
    if combined_score <= -0.3:  # Strongly negative sentiment
        print("   ► Mood Intervention: Consider distraction techniques if distress increases")
        print("   ► Consider: Schedule enjoyable activity following difficult memory discussions")
        print("   ► Alert: Monitor for signs of depression or withdrawal")
    
    print("\n✅ Data saved to structured_memories.csv")
    
    return results

def save_to_csv(results, csv_path="structured_memories.csv"):
    """Save analysis results to CSV file"""
    file_exists = os.path.isfile(csv_path)
    
    fieldnames = [
        "timestamp", "input_text", "sentiment_label", "sentiment_score", 
        "vader_score", "distilbert_score", "importance_label", "importance_score",
        "event_age", "event_date", "location_indicators", "routine_references", 
        "time_indicators", "memory_references", "grief_indicators", 
        "trauma_indicators", "severity_indicators", "potential_concerns"
    ]
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)

if __name__ == "__main__":
    print("\n💬 Alzheimer's Patient Monitoring - Sentiment & Memory Analysis\n")
    print("📊 Results will be saved to structured_memories.csv for narration system")
    while True:
        try:
            user_input = input("\nEnter a patient statement (or type 'exit' to quit):\n> ")
            if user_input.strip().lower() == "exit":
                print("👋 Exiting...")
                break
            if user_input.strip() == "":
                print("⚠️ Please enter a non-empty statement.")
                continue
            
            results = analyze_sentiment(user_input)
            save_to_csv(results)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break