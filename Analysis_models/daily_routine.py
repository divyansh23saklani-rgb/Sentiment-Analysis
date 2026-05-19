import nltk
import re
import pandas as pd
import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK resources - make these run every time to ensure they're available
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

class MemorySentimentAnalyzer:
    def __init__(self, log_file="memory_log.csv"):
        self.sia = SentimentIntensityAnalyzer()
        self.log_file = log_file
        self.memory_keywords = [
            'forget', 'forgot', 'forgotten', 'remember', 'memory', 'recall',
            'misplaced', 'lost', 'confusion', 'confused', 'mix', 'mixed',
            'appointment', 'schedule', 'diary', 'calendar', 'reminder'
        ]
        self.routine_keywords = [
            'routine', 'daily', 'task', 'chore', 'activity', 'habit',
            'morning', 'evening', 'night', 'afternoon', 'medication', 'pills',
            'shower', 'bath', 'cook', 'cooking', 'clean', 'cleaning', 'walk',
            'exercise', 'eat', 'eating', 'sleep', 'sleeping', 'wake', 'waking',
            'breakfast', 'lunch', 'dinner'
        ]
        self.time_indicators = [
            'today', 'yesterday', 'tomorrow', 'last week', 'next week',
            'morning', 'afternoon', 'evening', 'night', 'always', 'never',
            'sometimes', 'often', 'rarely', 'frequently', 'occasionally',
            'now', 'later', 'soon', 'right now'
        ]
        self.location_indicators = [
            'home', 'house', 'kitchen', 'bedroom', 'bathroom', 'living room',
            'office', 'work', 'store', 'shop', 'market', 'hospital', 'doctor',
            'outside', 'inside', 'upstairs', 'downstairs', 'room', 'delhi'
        ]
        self.severity_indicators = [
            'very', 'extremely', 'really', 'completely', 'totally',
            'slightly', 'somewhat', 'a bit', 'a little', 'mild',
            'severe', 'serious', 'concerning', 'worried', 'scared',
            'hard', 'difficult', 'easy', 'simple', 'tough'
        ]
        
        # Try to load existing log or create new one
        try:
            self.log_df = pd.read_csv(self.log_file)
        except FileNotFoundError:
            self.log_df = pd.DataFrame(columns=[
                'timestamp', 'input_text', 'sentiment_score', 'sentiment_label',
                'memory_references', 'routine_references', 'time_indicators',
                'location_indicators', 'severity_indicators', 'potential_concerns'
            ])
    
    def analyze_input(self, text):
        """Analyze user input for sentiment and memory/routine information."""
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Clean text
        text = text.lower()
        
        # Perform sentiment analysis
        sentiment = self.sia.polarity_scores(text)
        compound_score = sentiment['compound']
        
        # Determine sentiment label
        if compound_score >= 0.05:
            sentiment_label = "Positive"
        elif compound_score <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Extract memory and routine references - fixed to avoid tokenization issues
        memory_refs = self.extract_references(text, self.memory_keywords)
        routine_refs = self.extract_references(text, self.routine_keywords)
        time_refs = self.extract_references(text, self.time_indicators)
        location_refs = self.extract_references(text, self.location_indicators)
        severity_refs = self.extract_references(text, self.severity_indicators)
        
        # Check for potential concerns
        potential_concerns = self.identify_concerns(text, compound_score, memory_refs, routine_refs)
        
        # Store analysis in log
        new_entry = {
            'timestamp': timestamp,
            'input_text': text,
            'sentiment_score': compound_score,
            'sentiment_label': sentiment_label,
            'memory_references': ', '.join(memory_refs) if memory_refs else 'None',
            'routine_references': ', '.join(routine_refs) if routine_refs else 'None',
            'time_indicators': ', '.join(time_refs) if time_refs else 'None',
            'location_indicators': ', '.join(location_refs) if location_refs else 'None',
            'severity_indicators': ', '.join(severity_refs) if severity_refs else 'None',
            'potential_concerns': ', '.join(potential_concerns) if potential_concerns else 'None'
        }
        
        self.log_df = pd.concat([self.log_df, pd.DataFrame([new_entry])], ignore_index=True)
        self.log_df.to_csv(self.log_file, index=False)
        
        return self.format_analysis_results(new_entry)
    
    def extract_references(self, text, keyword_list):
        """Extract phrases containing keywords - fixed to avoid tokenization issues."""
        references = []
        
        # Simple word-based matching without tokenization
        words = text.split()
        
        # First look for exact keywords
        for keyword in keyword_list:
            if ' ' in keyword:  # Multi-word keyword
                if keyword in text:
                    references.append(keyword)
            else:  # Single word keyword
                for word in words:
                    # Strip punctuation for comparison
                    clean_word = word.strip('.,!?;:"\'()[]{}')
                    if clean_word == keyword:
                        references.append(keyword)
                        break
        
        # Then look for phrases containing keywords
        for keyword in keyword_list:
            if keyword in text:
                # Find phrases containing this keyword (simple approach)
                pattern = r'[^.!?]*\b' + re.escape(keyword) + r'\b[^.!?]*[.!?]?'
                matches = re.findall(pattern, text)
                for match in matches:
                    clean_match = match.strip()
                    if clean_match and len(clean_match) < 100:  # Avoid overly long matches
                        references.append(clean_match)
        
        return list(set(references))  # Remove duplicates
    
    def identify_concerns(self, text, sentiment_score, memory_refs, routine_refs):
        """Identify potential concerns based on patterns in the text."""
        concerns = []
        
        # Check for negative sentiment about memory
        if sentiment_score < -0.3 and memory_refs:
            concerns.append("Negative feelings about memory")
        
        # Check for negative sentiment about routine
        if sentiment_score < -0.3 and routine_refs:
            concerns.append("Negative experience with daily activities")
        
        # Check for missed activities or appointments
        missed_patterns = [
            r'miss(ed)?\s\w+\s(appointment|meeting|medication)',
            r'forgot\s(to|my|the)\s\w+',
            r'didn\'t\s(remember|recall)\s\w+'
        ]
        
        for pattern in missed_patterns:
            if re.search(pattern, text):
                concerns.append("Missed activities or appointments")
                break
        
        # Check for confusion about time or place
        confusion_patterns = [
            r'confus(ed|ing)\s(about|with)',
            r'(don\'t|didn\'t)\sknow\s(where|when|what)',
            r'lost\s(track|sense)\sof\s(time|place|day)'
        ]
        
        for pattern in confusion_patterns:
            if re.search(pattern, text):
                concerns.append("Confusion about time or place")
                break
        
        # Check for difficulties with eating or swallowing
        if ('hard to swallow' in text or 'difficult to eat' in text or 
            'trouble eating' in text or 'not tasty' in text):
            concerns.append("Difficulties with eating or appetite")
            
        # Check for routine disruption
        if routine_refs and sentiment_score < -0.2:
            concerns.append("Disrupted daily routine")
            
        # Check for repetitive forgetting
        if "always forget" in text or "keep forgetting" in text:
            concerns.append("Pattern of repetitive forgetting")
        
        return list(set(concerns))  # Remove duplicates
        
    def format_analysis_results(self, analysis):
        """Format the analysis results for display."""
        result = f"--- MEMORY & ROUTINE SENTIMENT ANALYSIS ---\n\n"
        result += f"Date/Time: {analysis['timestamp']}\n\n"
        result += f"User Statement: \"{analysis['input_text']}\"\n\n"
        result += f"SENTIMENT ANALYSIS:\n"
        result += f"• Overall sentiment: {analysis['sentiment_label']} (score: {analysis['sentiment_score']:.2f})\n\n"
        
        result += f"MEMORY REFERENCES:\n"
        memory_refs = analysis['memory_references'] if analysis['memory_references'] != 'None' else None
        if memory_refs:
            result += f"• Found references: {memory_refs}\n"
        else:
            result += "• No specific memory references detected\n"
        
        result += f"\nROUTINE REFERENCES:\n"
        routine_refs = analysis['routine_references'] if analysis['routine_references'] != 'None' else None
        if routine_refs:
            result += f"• Found references: {routine_refs}\n"
        else:
            result += "• No specific routine references detected\n"
        
        result += f"\nCONTEXTUAL INDICATORS:\n"
        time_refs = analysis['time_indicators'] if analysis['time_indicators'] != 'None' else None
        location_refs = analysis['location_indicators'] if analysis['location_indicators'] != 'None' else None
        severity_refs = analysis['severity_indicators'] if analysis['severity_indicators'] != 'None' else None
        
        if time_refs:
            result += f"• Time indicators: {time_refs}\n"
        if location_refs:
            result += f"• Location indicators: {location_refs}\n"
        if severity_refs:
            result += f"• Severity indicators: {severity_refs}\n"
        if not time_refs and not location_refs and not severity_refs:
            result += "• No specific contextual indicators detected\n"
        
        result += f"\nPOTENTIAL CONCERNS:\n"
        concerns = analysis['potential_concerns'] if analysis['potential_concerns'] != 'None' else None
        if concerns:
            result += f"• {concerns}\n"
        else:
            result += "• No specific concerns identified\n"
        
        result += f"\n--- DATA LOGGED FOR MEDICAL RESEARCH ---"
        return result
    
    def get_historical_trends(self):
        """Analyze historical data for trends."""
        if len(self.log_df) < 3:
            return "Insufficient data for trend analysis. Please continue logging entries."
        
        # Basic trend analysis
        result = "=== HISTORICAL TRENDS ANALYSIS ===\n\n"
        
        # Sentiment trend
        sentiment_trend = self.log_df['sentiment_score'].rolling(window=3).mean().iloc[-1]
        prev_sentiment = self.log_df['sentiment_score'].rolling(window=3).mean().iloc[-4] if len(self.log_df) >= 6 else None
        
        result += "SENTIMENT TREND:\n"
        if prev_sentiment is not None:
            trend_direction = "improving" if sentiment_trend > prev_sentiment else "worsening" if sentiment_trend < prev_sentiment else "stable"
            result += f"• Recent sentiment is {trend_direction} (average: {sentiment_trend:.2f})\n"
        else:
            result += f"• Recent average sentiment: {sentiment_trend:.2f}\n"
        
        # Most common concerns
        all_concerns = []
        for concerns in self.log_df['potential_concerns']:
            if concerns != 'None':
                all_concerns.extend(concerns.split(', '))
        
        if all_concerns:
            from collections import Counter
            concern_counts = Counter(all_concerns)
            most_common = concern_counts.most_common(3)
            
            result += "\nMOST COMMON CONCERNS:\n"
            for concern, count in most_common:
                result += f"• {concern}: {count} occurrences\n"
        
        # Frequently mentioned memory/routine issues
        result += "\nFREQUENTLY MENTIONED ISSUES:\n"
        
        memory_mentions = []
        for refs in self.log_df['memory_references']:
            if refs != 'None':
                memory_mentions.extend(refs.split(', '))
        
        routine_mentions = []
        for refs in self.log_df['routine_references']:
            if refs != 'None':
                routine_mentions.extend(refs.split(', '))
        
        if memory_mentions:
            from collections import Counter
            memory_counts = Counter(memory_mentions)
            most_common = memory_counts.most_common(3)
            
            result += "Memory-related:\n"
            for item, count in most_common:
                result += f"• {item}: {count} mentions\n"
        
        if routine_mentions:
            from collections import Counter
            routine_counts = Counter(routine_mentions)
            most_common = routine_counts.most_common(3)
            
            result += "\nRoutine-related:\n"
            for item, count in most_common:
                result += f"• {item}: {count} mentions\n"
        
        return result

def main():
    analyzer = MemorySentimentAnalyzer()
    
    print("Memory & Routine Sentiment Analysis Tool")
    print("Please share your thoughts about your memory or daily routine.")
    print("Type 'quit' to exit or 'trends' to see historical analysis.\n")
    
    while True:
        user_input = input("\nYour statement: ")
        
        if user_input.lower() == 'quit':
            print("Thank you for using the Memory Sentiment Analyzer. Goodbye!")
            break
        elif user_input.lower() == 'trends':
            print("\n" + analyzer.get_historical_trends())
        else:
            analysis_result = analyzer.analyze_input(user_input)
            print("\n" + analysis_result)

if __name__ == "__main__":
    main()