import pandas as pd
import google.generativeai as genai
import pyttsx3
import time

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)  # Speaking speed (words per minute)
tts_engine.setProperty('volume', 1.0)  # Max volume (0.0 to 1.0)

# (Optional) Set a nicer voice
voices = tts_engine.getProperty('voices')
for voice in voices:
    if 'female' in voice.name.lower():
        tts_engine.setProperty('voice', voice.id)
        break

# Configure Gemini API
YOUR_GEMINI_API_KEY = ""  # add your Gemini API key here
genai.configure(api_key=YOUR_GEMINI_API_KEY)

# Load memories from CSV
csv_path = "structured_memories.csv"  # Ensure this file exists
def load_memories(csv_path):
    return pd.read_csv(csv_path)

# Create a prompt for Gemini (updated to second-person narration)
def create_prompt_for_row(row):
    return f"""
You are a warm and empathetic AI narrator.

Here is a memory entry:
- Timestamp: {row['timestamp']}
- Sentiment: {row['sentiment_label']} ({row['sentiment_score']})
- Input: "{row['input_text']}"
- Memory References: {row['memory_references']}
- Routine References: {row['routine_references']}
- Time Indicators: {row['time_indicators']}
- Location Indicators: {row['location_indicators']}
- Severity Indicators: {row['severity_indicators']}
- Potential Concerns: {row['potential_concerns']}

Narrate this memory by talking **directly to the user**, using "you" (second-person perspective).
Speak warmly, gently reminding them of what they experienced, and make it vivid and slightly poetic.
Do not use "I" — only focus on "you".
Keep it natural and empathetic.
"""

# Generate narration from Gemini
def narrate(model, prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Error generating content: {e}]"

# Speak text properly
def speak_text(text):
    sentences = text.split('. ')
    for sentence in sentences:
        if sentence.strip():
            tts_engine.say(sentence.strip())
            tts_engine.runAndWait()
            time.sleep(0.2)

# Main logic
def main():
    df = load_memories("structured_memories.csv")
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    print(f"\n✅ Total entries: {len(df)}")
    narr_count = 0

    for idx, row in df.iterrows():
        if str(row['sentiment_label']).strip().lower() == "negative":
            continue

        input(f"\n👉 Press Enter to narrate memory #{narr_count + 1}...\n")

        prompt = create_prompt_for_row(row)
        narration = narrate(model, prompt)

        print(f"\n🧠 Narration #{narr_count + 1}:\n{narration}\n{'='*60}")

        speak_text(narration)
        narr_count += 1

    print("\n✅ Done narrating all positive/neutral memories.")

if __name__ == "__main__":
    main()