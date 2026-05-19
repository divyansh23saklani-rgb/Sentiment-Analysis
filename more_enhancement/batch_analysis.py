# batch_analysis.py

import pandas as pd
from joblib import load
import os
from datetime import datetime
from tqdm import tqdm
from utils.text_preprocessing import preprocess_text

# Load model and vectorizer
model = load('model/sentiment_model.pkl')
vectorizer = load('model/vectorizer.pkl')

def batch_sentiment(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    if 'Text' not in df.columns:
        print("❌ 'Text' column not found in CSV.")
        return

    sentiments = []
    confidences = []

    print("🔍 Performing batch sentiment analysis...")
    for text in tqdm(df['Text'], desc="Processing"):
        try:
            clean_text = preprocess_text(str(text))
            features = vectorizer.transform([clean_text])
            prediction = model.predict(features)[0]
            confidence = round(max(model.predict_proba(features)[0]), 4)
        except Exception as e:
            prediction = "error"
            confidence = 0.0

        sentiments.append(prediction)
        confidences.append(confidence)

    df['Sentiment'] = sentiments
    df['Confidence'] = confidences

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = file_path.replace(".csv", f"_with_sentiments_{timestamp}.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Sentiment analysis complete. File saved to: {output_path}")

if __name__ == "__main__":
    file_path = input("📄 Enter CSV file path: ").strip()
    batch_sentiment(file_path)
