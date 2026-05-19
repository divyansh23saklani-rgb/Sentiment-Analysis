# emotion_classifier.py

from transformers import pipeline
from tqdm import tqdm
from typing import List, Union

# Load the model once globally for reuse
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

def predict_emotion(text: Union[str, List[str]]) -> Union[str, List[dict]]:
    if isinstance(text, str):
        if not text.strip():
            return "No input"
        result = classifier(text)[0]
        return f"{result['label']} (Confidence: {result['score']:.2f})"
    elif isinstance(text, list):
        return [
            {
                "text": t,
                "emotion": classifier(t)[0]["label"],
                "score": round(classifier(t)[0]["score"], 2)
            }
            for t in tqdm(text, desc="Predicting emotions")
        ]
    else:
        raise TypeError("Input must be a string or a list of strings")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Emotion Classifier")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="Path to a text file (one line per input)")
    args = parser.parse_args()

    if args.text:
        emotion = predict_emotion(args.text)
        print(f"Detected Emotion: {emotion}")

    elif args.file and os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        results = predict_emotion(lines)
        for res in results:
            print(f"📝 {res['text']}\n➡️ Emotion: {res['emotion']} (Confidence: {res['score']})\n")

    else:
        print("❗ Please provide --text or --file as input.")
