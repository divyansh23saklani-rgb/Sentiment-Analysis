# explainability_tool.py

import lime.lime_text
import numpy as np
from joblib import load
import argparse
import os

# Load trained sentiment model and vectorizer
model = load('model/sentiment_model.pkl')
vectorizer = load('model/vectorizer.pkl')

class_names = ['Negative', 'Neutral', 'Positive']

def predict_proba(texts):
    """
    Vectorize input texts and return class probabilities.
    """
    vectors = vectorizer.transform(texts)
    return model.predict_proba(vectors)

def explain_instance(text, save_path=None, show_inline=True):
    """
    Generates a LIME explanation for a single text input.
    
    Parameters:
    - text: The input string to explain
    - save_path: If provided, saves the explanation HTML to this file
    - show_inline: If True, displays explanation inline (e.g. in notebook)
    """
    explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba, num_features=6)
    
    if show_inline:
        try:
            from IPython.display import display
            display(exp.show_in_notebook(text=True))
        except ImportError:
            print("[⚠️] Inline display only available in Jupyter.")
    
    if save_path:
        exp.save_to_file(save_path)
        print(f"[✅] Explanation saved to {save_path}")

    return exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIME Explainability for Sentiment Analysis")
    parser.add_argument("--text", type=str, required=True, help="Text to explain")
    parser.add_argument("--save", type=str, help="Path to save explanation HTML")
    parser.add_argument("--no-show", action="store_true", help="Disable inline notebook display")

    args = parser.parse_args()

    if not args.text.strip():
        print("❌ Please provide valid input text.")
    else:
        explain_instance(args.text, save_path=args.save, show_inline=not args.no_show)
