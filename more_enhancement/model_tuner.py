# model_tuner.py

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import argparse

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    X_raw = df['Text']
    y = df['Sentiment']

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    X_vec = vectorizer.fit_transform(X_raw)

    return X_vec, y, vectorizer

def benchmark_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (RBF Kernel)": SVC(),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SGD Classifier": SGDClassifier(max_iter=1000, tol=1e-3)
    }

    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()

        preds = model.predict(X_test)
        print(f"✅ {name} completed in {end - start:.2f}s")
        print(classification_report(y_test, preds, digits=3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ML models for sentiment classification")
    parser.add_argument("--data", required=True, help="Path to CSV with 'Text' and 'Sentiment' columns")
    args = parser.parse_args()

    X, y, _ = load_and_preprocess(args.data)
    benchmark_models(X, y)
