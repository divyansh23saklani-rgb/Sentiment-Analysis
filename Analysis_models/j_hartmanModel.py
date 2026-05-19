import os
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Validate required env variables
required_env_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "CSV_PATH"]
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
CSV_PATH = os.getenv("CSV_PATH")

# ✅ Correct Pinecone initialization for v3
try:
    pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")  # Replace with your actual environment
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        raise ValueError(f"Index '{PINECONE_INDEX_NAME}' not found in Pinecone.")
    index = pinecone.Index(PINECONE_INDEX_NAME)
    print(f"🟢 Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    print(f"❌ Pinecone init failed: {e}")
    exit(1)


# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load sentiment model (J-Hartmann RoBERTa)
print("⚙️ Loading J-Hartmann sentiment model...")
try:
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/sentiment-roberta-large-english-3-epochs")
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/sentiment-roberta-large-english-3-epochs")
except Exception as e:
    print(f"❌ Failed to load sentiment model: {e}")
    exit(1)

def analyze_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()

        labels = ["negative", "neutral", "positive"]
        label_index = torch.argmax(outputs.logits).item()
        return {
            "label": labels[label_index],
            "confidence": round(probs[label_index], 4),
            "score": round(probs[2] - probs[0], 4)  # Compound-like score: pos - neg
        }
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        return {"label": "neutral", "confidence": 0.0, "score": 0.0}

# Load Data
try:
    df = pd.read_csv(CSV_PATH)
    print(f"📊 Loaded CSV with {df.shape[0]} rows from {CSV_PATH}")
except Exception as e:
    print(f"❌ Failed to read CSV: {e}")
    exit(1)

# Process and upsert
for i, row in df.iterrows():
    try:
        text = str(row["chunk"])
        if not text.strip():
            print(f"⚠️ Skipping empty text at row {i}")
            continue

        # Generate embedding
        text_embedding = embedding_model.encode(text).tolist()

        # Sentiment
        sentiment = analyze_sentiment(text)

        # Metadata
        metadata = {
            "url": row["url"] if "url" in row and pd.notna(row["url"]) else "",
            "title": row["title"] if "title" in row and pd.notna(row["title"]) else "",
            "summary": row["summary"] if "summary" in row and pd.notna(row["summary"]) else "",
            "date": row["date"] if "date" in row and pd.notna(row["date"]) else "",
            "sentiment_score": sentiment["score"],
            "sentiment_label": sentiment["label"],
            "text": text,
        }

        # Upsert into Pinecone
        index.upsert([(str(i), text_embedding, metadata)])
        print(f"✅ Upserted row {i}: Sentiment={sentiment['label']}, Score={sentiment['score']}")
    except Exception as e:
        print(f"❌ Failed at row {i}: {e}")
