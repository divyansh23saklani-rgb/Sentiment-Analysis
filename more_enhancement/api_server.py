# main/api_server.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import uvicorn
from datetime import datetime
import logging
import traceback

# Utility imports
from utils.text_preprocessing import preprocess_text
from utils.logger import log_prediction

# Load model and vectorizer
model = load('model/sentiment_model.pkl')
vectorizer = load('model/vectorizer.pkl')

# FastAPI setup
app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment of a given text with logging and confidence.",
    version="2.0"
)

# CORS (Optional: Adjust origins for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment: str
    confidence: float
    timestamp: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}

# Health check
@app.get("/health")
def health_check():
    return {"status": "OK"}

# Metadata endpoint
@app.get("/model/info")
def model_info():
    return {
        "model_name": "SentimentClassifierV1",
        "version": "1.0.0",
        "framework": "scikit-learn",
        "classes": model.classes_.tolist()
    }

# Prediction endpoint
@app.post("/predict/", response_model=PredictionOutput)
async def predict_sentiment(input: TextInput, request: Request):
    try:
        clean_text = preprocess_text(input.text)
        features = vectorizer.transform([clean_text])
        prediction = model.predict(features)[0]
        confidence = round(max(model.predict_proba(features)[0]), 4)
        timestamp = datetime.now().isoformat()

        log_prediction(input.text, prediction, confidence, timestamp)

        return {
            "sentiment": prediction,
            "confidence": confidence,
            "timestamp": timestamp
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed.")
