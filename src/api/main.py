import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from src.api.model import load_model, predict_phishing
from src.config.loader import load_config

CONFIG = load_config("src/config/settings.yaml")
MODEL_PATH = f"{CONFIG.paths.model_dir}/{CONFIG.model.name}.pkl"

class URLPredictionRequest(BaseModel):
    url: str = Field(..., description="URL to analyze for phishing")

class URLPredictionResponse(BaseModel):
    url: str
    label: str
    probability: float
    timestamp: float

class BatchPredictionRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to analyze")

class BatchPredictionResponse(BaseModel):
    predictions: List[URLPredictionResponse]

app = FastAPI(title="Phishing Detection API", version="1.0.0")

@app.on_event("startup")
def startup_event():
    load_model(MODEL_PATH)

@app.post("/predict", response_model=URLPredictionResponse)
def predict(request: URLPredictionRequest):
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    try:
        result = predict_phishing(url)
        return URLPredictionResponse(
            url=url,
            label=result["label"],
            probability=result["probability"],
            timestamp=time.time()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest):
    if not request.urls:
        raise HTTPException(status_code=400, detail="URLs list cannot be empty")
    predictions = []
    for url in request.urls:
        url = url.strip()
        if not url:
            continue
        try:
            result = predict_phishing(url)
            predictions.append(URLPredictionResponse(
                url=url,
                label=result["label"],
                probability=result["probability"],
                timestamp=time.time()
            ))
        except Exception:
            predictions.append(URLPredictionResponse(
                url=url,
                label="error",
                probability=0.0,
                timestamp=time.time()
            ))
    return BatchPredictionResponse(predictions=predictions)

@app.get("/")
def root():
    return {"message": "Phishing Detection API running", "version": "1.0.0"}