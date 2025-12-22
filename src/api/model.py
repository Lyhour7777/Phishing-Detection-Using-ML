from logging import config
from pathlib import Path
import joblib
import numpy as np
from src.preprocess.decomposeURL import PhishingFeatureExtractor

MODEL = None
FEATURES = None
EXTRACTOR = PhishingFeatureExtractor()

def load_model(model_path: str):
    global MODEL, FEATURES
    print("[INFO] Loading phishing detection model...")
    MODEL_PATH = Path(model_path)
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        MODEL = bundle["model"]
        FEATURES = bundle["features"]
    else:
        MODEL = bundle
        FEATURES = EXTRACTOR.FEATURE_NAMES
    print("[INFO] Model loaded successfully")

def predict_phishing(url: str) -> dict:
    if not url:
        return {"probability": 0.0, "label": "unknown"}
    features = EXTRACTOR.extract(url)
    X = np.array(features).reshape(1, -1)
    pred = MODEL.predict(X)[0]
    proba = MODEL.predict_proba(X)[0]
    return {
        "probability": float(proba[1]) if pred == 1 else float(proba[0]),
        "label": "legitimate" if pred == 1 else "phishing"
    }