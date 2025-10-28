"""FastAPI entrypoint for Phishing Detection API."""

from fastapi import FastAPI
from src.api.schemas import PhishingRequest, PhishingResponse
from src.api.model import predict_phishing
from src.config.loader import CONFIG


app = FastAPI(
    title="Phishing Detection API",
    description="API for predicting phishing URLs and emails",
    version="0.1.0"
)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict", response_model=PhishingResponse)
def predict_endpoint(request: PhishingRequest):
    """Predict phishing probability for a URL or email."""
    result = predict_phishing(url=request.url, email=request.email)
    return result

if __name__ == "__main__":
    import uvicorn

    api_conf = CONFIG.get("api", {})
    host = api_conf.get("host", "127.0.0.1")
    port = api_conf.get("port", 8000)
    reload_flag = api_conf.get("reload", False)

    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload_flag)
    