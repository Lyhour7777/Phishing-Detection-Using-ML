#!/usr/bin/env python3
"""
FastAPI backend for Phishing Detection System.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import random
import time
import sys
from pathlib import Path

# Add project root to path for config import
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.config.loader import CONFIG
    API_CONFIG = CONFIG.api
    MODEL_CONFIG = CONFIG.model
    print(f"✅ API Config loaded: {API_CONFIG.host}:{API_CONFIG.port}")
except ImportError as e:
    print(f"⚠️  Config import error: {e}. Using defaults.")
    API_CONFIG = type('obj', (object,), {'host': '127.0.0.1', 'port': 8000, 'reload': True})
    MODEL_CONFIG = type('obj', (object,), {'name': 'distilbert-base-uncased', 'provider': 'huggingface'})

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Phishing Detection API",
    description="API for detecting phishing URLs using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---
class URLPredictionRequest(BaseModel):
    """Request model for single URL prediction."""
    url: str = Field(..., description="URL to analyze for phishing")
    analyze_features: Optional[bool] = Field(False, description="Return detailed feature analysis")

class BatchPredictionRequest(BaseModel):
    """Request model for batch URL prediction."""
    urls: List[str] = Field(..., description="List of URLs to analyze")
    analyze_features: Optional[bool] = Field(False, description="Return detailed feature analysis")

class FeatureAnalysis(BaseModel):
    """Model for detailed feature analysis."""
    has_https: bool
    domain_length: int
    has_suspicious_keywords: bool
    tld_risk_score: float
    path_length: int
    has_subdomain: bool
    num_special_chars: int

class URLPredictionResponse(BaseModel):
    """Response model for URL prediction."""
    url: str
    label: str
    probability: float
    is_phishing: bool
    confidence: str  # high, medium, low
    features: Optional[FeatureAnalysis] = None
    timestamp: float
    model_used: str

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[URLPredictionResponse]
    summary: Dict[str, Any]
    total_time: float

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    model: str

# --- Mock Data for Demonstration ---
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "bank", "paypal", "account", 
    "update", "confirm", "password", "credential", "click", "urgent"
]

SAFE_DOMAINS = [
    "google.com", "github.com", "microsoft.com", "apple.com", 
    "amazon.com", "facebook.com", "twitter.com", "linkedin.com"
]

SUSPICIOUS_TLDS = [".xyz", ".top", ".club", ".click", ".gq", ".ml"]

# --- Helper Functions ---
def extract_url_features(url: str) -> Dict[str, Any]:
    """Extract features from URL."""
    from urllib.parse import urlparse, urlunparse
    
    # Ensure URL has scheme
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        
        # Calculate features
        features = {
            "has_https": url.startswith("https"),
            "domain_length": len(domain),
            "has_suspicious_keywords": any(
                keyword in url.lower() for keyword in SUSPICIOUS_KEYWORDS
            ),
            "tld_risk_score": 0.8 if any(tld in domain for tld in SUSPICIOUS_TLDS) else 0.2,
            "path_length": len(path),
            "has_subdomain": domain.count('.') > 1,
            "num_special_chars": sum(1 for c in url if c in '@!$%&*()_+-=[]{}|;:,.<>?'),
            "domain_parts": len(domain.split('.')),
            "is_ip_address": any(c.isdigit() for c in domain.split('.')[0]) if '.' in domain else False
        }
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features from {url}: {e}")
        return {
            "has_https": False,
            "domain_length": 0,
            "has_suspicious_keywords": False,
            "tld_risk_score": 0.5,
            "path_length": 0,
            "has_subdomain": False,
            "num_special_chars": 0,
            "domain_parts": 0,
            "is_ip_address": False
        }

def mock_predict_phishing(url: str, analyze_features: bool = False) -> URLPredictionResponse:
    """Mock prediction function (replace with actual ML model)."""
    start_time = time.time()
    
    # Extract features
    raw_features = extract_url_features(url)
    
    # Calculate probability based on features
    base_prob = 0.5
    
    # Adjust based on features
    if not raw_features["has_https"]:
        base_prob += 0.2
    
    if raw_features["has_suspicious_keywords"]:
        base_prob += 0.25
    
    if raw_features["tld_risk_score"] > 0.7:
        base_prob += 0.15
    
    if raw_features["has_subdomain"]:
        base_prob += 0.1
    
    if raw_features["is_ip_address"]:
        base_prob += 0.3
    
    # Add some randomness but keep within bounds
    base_prob += random.uniform(-0.1, 0.1)
    base_prob = max(0.01, min(0.99, base_prob))
    
    # Determine result
    is_phishing = base_prob > 0.65
    label = "Phishing" if is_phishing else "Safe"
    
    # Determine confidence
    if base_prob > 0.8 or base_prob < 0.2:
        confidence = "high"
    elif base_prob > 0.65 or base_prob < 0.35:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Prepare features if requested
    features = None
    if analyze_features:
        features = FeatureAnalysis(
            has_https=raw_features["has_https"],
            domain_length=raw_features["domain_length"],
            has_suspicious_keywords=raw_features["has_suspicious_keywords"],
            tld_risk_score=raw_features["tld_risk_score"],
            path_length=raw_features["path_length"],
            has_subdomain=raw_features["has_subdomain"],
            num_special_chars=raw_features["num_special_chars"]
        )
    
    return URLPredictionResponse(
        url=url,
        label=label,
        probability=round(base_prob, 4),
        is_phishing=is_phishing,
        confidence=confidence,
        features=features,
        timestamp=time.time(),
        model_used=MODEL_CONFIG.name if hasattr(MODEL_CONFIG, 'name') else "mock-model"
    )

# --- API Endpoints ---
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Phishing Detection API",
        "version": "1.0.0",
        "status": "running",
        "model": MODEL_CONFIG.name if hasattr(MODEL_CONFIG, 'name') else "mock-model",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "features": "/features/{url}"
        },
        "config": {
            "host": API_CONFIG.host,
            "port": API_CONFIG.port
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        model=MODEL_CONFIG.name if hasattr(MODEL_CONFIG, 'name') else "mock-model"
    )

@app.post("/predict", response_model=URLPredictionResponse)
async def predict(request: URLPredictionRequest):
    """
    Predict if a single URL is phishing.
    
    Example:
    ```json
    {
        "url": "https://example.com",
        "analyze_features": true
    }
    ```
    """
    logger.info(f"Received prediction request for URL: {request.url}")
    
    if not request.url or len(request.url.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="URL parameter is required and cannot be empty"
        )
    
    # Basic URL validation
    url = request.url.strip()
    if len(url) < 4 or "." not in url:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid URL format: {url}"
        )
    
    try:
        # Use mock prediction (replace with actual model)
        result = mock_predict_phishing(url, request.analyze_features)
        logger.info(f"Prediction complete: {url} -> {result.label} (confidence: {result.confidence})")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error for {url}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict if multiple URLs are phishing.
    
    Example:
    ```json
    {
        "urls": ["https://google.com", "http://suspicious-site.com"],
        "analyze_features": false
    }
    ```
    """
    start_time = time.time()
    
    if not request.urls:
        raise HTTPException(
            status_code=400,
            detail="URLs list cannot be empty"
        )
    
    # Limit batch size
    if len(request.urls) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum 100 URLs allowed. Received {len(request.urls)}"
        )
    
    logger.info(f"Processing batch prediction for {len(request.urls)} URLs")
    
    predictions = []
    stats = {
        "phishing": 0,
        "safe": 0,
        "errors": 0
    }
    
    for i, url in enumerate(request.urls):
        try:
            if not url or len(url.strip()) == 0:
                stats["errors"] += 1
                continue
                
            prediction = mock_predict_phishing(url.strip(), request.analyze_features)
            predictions.append(prediction)
            
            if prediction.is_phishing:
                stats["phishing"] += 1
            else:
                stats["safe"] += 1
                
        except Exception as e:
            logger.warning(f"Error processing URL {url}: {str(e)}")
            stats["errors"] += 1
    
    total_time = time.time() - start_time
    
    summary = {
        "total_urls": len(request.urls),
        "phishing_count": stats["phishing"],
        "safe_count": stats["safe"],
        "error_count": stats["errors"],
        "phishing_rate": round(stats["phishing"] / len(request.urls), 4) if request.urls else 0,
        "processing_time_per_url": round(total_time / len(request.urls), 4) if request.urls else 0
    }
    
    return BatchPredictionResponse(
        predictions=predictions,
        summary=summary,
        total_time=total_time
    )

@app.get("/features/{url:path}")
async def analyze_features(url: str):
    """
    Analyze URL features without prediction.
    
    Example: /features/https://example.com
    """
    try:
        features = extract_url_features(url)
        
        # Calculate risk indicators
        risk_indicators = []
        if not features["has_https"]:
            risk_indicators.append("No HTTPS (insecure connection)")
        if features["has_suspicious_keywords"]:
            risk_indicators.append("Contains suspicious keywords")
        if features["tld_risk_score"] > 0.7:
            risk_indicators.append("High-risk TLD detected")
        if features["has_subdomain"]:
            risk_indicators.append("Uses subdomains")
        if features["is_ip_address"]:
            risk_indicators.append("Uses IP address instead of domain name")
        
        return {
            "url": url,
            "features": features,
            "risk_indicators": risk_indicators,
            "risk_level": "high" if len(risk_indicators) >= 3 else 
                          "medium" if len(risk_indicators) >= 2 else 
                          "low"
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Feature analysis failed: {str(e)}"
        )

@app.get("/stats")
async def get_statistics():
    """Get API statistics (mock data)."""
    return {
        "total_predictions": random.randint(1000, 10000),
        "phishing_detected": random.randint(100, 2000),
        "accuracy_rate": round(random.uniform(0.85, 0.95), 3),
        "average_response_time_ms": round(random.uniform(50, 200), 2),
        "active_model": MODEL_CONFIG.name if hasattr(MODEL_CONFIG, 'name') else "mock-model",
        "uptime_hours": round(random.uniform(100, 1000), 1),
        "endpoint_usage": {
            "/predict": random.randint(500, 5000),
            "/batch_predict": random.randint(50, 500),
            "/features": random.randint(100, 1000)
        }
    }

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return {
        "error": True,
        "message": exc.detail,
        "status_code": exc.status_code,
        "timestamp": time.time(),
        "path": request.url.path
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {
        "error": True,
        "message": "Internal server error",
        "status_code": 500,
        "timestamp": time.time(),
        "path": request.url.path
    }

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    
    host = getattr(API_CONFIG, 'host', '127.0.0.1')
    port = getattr(API_CONFIG, 'port', 8000)
    reload = getattr(API_CONFIG, 'reload', True)
    
    print(f"🚀 Starting Phishing Detection API on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health check: http://{host}:{port}/health")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        reload=reload,
        log_level="info"
    )