"""
Testing Api
"""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """test_health_check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_phishing():
    """test_predict_phishing"""
    payload = {
        "url": "http://example.com/phishing",
        "email": "example@example.com"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Check keys exist
    assert "probability" in data
    assert "label" in data
    # Probability should be between 0 and 1
    assert 0 <= data["probability"] <= 1
