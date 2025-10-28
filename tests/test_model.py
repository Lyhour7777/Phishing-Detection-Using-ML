"""Model"""
from src.api.model import predict_phishing, load_model

def test_load_model(capsys):
    """Test that load_model prints the info message."""
    load_model()
    captured = capsys.readouterr()
    assert "[INFO] Loading model..." in captured.out

def test_predict_with_url_or_email():
    """Test normal case when url or email is provided."""
    result = predict_phishing(url="http://example.com")
    assert result["probability"] == 0.85
    assert result["label"] == "phishing"

    result = predict_phishing(email="test@example.com")
    assert result["probability"] == 0.85
    assert result["label"] == "phishing"

def test_predict_none_inputs():
    """Test else branch when both url and email are None."""
    result = predict_phishing(url=None, email=None)
    assert result["probability"] == 0.0
    assert result["label"] == "safe"
