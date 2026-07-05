import sys
from pathlib import Path
import pytest

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.bilstm_service import BiLSTMService
from models.bert_service import BERTService
from services.gemini_service import generate_ai_response, FALLBACKS

def test_missing_bilstm_model_error():
    # Instantiating and loading without trained files should raise FileNotFoundError
    service = BiLSTMService()
    # Force dummy non-existent paths for both standard and fallback models
    service.model_path = Path("non_existent_bilstm_model.keras")
    service.fallback_path = Path("non_existent_bilstm_fallback_model.pkl")
    with pytest.raises(FileNotFoundError) as exc_info:
        service.load()
    assert "BiLSTM model" in str(exc_info.value) or "fallback BiLSTM model" in str(exc_info.value)

def test_missing_bert_model_error():
    # Instantiating and loading without trained folder should raise FileNotFoundError
    service = BERTService()
    service.path = Path("non_existent_bert_model")
    service.fallback_path = Path("non_existent_bert_fallback_model.pkl")
    with pytest.raises(FileNotFoundError) as exc_info:
        service.load()
    assert "BERT model" in str(exc_info.value) or "fallback BERT model" in str(exc_info.value)

def test_gemini_fallback_without_key(monkeypatch):
    # Set key to empty to force fallback
    import config
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    
    # Run generate_ai_response
    response = generate_ai_response("Computer Science", "how to build a class", "Confused", 0.95)
    
    # It should return the fallback message for Confused
    assert response == FALLBACKS["Confused"]
    
    # Test for Frustrated
    response_frustrated = generate_ai_response("Math", "wrong equations", "Frustrated", 0.60)
    assert response_frustrated == FALLBACKS["Frustrated"]

def test_app_imports():
    # Verify we can import main streamlit files and dashboard module without syntax/import errors
    import app
    import dashboard
    assert app is not None
    assert dashboard is not None
