import pickle
import numpy as np
from config import MODEL_DIR, EMOTIONS
from utils.preprocessing import clean_text

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # Verify torch runs fine
    x = torch.tensor([1.0])
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

class BERTService:
    def __init__(self):
        self.path = MODEL_DIR / "bert_emotion_model"
        self.fallback_path = MODEL_DIR / "bert_fallback_model.pkl"
        self.tokenizer = None
        self.model = None
        self.fallback_data = None
        self.is_fallback = False

    def load(self):
        # Determine whether to use standard BERT or fallback LogisticRegression
        if not HAS_TORCH or self.fallback_path.exists():
            if not self.fallback_path.exists():
                raise FileNotFoundError("Train/export fallback BERT model first (run training/train_bert.py).")
            
            with open(self.fallback_path, "rb") as f:
                self.fallback_data = pickle.load(f)
            self.model = self.fallback_data["model"]
            self.tokenizer = self.fallback_data["vectorizer"]
            self.is_fallback = True
            return self

        # Standard loading
        if not self.path.exists():
            raise FileNotFoundError("Train/export BERT model first (run training/train_bert.py).")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.path))
        self.model.eval()
        self.is_fallback = False
        return self

    def predict(self, text):
        cleaned = clean_text(text)
        
        # Load labels from labels mapping or default list
        labels = EMOTIONS
        
        if self.is_fallback:
            features = self.tokenizer.transform([cleaned]).toarray()
            probs = self.model.predict_proba(features)[0]
            # Map predictions to emotions
            return cleaned, {label: float(p) for label, p in zip(labels, probs)}

        # Standard PyTorch inference
        encoded = self.tokenizer(
            cleaned, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            
        if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
            id2label = self.model.config.id2label
            labels = [id2label.get(i, EMOTIONS[i]) for i in range(len(probs))]
            
        return cleaned, {label: float(p) for label, p in zip(labels, probs)}
