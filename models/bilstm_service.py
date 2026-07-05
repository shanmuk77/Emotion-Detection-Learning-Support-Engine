import pickle
import numpy as np
from config import MODEL_DIR, MAX_LEN, EMOTIONS
from utils.preprocessing import clean_text

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

class BiLSTMService:
    def __init__(self):
        self.model_path = MODEL_DIR / "bilstm_model.keras"
        self.fallback_path = MODEL_DIR / "bilstm_fallback_model.pkl"
        self.tokenizer_path = MODEL_DIR / "tokenizer.pkl"
        self.encoder_path = MODEL_DIR / "label_encoder.pkl"
        
        self.model = None
        self.tokenizer = None
        self.encoder = None
        self.fallback_data = None
        self.is_fallback = False

    def load(self):
        # Determine whether to use standard BiLSTM or fallback MLP
        if not HAS_TENSORFLOW or self.fallback_path.exists():
            if not self.fallback_path.exists():
                raise FileNotFoundError("Train/export fallback BiLSTM model first (run training/train_bilstm.py).")
            
            with open(self.fallback_path, "rb") as f:
                self.fallback_data = pickle.load(f)
            self.model = self.fallback_data["model"]
            self.tokenizer = self.fallback_data["vectorizer"]
            self.is_fallback = True
            
            if self.encoder_path.exists():
                with open(self.encoder_path, "rb") as f:
                    self.encoder = pickle.load(f)
            return self
        
        # Standard TensorFlow loading
        if not self.model_path.exists():
            raise FileNotFoundError("Train/export BiLSTM model first (run training/train_bilstm.py).")
        self.model = tf.keras.models.load_model(str(self.model_path))
        with open(self.tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        if self.encoder_path.exists():
            with open(self.encoder_path, "rb") as f:
                self.encoder = pickle.load(f)
        self.is_fallback = False
        return self

    def predict(self, text):
        cleaned = clean_text(text)
        
        # Load labels from label encoder or default list
        labels = list(self.encoder.classes_) if self.encoder is not None else EMOTIONS
        
        if self.is_fallback:
            # Predict using MLP
            features = self.tokenizer.transform([cleaned]).toarray()
            probs = self.model.predict_proba(features)[0]
            return cleaned, {label: float(p) for label, p in zip(labels, probs)}
            
        # Predict using Keras BiLSTM
        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        probs = self.model.predict(padded, verbose=0)[0]
        return cleaned, {label: float(p) for label, p in zip(labels, probs)}
