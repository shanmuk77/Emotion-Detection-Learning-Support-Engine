from datetime import datetime
import pandas as pd
from config import HISTORY_CSV

COLUMNS = [
    "timestamp", "field", "problem", "model",
    "emotion", "primary_emotion", "confidence"
]

def ensure_history():
    if not HISTORY_CSV.exists():
        # Ensure directory exists as well
        HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=COLUMNS).to_csv(HISTORY_CSV, index=False)

def append_interaction(field, problem, model, emotion, primary, confidence):
    ensure_history()
    # Format and save interaction record
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "field": field,
        "problem": problem,
        "model": model,
        "emotion": emotion,
        "primary_emotion": primary,
        "confidence": float(confidence),
    }])
    row.to_csv(HISTORY_CSV, mode="a", header=False, index=False)

def load_history():
    ensure_history()
    try:
        df = pd.read_csv(HISTORY_CSV)
        # Verify columns exist, if file is malformed, recreate
        for col in COLUMNS:
            if col not in df.columns:
                raise ValueError("Malformed CSV columns")
        return df
    except Exception:
        # Recreate if file corrupted or empty
        pd.DataFrame(columns=COLUMNS).to_csv(HISTORY_CSV, index=False)
        return pd.read_csv(HISTORY_CSV)

def nearest_saved_examples(text, n=3):
    """Simple TF-IDF similarity over saved problems."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    df = load_history()
    if df.empty:
        return df
        
    # Filter only unique problems or just take all
    problems = df["problem"].fillna("").astype(str).tolist()
    corpus = problems + [text]
    
    try:
        # Set min_df=1 to prevent issues with extremely small corpora
        vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b")
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[-1], matrix[:-1]).reshape(-1)
        
        # Sort indices descending by similarity
        idx = sims.argsort()[::-1][:n]
        
        result = df.iloc[idx].copy()
        result["similarity"] = sims[idx]
        return result
    except Exception:
        # Fallback if TF-IDF fails (e.g. empty vocab)
        df["similarity"] = 0.0
        return df.head(n)
