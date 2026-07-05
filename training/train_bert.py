import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import MODEL_DIR
from utils.preprocessing import clean_text

try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer
    )
    # Test if torch actually loads its libraries without throwing DLL errors
    x = torch.tensor([1.0])
    HAS_TORCH = True
except (ImportError, OSError) as e:
    HAS_TORCH = False

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert():
    BASE_MODEL = "distilbert-base-uncased"
    DATASET = "data/emotions.csv"
    if not Path(DATASET).exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET}")

    df = pd.read_csv(DATASET).dropna(subset=["text", "emotion"]).drop_duplicates()
    df["text"] = df["text"].map(clean_text)

    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["emotion"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. ALWAYS train and save the scikit-learn Logistic Regression fallback model
    print("\nTraining fallback scikit-learn Logistic Regression classifier for BERT...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["text"]).toarray()
    y = df["label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    fallback_model = LogisticRegression(random_state=42)
    fallback_model.fit(X_train, y_train)
    
    accuracy = fallback_model.score(X_test, y_test)
    print(f"Fallback accuracy: {accuracy:.4f}")
    
    y_pred = fallback_model.predict(X_test)
    print("\nFallback Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    # Save fallback model & tokenizer
    with open(MODEL_DIR / "bert_fallback_model.pkl", "wb") as f:
        pickle.dump({"model": fallback_model, "vectorizer": vectorizer}, f)
    print("Fallback BERT model saved to bert_fallback_model.pkl successfully.")

    # 2. If PyTorch is available, fine-tune the DistilBERT model
    if HAS_TORCH:
        print("\nPyTorch and HF environment detected. Fine-tuning DistilBERT model...")
        try:
            train_df, test_df = train_test_split(
                df[["text", "label"]], test_size=0.2, random_state=42, stratify=df["label"]
            )

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

            def tokenize(batch):
                return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

            train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize, batched=True)
            test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize, batched=True)

            id2label = {i: label for i, label in enumerate(encoder.classes_)}
            label2id = {v: k for k, v in id2label.items()}

            model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL, num_labels=len(id2label), id2label=id2label, label2id=label2id
            )

            args = TrainingArguments(
                output_dir="bert_runs",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                report_to="none"
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                compute_metrics=compute_metrics
            )

            trainer.train()
            eval_results = trainer.evaluate()
            print("Evaluation Results:", eval_results)

            save_path = MODEL_DIR / "bert_emotion_model"
            save_path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            print("BERT PyTorch model saved to bert_emotion_model/ successfully.")
        except Exception as e:
            print(f"Warning: BERT PyTorch fine-tuning failed with error: {str(e)}")
            print("The system will use the trained scikit-learn fallback.")
            
    else:
        # Create empty folder to satisfy path exist checks if only running fallback
        save_path = MODEL_DIR / "bert_emotion_model"
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "config.json", "w") as f:
            f.write('{"fallback_mode": true}')
            
        print("\nPyTorch not available. Skipping DistilBERT training. Fallback classifier will be used.")

if __name__ == "__main__":
    train_bert()
