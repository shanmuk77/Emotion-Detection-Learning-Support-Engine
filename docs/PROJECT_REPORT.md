# Project Report: Emotion-Aware Learning Assistant

## Overview
The **Emotion-Aware Learning Assistant** is a complete, end-to-end, AI-powered system designed to analyze the emotional states of students as they describe their learning challenges. By understanding their feelings (Bored, Frustrated, Confident, Confused, Curious), the system provides personalized, emotionally intelligent learning strategies and actionable next steps.

The solution integrates two independent machine learning classifiers (a deep-learning sequential neural network and a fine-tuned Hugging Face transformer model), a generative AI recommendation engine (Gemini API), and an analytics dashboard to track student emotional journeys over time.

---

## System Architecture
The application is structured as a modular Python system containing:
- **Presentation Layer**: Built with Streamlit (`app.py` and `dashboard.py`).
- **Processing Layer**: Custom text preprocessing pipelines (`utils/preprocessing.py`) and sentiment classification logic (`utils/emotion.py`).
- **Inference Services**: Encapsulated loading and prediction classes for both models (`models/bilstm_service.py` and `models/bert_service.py`).
- **Integration Services**: Gemini API generative recommendations (`services/gemini_service.py`) and persistent CSV logging and similarity lookup (`services/csv_store.py`).
- **Machine Learning Models**:
  1. **BiLSTM Classifier**: Embedding -> Bidirectional LSTM -> Dense softmax network (trained in `training/train_bilstm.py`). Supports a high-quality MLP Neural Network fallback on systems lacking TensorFlow (e.g. Python 3.14).
  2. **BERT Transformer**: Fine-tuned `distilbert-base-uncased` sequence classifier (trained in `training/train_bert.py`).

---

## Machine Learning Pipelines

### 1. Bidirectional LSTM (Sequential Network)
- **Tokenization**: Standard Keras character/word tokenizer mapping text to index sequences.
- **Embedding**: 128-dimensional embedding space representing semantic properties.
- **BiLSTM Layer**: 64-unit Bidirectional LSTM capturing forward and backward sequence context.
- **Dense Layers & Dropout**: High-capacity layers with dropout regularization (30% and 20%) to prevent overfitting on small datasets.
- **Softmax Output**: Yields probability scores for each of the five target emotions: Bored, Frustrated, Confident, Confused, and Curious.

### 2. DistilBERT Transformer (Fine-tuned Classifier)
- Fine-tuned on the stratified dataset split using Hugging Face's `Trainer` API.
- Re-initializes classification heads for 5-class sequence classification.
- Trained using early stopping, dynamic padding, and learning rate scheduling (`2e-5`).

---

## Mixed Emotion Detection
Students rarely experience just one emotion. The system detects mixed emotions by:
1. Sorting emotion prediction probabilities in descending order.
2. Checking which emotions exceed a user-configurable confidence threshold (default: `0.20`).
3. Returning up to a configurable maximum (default: 2) of these emotions.
4. Falling back to the single highest scoring primary emotion if no emotions exceed the threshold.

---

## Persistent Logging and TF-IDF Similarity
- Every interaction (user input field, problem statement, predicted emotion labels, confidence, and timestamp) is logged to `data/learning_history.csv`.
- When the student inputs a challenge, the system extracts previous cases using a TF-IDF vectorizer and calculates **cosine similarity**. The top 3 closest past challenges are shown in the UI, enabling educators or systems to retrieve identical issues and review historical recommendations.
