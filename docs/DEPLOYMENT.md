# Deployment Guide: Emotion-Aware Learning Assistant

This guide describes how to configure, package, and deploy the assistant.

## Prerequisites
- Python 3.10 to 3.14 (Note: Python 3.14 will use the robust scikit-learn MLP fallback for BiLSTM because TensorFlow is not yet compiled for 3.14).
- Streamlit and dependency libraries in `requirements.txt`.
- Gemini API Key from Google AI Studio.

## Environment Configuration
Create a `.env` file in the root folder of the project containing:
```env
GEMINI_API_KEY=your_actual_api_key_here
```
Ensure `.env` is listed in your `.gitignore` so no secrets are accidentally committed.

---

## Deployment Steps (Streamlit Community Cloud)
Streamlit apps can be deployed for free on Streamlit Community Cloud:

1. **Host on GitHub**: Commit the project repository (excluding `.env`, venv, and cache files) to a private or public GitHub repository.
2. **Train Models**: Make sure you have run the training scripts locally or as part of a setup script to populate:
   - `models/bilstm_fallback_model.pkl` (or `bilstm_model.keras`)
   - `models/tokenizer.pkl`
   - `models/label_encoder.pkl`
   - `models/bert_emotion_model/`
   Because GitHub does not track these files (per `.gitignore`), you must include them in the Git repository if you want the app to run immediately on Streamlit Cloud without retraining, or run training as a build step. If files are missing, the app will output an error asking to train the models.
3. **Connect Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Click "New app", select your repository, branch, and entry point (`app.py`).
4. **Configure Secrets**:
   - In Streamlit Cloud settings under "Secrets", paste:
     ```toml
     GEMINI_API_KEY = "your_gemini_api_key"
     ```
   - Streamlit will map these secrets to environment variables automatically.

---

## Containerized Deployment (Docker)

If you want to package the application inside a Docker container, use the following configurations.

### 1. [NEW] Dockerfile
Create a `Dockerfile` in the root:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run model training during container build
RUN python training/train_bilstm.py
RUN python training/train_bert.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. [NEW] docker-compose.yml
```yaml
version: '3.8'

services:
  learning-assistant:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./data:/app/data
```
