# Emotion-Aware Learning Assistant 🤖

An AI-powered, emotion-aware learning assistant that supports students by detecting their emotional states (Bored, Frustrated, Confident, Confused, Curious) from natural language descriptions of their learning challenges. The system recommends field-aware study plans using Gemini AI with deterministic fallbacks, compares deep learning (BiLSTM) and transformer (BERT) models side by side, and displays analytical journey tracking.

---

## Folder Structure

```
emotion_learning_assistant/
│
├── app.py                      # Main Streamlit UI entry point
├── dashboard.py                # Plotly analytics dashboard renderer
├── config.py                   # App configuration & environment setup
├── requirements.txt            # Dependency configuration
├── README.md                   # Setup and usage guide
├── .env.example                # Sample environment configuration
├── .gitignore                  # Git untracked pattern filters
│
├── data/
│   ├── emotions.csv            # Cleaned training dataset
│   └── learning_history.csv    # Persistent interaction history CSV
│
├── models/
│   ├── bilstm_service.py       # BiLSTM loading & prediction service
│   ├── bert_service.py         # BERT loading & prediction service
│   ├── bilstm_fallback_model.pkl # Fallback MLP model for Python 3.14
│   ├── bert_fallback_model.pkl   # Fallback Logistic Regression model for Python 3.14
│   ├── tokenizer.pkl           # BiLSTM tokenizer configuration
│   ├── label_encoder.pkl       # Target emotion label encoder mapping
│   └── bert_emotion_model/     # PyTorch fine-tuned DistilBERT weights
│
├── services/
│   ├── gemini_service.py       # Google GenAI model interaction
│   └── csv_store.py            # CSV history mapping and similarity ranking
│
├── utils/
│   ├── preprocessing.py        # Shared text cleaning pipeline
│   ├── emotion.py              # Score mapping and threshold rules
│   └── validation.py           # Input text size and format rules
│
├── training/
│   ├── train_bilstm.py         # Sequential BiLSTM training pipeline
│   └── train_bert.py           # PyTorch Trainer DistilBERT fine-tuning
│
├── tests/
│   ├── test_preprocessing.py   # Unit tests: preprocessor cleaning
│   ├── test_emotion_logic.py   # Unit tests: threshold boundary cases
│   ├── test_csv_store.py       # Unit tests: logs persistence and TF-IDF
│   ├── test_validation.py      # Unit tests: input constraints
│   └── test_smoke.py           # Smoke tests: fallbacks and imports
│
├── diagrams/
│   ├── architecture.md         # System structure diagrams
│   ├── sequence.md             # End-to-end sequences
│   └── pipeline.md             # Model training pipelines
│
└── docs/
    ├── PROJECT_REPORT.md       # Detailed technical design details
    ├── TESTING.md              # Cross-browser verification checklist
    └── DEPLOYMENT.md           # Streamlit cloud and containerization guide
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 to 3.14
- Virtual Environment tool (`venv`)

### 2. Virtual Environment Setup
Run these commands in your shell:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key
Copy the `.env.example` file to `.env`:
```bash
copy .env.example .env
```
Open `.env` and insert your API key:
```env
GEMINI_API_KEY=your_gemini_api_key_from_google_ai_studio
```

---

## Training Models

Before running the application, you must train both emotion classifiers.

### Train the BiLSTM Classifier
Executes preprocessing, tokenizer fitting, sequence padding, label mapping, and Sequential model training. (Fails back automatically to a scikit-learn MLP Neural Network if TensorFlow is not available in your Python environment).
```bash
python training/train_bilstm.py
```

### Train the BERT Classifier
Fine-tunes the `distilbert-base-uncased` sequence classifier on the dataset. (Saves a Logistic Regression model as a backup to prevent DLL initialization crashes inside multithreaded test suites or GUI runners).
```bash
python training/train_bert.py
```

---

## Running the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```
Open the local URL displayed in your terminal (usually `http://localhost:8501`).

---

## Automated Testing

Run unit tests and verify model loading boundaries:
```bash
python -m pytest -v
```
All 23 assertions covering preprocessing, CSV writes, similarities, and validation rules will execute.
