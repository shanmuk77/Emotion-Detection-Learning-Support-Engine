# Testing Guide: Emotion-Aware Learning Assistant

This document details the automated and manual testing strategies for verification of the assistant.

## Automated Tests
Automated tests are implemented using the `pytest` framework and verify key logic components isolated from Streamlit rendering.

### Test Suite Structure
Located in `tests/`:
- `test_preprocessing.py`: Verifies lowercase normalization, space trimming, URL/punctuation removal, and whitespace mapping.
- `test_validation.py`: Tests empty inputs, whitespace-only entries, short-length inputs (<5 characters), and long-length limits (>1000 characters).
- `test_emotion_logic.py`: Evaluates primary emotion lookup, confidence formatting, and boundary conditions for mixed emotion detection (exactly on the threshold, and falling back when all scores are below).
- `test_csv_store.py`: Asserts correct logging table schema creation, append operations, and TF-IDF cosine similarity ranking.
- `test_smoke.py`: Asserts standard imports work, missing-model exceptions are thrown, and Gemini API fallbacks trigger deterministic responses correctly.

### Run Tests Command
To execute the tests, activate your environment and run:
```bash
pytest -v
```

---

## Manual Verification Checklist
Verify the following behaviors in Chrome, Microsoft Edge, and Firefox:

| Area | Component | Expected Behavior | Checked |
|---|---|---|---|
| **Input Form** | Text Area | Reject empty submissions and inputs <5 or >1000 characters with user-friendly warnings. | [ ] |
| | Dropdown | Select study field options (Computer Science, Mathematics, Physics, etc.). | [ ] |
| **Model Panels** | BiLSTM View | Shows primary emotion, confidence percentage, and progressive bar indicators. | [ ] |
| | BERT View | Shows primary emotion, confidence, and bars. Demonstrates disagreements if they exist. | [ ] |
| **Mixed Emotions** | Settings Threshold | Toggling threshold slider/input changes mixed emotions list dynamically (e.g. "Bored + Frustrated"). | [ ] |
| **Generative AI** | LLM Toggle | When "Use AI Response (Gemini)" is checked and API key is set, generates custom responses. | [ ] |
| | Fallback Logic | When Gemini is unchecked or offline, displays deterministic templates (e.g., "Try a 10-minute interactive example..."). | [ ] |
| **Persistence** | CSV Logging | Interacting appends rows to `data/learning_history.csv` without erasing prior entries. | [ ] |
| | Saved Similarity | Table displays top 3 matches from history with accurate similarity scores. | [ ] |
| **Dashboard** | Tab 1: Emotions | Renders primary emotion Pie chart and confidence Line chart. | [ ] |
| | Tab 2: Fields | Renders grouped Bar charts representing study field and model variables. | [ ] |
| | Tab 3: Summary | Key-metric cards show correct statistics (Interactions count, mode emotion, active field, model counts). | [ ] |
