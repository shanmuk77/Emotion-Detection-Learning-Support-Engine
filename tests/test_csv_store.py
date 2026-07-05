import sys
import pandas as pd
from pathlib import Path
import pytest

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import services.csv_store as csv_store

@pytest.fixture
def temp_history_csv(tmp_path, monkeypatch):
    temp_file = tmp_path / "test_history.csv"
    monkeypatch.setattr(csv_store, "HISTORY_CSV", temp_file)
    return temp_file

def test_ensure_history(temp_history_csv):
    # Should not exist initially
    assert not temp_history_csv.exists()
    csv_store.ensure_history()
    assert temp_history_csv.exists()
    
    # Check headers
    df = pd.read_csv(temp_history_csv)
    assert list(df.columns) == csv_store.COLUMNS

def test_append_and_load(temp_history_csv):
    csv_store.append_interaction(
        field="Computer Science",
        problem="stuck on recursion call stack depth",
        model="BiLSTM",
        emotion="Confused",
        primary="Confused",
        confidence=0.85
    )
    
    df = csv_store.load_history()
    assert len(df) == 1
    assert df.iloc[0]["field"] == "Computer Science"
    assert df.iloc[0]["problem"] == "stuck on recursion call stack depth"
    assert df.iloc[0]["model"] == "BiLSTM"
    assert df.iloc[0]["primary_emotion"] == "Confused"
    assert float(df.iloc[0]["confidence"]) == pytest.approx(0.85)

def test_similarity_retrieval(temp_history_csv):
    # Add multiple history items
    csv_store.append_interaction("Math", "calculating derivatives and integrals", "BERT", "Confused", "Confused", 0.70)
    csv_store.append_interaction("CS", "writing code for binary search tree", "BiLSTM", "Confident", "Confident", 0.90)
    csv_store.append_interaction("CS", "how recursion works", "BiLSTM", "Confused", "Confused", 0.80)
    
    # Query for similar problems
    results = csv_store.nearest_saved_examples("recursion concept help", n=2)
    assert len(results) == 2
    
    # The first result should be the recursion one because it's most similar
    assert "recursion" in results.iloc[0]["problem"]
    assert results.iloc[0]["similarity"] > results.iloc[1]["similarity"]
