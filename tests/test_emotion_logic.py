import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.emotion import primary_emotion, mixed_emotions, scores_to_dict

def test_scores_to_dict():
    scores = [0.1, 0.5, 0.2, 0.1, 0.1]
    labels = ["Bored", "Frustrated", "Confident", "Confused", "Curious"]
    res = scores_to_dict(scores, labels)
    assert res["Frustrated"] == 0.5
    assert res["Confident"] == 0.2

def test_primary_emotion():
    scores = {"Bored": 0.1, "Frustrated": 0.5, "Confident": 0.2}
    assert primary_emotion(scores) == "Frustrated"

def test_mixed_emotions_below_limit():
    # Bored and Frustrated above 0.20
    scores = {"Bored": 0.655, "Frustrated": 0.241, "Confident": 0.05, "Confused": 0.05, "Curious": 0.004}
    assert mixed_emotions(scores, threshold=0.20, max_items=2) == ["Bored", "Frustrated"]

def test_mixed_emotions_max_limit():
    # Bored, Frustrated, and Confused above 0.20. Limit is 2.
    scores = {"Bored": 0.40, "Frustrated": 0.30, "Confused": 0.25, "Confident": 0.03, "Curious": 0.02}
    assert mixed_emotions(scores, threshold=0.20, max_items=2) == ["Bored", "Frustrated"]

def test_mixed_emotions_none_above_threshold():
    # All below threshold 0.30, should return highest scoring (Confident)
    scores = {"Bored": 0.15, "Frustrated": 0.10, "Confident": 0.29, "Confused": 0.23, "Curious": 0.23}
    assert mixed_emotions(scores, threshold=0.30) == ["Confident"]

def test_mixed_emotions_boundary():
    # score exactly on threshold 0.20 should be included
    scores = {"Bored": 0.50, "Frustrated": 0.20, "Confident": 0.15, "Confused": 0.10, "Curious": 0.05}
    assert mixed_emotions(scores, threshold=0.20) == ["Bored", "Frustrated"]
