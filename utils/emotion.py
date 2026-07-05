import numpy as np

def scores_to_dict(scores, labels):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    return {label: float(score) for label, score in zip(labels, scores)}

def primary_emotion(score_dict):
    return max(score_dict, key=score_dict.get)

def mixed_emotions(score_dict, threshold=0.20, max_items=2):
    ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    selected = [name for name, score in ranked if score >= threshold][:max_items]
    if not selected:
        selected = [ranked[0][0]]
    return selected
