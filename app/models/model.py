import numpy as np
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "career_model.joblib"

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_roles(texts: list[str], top_k: int = 3):
    model = load_model()

    probs = model.predict_proba(texts)[0]     # probabilities for one input
    classes = model.classes_                 # career labels

    # Sort by confidence (descending)
    sorted_idx = np.argsort(probs)[::-1][:top_k]

    results = []
    for idx in sorted_idx:
        results.append({
            "role": classes[idx],
            "confidence": float(probs[idx])
        })

    return results
