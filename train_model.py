from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import joblib

# Paths are relative to the backend folder
BASE_DIR = Path(__file__).resolve().parent          # backend/
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "app" / "models"
MODEL_PATH = MODEL_DIR / "career_model.joblib"
METRICS_PATH = BASE_DIR / "data" / "model_metrics.json"


def main():
    print(f"[INFO] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Combine skills and interests into one text field
    df["text"] = df["skills"].fillna("") + " " + df["interests"].fillna("")
    X = df["text"]
    y = df["role"]

    print(f"[INFO] Samples: {len(df)}, Classes: {df['role'].nunique()}")

    model = make_pipeline(
        CountVectorizer(token_pattern=r"[^; ]+"),
        LogisticRegression(max_iter=1000)
    )

    # Cross-validation if data is reasonable
    class_counts = y.value_counts()
    min_class = class_counts.min()

    if len(df) >= 6 and min_class >= 2:
        n_splits = min(5, int(min_class))
        print(f"[INFO] Running {n_splits}-fold cross-validation...")
        scores = cross_val_score(model, X, y, cv=n_splits)
        cv_mean = float(scores.mean())
        print(f"[METRIC] CV Mean Accuracy: {cv_mean:.3f}")
    else:
        print("[WARN] Dataset too small or imbalanced for reliable CV.")
        cv_mean = None

    print("[INFO] Training final model on full dataset...")
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Model saved to: {MODEL_PATH}")

    metrics = {
        "cv_mean_accuracy": cv_mean,
        "n_samples": int(len(df)),
        "n_classes": int(df['role'].nunique()),
        "min_class_count": int(min_class),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to: {METRICS_PATH}")
    print("[INFO] Training pipeline complete.")


if __name__ == "__main__":
    main()
