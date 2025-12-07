import sys
from pathlib import Path

# Ensure the backend root (which contains the 'app' package) is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Root endpoint should return a simple health message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "CareerPathAI" in data["message"]


def test_predict_basic():
    """Predict endpoint should return at least one suggested role."""
    payload = {
      "skills": ["python", "sql"],
      "interests": ["data-analysis"],
      "experience_years": 1,
      "top_k": 3
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) > 0

    first = data["predictions"][0]
    assert "role" in first
    assert "confidence" in first
    assert 0.0 <= first["confidence"] <= 1.0


def test_history():
    """
    History endpoint should return a list.
    Assumes at least one prediction has been made earlier,
    but even if not, it should still return a JSON object with 'results'.
    """
    response = client.get("/api/history?limit=5")
    assert response.status_code == 200

    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
