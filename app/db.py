import sqlite3
from pathlib import Path

from app.config import get_settings

settings = get_settings()

# Base directory = backend/
BASE_DIR = Path(__file__).resolve().parent.parent

# Database directory (backend/data)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "careerpath.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skills TEXT,
            interests TEXT,
            experience_years INTEGER,
            prediction TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def save_result(skills, interests, experience_years, prediction, confidence):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (skills, interests, experience_years, prediction, confidence)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            ";".join(skills),
            ";".join(interests),
            experience_years,
            prediction,
            confidence,
        ),
    )
    conn.commit()
    conn.close()


def get_recent_results(limit=10):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT skills, interests, experience_years, prediction, confidence, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows
