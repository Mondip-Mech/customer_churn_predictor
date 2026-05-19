"""Prediction logger — persists every inference request to SQLite.

This lightweight store lets you:
  - Audit all predictions made via the API
  - Detect distribution drift over time
  - Build a feedback loop (label actual outcomes later)
  - Feed the Streamlit monitoring dashboard
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "predictions.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the predictions table if it doesn't already exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                tenure           INTEGER,
                monthly_charges  REAL,
                contract         TEXT,
                internet_service TEXT,
                features_json    TEXT    NOT NULL,
                churn_prediction INTEGER NOT NULL,
                churn_probability REAL   NOT NULL,
                risk_level       TEXT    NOT NULL,
                latency_ms       REAL,
                actual_outcome   INTEGER DEFAULT NULL  -- filled in later for monitoring
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)
        """)
        conn.commit()


def log_prediction(
    features: dict,
    prediction: bool,
    probability: float,
    risk_level: str,
    latency_ms: float,
) -> int:
    """Insert one prediction record; return the new row id."""
    init_db()
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (timestamp, tenure, monthly_charges, contract, internet_service,
                 features_json, churn_prediction, churn_probability, risk_level, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                features.get("tenure"),
                features.get("MonthlyCharges"),
                features.get("Contract"),
                features.get("InternetService"),
                json.dumps(features),
                int(prediction),
                round(probability, 6),
                risk_level,
                round(latency_ms, 3),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_recent_predictions(limit: int = 50) -> list[dict]:
    """Return the most recent N predictions as a list of dicts."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_summary_stats() -> dict:
    """Aggregate stats for the monitoring dashboard."""
    init_db()
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                          AS total_predictions,
                ROUND(AVG(churn_probability), 4)  AS avg_probability,
                SUM(churn_prediction)             AS total_predicted_churn,
                ROUND(AVG(latency_ms), 2)         AS avg_latency_ms,
                COUNT(CASE WHEN risk_level='High'   THEN 1 END) AS high_risk,
                COUNT(CASE WHEN risk_level='Medium' THEN 1 END) AS medium_risk,
                COUNT(CASE WHEN risk_level='Low'    THEN 1 END) AS low_risk
            FROM predictions
        """).fetchone()
    return dict(row) if row else {}
