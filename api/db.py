"""PostgreSQL prediction audit logger."""

import logging
import os

import psycopg2

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fraud_user:fraud_password@localhost:5432/fraud_db",
)


def init_db() -> bool:
    """Create the prediction_logs table if it does not already exist."""
    try:
        with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS prediction_logs (
                        id SERIAL PRIMARY KEY,
                        transaction_id TEXT NOT NULL,
                        fraud_probability DOUBLE PRECISION NOT NULL,
                        risk_tier TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        latency_ms DOUBLE PRECISION NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
            )
        return True
    except Exception as exc:
        logger.warning("Database initialization skipped/failed: %s", exc)
        return False


def log_prediction(
    transaction_id: str,
    prob: float,
    tier: str,
    decision: str,
    latency_ms: float,
) -> None:
    """Persist a single scoring event for latency and drift auditing."""
    try:
        with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO prediction_logs
                    (transaction_id, fraud_probability, risk_tier, decision, latency_ms)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                (transaction_id, prob, tier, decision, latency_ms),
            )
    except Exception as exc:
        logger.warning("Failed to persist prediction log for %s: %s", transaction_id, exc)
