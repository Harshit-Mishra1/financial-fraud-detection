"""Real-Time Fraud Scoring FastAPI service with SHAP reason codes and Postgres audit logging."""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import mlflow.lightgbm
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from api.db import init_db, log_prediction
from api.schemas import (
    BatchScoreResponse,
    BatchTransactionRequest,
    ReasonCode,
    ScoreResponse,
    TransactionRequest,
)

logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.75"))
MEDIUM_RISK_THRESHOLD = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.35"))

state: dict[str, Any] = {}


def _classify_risk(prob: float) -> tuple[str, str]:
    if prob >= HIGH_RISK_THRESHOLD:
        return "HIGH", "BLOCK"
    if prob >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM", "REVIEW"
    return "LOW", "APPROVE"


def _compute_top_reasons(X: np.ndarray, top_k: int = 3) -> list[ReasonCode]:
    explainer = state.get("explainer")
    feature_names = state.get("feature_names", [])
    if explainer is None:
        return []

    shap_vals = explainer.shap_values(X)
    vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    top_idx = np.argsort(np.abs(vals))[::-1][:top_k]
    return [
        ReasonCode(
            feature=str(feature_names[i]) if i < len(feature_names) else f"f_{i}",
            shap_impact=round(float(vals[i]), 4),
        )
        for i in top_idx
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["db_ready"] = init_db()

    if os.getenv("SKIP_MODEL_LOAD", "false").lower() != "true":
        try:
            local_path = os.getenv("LOCAL_MODEL_PATH")
            if local_path and os.path.exists(local_path):
                model = joblib.load(local_path)
            else:
                mlflow.set_tracking_uri(
                    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
                )
                model_uri = os.getenv("MODEL_URI", "models:/fraud-lightgbm/latest")
                model = mlflow.lightgbm.load_model(model_uri)

            state["model"] = model
            state["explainer"] = shap.TreeExplainer(model)
            state["feature_names"] = list(getattr(model, "feature_name_", []))
        except Exception as exc:
            logger.error("Failed to load model on startup: %s", exc)

    yield
    state.clear()


app = FastAPI(
    title="Real-Time Fraud Scoring Engine",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "db_ready": bool(state.get("db_ready", False)),
    }


@app.post("/score", response_model=ScoreResponse)
def score_transaction(req: TransactionRequest):
    start = time.perf_counter()
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    expected_len = len(state.get("feature_names", []))
    if expected_len and len(req.features) != expected_len:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected_len} features, received {len(req.features)}.",
        )

    X = np.array([req.features], dtype=np.float32)
    prob = float(model.predict_proba(X)[0, 1])
    tier, decision = _classify_risk(prob)
    top_reasons = _compute_top_reasons(X, top_k=3)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    if state.get("db_ready"):
        log_prediction(req.transaction_id, prob, tier, decision, latency_ms)

    return ScoreResponse(
        transaction_id=req.transaction_id,
        fraud_probability=round(prob, 4),
        risk_tier=tier,
        decision=decision,
        top_reasons=top_reasons,
        latency_ms=latency_ms,
    )


@app.post("/batch-score", response_model=BatchScoreResponse)
def score_batch(req: BatchTransactionRequest):
    batch_start = time.perf_counter()
    results = [score_transaction(txn) for txn in req.transactions]
    total_ms = round((time.perf_counter() - batch_start) * 1000, 2)
    return BatchScoreResponse(
        count=len(results),
        total_latency_ms=total_ms,
        results=results,
    )