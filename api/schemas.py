"""Pydantic v2 request and response contracts for the Fraud Scoring API."""

from typing import Literal

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, examples=["txn_2026_0001"])
    features: list[float] = Field(
        ...,
        min_length=1,
        description="Ordered numeric feature vector matching the training schema.",
    )


class BatchTransactionRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=500)


class ReasonCode(BaseModel):
    feature: str
    shap_impact: float


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    top_reasons: list[ReasonCode]
    latency_ms: float


class BatchScoreResponse(BaseModel):
    count: int
    total_latency_ms: float
    results: list[ScoreResponse]
