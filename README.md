# Real-Time Financial Fraud Scoring Engine
[![CI](https://github.com/Harshit-Mishra1/financial-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Harshit-Mishra1/financial-fraud-detection/actions/workflows/ci.yml)
[![Live API](https://img.shields.io/badge/Live%20API-Swagger%20UI-009688?logo=fastapi)](https://financial-fraud-detection-7kik.onrender.com/docs)

> **Live Demo:** [https://financial-fraud-detection-7kik.onrender.com/docs](https://financial-fraud-detection-7kik.onrender.com/docs) *(Render free tier — allow ~30s for cold start)*
An end-to-end, containerized machine learning backend for scoring financial transactions in real time using **LightGBM**, **SHAP** explainability, **MLflow** model registry, **FastAPI**, and **PostgreSQL** audit logging.

## Architecture
- **Training & Experiment Tracking (`src/train.py`):** Stratified train/validation/test split, class-imbalance weighting (`scale_pos_weight`), validation-only F1 threshold optimization, and automatic model registration to MLflow.
- **Real-Time Inference API (`api/main.py`):** FastAPI service with `lifespan` model + SHAP `TreeExplainer` loading, Pydantic v2 schema validation, `/health`, `/score`, and `/batch-score` endpoints.
- **Audit Persistence (`api/db.py`):** Logs transaction IDs, fraud probabilities, risk tiers, decisions, and inference latency to PostgreSQL (`prediction_logs`).
- **CI & Containerization:** Split `requirements-train.txt` / `requirements-serve.txt`, Docker Compose stack (`postgres`, `mlflow`, `api`), and GitHub Actions (`ruff` + `pytest`).

## Quick Start (Windows / VS Code)

1. **Place Dataset:** Download a binary fraud dataset (e.g., Kaggle Credit Card Fraud `creditcard.csv`) into `data/raw/creditcard.csv`.
2. **Start PostgreSQL & MLflow:**
   ```powershell
   docker compose up -d --build postgres mlflow
   ```
   Verify the MLflow UI at `http://localhost:5000`.
3. **Train & Register Model:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements-train.txt
   python -m src.train
   ```
4. **Launch FastAPI Scoring Container:**
   ```powershell
   docker compose up -d --build api
   ```
   Open interactive Swagger docs at `http://localhost:8000/docs`.

## API Example

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn_1001", "features": [0.0, -1.35, 1.19, 0.26, 0.16, 0.46, 0.23, 0.09, 0.36, 0.09, -0.55, -0.61, -0.99, -0.31, 1.46, -0.47, 0.20, 0.02, 0.40, 0.25, -0.01, 0.27, -0.11, 0.06, 0.12, -0.18, 0.13, -0.02, 149.62, 0.0]}'
```

## Running Tests

```powershell
pytest -v
```
