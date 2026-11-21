import os

import numpy as np
from fastapi.testclient import TestClient

os.environ["SKIP_MODEL_LOAD"] = "true"

from api.main import app, state


class DummyModel:
    feature_name_ = ("V1", "V2", "V3", "V4")

    def predict_proba(self, X):
        prob = float(np.clip(np.mean(X), 0.0, 1.0))
        return np.array([[1.0 - prob, prob]])


class DummyExplainer:
    def shap_values(self, X):
        return np.array([[0.12, -0.05, 0.44, 0.02]])


def test_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


def test_score_valid_transaction():
    with TestClient(app) as client:
        state["model"] = DummyModel()
        state["explainer"] = DummyExplainer()
        state["feature_names"] = DummyModel.feature_name_
        state["db_ready"] = False

        res = client.post(
            "/score",
            json={"transaction_id": "txn_001", "features": [0.8, 0.9, 0.85, 0.9]},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["decision"] == "BLOCK"
        assert body["risk_tier"] == "HIGH"
        assert len(body["top_reasons"]) == 3


def test_score_rejects_wrong_feature_length():
    with TestClient(app) as client:
        state["model"] = DummyModel()
        state["explainer"] = DummyExplainer()
        state["feature_names"] = DummyModel.feature_name_
        state["db_ready"] = False

        res = client.post(
            "/score",
            json={"transaction_id": "txn_002", "features": [1.0, 2.0]},
        )
        assert res.status_code == 422
