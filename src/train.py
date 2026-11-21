"""Train a cost-sensitive LightGBM fraud classifier and register it in MLflow."""

import argparse
import os

import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from src.data import load_transactions, split_dataset
from src.evaluate import evaluate_predictions, find_optimal_threshold

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "financial-fraud-detection"
REGISTERED_MODEL_NAME = "fraud-lightgbm"


def train_and_log(dataset_path: str = "raw/creditcard.csv", target_col: str = "Class") -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_transactions(dataset_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, target_col=target_col)


    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="lightgbm-cost-sensitive"):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        # Tune decision threshold strictly on validation split
        val_probs = model.predict_proba(X_val)[:, 1]
        best_threshold, _ = find_optimal_threshold(y_val, val_probs)
        val_metrics = evaluate_predictions(y_val, val_probs, threshold=best_threshold)

        # Evaluate final generalization on untouched test split
        test_probs = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_predictions(y_test, test_probs, threshold=best_threshold)

        mlflow.log_params(params)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(
            f"Run complete | Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
            f"Test PR-AUC: {test_metrics['pr_auc']:.4f} | "
            f"Test F1: {test_metrics['f1']:.4f} | "
            f"Optimal Threshold: {best_threshold:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM fraud model and log to MLflow.")
    parser.add_argument("--data", default="raw/creditcard.csv", help="Path relative to data/")
    parser.add_argument("--target", default="Class", help="Binary target column name")
    args = parser.parse_args()
    train_and_log(dataset_path=args.data, target_col=args.target)
