"""Dataset loading and validation utilities for Financial Fraud Detection."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_transactions(relative_path: str = "raw/creditcard.csv") -> pd.DataFrame:
    """Load a transaction CSV from the local data/ directory and run basic validation."""
    path = DATA_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place your CSV (e.g. creditcard.csv) inside data/raw/."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded dataset at {path} is empty.")
    return df


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train / validation / test split to prevent threshold-tuning leakage."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val, stratify=y_train_val, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
