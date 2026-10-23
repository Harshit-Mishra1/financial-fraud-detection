"""Dataset loading helpers for the Financial Fraud Detection project."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_transactions(filename: str) -> pd.DataFrame:
    """Load a CSV from the local data directory.

    Put your public dataset in data/ and pass its filename, for example:
    load_transactions("transactions.csv")
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Add a permitted CSV file to the data/ folder."
        )
    return pd.read_csv(path)
