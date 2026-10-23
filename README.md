# Financial Fraud Detection

An educational end-to-end machine learning project for detecting potentially fraudulent financial transactions.

## Planned features
- Exploratory data analysis (EDA)
- Class-imbalance handling
- Traditional ML models and a PyTorch neural network
- Evaluation with precision, recall, F1-score, PR-AUC, and confusion matrices
- Decision-threshold tuning
- Model explainability with SHAP
- Optional Streamlit dashboard

## Project status
Starter scaffold only. Dataset, trained models, and verified evaluation results have not been included yet.

## Folder structure
- `data/`: local datasets (not committed to Git)
- `notebooks/`: Jupyter notebooks
- `src/`: data loading, training, evaluation utilities
- `models/`: locally saved model files (not committed to Git)
- `app/`: Streamlit demo
- `tests/`: starter tests

## Setup (Windows)
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Start with `notebooks/01_eda.ipynb` (create your notebook there) or the scripts in `src/`.

## Dataset
This ZIP intentionally does not include a transaction dataset. Use a reputable public dataset and read its license/terms before using it. Place your local CSV file in `data/` and do not commit sensitive or private transaction data.

## Disclaimer
Educational project only; not suitable for real-world financial decisions or production fraud detection without extensive validation, security, privacy, and compliance work.
