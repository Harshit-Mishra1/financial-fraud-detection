"""Thin Streamlit demo client that calls the FastAPI Fraud Scoring Engine."""

import os
import requests
import streamlit as st

API_URL = os.getenv("FRAUD_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Real-Time Fraud Scoring Client", layout="wide")
st.title("Real-Time Fraud Scoring Engine")
st.caption(f"Connected to FastAPI backend: `{API_URL}`")

txn_id = st.text_input("Transaction ID", value="txn_demo_001")
raw_features = st.text_area(
    "Comma-separated feature vector (30 numeric values for creditcard.csv)",
    value=", ".join(["0.0"] * 30),
    height=100,
)

if st.button("Score Transaction", type="primary"):
    try:
        features = [float(x.strip()) for x in raw_features.split(",") if x.strip()]
        resp = requests.post(
            f"{API_URL}/score",
            json={"transaction_id": txn_id, "features": features},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Fraud Probability", f"{data['fraud_probability']:.4f}")
            col2.metric("Risk Tier", data["risk_tier"])
            col3.metric("Decision", data["decision"])
            col4.metric("Latency", f"{data['latency_ms']} ms")
            st.subheader("Top SHAP Risk Drivers")
            st.json(data["top_reasons"])
        else:
            st.error(f"API Error {resp.status_code}: {resp.text}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")
