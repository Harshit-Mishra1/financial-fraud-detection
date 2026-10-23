"""Starter Streamlit page. A trained model is not included yet."""

import streamlit as st

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🔎", layout="wide")
st.title("Financial Fraud Detection")
st.write(
    "Starter dashboard. Add a validated model and input schema before enabling predictions."
)
st.info(
    "No trained model or dataset is included in this starter ZIP. "
    "Build and validate the training pipeline before using this dashboard."
)
