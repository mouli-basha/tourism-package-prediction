# tourism_project/deployment/app.py

import pandas as pd
import joblib
import streamlit as st
import requests
from io import BytesIO

# Page & Runtime config
st.set_page_config(page_title="Tourism Package Prediction", layout="centered")
st.set_option("server.enableXsrfProtection", False)  # disable XSRF for HF Spaces

# HF locations
MODEL_URL = "https://huggingface.co/moulibasha/tourism-package-prediction-model/resolve/main/model.pkl"
TRAIN_CSV_URL = "https://huggingface.co/datasets/moulibasha/tourism-package-prediction-train-test/resolve/main/train.csv"
TARGET_COL = "ProdTaken"

@st.cache_resource
def load_model():
    resp = requests.get(MODEL_URL, timeout=60)
    resp.raise_for_status()
    return joblib.load(BytesIO(resp.content))

@st.cache_data
def infer_feature_columns():
    try:
        df = pd.read_csv(TRAIN_CSV_URL, nrows=300)
        cols = [c for c in df.columns if c != TARGET_COL]
        if cols:
            return cols
    except Exception:
        pass
    return [
        "Age","TypeofContact","CityTier","DurationOfPitch","Occupation","Gender",
        "NumberOfPersonVisiting","NumberOfFollowups","ProductPitched","PreferredPropertyStar",
        "MaritalStatus","NumberOfTrips","Passport","PitchSatisfactionScore","OwnCar",
        "NumberOfChildrenVisiting","Designation","MonthlyIncome"
    ]

st.title("Tourism Package Prediction")
st.caption("Predict whether a customer will purchase the tourism package (ProdTaken = 1).")

model = load_model()
feature_order = infer_feature_columns()

with st.form("inference_form"):
    st.subheader("Enter customer details")
    inputs = {c: st.text_input(c, "") for c in feature_order}
    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([inputs])
    # best-effort casting
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col])
        except Exception:
            X[col] = X[col].astype(str).str.strip().str.lower()

    y_pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else None

    st.success(f"Prediction: {'1 (Taken)' if y_pred == 1 else '0 (Not Taken)'}")
    if prob is not None:
        st.write(f"Probability of Taken (class=1): **{prob:.3f}**")
    with st.expander("View submitted features"):
        st.dataframe(X)
