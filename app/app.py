from pandas.core.indexes import category
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

@st.cache_resource
def load_artifacts():
    # Locate the project root directory
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"

    # Load the trained XGBoost model
    with open(MODELS_DIR / "xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load the label encoders
    with open(MODELS_DIR / "label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders

model, label_encoders = load_artifacts()

st.title("Demand Forcasting App")

st.divider()

st.header("Input Features")

price = st.number_input("Price", min_value=0.0, value = 50.0)
discount = st.number_input("Discount (%)", min_value=0, max_value=100, value=10)
inventory_level = st.number_input("Inventory Level",min_value=0,value=100)
promotion = st.selectbox("Promotion", [0,1])
competitor_pricing = st.number_input("Competitor Price", min_value=0.0, value= 50.0)

category = st.selectbox(
    "Category", label_encoders["Category"].classes_.tolist()

)

input_data = pd.DataFrame({
    "Price" :[price],
    "Discount" : [discount],
    "Inventory Level" :[inventory_level],
    "Promotion" :[promotion],
    "Competitor Pricing": [competitor_pricing],
    "Category" : [category]}

)

for col, encoder in label_encoders.items():
  if col in input_data.columns:
    input_data[col] = encoder.transform(input_data[col])

st.divider()

if st.button("Predict Demand"):
  prediction = model.predict(input_data)[0]
  st.success(f"Predict Demand: {int(prediction)} Units")
