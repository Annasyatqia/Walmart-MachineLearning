import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model, scaler, and features
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# Custom CSS
st.set_page_config(page_title="Walmart Behavior App", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #fff0f5;
        }
        h1, h2, h3 {
            color: #d63384;
        }
        .stButton>button {
            background-color: #d63384;
            color: white;
        }
        .css-1d391kg {
            background-color: #ffe6f0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💖 Walmart Customer Prediction & Segmentation")

# Input options
mode = st.radio("Select Input Mode:", ["Upload CSV", "Manual Input"])

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("📁 Upload your Walmart CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

        try:
            df = df[features]
            df_scaled = scaler.transform(df)
            predictions = model.predict(df_scaled)
            df["Predicted_Purchase"] = predictions
            st.subheader("📈 Prediction Results")
            st.dataframe(df[["Predicted_Purchase"]])
        except Exception as e:
            st.error(f"🚨 Error during prediction: {e}")

elif mode == "Manual Input":
    st.subheader("📝 Manual Input Form")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 80, 25)
    city = st.selectbox("City", ["New York", "Los Angeles", "Chicago"])
    category = st.selectbox("Category", ["Electronics", "Clothing", "Groceries"])
    payment = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Cash"])
    discount = st.selectbox("Discount Applied", ["Yes", "No"])
    rating = st.slider("Rating", 1, 5, 3)
    repeat = st.selectbox("Repeat Customer", ["Yes", "No"])

    # Encode manual input
    input_dict = {
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        "City": {"New York": 0, "Los Angeles": 1, "Chicago": 2}[city],
        "Category": {"Electronics": 0, "Clothing": 1, "Groceries": 2}[category],
        "Payment_Method": {"Credit Card": 0, "Debit Card": 1, "Cash": 2}[payment],
        "Discount_Applied": 1 if discount == "Yes" else 0,
        "Rating": rating,
        "Repeat_Customer": 1 if repeat == "Yes" else 0
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        result = model.predict(input_scaled)[0]
        st.success(f"🎯 Predicted Purchase Amount: **${result:.2f}**")

