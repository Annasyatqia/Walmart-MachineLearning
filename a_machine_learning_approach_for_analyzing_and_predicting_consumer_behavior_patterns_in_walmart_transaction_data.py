import streamlit as st
import pandas as pd
import numpy as np
import pickle

# === Load model dan scaler ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

st.set_page_config(page_title="Walmart Purchase Prediction", layout="wide")
st.title("🛍️ Walmart Customer Prediction")

# === Custom Pink CSS ===
st.markdown("""
    <style>
        body { background-color: #fff0f5; }
        h1, h2, h3, h4 { color: #d63384; }
        .stFileUploader label, .stTextInput label, .stSelectbox label, .stNumberInput label {
            color: #d63384; font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("📥 Pilih Metode Input")
mode = st.sidebar.radio("Metode:", ["Upload CSV", "Input Manual"])

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("📁 Upload file CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File berhasil di-upload!")
        st.dataframe(df.head())

        # Preprocess
        df = df[feature_cols]
        df_scaled = scaler.transform(df)

        # Predict
        predictions = model.predict(df_scaled)
        df["Predicted_Purchase"] = predictions
        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df)

elif mode == "Input Manual":
    st.subheader("📝 Formulir Data Pelanggan")

    form = {}
    form["Gender"] = st.selectbox("Gender", ["Male", "Female"])
    form["Age"] = st.number_input("Age", min_value=10, max_value=100, value=30)
    form["City"] = st.selectbox("City", ["A", "B", "C"])
    form["Category"] = st.selectbox("Category", ["Clothing", "Electronics", "Grocery"])
    form["Payment_Method"] = st.selectbox("Payment Method", ["Cash", "Credit Card", "E-Wallet"])
    form["Discount_Applied"] = st.selectbox("Discount Applied", ["Yes", "No"])
    form["Rating"] = st.slider("Rating", 1.0, 5.0, 3.0)
    form["Repeat_Customer"] = st.selectbox("Repeat Customer", ["Yes", "No"])

    # Encoding manual (pastikan sesuai urutan labelencoder dulu!)
    encode_map = {
        "Gender": {"Male": 1, "Female": 0},
        "City": {"A": 0, "B": 1, "C": 2},
        "Category": {"Clothing": 0, "Electronics": 1, "Grocery": 2},
        "Payment_Method": {"Cash": 0, "Credit Card": 1, "E-Wallet": 2},
        "Discount_Applied": {"Yes": 1, "No": 0},
        "Repeat_Customer": {"Yes": 1, "No": 0}
    }

    for k, v in encode_map.items():
        form[k] = v[form[k]]

    input_df = pd.DataFrame([form])[feature_cols]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🎯 Prediksi Pembelian: **${prediction:.2f}**")
