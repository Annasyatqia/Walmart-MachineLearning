import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Model, Scaler, dan Fitur
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# Setup Streamlit
st.set_page_config(page_title="Walmart Purchase Predictor", layout="wide")
st.title("🛒 Walmart Customer Behavior Prediction")

# Tab navigasi
tab1, tab2 = st.tabs(["📁 Upload CSV", "🧾 Input Manual"])

# ============================
# Tab 1: Upload CSV
# ============================
with tab1:
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded!")

        st.write("📊 Sample Data")
        st.dataframe(df.head())

        # Drop NA dan encode jika perlu
        df = df.dropna()
        for col in ['Gender', 'City', 'Category', 'Payment_Method', 'Discount_Applied', 'Repeat_Customer']:
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col])[0]

        # Prediksi
        X = df[features]
        X_scaled = scaler.transform(X)
        df['Predicted_Purchase'] = model.predict(X_scaled)

        st.subheader("🎯 Prediction Result")
        st.dataframe(df[['Customer_ID'] + features + ['Predicted_Purchase']])

        # Clustering
        st.subheader("🧠 Customer Segmentation")
        cluster_data = df[['Customer_ID', 'Predicted_Purchase', 'Age']].copy()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_scaled = scaler.fit_transform(cluster_data[['Predicted_Purchase', 'Age']])
        cluster_data['Cluster'] = kmeans.fit_predict(cluster_scaled)

        st.dataframe(cluster_data)

# ============================
# Tab 2: Input Manual
# ============================
with tab2:
    st.write("Isi form berikut untuk prediksi individual")

    input_data = {}
    input_data['Gender'] = st.selectbox("Gender", ["Male", "Female"])
    input_data['Age'] = st.slider("Age", 18, 65, 30)
    input_data['City'] = st.selectbox("City", ["Urban", "Suburban", "Rural"])
    input_data['Category'] = st.selectbox("Category", ["Electronics", "Clothing", "Grocery"])
    input_data['Payment_Method'] = st.selectbox("Payment Method", ["Credit Card", "Cash", "E-wallet"])
    input_data['Discount_Applied'] = st.selectbox("Discount Applied", ["Yes", "No"])
    input_data['Rating'] = st.slider("Rating", 1, 5, 3)
    input_data['Repeat_Customer'] = st.selectbox("Repeat Customer", ["Yes", "No"])

    if st.button("🔮 Predict"):
        # Encode manual input
        map_dict = {
            'Gender': {"Male": 1, "Female": 0},
            'City': {"Urban": 0, "Suburban": 1, "Rural": 2},
            'Category': {"Electronics": 0, "Clothing": 1, "Grocery": 2},
            'Payment_Method': {"Credit Card": 0, "Cash": 1, "E-wallet": 2},
            'Discount_Applied': {"Yes": 1, "No": 0},
            'Repeat_Customer': {"Yes": 1, "No": 0}
        }

        for col in map_dict:
            input_data[col] = map_dict[col][input_data[col]]

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df[features])
        prediction = model.predict(input_scaled)[0]
        st.success(f"💰 Predicted Purchase Amount: **${prediction:,.2f}**")
