# Walmart Customer Behavior Streamlit App - Lengkap
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Streamlit config
st.set_page_config(page_title="Walmart Analysis", layout="wide")
st.title("🛍️ Walmart Customer Behavior Analysis and Prediction")

# Tabs
tab1, tab2 = st.tabs(["📁 Upload Dataset", "📝 Input Manual"])

# ======================
# 📁 Upload CSV
# ======================
with tab1:
    uploaded_file = st.file_uploader("Upload your Walmart CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File berhasil diupload")
        st.dataframe(df.head())

        # Preprocessing
        df = df.dropna()
        for col in ['Gender', 'City', 'Category', 'Payment_Method', 'Discount_Applied', 'Repeat_Customer']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Prediction
        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        df['Predicted_Purchase_Amount'] = model.predict(X_scaled)

        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df[['Customer_ID', 'Predicted_Purchase_Amount']].head())

        # Visualisasi Actual vs Predicted jika kolom asli ada
        if 'Purchase_Amount' in df.columns:
            st.subheader("🎯 Actual vs Predicted")
            fig1, ax1 = plt.subplots()
            ax1.scatter(df['Purchase_Amount'], df['Predicted_Purchase_Amount'], alpha=0.5, color='deeppink')
            ax1.plot([df['Purchase_Amount'].min(), df['Purchase_Amount'].max()],
                     [df['Purchase_Amount'].min(), df['Purchase_Amount'].max()], 'r--')
            ax1.set_xlabel("Actual")
            ax1.set_ylabel("Predicted")
            st.pyplot(fig1)

        # Heatmap korelasi
        st.subheader("💖 Correlation Matrix")
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="pink", fmt=".2f", ax=ax2)
        st.pyplot(fig2)

        # Clustering
        st.subheader("🧠 Customer Segmentation")
        cluster_df = df.groupby("Customer_ID").agg({"Predicted_Purchase_Amount":"sum", "Age":"mean"}).reset_index()
        cluster_scaled = StandardScaler().fit_transform(cluster_df[['Predicted_Purchase_Amount', 'Age']])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_df['Cluster'] = kmeans.fit_predict(cluster_scaled)

        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=cluster_df, x='Age', y='Predicted_Purchase_Amount', hue='Cluster', palette='pastel', ax=ax3)
        st.pyplot(fig3)

        st.write("📌 Centroid Location")
        centroids = StandardScaler().fit(cluster_df[['Predicted_Purchase_Amount','Age']]).inverse_transform(kmeans.cluster_centers_)
        st.dataframe(pd.DataFrame(centroids, columns=['Predicted_Purchase_Amount', 'Age']))

# ======================
# 📝 Input Manual
# ======================
with tab2:
    st.subheader("Masukkan Data Customer")
    form = st.form("manual_input")
    gender = form.selectbox("Gender", ['Male', 'Female'])
    age = form.slider("Age", 18, 70, 25)
    city = form.selectbox("City", ['New York', 'Los Angeles', 'Chicago'])
    category = form.selectbox("Category", ['Electronics', 'Clothing', 'Grocery'])
    payment_method = form.selectbox("Payment Method", ['Credit Card', 'Cash', 'Online'])
    discount = form.selectbox("Discount Applied", ['Yes', 'No'])
    rating = form.slider("Rating", 1.0, 5.0, 3.0)
    repeat_customer = form.selectbox("Repeat Customer", ['Yes', 'No'])

    submitted = form.form_submit_button("Predict")

    if submitted:
        manual_df = pd.DataFrame([{
            'Gender': gender,
            'Age': age,
            'City': city,
            'Category': category,
            'Payment_Method': payment_method,
            'Discount_Applied': discount,
            'Rating': rating,
            'Repeat_Customer': repeat_customer
        }])

        # Encoding manual
        for col in ['Gender', 'City', 'Category', 'Payment_Method', 'Discount_Applied', 'Repeat_Customer']:
            le = LabelEncoder()
            manual_df[col] = le.fit_transform(manual_df[col])

        X_manual = manual_df[feature_cols]
        X_manual_scaled = scaler.transform(X_manual)
        pred = model.predict(X_manual_scaled)[0]

        st.success(f"🎉 Predicted Purchase Amount: ${pred:,.2f}")
