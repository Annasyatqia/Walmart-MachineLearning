import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# Streamlit config
st.set_page_config(page_title="Walmart Consumer Behavior", layout="wide")
st.title("🛒 Walmart Customer Behavior Analysis and Prediction")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your Walmart CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")

    st.subheader("📊 Sample Data")
    st.dataframe(df.head())

    # Preprocessing
    df = df.dropna()
    le = LabelEncoder()
    categorical_cols = ['Gender', 'City', 'Category', 'Payment_Method', 'Discount_Applied', 'Repeat_Customer']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # ==========================
    # 📈 Predictive Modelling
    # ==========================
    st.header("📈 Predict Purchase Amount")

    X = df[['Gender', 'Age', 'City', 'Category', 'Payment_Method', 'Discount_Applied', 'Rating', 'Repeat_Customer']]
    y = df['Purchase_Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_rf = RandomForestRegressor(random_state=42)
    model_rf.fit(X_train, y_train)
    rf_preds = model_rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    lr_preds = model_lr.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

    st.write(f"📊 **Random Forest RMSE**: {rf_rmse:.2f}")
    st.write(f"📉 **Linear Regression RMSE**: {lr_rmse:.2f}")

    # Visual: Actual vs Predicted
    st.subheader("🎯 Actual vs Predicted (Random Forest)")
    fig1, ax1 = plt.subplots()
    ax1.hexbin(y_test, rf_preds, gridsize=40, cmap='Blues', bins='log')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Hexbin Plot: Actual vs Predicted")
    st.pyplot(fig1)

    # Feature Importance
    st.subheader("🔍 Feature Importance")
    importances = model_rf.feature_importances_
    feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)
    fig2, ax2 = plt.subplots()
    sns.barplot(data=feature_df, x='Importance', y='Feature', palette='Blues_d', ax=ax2)
    st.pyplot(fig2)

    # Heatmap
    st.subheader("🔗 Correlation Matrix")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
    st.pyplot(fig3)

    # ==========================
    # 🧠 Customer Segmentation
    # ==========================
    st.header("🧠 Customer Segmentation (K-Means)")

    cluster_df = df.groupby("Customer_ID").agg({
        "Purchase_Amount": "sum",
        "Age": "mean"
    }).reset_index()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_df[['Purchase_Amount', 'Age']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_df['Cluster'] = kmeans.fit_predict(scaled_features)

    st.write("📋 **Cluster Summary**")
    summary = cluster_df.groupby('Cluster').agg({
        'Purchase_Amount': ['mean', 'min', 'max'],
        'Age': ['mean', 'min', 'max'],
        'Customer_ID': 'count'
    })
    summary.columns = ['_'.join(col) for col in summary.columns]
    st.dataframe(summary)

    # Scatter plot
    st.subheader("📌 Cluster Visualization")

    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=cluster_df, x='Age', y='Purchase_Amount', hue='Cluster', palette='Set2', ax=ax4)
    st.pyplot(fig4)

    # Centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    fig5, ax5 = plt.subplots()
    sns.scatterplot(data=cluster_df, x='Purchase_Amount', y='Age', hue='Cluster', palette='Set2', alpha=0.6, ax=ax5)
    ax5.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
    ax5.legend()
    ax5.set_title("K-Means Clustering with Centroids")
    st.pyplot(fig5)

else:
    st.warning("Please upload a dataset to continue.")
