import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------------
# Load dataset
# -------------------------------
st.title("Customer Segmentation App")

df = pd.read_excel("marketing_campaign.xlsx")

st.subheader("Raw Data Preview")
st.write(df.head())

# -------------------------------
# Preprocessing
# -------------------------------
if 'Age' not in df.columns and 'Year_Birth' in df.columns:
	df['Age'] = 2025 - df['Year_Birth']

	if 'Total_Spending' not in df.columns:
		spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts','MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

		df['Total_Spending'] = df[spend_cols].sum(axis=1)

		features = ['Age', 'Income', 'Total_Spending','NumWebPurchases', 'NumDealsPurchases', 'NumStorePurchases']

	missing_cols = [col for col in features if col not in df.columns]

	if missing_cols:
		st.error(f"Missing columns in dataset: {missing_cols}")

	else:
		X = df[features].copy()
		X = X.fillna(X.mean())

		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)

# -------------------------------
# KMeans
# -------------------------------
st.sidebar.subheader("Clustering Options")
k = st.sidebar.slider("Select number of clusters (K)", 2, 10, 4)

kmeans = KMeans(n_clusters=k, random_state=42)

df['Cluster'] = kmeans.fit_predict(X_scaled)

# -------------------------------
# Map clusters to descriptive names
# -------------------------------
cluster_names = {
	0: "Budget-Conscious / Moderate Buyers",
	1: "High-Value Traditional Buyers",
	2: "Affluent Active Shoppers",
	3: "Deal-Oriented / Frequent Online Shoppers"
	}

df['Cluster_Name'] = df['Cluster'].map(cluster_names)

st.subheader("Clustered Data (first 20 rows)")
st.write(df[features + ['Cluster', 'Cluster_Name']].head(20))

# -------------------------------
# Cluster Summary
# -------------------------------
st.subheader("Cluster Summary")
cluster_summary = df.groupby('Cluster')[features].mean()
st.write(cluster_summary)

# -------------------------------
# PCA Visualization
# -------------------------------
st.subheader("Cluster Visualization (PCA - 2D)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7, 5))

# Plot each cluster separately for proper legend
for cluster_id, cluster_label in cluster_names.items():
	idx = df['Cluster'] == cluster_id
	ax.scatter(
		X_pca[idx, 0], X_pca[idx, 1],
		label=cluster_label,
		alpha=0.7
		)

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("Customer Clusters in 2D space")
ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

# -------------------------------
# Download option
# -------------------------------
st.download_button(
	label="Download Data with Clusters",
	data=df.to_csv(index=False).encode("utf-8"),
	file_name="clustered_customers.csv",
	mime="text/csv"
	)

# -------------------------------
# Predict a new customer cluster
# -------------------------------
st.sidebar.subheader("Predict New Customer Cluster")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Income", min_value=0, max_value=200000, value=50000)
spending = st.sidebar.number_input("Total Spending", min_value=0, max_value=5000, value=1000)
web_purchases = st.sidebar.number_input("Web Purchases", min_value=0, max_value=50, value=5)
deals_purchases = st.sidebar.number_input("Deals Purchases", min_value=0, max_value=50, value=3)
store_purchases = st.sidebar.number_input("Store Purchases", min_value=0, max_value=50, value=10)

if st.sidebar.button("Predict Cluster"):
    new_customer = np.array([[age, income, spending, web_purchases, deals_purchases, store_purchases]])
    new_customer_scaled = scaler.transform(new_customer)
    cluster_label = kmeans.predict(new_customer_scaled)[0]

    st.subheader("Prediction Result")
    st.success(f"This customer belongs to **Cluster {cluster_label}: {cluster_names.get(cluster_label, 'Unknown')}**")

    st.write("Cluster Characteristics:")
    st.write(cluster_summary.loc[cluster_label])

    # Project new customer to PCA space
    new_customer_pca = pca.transform(new_customer_scaled)

    # Plot clusters with new customer
    fig, ax = plt.subplots(figsize=(7,5))
    for cluster_id, cluster_label in cluster_names.items():
        idx = df['Cluster'] == cluster_id
        ax.scatter(
            X_pca[idx, 0], X_pca[idx, 1],
            label=cluster_label,
            alpha=0.6
        )

    ax.scatter(
        new_customer_pca[0, 0], new_customer_pca[0, 1],
        color="red", s=200, marker="*", label="New Customer"
    )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Customer Clusters with New Customer")
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
