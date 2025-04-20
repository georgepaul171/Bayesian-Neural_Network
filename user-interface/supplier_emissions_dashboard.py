import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/georgepaul/Desktop/bayesian-proposal/output-data/supplier_emissions_risk.csv")

st.title("Supplier Emissions Risk Dashboard")

st.write("Upload new supplier emissions CSV:")
uploaded = st.file_uploader("Choose a file", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

st.subheader("Top 10 Risky Suppliers")
top_10 = df.sort_values("Risk_Score", ascending=False).head(10)
st.dataframe(top_10)

st.subheader("Prediction vs Uncertainty")
plt.figure(figsize=(8, 5))
plt.scatter(df["Predicted_Emissions"], df["Uncertainty"], alpha=0.6)
plt.xlabel("Predicted Emissions (kgCO₂e)")
plt.ylabel("Uncertainty")
st.pyplot(plt)

# To run: streamlit run supplier_emissions_dashboard.py