import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the best trained model
with open("best_breast_cancer_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load dataset to get feature names and scaler
df = pd.read_csv(r"C:\Users\lenovo\Downloads\data.csv")
df.drop(columns=["id", "Unnamed: 32"], inplace=True)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Top 10 features from training
top_10_features = [
    "concave points_worst", "perimeter_worst", "concave points_mean",
    "radius_worst", "perimeter_mean", "area_worst", "radius_mean",
    "area_mean", "concavity_mean", "concavity_worst"
]
X = df[top_10_features]

# Fit scaler on training data
scaler = StandardScaler()
scaler.fit(X)

# Streamlit App Title
st.title("Breast Cancer Prediction App")
st.write("Enter the tumor characteristics to predict whether it is benign or malignant.")

# Create input fields for top 10 features
user_input = []
for feature in top_10_features:
    value = st.number_input(f"Enter {feature}", min_value=0.0, max_value=5000.0, value=1.0)
    user_input.append(value)

# Convert input to NumPy array
user_input = np.array(user_input).reshape(1, -1)

# Scale user input
user_input_scaled = scaler.transform(user_input)

# Button to make prediction
if st.button("Predict"):
    prediction = model.predict(user_input_scaled)[0]
    if prediction == 1:
        st.error("The tumor is **Malignant** (Cancerous). Consult a doctor immediately.")
    else:
        st.success("The tumor is **Benign** (Non-cancerous).")