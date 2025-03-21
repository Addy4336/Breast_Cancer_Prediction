import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("breast_cancer_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Streamlit App Title
st.title("Breast Cancer Prediction App")
st.write("Enter the tumor characteristics to predict whether it is benign or malignant.")

# Create input fields for all 30 features
features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Store user inputs
user_input = []
for feature in features:
    value = st.number_input(f"Enter {feature}", min_value=0.0, max_value=5000.0, value=1.0)
    user_input.append(value)

# Convert user input to NumPy array
user_input = np.array(user_input).reshape(1, -1)

# Button to make prediction
if st.button("Predict"):
    # Make prediction
    prediction = model.predict(user_input)[0]

    # Display result
    if prediction == 1:
        st.error("The tumor is **Malignant** (Cancerous). Consult a doctor immediately.")
    else:
        st.success("The tumor is **Benign** (Non-cancerous).")
