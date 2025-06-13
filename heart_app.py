# heart_disease_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
def load_model():
    return joblib.load("heart_disease_model.pkl")

model = load_model()

# Page title
st.title("💓 Heart Disease Prediction App")
st.write("Enter the patient's details below to predict if they may have heart disease.")

# Input features
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.slider("Cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels colored by fluoroscopy", [0, 1, 2, 3])
thall = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Convert categorical values to numeric
sex = 1 if sex == "Male" else 0

# Create input array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thall]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease. Probability: {probability:.2f}")
    else:
        st.success(f"✅ Low risk of heart disease. Probability: {probability:.2f}")
