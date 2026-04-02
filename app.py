import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("❤️ Heart Disease Prediction App")

# Inputs
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting BP", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalachh = st.number_input("Max Heart Rate", 60, 220)
exang = st.selectbox("Exercise Angina", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0)
slope = st.selectbox("Slope", [0,1,2,3])
ca = st.selectbox("CA", [0,1,2,3,4])
thal = st.selectbox("Thal", [0,1,2,3,6,7])

if st.button("Predict"):
    input_dict = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalachh': thalachh,
        'oldpeak': oldpeak,
        'sex': sex,
        'cp': cp,
        'fbs': fbs,
        'restecg': restecg,
        'exang': exang,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Align columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale
    input_df[['age','trestbps','chol','thalachh','oldpeak']] = scaler.transform(
        input_df[['age','trestbps','chol','thalachh','oldpeak']]
    )

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk ({prob:.2f})")
