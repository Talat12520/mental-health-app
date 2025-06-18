import streamlit as st
import joblib
import numpy as np

# Load model and features
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

# Title
st.title("🧠 Mental Health Risk Predictor")

# Form input
def user_input():
    age = st.slider("Age", 18, 65, 25)
    gender = st.selectbox("Gender", ['male', 'female', 'other'])
    family_history = st.selectbox("Family History of Mental Illness", ['Yes', 'No'])
    benefits = st.selectbox("Does Employer Provide Mental Health Benefits?", ['Yes', 'No'])
    care_options = st.selectbox("Access to Care Options", ['Yes', 'No', "Don't know"])
    anonymity = st.selectbox("Is Anonymity Maintained?", ['Yes', 'No', "Don't know"])
    leave = st.selectbox("Ease of Taking Leave", ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"])
    work_interfere = st.selectbox("Mental Health Interference with Work", ['Never', 'Rarely', 'Sometimes', 'Often'])

    input_data = np.array([[age, gender, family_history, benefits, care_options, anonymity, leave, work_interfere]])
    return input_data

# Encode input
def encode_input(input_data, features):
    encoded = []
    for i, col in enumerate(features):
        val = input_data[0][i]
        if isinstance(val, str):
            le = joblib.load(f'encoders/{col}_encoder.pkl')
            encoded.append(le.transform([val])[0])
        else:
            encoded.append(val)
    return np.array([encoded])

# Run prediction
input_data = user_input()
if st.button("Predict"):
    encoded_input = encode_input(input_data, features)
    prediction = model.predict(encoded_input)[0]

    if prediction == 1:
        st.error("❗ You are likely to need mental health treatment.")
    else:
        st.success("✅ You are unlikely to need mental health treatment.")