import streamlit as st
import joblib
import pandas as pd

# Load trained model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠")

# Title
st.title("🧠 Mental Health Risk Predictor")
st.write("Predict the likelihood of a person needing mental health treatment based on workplace and lifestyle factors.")

# User input section
def user_inputs():
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
    family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Does your work interfere with mental health?", ["Often", "Rarely", "Never", "Sometimes"])
    no_employees = st.selectbox("Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
    tech_company = st.selectbox("Is it a tech company?", ["Yes", "No"])
    benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Access to mental health care options?", ["Not sure", "Yes", "No"])
    wellness_program = st.selectbox("Wellness program at work?", ["Yes", "No", "Don't know"])
    seek_help = st.selectbox("Encouraged to seek help?", ["Yes", "No", "Don't know"])
    anonymity = st.selectbox("Anonymity protected when seeking help?", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Ease of taking leave for mental health?", ["Somewhat easy", "Somewhat difficult", "Very difficult", "Very easy", "Don't know"])
    mental_health_consequence = st.selectbox("Fear of mental health consequence at work?", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Fear of physical health consequence?", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Comfort discussing with coworkers?", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Comfort discussing with supervisor?", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Would you discuss mental health in interview?", ["Yes", "No", "Maybe"])
    phys_health_interview = st.selectbox("Would you discuss physical health in interview?", ["Yes", "No", "Maybe"])
    mental_vs_physical = st.selectbox("Mental vs Physical illness importance?", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Observed mental health consequences?", ["Yes", "No"])

    user_data = {
        "Age": age,
        "Gender": gender,
        "self_employed": self_employed,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "no_employees": no_employees,
        "remote_work": remote_work,
        "tech_company": tech_company,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "seek_help": seek_help,
        "anonymity": anonymity,
        "leave": leave,
        "mental_health_consequence": mental_health_consequence,
        "phys_health_consequence": phys_health_consequence,
        "coworkers": coworkers,
        "supervisor": supervisor,
        "mental_health_interview": mental_health_interview,
        "phys_health_interview": phys_health_interview,
        "mental_vs_physical": mental_vs_physical,
        "obs_consequence": obs_consequence
    }

    return pd.DataFrame([user_data])

# Collect input
input_df = user_inputs()

# Encode input like training data
def preprocess_input(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.factorize(df[col])[0]
    return df

processed_input = preprocess_input(input_df)

# Predict
if st.button("Predict Mental Health Need"):
    prediction = model.predict(processed_input)[0]
    if prediction == 1:
        st.error("🚨 Person is likely to *need mental health support.*")
    else:
        st.success("✅ Person is *unlikely to need mental health treatment.*")