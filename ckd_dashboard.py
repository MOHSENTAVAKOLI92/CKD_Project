import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
model, feature_names = joblib.load('data/ckd_best_model_with_features.pkl')

# Load the exact feature template for input
input_data = pd.DataFrame(columns=feature_names)
input_data.loc[0] = 0  # Set all initial values to 0

st.title("Chronic Kidney Disease (CKD) Prediction Dashboard")

# User input for specific features
st.header("Enter Patient Data for Prediction")

# Collecting user inputs
age = st.number_input("Age", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
bgr = st.number_input("Blood Glucose Random", min_value=0)
al = st.number_input("Albumin Level", min_value=0)
hemo = st.number_input("Hemoglobin", min_value=0.0, format="%.1f")
pot = st.number_input("Potassium Level", min_value=0.0, format="%.1f")
sod = st.number_input("Sodium Level", min_value=0.0, format="%.1f")
pe = st.selectbox("Pedal Edema (Swelling of feet)", ["No", "Yes"])
ba = st.selectbox("Presence of Bacteria", ["No", "Yes"])

# Update input data with user values
input_data.at[0, 'age'] = age
input_data.at[0, 'bp'] = blood_pressure
input_data.at[0, 'bgr'] = bgr
input_data.at[0, 'al'] = al
input_data.at[0, 'hemo'] = hemo
input_data.at[0, 'pot'] = pot
input_data.at[0, 'sod'] = sod
input_data.at[0, 'pe'] = 1 if pe == "Yes" else 0
input_data.at[0, 'ba'] = 1 if ba == "Yes" else 0

# Ensure the input_data has the correct order of features as expected by the model
input_data = input_data[feature_names]

# Prediction button
if st.button("Predict CKD"):
    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "CKD Detected" if prediction == 1 else "No CKD"

    st.write(f"Prediction Result: {result}")
