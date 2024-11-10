# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_resources():
    # Load model and related data
    model_data = joblib.load('data/ckd_model.pkl')
    scaling_params = joblib.load('data/scaling_params.pkl')
    return model_data, scaling_params

def create_feature_input(feature_name, min_val, max_val, step, default_val):
    return st.number_input(
        f"{feature_name}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_val),
        step=float(step)
    )

def main():
    st.set_page_config(page_title="CKD Prediction Dashboard", layout="wide")

    st.title("Chronic Kidney Disease (CKD) Prediction Dashboard")

    try:
        # Load resources
        model_data, scaling_params = load_resources()
        model = model_data['model']
        feature_importance = model_data['feature_importance']
        scaler = scaling_params['scaler']
        numerical_columns = scaling_params['numerical_columns']
        categorical_columns = scaling_params['categorical_columns']

        # Create two columns
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("Patient Data Input")

            # Create sub-columns for input fields
            input_col1, input_col2, input_col3 = st.columns(3)

            with input_col1:
                # Clinical Measurements
                st.subheader("Clinical Measurements")
                age = create_feature_input("Age", 0, 100, 1, 60)
                bp = create_feature_input("Blood Pressure (mmHg)", 50, 200, 1, 120)
                sg = create_feature_input("Specific Gravity", 1.005, 1.025, 0.005, 1.015)
                al = create_feature_input("Albumin", 0, 5, 1, 0)
                su = create_feature_input("Sugar", 0, 5, 1, 0)

            with input_col2:
                # Blood Tests
                st.subheader("Blood Tests")
                sc = create_feature_input("Serum Creatinine", 0, 15, 0.1, 1.2)
                bgr = create_feature_input("Blood Glucose Random", 70, 500, 1, 120)
                bu = create_feature_input("Blood Urea", 1, 100, 1, 20)
                sod = create_feature_input("Sodium", 125, 145, 1, 135)
                pot = create_feature_input("Potassium", 2.5, 6.5, 0.1, 4.0)

            with input_col3:
                # Blood Counts
                st.subheader("Blood Counts")
                hemo = create_feature_input("Hemoglobin", 3, 20, 0.1, 12)
                pcv = create_feature_input("Packed Cell Volume", 20, 60, 1, 40)
                wc = create_feature_input("White Blood Cells", 2000, 20000, 100, 8000)
                rc = create_feature_input("Red Blood Cells", 2, 8, 0.1, 4.5)

            # Categorical Variables
            st.subheader("Medical History")
            cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

            with cat_col1:
                htn = st.selectbox("Hypertension", ["No", "Yes"])
                dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
                cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])

            with cat_col2:
                appet = st.selectbox("Appetite", ["Good", "Poor"])
                pe = st.selectbox("Pedal Edema", ["No", "Yes"])
                ane = st.selectbox("Anemia", ["No", "Yes"])

            with cat_col3:
                rbc = st.selectbox("RBC in Urine", ["Normal", "Abnormal"])
                pc = st.selectbox("Pus Cells", ["Normal", "Abnormal"])
                pcc = st.selectbox("Pus Cell Clumps", ["NotPresent", "Present"])

            with cat_col4:
                ba = st.selectbox("Bacteria", ["NotPresent", "Present"])

            # Create input dataframe
            if st.button("Predict"):
                # Convert categorical variables to numerical
                categorical_inputs = {
                    'htn': 1 if htn == "Yes" else 0,
                    'dm': 1 if dm == "Yes" else 0,
                    'cad': 1 if cad == "Yes" else 0,
                    'pe': 1 if pe == "Yes" else 0,
                    'ane': 1 if ane == "Yes" else 0,
                    'appet': 1 if appet == "Good" else 0,
                    'rbc': 1 if rbc == "Normal" else 0,
                    'pc': 1 if pc == "Normal" else 0,
                    'pcc': 1 if pcc == "Present" else 0,
                    'ba': 1 if ba == "Present" else 0
                }

                # Create input DataFrame
                input_data = pd.DataFrame({
                    'age': [age],
                    'bp': [bp],
                    'sg': [sg],
                    'al': [al],
                    'su': [su],
                    'bgr': [bgr],
                    'bu': [bu],
                    'sc': [sc],
                    'sod': [sod],
                    'pot': [pot],
                    'hemo': [hemo],
                    'pcv': [pcv],
                    'wc': [wc],
                    'rc': [rc],
                    'rbc': [categorical_inputs['rbc']],
                    'pc': [categorical_inputs['pc']],
                    'pcc': [categorical_inputs['pcc']],
                    'ba': [categorical_inputs['ba']],
                    'htn': [categorical_inputs['htn']],
                    'dm': [categorical_inputs['dm']],
                    'cad': [categorical_inputs['cad']],
                    'appet': [categorical_inputs['appet']],
                    'pe': [categorical_inputs['pe']],
                    'ane': [categorical_inputs['ane']]
                })

                # Ensure columns are in the correct order
                input_data = input_data[model_data['feature_names']]

                # Scale numerical features
                input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

                # Make prediction
                prediction = model.predict(input_data)
                prediction_prob = model.predict_proba(input_data)

                # Display results
                with col2:
                    st.header("Prediction Results")

                    if prediction[0] == 1:
                        st.error("⚠️ High Risk of CKD")
                        risk_probability = prediction_prob[0][1] * 100
                    else:
                        st.success("✅ Low Risk of CKD")
                        risk_probability = prediction_prob[0][0] * 100

                    st.metric(
                        label="Risk Probability",
                        value=f"{risk_probability:.1f}%"
                    )

                    # Display feature importance
                    st.subheader("Key Factors")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    top_features = feature_importance.head(5)
                    sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
                    ax.set_title("Top 5 Important Features")
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please ensure all required files are present in the data directory.")

if __name__ == "__main__":
    main()