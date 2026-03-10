import streamlit as st
import pandas as pd
import pickle

# Load trained (FITTED) pipeline
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk")

# ---- User Inputs ----
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)

hypertension = st.selectbox(
    "Hypertension",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

heart_disease = st.selectbox(
    "Heart Disease",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

smoking = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "No Info"]
)

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
HbA1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120)

# ---- Convert input to DataFrame (MATCH TRAINING COLUMNS) ----
input_data = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "smoking_history": [smoking],
    "bmi": [bmi],
    "HbA1c_level": [HbA1c],
    "blood_glucose_level": [glucose]
})

# ---- Prediction ----
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.subheader("Result")
    st.write(f"**Diabetes Probability:** {prob:.2f}")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")