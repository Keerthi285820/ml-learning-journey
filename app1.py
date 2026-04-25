import streamlit as st
import pickle
import numpy as np
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
st.title("💰 Medical Insurance Cost Prediction")
st.write("Enter details to estimate medical insurance charges")
age = st.number_input("Age", 18, 100)
bmi = st.number_input("BMI", 10.0, 50.0)
children = st.number_input("Number of Children", 0, 10)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "southeast", "southwest"])
input_data = np.zeros(len(columns))
input_dict = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0
}
for i, col in enumerate(columns):
    if col in input_dict:
        input_data[i] = input_dict[col]
if st.button("Predict Cost"):
    prediction = model.predict([input_data])[0]
    st.success(f"💵 Estimated Cost (USD): ${prediction:.2f}")
    inr = prediction * 83
    st.success(f"🇮🇳 Estimated Cost (INR): ₹{inr:.2f}")
    if smoker == "yes":
        st.warning("⚠️ Smoking significantly increases insurance cost")
st.write("📌 Note: This prediction is based on a US-based insurance dataset.")