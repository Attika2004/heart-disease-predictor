import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")

saved = load_model()
model, le_sex, le_cp, features = saved["model"], saved["le_sex"], saved["le_cp"], saved["features"]

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the 4 patient attributes below. This is a learning project, not a medical diagnosis tool.")

age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex", list(le_sex.classes_))
cp = st.selectbox("Chest Pain Type", list(le_cp.classes_))
thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)

if st.button("Predict"):
    row = pd.DataFrame([[age, le_sex.transform([sex])[0], le_cp.transform([cp])[0], thalach]],
                       columns=features)
    if model.predict(row)[0] == 1:
        st.error("Prediction: Heart Disease Likely")
    else:
        st.success("Prediction: No Heart Disease")
