import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("random_forest_under_25mb.pkl")

st.set_page_config(page_title="Random Forest Regression App", layout="centered")

st.title("🌲 Random Forest Regression App")
st.write("Predict values using a trained Random Forest Regressor")

st.divider()

# Input fields (change names according to your dataset)
st.subheader("Enter Input Features")

# Example input fields (EDIT these to match your features)
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)

# Create dataframe
input_data = pd.DataFrame({
    "Feature 1": [feature_1],
    "Feature 2": [feature_2],
    "Feature 3": [feature_3]
})

st.write("### Input Data")
st.dataframe(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Value: {prediction[0]:.2f}")
