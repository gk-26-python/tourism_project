import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Wellness Tourism Package Purchase Predictor",
    page_icon="✈️",
    layout="centered"
)

# 1. Load the trained model
MODEL_PATH = 'tourism_project/deployment/best_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        # Fallback to absolute path or display warning if missing
        return None

model = load_model()

st.title(" Wellness Tourism Package Predictor")
st.write("Predict whether a customer is likely to purchase the new Wellness Tourism Package.")

if model is None:
    st.error(f"Trained model file could not be found at `{MODEL_PATH}`. Please train the model and commit it to the repository first.")
else:
    st.success("Trained model loaded successfully!")
    
    st.subheader("Customer Profile Information")
    
    # Divide form inputs into two columns
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        TypeofContact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
        CityTier = st.selectbox("City Tier", [1, 2, 3], index=0)
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        NumberOfPersonVisiting = st.slider("Number of Persons Accompanying", 1, 10, 2)
        PreferredPropertyStar = st.selectbox("Preferred Hotel Star Rating", [3.0, 4.0, 5.0], index=0)
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        NumberOfTrips = st.number_input("Average Annual Trips", min_value=0, max_value=20, value=3, step=1)

    with col2:
        Passport = st.selectbox("Has Valid Passport?", ["Yes", "No"])
        OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
        NumberOfChildrenVisiting = st.slider("Number of Children Accompanying (Under 5)", 0, 5, 0)
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Gross Monthly Income (INR)", min_value=0.0, value=25000.0, step=1000.0)
        PitchSatisfactionScore = st.slider("Sales Pitch Satisfaction Score", 1, 5, 3)
        ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
        NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
        DurationOfPitch = st.number_input("Duration of Pitch (Minutes)", min_value=0, max_value=120, value=15, step=1)

    # Map inputs to DataFrame representation
    passport_val = 1 if Passport == "Yes" else 0
    own_car_val = 1 if OwnCar == "Yes" else 0
    
    input_data = pd.DataFrame({
        'Age': [Age],
        'TypeofContact': [TypeofContact],
        'CityTier': [CityTier],
        'Occupation': [Occupation],
        'Gender': [Gender],
        'NumberOfPersonVisiting': [NumberOfPersonVisiting],
        'PreferredPropertyStar': [PreferredPropertyStar],
        'MaritalStatus': [MaritalStatus],
        'NumberOfTrips': [NumberOfTrips],
        'Passport': [passport_val],
        'OwnCar': [own_car_val],
        'NumberOfChildrenVisiting': [NumberOfChildrenVisiting],
        'Designation': [Designation],
        'MonthlyIncome': [MonthlyIncome],
        'PitchSatisfactionScore': [PitchSatisfactionScore],
        'ProductPitched': [ProductPitched],
        'NumberOfFollowups': [NumberOfFollowups],
        'DurationOfPitch': [DurationOfPitch]
    })

    if st.button("Predict Purchase Likelihood"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        st.write("---")
        if prediction == 1:
            st.success(f"**Prediction:** The customer is likely to purchase the Wellness Tourism Package! (Probability: {prediction_proba:.2%})")
        else:
            st.warning(f" **Prediction:** The customer is unlikely to purchase the Wellness Tourism Package. (Probability: {prediction_proba:.2%})")
