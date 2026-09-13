import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# --- IMAGE TITLE: Disease Prediction from Symptoms ---
st.title("26. Disease Prediction from Symptoms")
st.caption("Description: Predict potential diseases based on patient-reported symptoms using ML.")

# --- DATASET (Sample for Logic) ---
# Ye data "Multi-disease prediction" ke liye hai
data = {
    'itching': [1,0,0,1,0],
    'skin_rash': [1,1,0,0,0],
    'nodal_skin_eruptions': [1,0,0,0,0],
    'continuous_sneezing': [0,0,1,0,1],
    'shivering': [0,0,1,0,0],
    'Disease': ['Fungal infection', 'Allergy', 'Allergy', 'Drug Reaction', 'Common Cold']
}
df = pd.DataFrame(data)

# --- TECH STACK: Scikit-learn (LabelEncoder) ---
le = LabelEncoder()
df['Disease'] = le.fit_transform(df['Disease'])

X = df.drop('Disease', axis=1)
y = df['Disease']

# --- TECH STACK: XGBoost Model ---
model = XGBClassifier()
model.fit(X, y)

# --- KEY FEATURE: Patient-reported symptoms (UI) ---
st.write("### Select Your Symptoms")
s1 = st.checkbox("Itching")
s2 = st.checkbox("Skin Rash")
s3 = st.checkbox("Nodal Skin Eruptions")
s4 = st.checkbox("Continuous Sneezing")
s5 = st.checkbox("Shivering")

if st.button("Predict"):
    # Input Processing
    user_input = np.array([[1 if s1 else 0, 1 if s2 else 0, 1 if s3 else 0, 
                            1 if s4 else 0, 1 if s5 else 0]])
    
    # --- KEY FEATURE: Multi-disease prediction ---
    prediction_idx = model.predict(user_input)[0]
    disease_name = le.inverse_transform([prediction_idx])[0]
    
    # --- KEY FEATURE: Confidence scores ---
    probability = model.predict_proba(user_input)
    confidence = np.max(probability) * 100
    
    st.success(f"Predicted Disease: {disease_name}")
    st.info(f"Confidence Score: {confidence:.2f}%")
    
    # --- KEY FEATURE: Doctor recommendations ---
    st.warning("Doctor Recommendations: Please consult a Dermatologist or General Physician immediately for a detailed checkup.")

# --- TECH STACK MENTION ---
st.sidebar.markdown("""
**Tech Stack Used:**
* Python
* Scikit-learn
* XGBoost
* Streamlit
""")