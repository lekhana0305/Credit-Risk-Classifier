import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Risk Classifier", layout="centered", page_icon="🏦")

# --- Load Model and Features ---
@st.cache_resource
def load_model_and_features():
    # Use paths relative to the current file (frontend/app.py -> ../model_pipeline)
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_pipeline", "model.pkl")
    FEATURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_pipeline", "features.pkl")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, features
    else:
        return None, None

model, features = load_model_and_features()

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .result-good {
        background-color: #d4edda;
        color: #155724;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .result-bad {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Credit Risk Classifier")
st.write("Enter the customer's details below to predict their credit risk.")

with st.form("credit_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.selectbox("Job Type", [0, 1, 2, 3], format_func=lambda x: {0: "Unskilled/Non-resident", 1: "Unskilled/Resident", 2: "Skilled", 3: "Highly Skilled"}[x])
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        
    with col2:
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "no_inf"])
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "no_inf"])
        credit_amount = st.number_input("Credit Amount (in DM)", min_value=100.0, max_value=20000.0, value=2000.0)
        duration = st.number_input("Duration (in months)", min_value=1, max_value=72, value=12)
        
    purpose = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "domestic appliances", "repairs", "education", "business", "vacation/others"])
    
    submitted = st.form_submit_button("Predict Risk")
    
if submitted:
    if model is None or features is None:
        st.error("Model not loaded. Please ensure `model.pkl` and `features.pkl` exist in the `model_pipeline` directory.")
    else:
        # 1. Create DataFrame from input
        data = {
            "Age": [age],
            "Sex": [sex],
            "Job": [job],
            "Housing": [housing],
            "Saving accounts": [saving_accounts],
            "Checking account": [checking_account],
            "Credit amount": [credit_amount],
            "Duration": [duration],
            "Purpose": [purpose]
        }
        df = pd.DataFrame(data)
        
        # 2. Preprocess data (matching backend logic)
        df_processed = df.copy()
        
        # Feature Engineering (Dummies)
        df_processed = df_processed.merge(pd.get_dummies(df_processed.Purpose, prefix='Purpose'), left_index=True, right_index=True)
        df_processed = df_processed.merge(pd.get_dummies(df_processed.Sex, prefix='Sex'), left_index=True, right_index=True)
        df_processed = df_processed.merge(pd.get_dummies(df_processed.Housing, prefix='Housing'), left_index=True, right_index=True)
        df_processed = df_processed.merge(pd.get_dummies(df_processed["Saving accounts"], prefix='Savings'), left_index=True, right_index=True)
        df_processed = df_processed.merge(pd.get_dummies(df_processed["Checking account"], prefix='Check'), left_index=True, right_index=True)
        
        # Age categories
        interval = (18, 25, 35, 60, 120)
        cats = ['Student', 'Young', 'Adult', 'Senior']
        df_processed["Age_cat"] = pd.cut(df_processed.Age, interval, labels=cats)
        df_processed = df_processed.merge(pd.get_dummies(df_processed["Age_cat"], prefix='Age_cat'), left_index=True, right_index=True)
        
        df_processed['Credit amount'] = np.log(df_processed['Credit amount'])
        
        # Drop categorical columns
        for col in ["Saving accounts", "Checking account", "Purpose", "Sex", "Housing", "Age_cat"]:
            if col in df_processed.columns:
                del df_processed[col]
                
        # Reindex to match training features exactly
        df_final = df_processed.reindex(columns=features, fill_value=0)
        
        # 3. Predict
        with st.spinner("Analyzing data..."):
            try:
                prediction = model.predict(df_final.values)
                if prediction[0] == 1: # 1 means Bad Risk in the original logic
                    st.markdown('<div class="result-bad">⚠️ Bad Credit Risk</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-good">✅ Good Credit Risk</div>', unsafe_allow_html=True)
                    st.balloons()
            except Exception as e:
                st.error(f"Error during prediction: {e}")
