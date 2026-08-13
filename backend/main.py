from fastapi import FastAPI
from .schemas import CreditApplication
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Credit Risk Classifier API")

# Load model and features
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_pipeline", "model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_pipeline", "features.pkl")

model = None
features = None

@app.on_event("startup")
def load_model():
    global model, features
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
    else:
        print("Warning: Model or features file not found. Ensure you have run train.py.")

@app.post("/predict")
def predict(application: CreditApplication):
    if model is None or features is None:
        return {"error": "Model not loaded. Please train the model first."}
    
    # Convert input to DataFrame
    data = application.dict(by_alias=True)
    df = pd.DataFrame([data])
    
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
    
    prediction = model.predict(df_final.values)
    risk = "Bad" if prediction[0] == 1 else "Good"
    
    return {"prediction": risk}
