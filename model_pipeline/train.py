import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
import os
import urllib.request

DATA_FILE = "../dataset/german_credit_data_with_target.csv"

def download_data():
    if not os.path.exists(DATA_FILE):
        print(f"Waiting for {DATA_FILE} to be placed in the current directory...")
        # We will not auto-download here since the user is fetching it


def main():
    download_data()
    print("Loading data...")
    df_credit = pd.read_csv(DATA_FILE, index_col=0)
    
    print("Preprocessing data...")
    df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
    df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')
    
    # Feature Engineering (Dummies)
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
    
    # Age categories
    interval = (18, 25, 35, 60, 120)
    cats = ['Student', 'Young', 'Adult', 'Senior']
    df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)
    
    # Drop old features
    del df_credit["Saving accounts"]
    del df_credit["Checking account"]
    del df_credit["Purpose"]
    del df_credit["Sex"]
    del df_credit["Housing"]
    del df_credit["Age_cat"]
    del df_credit["Risk"]
    del df_credit['Risk_good'] # Keep Risk_bad as target
    
    df_credit['Credit amount'] = np.log(df_credit['Credit amount'])
    
    print("Preparing X and y...")
    X = df_credit.drop('Risk_bad', axis=1)
    y = df_credit["Risk_bad"]
    
    # Save the feature columns to ensure prediction inputs match
    joblib.dump(list(X.columns), "features.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25, random_state=42)
    
    print("Training GaussianNB model...")
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")
    
    print("Saving model to model.pkl...")
    joblib.dump(model, "model.pkl")
    print("Done!")

if __name__ == "__main__":
    main()
