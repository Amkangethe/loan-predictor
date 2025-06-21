import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import joblib
import os

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Fill numeric columns with median
    for col in ['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio']:
        df[col] = df[col].fillna(df[col].median())
    # Encode Employment_Status
    df['Employment_Status'] = df['Employment_Status'].map({'employed': 1, 'unemployed': 0})
    # Encode Approval (target)
    df['Approval'] = df['Approval'].map({'Approved': 1, 'Rejected': 0})
    return df

def train_and_save_models(df):
    # Use only relevant features
    X = df[['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status']]
    y = df['Approval']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    log_pred = log_model.predict(X_test_scaled)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
    print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, log_pred))
    print("Logistic Regression Classification Report:\n", classification_report(y_test, log_pred))
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
    print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
    # Save the best model and scaler
    joblib.dump(rf_model, 'models/loan_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Best model and scaler saved as 'loan_model.pkl' and 'scaler.pkl'.")

def main():
    filepath = 'loan_data.csv'
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
    df = load_and_clean_data(filepath)
    train_and_save_models(df)

if __name__ == "__main__":
    main()