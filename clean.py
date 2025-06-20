import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

# Load the new data with real target
df = pd.read_csv("loan.csv")

# Fill categorical columns with mode
for col in ['Gender', 'Dependents', 'Self_Employed', 'Married']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill numeric columns with median
for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'ApplicantIncome', 'CoapplicantIncome']:
    df[col] = df[col].fillna(df[col].median())

# Encode categorical variables
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

# Encode the real target column (Loan_Status: Y=1, N=0)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Split features and target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)
try:
    model.fit(X_train_scaled, y_train)
    print("Logistic Regression model training successful!")
except Exception as e:
    print("Logistic Regression model training failed:", e)

# Predict based on the test set
y_pred = model.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred))

# Train a Random Forest model with class balancing
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
try:
    rf_model.fit(X_train_scaled, y_train)
    print("Random Forest model training successful!")
except Exception as e:
    print("Random Forest model training failed:", e)

# Predict and evaluate Random Forest
rf_y_pred = rf_model.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_y_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_y_pred))

joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(rf_model, 'rf_loan_model.pkl')