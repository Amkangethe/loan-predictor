import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import joblib


# Load the data
df = pd.read_csv("loan-test.csv")

# Fill categorical columns with mode
for col in ['Gender', 'Dependents', 'Self_Employed']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill numeric columns with median
for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    df[col] = df[col].fillna(df[col].median())

# Encode categorical variables
# Gender, Married, Education, Self_Employed: map to 0/1
# Dependents: replace '3+' with 3, then convert to int
# Property_Area: one-hot encoding
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

# Add a dummy target column for demonstration 
df['Loan_Status'] = np.random.choice([0, 1], size=len(df))

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
    print("Model training successful!")
except Exception as e:
    print("Model training failed:", e)



