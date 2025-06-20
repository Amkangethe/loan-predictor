import pandas as pd

df = pd.read_csv("loan-test.csv")


# Fill categorical columns with mode
for col in ['Gender', 'Dependents', 'Self_Employed']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill numeric columns with median
for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    df[col].fillna(df[col].median(), inplace=True)

# Check again for missing values
print("\nMissing values after filling:")
print(df.isnull().sum())


# Convert categorical columns to numeric
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)