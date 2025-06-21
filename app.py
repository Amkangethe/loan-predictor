from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('models/loan_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Defines home page
@app.route('/')
def home():
    return render_template('form.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Example: expects a list of features in the correct order
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features) 
    prediction = model.predict(features_scaled)[0]
    return jsonify({'prediction': int(prediction)})

@app.route('/predict_result', methods=['POST'])
def predict_form():
    # Get form data for new dataset
    income = float(request.form['Income'])
    credit_score = float(request.form['Credit_Score'])
    loan_amount = float(request.form['Loan_Amount'])
    dti_ratio = float(request.form['DTI_Ratio'])
    employment_status = int(request.form['Employment_Status'])

    features = [income, credit_score, loan_amount, dti_ratio, employment_status]
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    result = 'Approved' if prediction == 1 else 'Not Approved'
    return render_template('form.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
