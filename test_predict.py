import requests

# Replace these values with a real example from your dataset, in the correct order
features = [1, 1, 4583, 1508, 128.0, 360.0, 1.0, 0, 0, 1, 0]  # Example only

response = requests.post(
    "http://localhost:5000/predict",
    json={"features": features}
)

print("Prediction result:", response.json())
