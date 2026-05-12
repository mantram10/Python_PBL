import numpy as np
import pandas as pd
import pickle

# Needed for pipeline
def log_transform(X):
    return np.log1p(X)

# Load model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Enter user details:\n")

data = {
    "Administrative": int(input("Administrative pages: ")),
    "Administrative_Duration": float(input("Administrative duration: ")),
    "Informational": int(input("Informational pages: ")),
    "Informational_Duration": float(input("Informational duration: ")),
    "ProductRelated": int(input("Product related pages: ")),
    "ProductRelated_Duration": float(input("Product related duration: ")),
    "BounceRates": float(input("Bounce rate (0-1): ")),
    "ExitRates": float(input("Exit rate (0-1): ")),
    "PageValues": float(input("Page values: ")),
    "SpecialDay": float(input("Special day (0-1): ")),
    "Month": input("Month (Jan, Feb, Mar...): ").capitalize(),
    "OperatingSystems": int(input("Operating system (1-8): ")),
    "Browser": int(input("Browser (1-13): ")),
    "Region": int(input("Region (1-9): ")),
    "TrafficType": int(input("Traffic type: ")),
    "VisitorType": input("Visitor type (Returning_Visitor/New_Visitor/Other): "),
    "Weekend": input("Weekend (True/False): ") == "True"
}

# Convert to DataFrame
user_data = pd.DataFrame([data])

# Predict
prob = model.predict_proba(user_data)[:, 1]

threshold = 0.3
prediction = (prob >= threshold).astype(int)

print("\n--- RESULT ---")
print(f"Probability of purchase: {prob[0]:.2f}")

if prediction[0] == 1:
    print("Prediction: User is likely to PURCHASE")
else:
    print("Prediction: User is NOT likely to purchase")