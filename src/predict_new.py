import pandas as pd
import joblib
import os

artifact = joblib.load("outputs/best_churn_model.pkl")

model = artifact["model"]
threshold = artifact["threshold"]

new_data = pd.read_csv("data/new_customers.csv")

if "customerID" in new_data.columns:
    customer_ids = new_data["customerID"]
else:
    customer_ids = pd.Series(
        range(1, len(new_data) + 1),
        name="Customer"
    )

probabilities = model.predict_proba(new_data)[:, 1]

predictions = (probabilities >= threshold).astype(int)

results = pd.DataFrame({
    "customerID": customer_ids,
    "Churn_Probability": probabilities,
    "Churn_Prediction": predictions
})

results["Churn_Prediction"] = results["Churn_Prediction"].map({
    0: "No",
    1: "Yes"
})

os.makedirs("outputs", exist_ok=True)

results.to_csv(
    "outputs/predictions.csv",
    index=False
)

print("\nPredictions generated successfully!")
print(results.head())
