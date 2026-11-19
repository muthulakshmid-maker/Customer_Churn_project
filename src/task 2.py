import pandas as pd

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

new_customers = df.drop(
    columns=["Churn", "customerID"]
).head(5)

new_customers.insert(
    0,
    "customerID",
    df["customerID"].head(5)
)

new_customers.to_csv(
    "data/new_customers.csv",
    index=False
)

print("New customer sample created!")
