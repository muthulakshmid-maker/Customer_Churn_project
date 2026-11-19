import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report
)

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully")
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
print("Numeric columns:", len(numeric_features))
print("Categorical columns:", len(categorical_features))


negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass imbalance ratio:", scale_pos_weight)

logistic_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

random_forest_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    ))
])


models = {
    "Logistic Regression": logistic_pipeline,
    "Random Forest": random_forest_pipeline,
    "XGBoost": xgb_pipeline
}

print("\nThree model pipelines created successfully!")


for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    print(f"{name} training completed!")

print("\nAll three models trained successfully!")


results = []

for name, model in models.items():

    y_probability = model.predict_proba(X_test)[:, 1]

    y_prediction = (y_probability >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_probability)
    f1 = f1_score(y_test, y_prediction)

    results.append({
        "Model": name,
        "ROC-AUC": round(roc_auc, 4),
        "F1 Score": round(f1, 4)
    })

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison_baseline.csv",
    index=False
)

print("\nBaseline comparison saved successfully!")

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [3, 5, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}


random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=-1
)


param_grids = {
    "Random Forest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [5, 10, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    },

    "XGBoost": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.8, 1.0]
    }
}

tuned_models = {}

for name in ["Random Forest", "XGBoost"]:

    print(f"\nTuning {name}...")

    random_search = RandomizedSearchCV(
        estimator = models.get(name),
        param_distributions=param_grids[name],
        n_iter=10,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    tuned_models[name] = random_search.best_estimator_

    print("Best parameters:", random_search.best_params_)
    print("Best ROC-AUC:", random_search.best_score_)

print("\nRandomizedSearchCV completed!")



param_grids = {

    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100]
    },

    "XGBoost": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.8, 1.0]
    }
}

tuned_models = {}

best_two_models = [
    "Logistic Regression",
    "XGBoost"
]

for name in best_two_models:

    print(f"\nTuning {name}...")

    random_search = RandomizedSearchCV(
        estimator = models.get(name),
        param_distributions=param_grids[name],
        n_iter=10,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    tuned_models[name] = random_search.best_estimator_

    print("\nBest parameters:")
    print(random_search.best_params_)

    print("Best ROC-AUC:")
    print(random_search.best_score_)

print("\nBoth models tuned successfully!")


tuned_results = []

for name, model in tuned_models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    tuned_results.append({
        "Model": name,
        "ROC-AUC": roc,
        "F1 Score": f1
    })

tuned_comparison = pd.DataFrame(tuned_results)

print("\nTUNED MODEL COMPARISON")
print(tuned_comparison)

tuned_comparison.to_csv(
    "outputs/tuned_model_comparison.csv",
    index=False
)

print("\nTuned comparison saved successfully!")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

plt.figure(figsize=(8, 6))

for name, model in tuned_models.items():

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=name
    )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Tuned Models")

plt.legend()
plt.grid()

plt.savefig("outputs/tuned_roc_curve.png")
plt.show()

print("ROC curve saved successfully!")



best_model_name = tuned_comparison.loc[
    tuned_comparison["ROC-AUC"].idxmax(),
    "Model"
]

best_model = tuned_models[best_model_name]

print("\nBEST MODEL:", best_model_name)


from sklearn.metrics import recall_score, precision_score, f1_score

y_prob = best_model.predict_proba(X_test)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

threshold_results = []

for threshold in thresholds:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    recall = recall_score(y_test, y_pred_threshold)
    precision = precision_score(
        y_test, y_pred_threshold, zero_division=0
    )
    f1 = f1_score(y_test, y_pred_threshold)

    threshold_results.append({
        "Threshold": threshold,
        "Recall": recall,
        "Precision": precision,
        "F1 Score": f1
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nTHRESHOLD COMPARISON")
print(threshold_df)

eligible = threshold_df[threshold_df["Recall"] >= 0.75]

if not eligible.empty:
    selected_threshold = eligible.sort_values(
        "F1 Score", ascending=False
    ).iloc[0]["Threshold"]
else:
    selected_threshold = threshold_df.loc[
        threshold_df["Recall"].idxmax(), "Threshold"
    ]

print("\nSELECTED THRESHOLD:", selected_threshold)

threshold_df.to_csv(
    "outputs/threshold_comparison.csv",
    index=False
)


import numpy as np

preprocessor = best_model.named_steps["preprocessor"]
classifier = best_model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()

if hasattr(classifier, "feature_importances_"):
    importance = classifier.feature_importances_

elif hasattr(classifier, "coef_"):
    importance = np.abs(classifier.coef_[0])

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

top_5 = feature_importance.head(5)

print("\nTOP 5 IMPORTANT FEATURES")
print(top_5)

plt.figure(figsize=(10, 6))

plt.barh(
    top_5["Feature"],
    top_5["Importance"]
)

plt.xlabel("Importance")
plt.title("Top 5 Important Features")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("outputs/feature_importance.png")
plt.show()

print("Feature importance chart saved successfully!")



from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score
)

y_prob = best_model.predict_proba(X_test)[:, 1]

y_pred = (y_prob >= selected_threshold).astype(int)

print("\nFINAL MODEL EVALUATION")
print("Selected Model:", best_model_name)
print("Selected Threshold:", selected_threshold)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nCONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))


import joblib
import os

os.makedirs("outputs", exist_ok=True)

model_artifact = {
    "model": best_model,
    "model_name": best_model_name,
    "threshold": selected_threshold
}

joblib.dump(
    model_artifact,
    "outputs/best_churn_model.pkl"
)

print("\nModel saved successfully!")
