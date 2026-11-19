# Customer Churn Prediction – Machine Learning Project

## 1. Problem Statement
The objective is to predict whether a telecom customer is likely to leave the company using machine learning classification models.

## 2. Dataset
- Dataset: Telco Customer Churn
- Total records: 7,043
- Target variable: Churn
- Target values: Yes and No

## 3. Methodology
- Data cleaning and preprocessing
- Missing value handling using SimpleImputer
- Categorical encoding using OneHotEncoder
- Feature scaling using StandardScaler
- Train-test split
- Logistic Regression, Random Forest and XGBoost
- Hyperparameter tuning using RandomizedSearchCV
- Business-based threshold selection

## 4. Model Evaluation
Models were compared using ROC-AUC and F1 Score.

Baseline results:
- Logistic Regression ROC-AUC: 0.8413
- Random Forest ROC-AUC: 0.8225
- XGBoost ROC-AUC: 0.8375

Tuned XGBoost cross-validation ROC-AUC: 0.8485

The final selected model and held-out test metrics are available in the project output.

## 5. Limitations
- Predictions depend on the quality of the available customer data.
- Model performance may change with new customer behaviour.
- False positive and false negative predictions are possible.

## 6. Business Recommendations
- Identify customers with a high predicted churn probability.
- Offer suitable retention plans to customers at risk.
- Monitor customer satisfaction and service usage.
- Retrain the model when new data becomes available.

## 7. Project Outputs
- best_churn_model.pkl
- tuned_model_comparison.csv
- tuned_roc_curve.png
- feature_importance.png
- predictions.csv