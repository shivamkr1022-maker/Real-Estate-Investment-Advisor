import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import mlflow
import mlflow.sklearn

from preprocessing import load_and_clean_data

def train_models():
    print("Loading and preprocessing data...")
    df = load_and_clean_data("data/india_housing_prices.csv")

    # Select features for modeling
    features = [
        'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 
        'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
        'Nearby_Schools', 'Nearby_Hospitals', 'Property_Type', 'City',
        'Furnished_Status', 'Parking_Space'
    ]

    # Keep only available features
    features = [f for f in features if f in df.columns]
    
    X = df[features].copy()
    
    # One-Hot Encoding for categorical features
    X = pd.get_dummies(X, drop_first=True)

    # 1. Regression Target: Future_Price_5Y
    y_reg = df['Future_Price_5Y']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # 2. Classification Target: Good_Investment
    y_clf = df['Good_Investment']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    # Set MLflow Experiment
    mlflow.set_experiment("Real_Estate_Investment_Advisor")

    with mlflow.start_run(run_name="RandomForest_Models"):
        print("\n--- Training Regression Model (Future Price Prediction) ---")
        reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        reg_model.fit(X_train_r, y_train_r)
        
        y_pred_r = reg_model.predict(X_test_r)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        r2 = r2_score(y_test_r, y_pred_r)
        
        print(f"Regression RMSE: {rmse:.4f}")
        print(f"Regression R²: {r2:.4f}")

        mlflow.log_param("reg_model", "RandomForestRegressor")
        mlflow.log_metric("reg_rmse", rmse)
        mlflow.log_metric("reg_r2", r2)
        mlflow.sklearn.log_model(reg_model, "regression_model")

        print("\n--- Training Classification Model (Good Investment Prediction) ---")
        clf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_model.fit(X_train_c, y_train_c)
        
        y_pred_c = clf_model.predict(X_test_c)
        acc = accuracy_score(y_test_c, y_pred_c)
        
        print(f"Classification Accuracy: {acc:.4f}")
        print(classification_report(y_test_c, y_pred_c))

        mlflow.log_param("clf_model", "RandomForestClassifier")
        mlflow.log_metric("clf_accuracy", acc)
        mlflow.sklearn.log_model(clf_model, "classification_model")

        print("\nMLflow tracking completed and models logged successfully!")

if __name__ == "__main__":
    train_models()