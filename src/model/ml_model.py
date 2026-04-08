# -*- coding: utf-8 -*-
"""ML_model.py"""

import os
import pandas as pd
import sqlite3
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Set MLflow tracking URI (local sqlite DB)
mlflow.set_tracking_uri("sqlite:////root/WePP_Project_copy/instance/mlflow.db")

# Explicitly set artifact root directory
os.environ["MLFLOW_ARTIFACT_ROOT"] = "/root/WePP_Project_copy/mlruns"

# Load data
conn = sqlite3.connect('/root/WePP_Project_copy/instance/flights.db')
df = pd.read_sql_query("SELECT * FROM flight", conn)
conn.close()

# Clean price column
df['price'] = (
    df['price']
    .astype(str)
    .str.replace(r'[^\d.,]', '', regex=True)
    .str.replace(',', '.', regex=False)
)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(subset=['price'], inplace=True)

# Drop missing feature rows
df.dropna(subset=['airline', 'number_of_stops', 'destination', 'origin', 'days_until_flight'], inplace=True)

# Feature engineering
df['departure_hour'] = pd.to_datetime(df['departure_time'], errors='coerce').dt.hour
df['departure_datetime'] = pd.to_datetime(df['departure_date'] + ' 2025', format='%d %b %Y', errors='coerce')
df['departure_weekday'] = df['departure_datetime'].dt.dayofweek
df['is_weekend'] = df['departure_weekday'].isin([5, 6]).astype(int)
df.dropna(subset=['departure_hour', 'departure_weekday'], inplace=True)

# Encode features
X = pd.get_dummies(df[[
    'airline', 'number_of_stops', 'destination', 'origin',
    'days_until_flight', 'departure_hour', 'departure_weekday', 'is_weekend'
]])
y = df['price']

# Ensure numeric types float64
numeric_cols = ['days_until_flight', 'departure_hour', 'departure_weekday', 'is_weekend']
for col in numeric_cols:
    if col in X.columns:
        X[col] = X[col].astype('float64')

# Save feature names
feature_names_path = "/root/WePP_Project_copy/ML_model/feature_names.pkl"
joblib.dump(X.columns.tolist(), feature_names_path)
print(f"✅ Feature names saved to: {feature_names_path}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best R2 score (CV):", grid_search.best_score_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Test set R2: {r2:.4f}")
print(f"Test MAE: CHF {mae:.2f}")
print(f"Test MSE: {mse:.2f}")

# Create a new experiment named "Flights"
client = MlflowClient()
experiment_name = "Flights"

# Check if experiment already exists
existing_exp = client.get_experiment_by_name(experiment_name)
if existing_exp is None:
    experiment_id = client.create_experiment(experiment_name)
    print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
else:
    experiment_id = existing_exp.experiment_id
    print(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")

mlflow.set_experiment(experiment_name)

with mlflow.start_run(experiment_id=experiment_id) as run:
    best_rf.fit(X_train, y_train)
    predictions = best_rf.predict(X_test)

    # Log metrics and params
    mlflow.log_params(best_rf.get_params())
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Signature
    X_test_copy = X_test.copy()
    for col in numeric_cols:
        if col in X_test_copy.columns:
            X_test_copy[col] = X_test_copy[col].astype('float64')

    signature = infer_signature(X_test_copy, predictions)

    model_name = "FlightPrice"
    model_uri = mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="model",
        registered_model_name=model_name,
        input_example=X_test_copy.iloc[:1],
        signature=signature
    )

    run_id = run.info.run_id

print(f"MLflow run ID: {run_id}")
print(f"✅ Model registered with URI: {model_uri}")

time.sleep(2)

# Promote model to Production if better
latest_versions = client.search_model_versions(f"name='{model_name}'")
latest_model = max(latest_versions, key=lambda v: int(v.version))
version_to_promote = latest_model.version
print(f"✅ Latest model version fetched: {version_to_promote}")

prod_versions = [v for v in latest_versions if v.current_stage == "Production"]

promote = True
if prod_versions:
    prod_version = prod_versions[0]
    prod_run = client.get_run(prod_version.run_id)
    prod_r2 = float(prod_run.data.metrics.get("r2", 0))
    print(f"Current PROD model R2: {prod_r2:.4f}")

    if r2 <= prod_r2:
        print("⚠️ New model is not better than current production.")
        promote = False
else:
    print("✅ No Production model found — new model will be promoted.")

if promote:
    client.transition_model_version_stage(
        name=model_name,
        version=version_to_promote,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Model version {version_to_promote} promoted to Production!")
else:
    print(f"ℹ️ Model version {version_to_promote} remains in stage 'None' for now.")

# Save model locally
save_dir = "/root/WePP_Project_copy/ML_model/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "best_model.pkl")
joblib.dump(best_rf, save_path)
print(f"✅ Model saved as: {save_path}")

print("===== Model Evaluation Results =====")
print(f"Test set R²: {r2:.4f}")
print(f"Test MAE: CHF {mae:.2f}")
print(f"Test MSE: {mse:.2f}")
print("====================================")
