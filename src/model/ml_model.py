import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "full_list_of_clean_flight_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===== LOAD DATA =====
df = pd.read_csv(DATA_PATH)

print("✅ Data loaded:", df.shape)

# ===== CLEAN PRICE =====
df['price'] = (
    df['price']
    .astype(str)
    .str.replace(r'[^\d.,]', '', regex=True)
    .str.replace(',', '.', regex=False)
)

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(subset=['price'], inplace=True)

# ===== DROP MISSING =====
df.dropna(subset=[
    'airline', 'number_of_stops', 'destination',
    'origin', 'days_until_flight'
], inplace=True)

# ===== FEATURE ENGINEERING =====
df['departure_hour'] = pd.to_datetime(df['departure_time'], errors='coerce').dt.hour

df['departure_datetime'] = pd.to_datetime(
    df['departure_date'] + ' 2025',
    format='%d %b %Y',
    errors='coerce'
)

df['departure_weekday'] = df['departure_datetime'].dt.dayofweek
df['is_weekend'] = df['departure_weekday'].isin([5, 6]).astype(int)

df.dropna(subset=['departure_hour', 'departure_weekday'], inplace=True)

# ===== FEATURES =====
X = pd.get_dummies(df[[
    'airline', 'number_of_stops', 'destination', 'origin',
    'days_until_flight', 'departure_hour',
    'departure_weekday', 'is_weekend'
]])

y = df['price']

# ===== NUMERIC TYPES =====
numeric_cols = [
    'days_until_flight',
    'departure_hour',
    'departure_weekday',
    'is_weekend'
]

for col in numeric_cols:
    if col in X.columns:
        X[col] = X[col].astype('float64')

# ===== SAVE FEATURE NAMES =====
feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")
joblib.dump(X.columns.tolist(), feature_names_path)
print("✅ Feature names saved")

# ===== TRAIN TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== GRID SEARCH =====
param_grid = {
    'n_estimators': [100],
    'max_depth': [10, None],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ===== EVALUATION =====
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("===== RESULTS =====")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# ===== SAVE MODEL =====
model_path = os.path.join(MODEL_DIR, "best_model.pkl")
joblib.dump(best_model, model_path)

print(f"✅ Model saved to: {model_path}")
