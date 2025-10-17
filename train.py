import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json, os

# 1. Load dataset
df = pd.read_csv("data/house_prices.csv")  

# 2. Initial exploration
print(df.head())
print(df.info())
print(df.isna().sum())

# 3. Drop rows (or columns) with missing values
# Option A: drop rows with ANY missing values
df_clean = df.dropna(axis=0, how='any').reset_index(drop=True)
# Option B: alternatively, drop columns with too many missingness first, then drop rows
# (But you said “do not use imputor”, so this is pure drop.)

print("Dropped rows. New shape:", df_clean.shape)

df_clean.head()

df_clean.info()

df_clean=df_clean.iloc[:,[4,5,7,8,9,10,11,12,13,17]]

df_clean.head()

from sklearn.preprocessing import LabelEncoder

df_clean.columns

string_cols=["property_type","city","province_name","purpose"]

encoder=LabelEncoder()

# 7. Label encode each selected column (explicitly as strings)
encoders = {}
for col in string_cols:
    le = LabelEncoder()
    # convert all to string first so LabelEncoder is deterministic
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    encoders[col] = le
    print(f"Encoded '{col}' -> {len(le.classes_)} classes")




X = df_clean.drop("price",axis=1)
y = df_clean["price"]


# 5. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def slugify(name):
    name = str(name)
    name = name.strip()
    # replace non-alphanumeric with underscore
    return re.sub(r'[^0-9a-zA-Z_]+', '_', name)

# 6. Choose and train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R²:", r2)

import os
import re

features_list=X.columns
feature_field_map = {col: slugify(col) for col in features_list}

features_list

feature_field_map

OUTPUT_DIR="models/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(model, os.path.join(OUTPUT_DIR, "house_price_model.pkl"))
joblib.dump(features_list, os.path.join(OUTPUT_DIR, "model_features.pkl"))
joblib.dump(encoders, os.path.join(OUTPUT_DIR, "label_encoders.pkl"))
joblib.dump(feature_field_map, os.path.join(OUTPUT_DIR, "feature_field_map.pkl"))

mse = mean_squared_error(y_test, y_pred)
with open("models/metrics.json", "w") as f:
    json.dump({"mse": mse}, f)

print("Saved model and artifacts to", OUTPUT_DIR)
print("Files:")
for f in os.listdir(OUTPUT_DIR):
    print(" -", f)