# train.py
import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json
import os

# load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/house_prices.csv")   # adjust path/column names

# Example: simple feature/target selection (adjust for dataset)
X = df.drop(columns=["price"])    # change "price" to your target
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["seed"]
)

model = RandomForestRegressor(
    n_estimators=params["model"]["n_estimators"],
    max_depth=params["model"]["max_depth"],
    random_state=params["seed"],
)

model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")

# write metrics and params to files for DVC/experiments
with open("metrics.json", "w") as f:
    json.dump({"mse": mse}, f)

with open("params_out.yaml", "w") as f:
    yaml.safe_dump(params, f)
