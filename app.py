# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import subprocess

MODEL_PATH = "models/model.joblib"

def dvc_pull():
    # Pull latest model from DVC remote if available
    # safe to run; if dvc not configured it will be a no-op
    try:
        subprocess.run(["dvc","pull","-q"], check=True)
    except Exception as e:
        print("dvc pull failed (ok for local dev):", e)

def load_model():
    if not os.path.exists(MODEL_PATH):
        dvc_pull()
    return joblib.load(MODEL_PATH)

app = Flask(__name__)
model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # expect features as list or dict; adapt to your pipeline
    if isinstance(data, dict):
        # convert dict to ordered feature vector - must match training
        features = [data[k] for k in sorted(data.keys())]
    else:
        features = data
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)[0]
    return jsonify({"prediction": float(pred)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
