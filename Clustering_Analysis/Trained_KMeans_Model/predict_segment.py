"""
Assign a segment to new customers with the saved model.

Usage (after train_kmeans.py has created the model files):
    python predict_segment.py
"""
from pathlib import Path

import joblib
import pandas as pd

MOD = Path(__file__).resolve().parent
kmeans = joblib.load(MOD / "kmeans_model.pkl")
scaler = joblib.load(MOD / "scaler_clustering.pkl")

# Cluster numbers -> names, written by train_kmeans.py so they always match the model
names = pd.read_csv(MOD / "cluster_centres.csv", index_col="Cluster")["Segment"]

new_customers = pd.DataFrame({"tenure": [3, 6, 60, 70], "MonthlyCharges": [95, 25, 105, 20]})
new_customers["Cluster"] = kmeans.predict(scaler.transform(new_customers[["tenure", "MonthlyCharges"]]))
new_customers["Segment"] = new_customers["Cluster"].map(names)
print(new_customers.to_string(index=False))
