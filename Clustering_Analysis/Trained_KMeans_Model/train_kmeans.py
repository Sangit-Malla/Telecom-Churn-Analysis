"""
Customer Churn - Stage 2: Clustering Analysis (K-Means)
Script version of notebook/DA_Customer_churn_Clustering.ipynb.

Steps
  1. Load the raw dataset and encode the categorical columns
  2. Scale tenure and MonthlyCharges with StandardScaler
  3. Elbow method (k = 1..10) + silhouette score (k = 2..10) to choose k
  4. Train the final K-Means model (k = 4)
  5. Profile and label the clusters
  6. Save charts, results tables, the model and the scaler

Usage:
    python train_kmeans.py "path/to/Dataset_ATS_v2 1.csv"
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "Dataset_ATS_v2 1.csv"
# All outputs (model, scaler, charts, tables) are written next to this script
ROOT = Path(__file__).resolve().parent
FIG = RES = MOD = ROOT

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
FEATURES = ["tenure", "MonthlyCharges"]
OPTIMAL_K = 4

# ------------------------------------------------------------------ 1. load + encode
df = pd.read_csv(DATA_PATH)
print("Raw data:", df.shape)

df_enc = df.copy()
yes_no = {"Yes": 1, "No": 0}
for col in ["Dependents", "PhoneService", "MultipleLines", "Churn"]:
    df_enc[col] = df_enc[col].map(yes_no)
df_enc["gender"] = df_enc["gender"].map({"Male": 1, "Female": 0})
df_enc["InternetService"] = df_enc["InternetService"].map({"Fiber optic": 1, "DSL": 0})
df_enc["Contract"] = df_enc["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})

# ------------------------------------------------------------------ 2. scale
scaler_clust = StandardScaler()
X_clust = scaler_clust.fit_transform(df_enc[FEATURES])

# ------------------------------------------------------------------ 3. choose k
k_range = range(1, 11)
inertia = [KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_clust).inertia_
           for k in k_range]
sil = {k: silhouette_score(X_clust, KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
                           .fit_predict(X_clust)) for k in range(2, 11)}

k_table = pd.DataFrame({"k": list(k_range), "inertia": np.round(inertia, 1)})
k_table["drop_from_previous"] = -k_table.inertia.diff().round(1)
k_table["silhouette"] = k_table.k.map(sil).round(3)
k_table.to_csv(RES / "k_selection_scores.csv", index=False)
print(k_table.to_string(index=False))
print("Highest silhouette at k =", max(sil, key=sil.get))

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertia, marker="o")
plt.axvline(OPTIMAL_K, color="red", linestyle="--", label=f"elbow at k = {OPTIMAL_K}")
plt.title("Elbow Method – choosing k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.xticks(list(k_range))
plt.legend()
plt.savefig(FIG / "01_elbow_method.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(list(sil.keys()), list(sil.values()), marker="o", color="green")
plt.title("Silhouette score by k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.xticks(list(sil.keys()))
plt.savefig(FIG / "02_silhouette_scores.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------ 4. train
kmeans = KMeans(n_clusters=OPTIMAL_K, n_init=10, random_state=RANDOM_STATE)
df_enc["Cluster"] = kmeans.fit_predict(X_clust)

centres = pd.DataFrame(scaler_clust.inverse_transform(kmeans.cluster_centers_),
                       columns=FEATURES).round(1)
centres.index.name = "Cluster"

# ------------------------------------------------------------------ 5. profile + label
tenure_cut = df_enc["tenure"].mean()
charge_cut = df_enc["MonthlyCharges"].mean()

def name_cluster(row):
    long_stay = row["tenure"] >= tenure_cut
    high_spend = row["MonthlyCharges"] >= charge_cut
    if not long_stay and high_spend:
        return "New High-Spenders (At Risk)"
    if not long_stay and not high_spend:
        return "New Budget Customers"
    if long_stay and high_spend:
        return "Loyal Premium Customers"
    return "Loyal Budget Customers"

cluster_names = centres.apply(name_cluster, axis=1).to_dict()
df_enc["Segment"] = df_enc["Cluster"].map(cluster_names)
centres.insert(0, "Segment", centres.index.map(cluster_names))
centres.to_csv(RES / "cluster_centres.csv")

profile = df_enc.groupby("Cluster").agg(
    customers=("Churn", "size"),
    avg_tenure=("tenure", "mean"),
    avg_monthly_charges=("MonthlyCharges", "mean"),
    churn_rate=("Churn", "mean"),
    pct_fiber=("InternetService", "mean"),
    pct_month_to_month=("Contract", lambda s: (s == 0).mean()),
    pct_senior=("SeniorCitizen", "mean"),
    pct_dependents=("Dependents", "mean"),
).round(2)
profile.insert(0, "Segment", profile.index.map(cluster_names))
profile = profile.sort_values("churn_rate", ascending=False)
profile.to_csv(RES / "segment_profile.csv")
print(profile.to_string())

# ------------------------------------------------------------------ 6. visualise
seg_order = ["New High-Spenders (At Risk)", "New Budget Customers",
             "Loyal Premium Customers", "Loyal Budget Customers"]
palette = dict(zip(seg_order, ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]))

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_enc, x="tenure", y="MonthlyCharges", hue="Segment",
                hue_order=seg_order, palette=palette, alpha=0.5, s=20)
plt.scatter(centres["tenure"], centres["MonthlyCharges"], c="black", marker="X",
            s=250, label="Cluster centre")
plt.title(f"Customer Segments – K-Means (k = {OPTIMAL_K})")
plt.xlabel("Tenure (months)")
plt.ylabel("Monthly Charges ($)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(FIG / "03_cluster_scatter_labelled.png", dpi=150, bbox_inches="tight")
plt.close()

churn_by_seg = df_enc.groupby("Segment")["Churn"].mean().reindex(seg_order) * 100
plt.figure(figsize=(9, 5))
ax = sns.barplot(x=churn_by_seg.index, y=churn_by_seg.values,
                 hue=churn_by_seg.index, palette=palette, legend=False)
ax.axhline(df_enc["Churn"].mean() * 100, color="grey", linestyle="--", label="Overall churn rate")
for i, v in enumerate(churn_by_seg.values):
    ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
plt.title("Churn rate by customer segment")
plt.ylabel("Churn rate (%)")
plt.xlabel("")
plt.xticks(rotation=15)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "04_churn_by_segment.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.countplot(data=df_enc, x="Segment", order=seg_order, hue="Segment",
              palette=palette, legend=False, ax=axes[0])
axes[0].set_title("Number of customers per segment")
axes[0].tick_params(axis="x", rotation=20)
contract_mix = (pd.crosstab(df_enc["Segment"], df["Contract"], normalize="index")
                .reindex(seg_order) * 100)
contract_mix.plot(kind="bar", stacked=True, ax=axes[1], colormap="Pastel1")
axes[1].set_title("Contract type within each segment (%)")
axes[1].set_ylabel("% of segment")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend(title="Contract", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(FIG / "05_segment_size_and_contract_mix.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------ 7. save
out = df.copy()
out["Cluster"] = df_enc["Cluster"]
out["Segment"] = df_enc["Segment"]
out.to_csv(RES / "customer_segments.csv", index=False)
joblib.dump(kmeans, MOD / "kmeans_model.pkl")
joblib.dump(scaler_clust, MOD / "scaler_clustering.pkl")
print("Saved kmeans_model.pkl, scaler_clustering.pkl, charts and result tables")
