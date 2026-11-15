# Trained K-Means Model

**A K-Means model with 4 clusters was trained on `tenure` and `MonthlyCharges` for 7,043 customers and
saved as `kmeans_model.pkl`, with its scaler saved as `scaler_clustering.pkl`.**

## Files in this folder

| File | Contents |
|---|---|
| `README.md` | This documentation |
| `kmeans_model.pkl` | The trained K-Means model (k = 4) |
| `scaler_clustering.pkl` | The StandardScaler fitted on `tenure` and `MonthlyCharges` |
| `train_kmeans.py` | Script that trains the model and saves the model, scaler, charts and tables |
| `predict_segment.py` | Script that loads the saved model and assigns new customers to segments |
| `DA_Customer_churn_Clustering.ipynb` | The full notebook, with every step explained and run |
| `cluster_centres.csv` | Cluster number → segment name, used by `predict_segment.py` |

## 1. Model settings

| Setting | Value | Why |
|---|---|---|
| Algorithm | K-Means (`sklearn.cluster.KMeans`) | Simple, fast and easy to explain; suits two numeric features |
| `n_clusters` | 4 | Optimal k from the elbow method and silhouette score |
| `n_init` | 10 | 10 runs from different random starting centres; the best is kept |
| `random_state` | 42 | Makes the results reproducible |
| Features | `tenure`, `MonthlyCharges` | Customer lifetime and spend |
| Scaling | `StandardScaler` (mean 0, std 1) | Stops charges outweighing tenure in the distance calculation |
| Training rows | 7,043 | All customers |

## 2. How K-Means works

1. Place 4 starting centres (centroids).
2. Assign every customer to the nearest centre, using Euclidean distance.
3. Move each centre to the average position of its customers.
4. Repeat steps 2–3 until the assignments stop changing (**convergence**).

## 3. Training code

```python
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load and encode
df = pd.read_csv('Dataset_ATS_v2 1.csv')
df_enc = df.copy()
yes_no = {'Yes': 1, 'No': 0}
for col in ['Dependents', 'PhoneService', 'MultipleLines', 'Churn']:
    df_enc[col] = df_enc[col].map(yes_no)
df_enc['gender'] = df_enc['gender'].map({'Male': 1, 'Female': 0})
df_enc['InternetService'] = df_enc['InternetService'].map({'Fiber optic': 1, 'DSL': 0})
df_enc['Contract'] = df_enc['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

# Scale the clustering features
features = ['tenure', 'MonthlyCharges']
scaler_clust = StandardScaler()
X_clust = scaler_clust.fit_transform(df_enc[features])

# Train the final model
OPTIMAL_K = 4
kmeans = KMeans(n_clusters=OPTIMAL_K, n_init=10, random_state=42)
df_enc['Cluster'] = kmeans.fit_predict(X_clust)

# Save the model and the scaler
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler_clust, 'scaler_clustering.pkl')
```

`fit_predict` trains the model and returns a cluster number (0–3) for every customer.

## 4. Training results

**Cluster sizes**

| Cluster | Customers |
|---|---|
| 0 | 2,202 |
| 1 | 1,733 |
| 2 | 1,948 |
| 3 | 1,160 |

**Cluster centres.** The model stores centres in scaled units. `inverse_transform` converts them back to
months and dollars:

```python
centres = pd.DataFrame(scaler_clust.inverse_transform(kmeans.cluster_centers_),
                       columns=features).round(1)
```

| Cluster | Tenure (months) | Monthly charges ($) |
|---|---|---|
| 0 | 14.8 | 81.2 |
| 1 | 10.5 | 32.5 |
| 2 | 58.8 | 93.1 |
| 3 | 54.1 | 34.2 |

**Quality:** silhouette score 0.476, which means reasonable separation between clusters.

## 5. Using the saved model

```python
import joblib
import pandas as pd

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler_clustering.pkl')

new_customer = pd.DataFrame({'tenure': [3], 'MonthlyCharges': [95]})
cluster_id = kmeans.predict(scaler.transform(new_customer))[0]
print(cluster_id)   # 0 -> New High-Spenders (At Risk)
```

New data is scaled with `transform`, not `fit_transform`. That reuses the mean and standard deviation
learned from the training data. Refitting the scaler on new data would scale it differently and give
wrong segments.

## 6. How to rerun

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python train_kmeans.py "path/to/Dataset_ATS_v2 1.csv"
python predict_segment.py
```

`train_kmeans.py` recreates `kmeans_model.pkl`, `scaler_clustering.pkl`, all five charts and the result
tables in this folder. With `random_state=42` the results are identical every time.
