# Clustering Analysis

K-Means clustering of 7,043 telecom customers by tenure and monthly charges. **Optimal k = 4**, giving
four labelled segments with churn rates from 49% to 5%.

| Deliverable | Folder | Contents |
|---|---|---|
| Optimal Number of Clusters | [`Optimal_Number_of_Clusters/`](Optimal_Number_of_Clusters/) | Elbow method, silhouette check, k-selection table |
| Trained K-Means Model | [`Trained_KMeans_Model/`](Trained_KMeans_Model/) | `kmeans_model.pkl`, `scaler_clustering.pkl`, training and prediction scripts, notebook |
| Visualisation and Labelling of Clusters | [`Visualisation_and_Labelling_of_Clusters/`](Visualisation_and_Labelling_of_Clusters/) | Labelled cluster plot, churn and profile charts, insights and actions |

| Segment | Customers | Churn |
|---|---|---|
| New High-Spenders (At Risk) | 2,202 | 49% |
| New Budget Customers | 1,733 | 24% |
| Loyal Premium Customers | 1,948 | 16% |
| Loyal Budget Customers | 1,160 | 5% |

![Customer segments](Visualisation_and_Labelling_of_Clusters/03_cluster_scatter_labelled.png)
