"""
segmentation.py — RFM-based customer segmentation pipeline.

Steps
-----
1.  Scale RFM features (RobustScaler handles monetary outliers well)
2.  K-Means: elbow + silhouette to find optimal K
3.  DBSCAN: density-based alternative
4.  Hierarchical clustering (Ward linkage)
5.  UMAP 2D projection
6.  Rule-based business label assignment per cluster
7.  Segment summary statistics
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

CACHE_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
SEG_CACHE    = os.path.join(CACHE_DIR, "segmentation.pkl")
SCALER_CACHE = os.path.join(CACHE_DIR, "scaler.pkl")

# Business label palette
SEGMENT_COLORS = {
    "Champions":          "#00C9A7",
    "Potential Loyalists": "#845EC2",
    "New Customers":      "#4B9FEA",
    "At-Risk":            "#FFC75F",
    "Hibernating":        "#F9A24B",
    "Lost":               "#FF6B6B",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Scaling
# ─────────────────────────────────────────────────────────────────────────────
def scale_rfm(rfm: pd.DataFrame) -> np.ndarray:
    scaler = RobustScaler()
    X = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    joblib.dump(scaler, SCALER_CACHE)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Optimal K — Elbow + Silhouette
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_k(X: np.ndarray, k_range: range = range(2, 11)):
    inertias, silhouettes, dbs = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        dbs.append(davies_bouldin_score(X, labels))

    results = pd.DataFrame({
        "K":         list(k_range),
        "Inertia":   inertias,
        "Silhouette": silhouettes,
        "DaviesBouldin": dbs,
    })
    optimal_k = results.loc[results["Silhouette"].idxmax(), "K"]
    return int(optimal_k), results


def plot_elbow(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["K"], y=results["Inertia"],
        mode="lines+markers", name="Inertia",
        line=dict(color="#4B9FEA", width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title="Elbow Method — Inertia vs K",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.8)",
        font=dict(color="#e0e0e0"),
    )
    return fig


def plot_silhouette(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["K"], y=results["Silhouette"],
        mode="lines+markers", name="Silhouette",
        line=dict(color="#00C9A7", width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title="Silhouette Score vs K",
        xaxis_title="K",
        yaxis_title="Silhouette Score",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.8)",
        font=dict(color="#e0e0e0"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  K-Means fit
# ─────────────────────────────────────────────────────────────────────────────
def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, init="k-means++", n_init=15, random_state=42)
    labels = km.fit_predict(X)
    joblib.dump(km, os.path.join(CACHE_DIR, "kmeans.pkl"))
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DBSCAN fit
# ─────────────────────────────────────────────────────────────────────────────
def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"[DBSCAN] eps={eps}, min_samples={min_samples} → "
          f"{n_clusters} clusters, {n_noise} noise points")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 5.  UMAP 2D projection
# ─────────────────────────────────────────────────────────────────────────────
def run_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=False,
        )
        embedding = reducer.fit_transform(X)
        joblib.dump(reducer, os.path.join(CACHE_DIR, "umap.pkl"))
        return embedding
    except ImportError:
        # Fallback to PCA if umap-learn not installed
        from sklearn.decomposition import PCA
        print("[UMAP] umap-learn not found; falling back to PCA 2D.")
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Business label assignment
# ─────────────────────────────────────────────────────────────────────────────
def assign_segment_labels(rfm: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """
    Rule engine based on normalised R/F/M ranks per cluster centroid.
    Lower recency = more recent; higher frequency/monetary = better.
    """
    rfm = rfm.copy()

    # Cluster-level summary
    summary = rfm.groupby(cluster_col).agg(
        R=("Recency",   "median"),
        F=("Frequency", "median"),
        M=("Monetary",  "median"),
    ).reset_index()

    # Rank clusters: R ascending (lower recency → better), F/M descending
    summary["R_rank"] = summary["R"].rank(ascending=True)
    summary["F_rank"] = summary["F"].rank(ascending=False)
    summary["M_rank"] = summary["M"].rank(ascending=False)
    summary["Score"]  = summary["R_rank"] + summary["F_rank"] + summary["M_rank"]
    summary = summary.sort_values("Score")

    n = len(summary)
    labels_pool = [
        "Champions",
        "Potential Loyalists",
        "New Customers",
        "At-Risk",
        "Hibernating",
        "Lost",
    ]
    # Pad or trim to match number of clusters
    labels_pool = (labels_pool * ((n // len(labels_pool)) + 1))[:n]

    cluster_to_label = dict(zip(summary[cluster_col], labels_pool[:n]))
    rfm["Segment"] = rfm[cluster_col].map(cluster_to_label)
    return rfm, cluster_to_label


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Interactive UMAP scatter
# ─────────────────────────────────────────────────────────────────────────────
def plot_umap(rfm: pd.DataFrame, embedding: np.ndarray) -> go.Figure:
    plot_df = rfm.copy()
    plot_df["UMAP_1"] = embedding[:, 0]
    plot_df["UMAP_2"] = embedding[:, 1]

    color_map = {seg: SEGMENT_COLORS.get(seg, "#888") for seg in plot_df["Segment"].unique()}

    fig = px.scatter(
        plot_df,
        x="UMAP_1", y="UMAP_2",
        color="Segment",
        color_discrete_map=color_map,
        hover_data={"CustomerID": True, "Recency": True,
                    "Frequency": True, "Monetary": ":.2f"},
        title="Customer Segments — UMAP 2D Projection",
        template="plotly_dark",
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.8)",
        legend=dict(title="Segment"),
        font=dict(color="#e0e0e0"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Segment summary stats
# ─────────────────────────────────────────────────────────────────────────────
def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    total_revenue = rfm["Monetary"].sum()
    summary = (
        rfm.groupby("Segment")
        .agg(
            Count          = ("CustomerID",  "count"),
            Avg_Recency    = ("Recency",     "mean"),
            Avg_Frequency  = ("Frequency",   "mean"),
            Avg_Monetary   = ("Monetary",    "mean"),
            Total_Revenue  = ("Monetary",    "sum"),
        )
        .reset_index()
    )
    summary["Revenue_%"]     = (summary["Total_Revenue"] / total_revenue * 100).round(2)
    summary["Avg_Recency"]   = summary["Avg_Recency"].round(1)
    summary["Avg_Frequency"] = summary["Avg_Frequency"].round(2)
    summary["Avg_Monetary"]  = summary["Avg_Monetary"].round(2)
    return summary.sort_values("Total_Revenue", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Full pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────────
def run_segmentation_pipeline(rfm_raw: pd.DataFrame,
                               force_rebuild: bool = False) -> dict:
    if not force_rebuild and os.path.exists(SEG_CACHE):
        print("[Segmentation] Loading from cache …")
        return joblib.load(SEG_CACHE)

    print("[Segmentation] Running pipeline …")
    X = scale_rfm(rfm_raw)

    # K-Means
    optimal_k, k_results = find_optimal_k(X)
    optimal_k = max(optimal_k, 4)   # enforce min 4 segments for business value
    print(f"[Segmentation] Optimal K = {optimal_k}")
    km_labels = run_kmeans(X, optimal_k)

    # DBSCAN
    db_labels = run_dbscan(X, eps=0.6, min_samples=8)

    # UMAP
    print("[Segmentation] Running UMAP …")
    embedding = run_umap(X)

    # Attach labels
    rfm = rfm_raw.copy()
    rfm["Cluster"]       = km_labels
    rfm["DBSCAN_Cluster"] = db_labels
    rfm, cluster_map     = assign_segment_labels(rfm, "Cluster")

    # Metrics
    sil_kmeans = silhouette_score(X, km_labels)
    db_kmeans  = davies_bouldin_score(X, km_labels)
    db_valid   = db_labels[db_labels != -1]
    X_valid    = X[db_labels != -1]
    sil_dbscan = silhouette_score(X_valid, db_valid) if len(set(db_valid)) > 1 else 0.0

    result = {
        "rfm":         rfm,
        "X_scaled":    X,
        "embedding":   embedding,
        "k_results":   k_results,
        "optimal_k":   optimal_k,
        "cluster_map": cluster_map,
        "metrics": {
            "kmeans_silhouette":  round(sil_kmeans, 4),
            "kmeans_db":          round(db_kmeans,  4),
            "dbscan_silhouette":  round(sil_dbscan, 4),
        },
    }
    joblib.dump(result, SEG_CACHE)
    print("[Segmentation] Done. Cached.")
    return result