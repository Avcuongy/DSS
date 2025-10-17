import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prince import MCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from typing import Optional, Tuple
from utils import group_columns_by_type
from model.preprocessing import preprocessing_numerical, preprocessing_categorical


def get_pca_contribution_table(
    pca_model: PCA,
    variable_names: list[str],
    percent: bool = False,
) -> pd.DataFrame:
    """
    Computes the contribution of variables to principal components in a PCA model.

    Args:
        pca_model (PCA): Fitted PCA model from scikit-learn.
        variable_names (list[str]): List of variable names corresponding to PCA input features.
        percent (bool, optional): If True, returns contributions as percentages. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame of variable contributions to each principal component, optionally in percentages.
    """
    # Loadings = eigenvectors * sqrt(eigenvalues)
    loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)

    df_loadings = pd.DataFrame(
        loadings,
        index=variable_names,
        columns=[f"PC{i+1}" for i in range(loadings.shape[1])],
    )

    # Contribution = loadings^2
    df_contrib = df_loadings**2
    if percent:
        df_contrib = df_contrib.div(df_contrib.sum(axis=0), axis=1)
        df_contrib *= 100
        df_contrib.loc["Total"] = df_contrib.sum(axis=0)
        return df_contrib.round(2).astype(str) + " %"
    else:
        return df_contrib


def get_mca_contribution_table(mca_model: MCA, percent: bool = False) -> pd.DataFrame:
    """
    Computes the contribution of variables to dimensions in an MCA model.

    Args:
        mca_model (MCA): Fitted MCA model from the prince library.
        percent (bool, optional): If True, returns contributions as percentages. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame of variable contributions to each MCA dimension, optionally in percentages.
    """
    mca_col_contrib = mca_model.column_contributions_

    var_contrib = {}

    for col in mca_col_contrib.index:
        var_name = col.split("__")[0]
        if var_name not in var_contrib:
            var_contrib[var_name] = mca_col_contrib.loc[col].copy()
        else:
            var_contrib[var_name] += mca_col_contrib.loc[col]

    df_var_contrib = pd.DataFrame(var_contrib).T
    df_var_contrib.columns = [f"Dim{i+1}" for i in range(df_var_contrib.shape[1])]

    if percent:
        df_var_contrib *= 100
        df_var_contrib.loc["Total"] = df_var_contrib.sum(axis=0)
        return df_var_contrib.round(2).astype(str) + " %"
    else:
        return df_var_contrib


def run_kmeans_clustering(
    df: pd.DataFrame, cluster_range: Optional[list] = None
) -> pd.DataFrame:
    """
    Perform KMeans clustering on a raw DataFrame (not preprocessed).
    Automatically preprocesses numerical and categorical features,
    finds the best number of clusters using silhouette score,
    and returns the original DataFrame with cluster labels.

    Args:
        df (pd.DataFrame): Original raw DataFrame (unprocessed)
        cluster_range (list, optional): List of k values to try (default: range(2, 10))

    Returns:
        pd.DataFrame: Cluster assignment for each observation.
    """
    df_for_cluster = df.copy()

    num_cols, cate_cols, _ = group_columns_by_type(df_for_cluster, display_info=False)
    df_num = df_for_cluster[num_cols]
    df_cate = df_for_cluster[cate_cols]

    if num_cols:
        df_num_pre = preprocessing_numerical(df_num, num_cols)
    else:
        df_num_pre = pd.DataFrame()

    if cate_cols:
        df_cate_pre = preprocessing_categorical(df_cate, nominal_cols=cate_cols)
    else:
        df_cate_pre = pd.DataFrame()

    df_cluster = pd.concat([df_num_pre, df_cate_pre], axis=1)

    range_n_clusters = cluster_range or list(range(2, 10))
    silhouette_scores = []

    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df_cluster)
        silhouette_scores.append(silhouette_score(df_cluster, labels))

    best_k = range_n_clusters[np.argmax(silhouette_scores)]
    kmeans_best = KMeans(n_clusters=best_k, random_state=42)
    kmeans_labels = kmeans_best.fit_predict(df_cluster)
    silhouette_kmeans = silhouette_score(df_cluster, kmeans_labels)

    print(f"KMeans optimization:")
    print(f"- Optimal number of clusters: {best_k}")
    print(f"- Silhouette Score: {silhouette_kmeans:.3f}")

    X_2d = PCA(n_components=2).fit_transform(df_cluster)
    plt.figure(figsize=(15, 6))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap="coolwarm", s=50
    )

    centers_2d = (
        PCA(n_components=2).fit(df_cluster).transform(kmeans_best.cluster_centers_)
    )
    plt.scatter(
        centers_2d[:, 0], centers_2d[:, 1], c="red", s=150, marker="o", label="Centroid"
    )

    plt.title(f"KMeans Clustering (k = {best_k})")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.show()

    df_result = df.copy()
    df_result["cluster"] = kmeans_labels

    return df_result[["cluster"]]
