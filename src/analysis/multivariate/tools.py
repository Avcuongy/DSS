import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prince import MCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from typing import Optional, Tuple
from utils import group_columns_by_type


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
