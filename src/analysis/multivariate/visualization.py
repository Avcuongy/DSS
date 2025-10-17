import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from prince import MCA


def plot_pca_contributions(pca_model: PCA, threshold: float = 0.95):
    """
    Plots a scree plot and cumulative explained variance for a PCA model to visualize component contributions.

    Args:
        pca_model (PCA): Fitted PCA model from scikit-learn.
        threshold (float, optional): Threshold for cumulative explained variance (0.0 to 1.0).
    """
    explained_var = pca_model.explained_variance_ratio_ * 100
    cum_var = np.cumsum(explained_var)

    plt.figure(figsize=(16, 5))

    # Plot 1: Scree plot
    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, len(explained_var) + 1),
        explained_var,
        marker="o",
        label="Individual Component",
    )
    plt.title("Scree Plot (Variance by Component)")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.grid(True)

    # Plot 2: Cumulative variance
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(cum_var) + 1),
        cum_var,
        marker="o",
        color="green",
        label="Cumulative",
    )
    plt.axhline(
        y=threshold * 100,
        color="red",
        linestyle="--",
        label=f"{int(threshold * 100)}% threshold",
    )
    plt.title("Cumulative Explained Variance by Number of Components")
    plt.xlabel("Number of Components Retained")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_mca_contributions(mca_model: MCA, threshold: float = 0.95):
    """
    Plots a scree plot and cumulative explained variance for an MCA model to visualize dimension contributions.

    Args:
        mca_model (MCA): Fitted MCA model from the prince library.
        threshold (float, optional): Threshold for cumulative explained variance (0.0 to 1.0).
    """
    df_eigen = mca_model.eigenvalues_summary.copy()

    for col in ["% of variance", "% of variance (cumulative)"]:
        if (
            df_eigen[col].dtype == object
            or df_eigen[col].astype(str).str.contains("%|,").any()
        ):
            df_eigen[col] = (
                df_eigen[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace("%", "", regex=False)
                .astype(float)
            )

    plt.figure(figsize=(16, 5))

    # Scree Plot
    plt.subplot(1, 2, 1)
    plt.plot(df_eigen.index + 1, df_eigen["eigenvalue"], marker="o", label="Eigenvalue")
    plt.title("Scree Plot (Eigenvalues)")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.grid(True)

    # Cumulative Variance Plot
    plt.subplot(1, 2, 2)
    plt.plot(
        df_eigen.index + 1,
        df_eigen["% of variance (cumulative)"],
        marker="o",
        color="green",
        label="Cumulative Variance",
    )
    plt.axhline(
        y=threshold * 100,
        color="red",
        linestyle="--",
        label=f"{int(threshold * 100)}% threshold",
    )
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Dimensions Retained")
    plt.ylabel("Cumulative % of Variance")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
