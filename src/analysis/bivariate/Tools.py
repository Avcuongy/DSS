import pandas as pd

from typing import Optional, List


def calculate_correlation(
    df: pd.DataFrame,
    cols: List[str],
    target_col: Optional[str] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Calculate correlation matrix between numerical variables and target.
    If target is not provided or invalid, compute correlation matrix among all cols.

    Args:
        df (pd.DataFrame): Input DataFrame
        cols (List[str]): List of numerical columns
        target (Optional[str]): Target variable (must be in df and numerical).
        method (str): Correlation method: 'pearson', 'spearman', or 'kendall'

    Returns:
        pd.DataFrame: Correlation matrix or column-wise correlation to target
    """
    numeric_cols = [
        col
        for col in cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not numeric_cols:
        raise ValueError("No valid numerical variables found in 'cols'.")

    if (
        target_col
        and target_col in df.columns
        and pd.api.types.is_numeric_dtype(df[target_col])
    ):
        return (
            df[numeric_cols + [target_col]]
            .corr(method=method)
            .loc[numeric_cols, [target_col]]
        )
    else:
        return df[numeric_cols].corr(method=method)
