import numpy as np
import pandas as pd
from scipy import stats


def cramers_v_test(
    df: pd.DataFrame, categorical_col: str, target_col: str
) -> pd.DataFrame:
    """
    Calculate Cramér's V to measure association strength between two categorical variables.

    When to use:
        - When both variables are categorical
        - You want to quantify strength of association (not just test independence)

    Args:
        df (pd.DataFrame): Input DataFrame
        categorical_col (str): First categorical variable
        target_col (str): Second categorical variable

    Returns:
        pd.DataFrame: Cramér's V value (range: 0–1)
    """
    print(f"{'='*80}")
    print(f"CRAMÉR'S V: {categorical_col} vs {target_col}".center(80))
    print(f"{'='*80}")

    contingency = pd.crosstab(df[categorical_col], df[target_col])
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1

    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

    print(f"Cramér's V: {v:.4f}")  # (∈ [0, 1])
    print(f"{'='*80}\n")

    return pd.DataFrame({"test": ["cramers_v"], "value": [v]})
