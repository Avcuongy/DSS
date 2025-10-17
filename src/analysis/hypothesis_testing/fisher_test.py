import numpy as np
import pandas as pd
import scipy.stats as stats


def fisher_test(
    df: pd.DataFrame, categorical_col: str, target_col: str
) -> pd.DataFrame:
    """
    Fisher's exact test for 2x2 contingency tables (small samples).

    When to use:
        - Both variables are categorical
        - Sample sizes are small or expected frequencies < 5
        - ONLY for 2x2 tables

    Args:
        df (pd.DataFrame): Input DataFrame
        categorical_col (str): First categorical variable
        target_col (str): Second categorical variable

    Returns:
        pd.DataFrame: Fisher's exact test result
    """
    print(f"{'='*80}")
    print(f"FISHER'S EXACT TEST: {categorical_col} vs {target_col}".center(80))
    print(f"{'='*80}")

    table = pd.crosstab(df[categorical_col], df[target_col])

    if table.shape != (2, 2):
        print("Error: Fisher's exact test only supports 2x2 tables.")
        return pd.DataFrame()

    odds_ratio, p_value = stats.fisher_exact(table)

    print("Test Results:")
    print(f"- Odds Ratio: {odds_ratio:.4f}")
    print(f"- P-value: {p_value:.4f}")
    print(
        f"\nInterpretation:\n"
        f"- {'Dependent' if p_value < 0.05 else 'Independent'} (p-value {'<' if p_value < 0.05 else '≥'} 0.05)"
    )
    print(f"{'='*80}\n")

    return pd.DataFrame(
        {
            "test": ["fisher"],
            "odds_ratio": [odds_ratio],
            "p_value": [p_value],
            "significant": [p_value < 0.05],
        }
    )
