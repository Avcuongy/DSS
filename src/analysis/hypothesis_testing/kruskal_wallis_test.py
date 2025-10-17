import pandas as pd
import scipy.stats as stats
import pandas.api.types as pd_types
from typing import Union, List


def check_normality(data: pd.Series) -> bool:
    """
    Check if data follows a normal distribution using Shapiro-Wilk test.
    """
    if len(data) < 3:
        print("Warning: Shapiro-Wilk test requires at least 3 samples.")
        return False
    stat, p_value = stats.shapiro(data)
    return p_value > 0.05


def check_homogeneity(groups: List[pd.Series]) -> bool:
    """
    Check homogeneity of variances across groups using Levene's test.
    """
    stat, p_value = stats.levene(*groups)
    return p_value > 0.05


def kruskal_wallis_test(
    df: pd.DataFrame, categorical_col: str, target: str
) -> Union[pd.DataFrame, None]:
    """
    Perform Kruskal-Wallis test for 2 or more groups.
    """
    print(f"{'='*80}")
    print(f"KRUSKAL-WALLIS TEST: {categorical_col} vs {target}".center(80))
    print(f"{'='*80}")

    if categorical_col not in df.columns or target not in df.columns:
        print(
            f"Error: Column '{categorical_col}' or '{target}' not found in DataFrame."
        )
        return None

    if not pd_types.is_numeric_dtype(df[target]):
        print(f"Error: Target column '{target}' must be numerical.")
        return None

    group_counts = df.groupby(categorical_col)[target].count()
    valid_groups = group_counts[group_counts > 0].index.tolist()
    group_data = [
        df[df[categorical_col] == group][target].dropna() for group in valid_groups
    ]

    if len(valid_groups) < 2:
        print(
            f"Error: Kruskal-Wallis requires 2 or more groups, found {len(valid_groups)} in '{categorical_col}'."
        )
        return None

    if any(len(g) == 0 for g in group_data):
        print(
            f"Error: One or more groups have no valid data in '{categorical_col}' for '{target}'."
        )
        return None

    is_normal = all(check_normality(g) for g in group_data)
    is_homogeneous = check_homogeneity(group_data)

    print("Assumption Checks (Informational):")
    print(f"- Normality (Shapiro-Wilk): {'✓ Passed' if is_normal else '✗ Failed'}")
    print(
        f"- Homogeneity (Levene's Test): {'✓ Passed' if is_homogeneous else '✗ Failed'}"
    )
    # print("Note: Kruskal-Wallis is non-parametric and robust to these violations.")

    print("\nTest Results:")
    stat, p_value = stats.kruskal(*group_data)
    test_name = "Kruskal-Wallis"
    print(f"- Test Type: {test_name}")
    print(f"- H-statistic: {stat:.4f}")
    print(f"- P-value: {p_value:.4f}")

    print("\nInterpretation:")
    print(
        f"- {'Significant difference' if p_value < 0.05 else 'No significant difference'} "
        f"between groups (p-value {'<' if p_value < 0.05 else '≥'} 0.05)."
    )

    results = {}
    results[f"{categorical_col}_{target}_{test_name}"] = {
        "test_name": test_name,
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }

    print(f"{'='*80}\n")

    return pd.DataFrame(results)
