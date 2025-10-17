import pandas as pd
import scipy.stats as stats


def chi2_test(df: pd.DataFrame, categorical_col: str, target_col: str) -> pd.DataFrame:
    """
    Chi-square test of independence between two categorical variables.

    When to use:
        - When both variables are categorical
        - Expected frequencies in contingency table are sufficiently large (≥5)

    Args:
        df (pd.DataFrame): Input DataFrame
        categorical_col (str): First categorical variable
        target_col (str): Second categorical variable

    Returns:
        pd.DataFrame: Chi-square statistic, p-value, degrees of freedom
    """
    print(f"{'='*80}")
    print(f"CHI-SQUARE TEST: {categorical_col} vs {target_col}".center(80))
    print(f"{'='*80}")

    contingency = pd.crosstab(df[categorical_col], df[target_col])
    stat, p, dof, expected = stats.chi2_contingency(contingency)

    print("Test Results:")
    print(f"- Test Type: Chi-square test of independence")
    print(f"- Chi2 Statistic: {stat:.4f}")
    print(f"- Degrees of freedom: {dof}")
    print(f"- P-value: {p:.4f}")
    print(
        f"\nInterpretation:\n"
        f"- {'Dependent' if p < 0.05 else 'Independent'} (p-value {'<' if p < 0.05 else '≥'} 0.05)"
    )
    print(f"{'='*80}\n")

    return pd.DataFrame(
        {
            "test": ["chi2"],
            "statistic": [stat],
            "p_value": [p],
            "dof": [dof],
            "significant": [p < 0.05],
        }
    )
