import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau


def _interpret_trend(corr, p_value, significance_level=0.05):
    """
    Interprets the trend based on correlation coefficient and p-value.
    """
    if p_value >= significance_level:
        return "Not statistically significant"

    trend_strength = (
        "weak" if abs(corr) < 0.3 else "moderate" if abs(corr) < 0.7 else "strong"
    )
    trend_direction = "positive" if corr > 0 else "negative"

    return f"{trend_direction.capitalize()} {trend_strength}"


def check_pearson_correlation(df, numerical_cols, target, significance_level=0.05):
    """
    Checks linear correlation between numerical variables and target using Pearson's method.
    """
    results = []
    for col in numerical_cols:
        corr, p_value = pearsonr(df[col], df[target])
        trend = _interpret_trend(corr, p_value, significance_level)
        results.append(
            {
                "variable": col,
                "pearson_correlation": round(corr, 4),
                "p-value": round(p_value, 6),
                "significant": p_value < significance_level,
                "trend": trend,
            }
        )

    return pd.DataFrame(results)


def check_spearman_correlation(df, numerical_cols, target, significance_level=0.05):
    """
    Checks monotonic correlation between numerical variables and target using Spearman's method.
    """
    results = []
    for col in numerical_cols:
        corr, p_value = spearmanr(df[col], df[target])
        trend = _interpret_trend(corr, p_value, significance_level)
        results.append(
            {
                "variable": col,
                "spearman_correlation": round(corr, 4),
                "p-value": round(p_value, 6),
                "significant": p_value < significance_level,
                "trend": trend,
            }
        )

    return pd.DataFrame(results)


def check_kendall_correlation(df, numerical_cols, target, significance_level=0.05):
    """
    Checks rank correlation between numerical variables and target using Kendall's Tau.
    """
    results = []
    for col in numerical_cols:
        corr, p_value = kendalltau(df[col], df[target])
        trend = _interpret_trend(corr, p_value, significance_level)
        results.append(
            {
                "variable": col,
                "kendall_correlation": round(corr, 4),
                "p-value": round(p_value, 6),
                "significant": p_value < significance_level,
                "trend": trend,
            }
        )

    return pd.DataFrame(results)
