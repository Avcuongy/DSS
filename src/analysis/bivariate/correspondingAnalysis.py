import pandas as pd
from prince import CA

from itertools import combinations


def calculate_ca_inertia_pairs(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Computes the total inertia for pairs of categorical variables using Correspondence Analysis (CA).

    Args:
        df (pd.DataFrame): Input DataFrame containing categorical variables.
        n_components (int, optional): Number of components for CA.

    Returns:
        pd.DataFrame: DataFrame with variable pairs and their total inertia, sorted by inertia in descending order.
    """
    results = []

    for var1, var2 in combinations(df.columns, 2):
        try:
            ct = pd.crosstab(df[var1], df[var2])

            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                ca = CA(
                    n_components=n_components,
                    n_iter=10,
                    copy=True,
                    check_input=True,
                    engine="sklearn",
                    random_state=42,
                )
                ca.fit(ct)

                results.append(
                    {
                        "variable_1": var1,
                        "variable_2": var2,
                        "inertia": ca.total_inertia_,
                        "n_modal_1": ct.shape[0],
                        "n_modal_2": ct.shape[1],
                    }
                )

        except Exception as e:
            continue

    return (
        pd.DataFrame(results)
        .sort_values(by="inertia", ascending=False)
        .reset_index(drop=True)
    )
