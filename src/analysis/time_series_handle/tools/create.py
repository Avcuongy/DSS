import pandas as pd


def create_time_variables(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Creates time-based features from a datetime column.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        date_col (str): Name of the datetime column.

    Returns:
        pd.DataFrame: DataFrame with added time-based features: hour, dayofweek, dayofmonth, month, weekofyear, quarter, is_weekend, and season (categorical).
    """
    df = df.copy()
    dt = df[date_col]

    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["dayofmonth"] = dt.dt.day
    df["month"] = dt.dt.month
    df["weekofyear"] = dt.dt.isocalendar().week
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = dt.dt.dayofweek >= 5

    season_map = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }

    df["season"] = dt.dt.month.map(season_map)
    df["season"] = pd.Categorical(
        df["season"], categories=["spring", "summer", "autumn", "winter"], ordered=True
    )

    return df


def create_lag_variables(
    df: pd.DataFrame, group_col: str, target_cols: list[str], lags=[1, 2, 3]
) -> pd.DataFrame:
    """
    Creates lag features for specified columns within groups.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        group_col (str): Column name to group by (e.g., province).
        target_cols (list[str]): List of numerical columns to create lag features for.
        lags (list, optional): List of lag periods. Defaults to [1, 2, 3].

    Returns:
        pd.DataFrame: DataFrame with added lag features for each target column and lag period.
    """
    df = df.sort_values(by=[group_col, "date"])

    for col in target_cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df


def create_lead_variables(
    df: pd.DataFrame, group_col: str, target_cols: list[str], leads=[1]
) -> pd.DataFrame:
    """
    Creates lead features for specified columns within groups to assist in future predictions.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        group_col (str): Column name to group by (e.g., province).
        target_cols (list[str]): List of numerical columns to create lead features for.
        leads (list, optional): List of lead periods. Defaults to [1].

    Returns:
        pd.DataFrame: DataFrame with added lead features for each target column and lead period.
    """
    df = df.sort_values(by=[group_col, "date"])

    for col in target_cols:
        for lead in leads:
            df[f"{col}_lead_{lead}"] = df.groupby(group_col)[col].shift(-lead)
    return df
