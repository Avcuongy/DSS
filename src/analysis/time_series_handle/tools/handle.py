import numpy as np
import pandas as pd


def clean_abnormal_days(
    df: pd.DataFrame, date_col: str, expected_count: int
) -> pd.DataFrame:
    """
    Removes days with abnormal data counts based on an expected count per day.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        date_col (str): Name of the datetime column.
        expected_count (int): Expected number of records per day.

    Returns:
        pd.DataFrame: DataFrame with days having non-expected counts removed.
    """
    counts = df[date_col].value_counts()
    abnormal_dates = counts[counts != expected_count].index
    df_cleaned = df[~df[date_col].isin(abnormal_dates)].copy()
    return df_cleaned


"""
def aggregate_by_day(df: pd.DataFrame, date_col: str):
    pass
"""


def analysis_date_gaps(df: pd.DataFrame, date_col: str):
    """
    Analyzes gaps in a time series and displays a formatted table of gap statistics.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        date_col (str): Name of the datetime column.
    """
    df = df.sort_values(date_col)
    delta = df[date_col].diff().dt.days
    delta_df = (
        delta.value_counts()
        .reset_index()
        .rename(columns={"date": "days_diff", "count": "count"})
    )
    delta_df.columns = ["days_diff", "count"]

    consecutive_pairs = delta_df.loc[delta_df["days_diff"] == 1, "count"].sum()
    consecutive_pairs = 0 if pd.isna(consecutive_pairs) else consecutive_pairs

    delta_df["missing_per_gap"] = delta_df["days_diff"] - 1
    delta_df["total_missing"] = delta_df["missing_per_gap"] * delta_df["count"]
    total_missing = delta_df[delta_df["days_diff"] > 1]["total_missing"].sum()

    print(f"\n{' DATE GAP ANALYSIS ':=^100}")
    print(f"\n- Found {consecutive_pairs} consecutive day pairs")
    print(
        f"- Total missing days: {total_missing if not pd.isna(total_missing) else 0}\n"
    )

    gap_stats = delta_df[delta_df["days_diff"] > 0].sort_values("days_diff")

    if not gap_stats.empty:
        from tabulate import tabulate

        table_data = []
        for _, row in gap_stats.iterrows():
            table_data.append(
                [
                    f"{row['days_diff']:.0f} days",
                    row["count"],
                    f"{row['missing_per_gap']:.0f} days",
                    f"{row['total_missing']:.0f} days",
                ]
            )

        print("GAP DETAILS")
        print(
            tabulate(
                table_data,
                headers=["Gap size", "Occurrences", "Missing per gap", "Total missing"],
                tablefmt="grid",
                stralign="center",
                numalign="center",
            )
        )
    else:
        print("Perfect sequence, no gaps found")

    print(f"\n{'='*100}")
    print(f"First date: {df[date_col].min().date()}")
    print(f"Last date: {df[date_col].max().date()}")
    print(
        f"Date range coverage: {len(df)}/{((df[date_col].max() - df[date_col].min()).days + 1)} days"
    )
    print(f"{'='*100}\n")


def find_missing_date(df: pd.DataFrame, date_col: str) -> list:
    """
    Identifies specific missing dates within gaps in a time series.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        date_col (str): Name of the datetime column.

    Returns:
        list: Sorted list of missing dates within identified gaps.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).drop_duplicates(subset=date_col)

    df["date_diff"] = df[date_col].diff().dt.days

    missing_dates = []
    for _, row in df[df["date_diff"] > 1].iterrows():
        current_date = row[date_col]
        prev_date = df[df[date_col] < current_date][date_col].iloc[-1]
        gap_dates = pd.date_range(
            start=prev_date + pd.Timedelta(days=1),
            end=current_date - pd.Timedelta(days=1),
            freq="D",
        )
        missing_dates.extend(gap_dates)

    return sorted(list(set(missing_dates)))


def forward_fill_handle(
    df: pd.DataFrame, date_col: str, start_date=None, end_date=None
) -> pd.DataFrame:
    """
    Fills missing dates in a time series using forward fill for all columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        date_col (str): Name of the datetime column.
        start_date (str, optional): Start date for the date range. Defaults to min date in data.
        end_date (str, optional): End date for the date range. Defaults to max date in data.

    Returns:
        pd.DataFrame: DataFrame with missing dates filled using forward fill, with an 'is_missing' flag.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    original_dates = df[date_col].copy()

    start = pd.to_datetime(start_date) if start_date else df[date_col].min()
    end = pd.to_datetime(end_date) if end_date else df[date_col].max()

    full_dates = pd.date_range(start=start, end=end, freq="D")
    df = df.set_index(date_col).reindex(full_dates)

    df = df.fillna(method="ffill")

    df["is_missing"] = (~df.index.isin(original_dates)).astype(int)

    return df.reset_index().rename(columns={"index": date_col})


def backward_fill_handle(
    df: pd.DataFrame, date_col: str, start_date=None, end_date=None
) -> pd.DataFrame:
    """
    Fills missing dates in a time series using backward fill for all columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        date_col (str): Name of the datetime column.
        start_date (str, optional): Start date for the date range. Defaults to min date in data.
        end_date (str, optional): End date for the date range. Defaults to max date in data.

    Returns:
        pd.DataFrame: DataFrame with missing dates filled using backward fill, with an 'is_missing' flag.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    original_dates = df[date_col].copy()

    start = pd.to_datetime(start_date) if start_date else df[date_col].min()
    end = pd.to_datetime(end_date) if end_date else df[date_col].max()

    full_dates = pd.date_range(start=start, end=end, freq="D")
    df = df.set_index(date_col).reindex(full_dates)

    df = df.fillna(method="bfill")

    df["is_missing"] = (~df.index.isin(original_dates)).astype(int)

    return df.reset_index().rename(columns={"index": date_col})


def interpolate_handle(
    df: pd.DataFrame,
    date_col: str,
    start_date=None,
    end_date=None,
    method: str = "linear",
    spline_order=3,
) -> pd.DataFrame:
    """
    Fills missing dates in a time series using interpolation for numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        date_col (str): Name of the datetime column.
        start_date (str, optional): Start date for the date range. Defaults to min date in data.
        end_date (str, optional): End date for the date range. Defaults to max date in data.
        method (str, optional): Interpolation method ('linear', 'spline'). Defaults to 'linear'.
        spline_order (int, optional): Order for spline interpolation. Defaults to 3.

    Returns:
        pd.DataFrame: DataFrame with missing dates filled using interpolation for numerical columns, with an 'is_missing' flag.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    original_dates = df[date_col].copy()

    start = pd.to_datetime(start_date) if start_date else df[date_col].min()
    end = pd.to_datetime(end_date) if end_date else df[date_col].max()

    full_dates = pd.date_range(start=start, end=end, freq="D")
    df = df.set_index(date_col).reindex(full_dates)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        df[col] = df[col].interpolate(method=method, order=spline_order)

    df["is_missing"] = (~df.index.isin(original_dates)).astype(int)

    return df.reset_index().rename(columns={"index": date_col})


def mixture_handle(
    df: pd.DataFrame,
    date_col: str,
    start_date=None,
    end_date=None,
    interpolate_method: str = "linear",
    categorical_fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Fills missing dates using interpolation for numerical columns and forward/backward fill for categorical columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        date_col (str): Name of the datetime column.
        start_date (str, optional): Start date for the date range. Defaults to min date in data.
        end_date (str, optional): End date for the date range. Defaults to max date in data.
        interpolate_method (str, optional): Interpolation method for numerical columns ('linear', 'spline'). Defaults to 'linear'.
        categorical_fill_method (str, optional): Fill method for categorical columns ('ffill', 'bfill'). Defaults to 'ffill'.

    Returns:
        pd.DataFrame: DataFrame with missing dates filled, with an 'is_missing' flag.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    original_dates = df[date_col].copy()

    start = pd.to_datetime(start_date) if start_date else df[date_col].min()
    end = pd.to_datetime(end_date) if end_date else df[date_col].max()

    full_dates = pd.date_range(start=start, end=end, freq="D")
    df = df.set_index(date_col).reindex(full_dates)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        df[col] = df[col].interpolate(method=interpolate_method)

    for col in categorical_cols:
        df[col] = df[col].fillna(method=categorical_fill_method)

    df["is_missing"] = (~df.index.isin(original_dates)).astype(int)

    return df.reset_index().rename(columns={"index": date_col})
