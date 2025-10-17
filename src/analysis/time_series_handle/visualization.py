import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_time_continuity_by_level(
    df: pd.DataFrame, date_col: str = "date", level: str = "day"
):
    """
    Plots a timeline to check the continuity of data over time at a specified level (day, month, or year).

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        date_col (str): Name of the datetime column.
        level (str, optional): Time granularity for continuity check ('day', 'month', 'year').
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if level == "day":
        df["time_unit"] = df[date_col].dt.date
        title = "Continuity by Day"
    elif level == "month":
        df["time_unit"] = df[date_col].dt.to_period("M").dt.to_timestamp()
        title = "Continuity by Month"
    elif level == "year":
        df["time_unit"] = df[date_col].dt.to_period("Y").dt.to_timestamp()
        title = "Continuity by Year"
    else:
        raise ValueError("Level must be one of ['day', 'month', 'year']")

    df_unique = df.drop_duplicates(subset="time_unit")
    df_unique = df_unique.sort_values(by="time_unit").reset_index(drop=True)
    df_unique["time_index"] = df_unique.index

    plt.figure(figsize=(12, 4))
    plt.plot(
        df_unique["time_unit"],
        df_unique["time_index"],
        marker="o",
        linestyle="-",
        color="tab:purple",
    )
    plt.title(f"Timeline – Continuity Check ({level})")
    plt.xlabel(f"{level.capitalize()}")
    plt.ylabel("Time Order")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_boxplot_by_time(df: pd.DataFrame, time_col: str, target_col: str):
    """
    Plots a boxplot to visualize the distribution of a numerical target variable across a time-based categorical column.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        time_col (str): Name of the time-based categorical column (e.g., season, month, day of week).
        target_col (str): Name of the numerical target column.
    """
    plt.figure(figsize=(16, 5))
    sns.boxplot(x=df[time_col], y=df[target_col], palette="coolwarm", showfliers=False)
    plt.title(f"Rainfall Distribution by {time_col.capitalize()}")
    plt.xlabel(time_col.capitalize())
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.show()


def plot_lineplot_by_time(df: pd.DataFrame, time_col: str, target_col: str):
    """
    Plots a line plot to visualize the average trend of a numerical target variable over a time-based categorical column.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        time_col (str): Name of the time-based categorical column (e.g., day, month, week).
        target_col (str): Name of the numerical target column.
    """
    df_plot = df.groupby(time_col)[target_col].mean().reset_index()

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df_plot, x=time_col, y=target_col, marker="o", color="steelblue")
    plt.title(f"Average Rainfall by {time_col.capitalize()}")
    plt.xlabel(time_col.capitalize())
    plt.ylabel(f"Mean {target_col}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
