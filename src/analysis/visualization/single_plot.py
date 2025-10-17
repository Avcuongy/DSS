import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, Optional, Tuple


sns.set_style("whitegrid")


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    style: Optional[str] = None,
    palette: Optional[str] = None,
    title: str = "Scatter",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Creates a scatter plot for two numerical variables with optional grouping.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        x (str): Column name for x-axis (numerical).
        y (str): Column name for y-axis (numerical).
        hue (Optional[str]): Column name for color grouping (categorical).
        style (Optional[str]): Column name for marker style grouping (categorical).
        palette (Optional[str]): Color palette for hue groups.
        title (str): Plot title.
        xlabel (Optional[str]): X-axis label.
        ylabel (Optional[str]): Y-axis label.
        figsize (Tuple[int, int]): Figure size as (width, height).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if x is None or y is None:
        raise ValueError("Both x and y must be specified.")

    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        style=style,
        palette=palette,
    )
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)
    plt.tight_layout()
    plt.show()


def plot_histogram(
    df: Union[pd.DataFrame, pd.Series],
    x: Optional[str] = None,
    y: Optional[str] = None,
    bins: int = 30,
    hue: Optional[str] = None,
    multiple: str = "layer",
    kde: bool = False,
    common_norm: bool = True,
    palette: Optional[str] = None,
    title: str = "Histogram",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot a histogram for a given column from a DataFrame or Series.

    Args:
        df (Union[pd.DataFrame, pd.Series]): Input DataFrame or Series.
        x (Optional[str]): Column name for the x-axis histogram (vertical).
        y (Optional[str]): Column name for the y-axis histogram (horizontal).
        bins (int): Number of bins for the histogram.
        hue (Optional[str]): Column name for grouping (categorical).
        multiple (str): How to represent multiple distributions ('layer', 'stack', 'dodge', 'fill').
        kde (bool): Whether to overlay a Kernel Density Estimate curve.
        common_norm (bool): If True, normalize density across groups; if False, normalize separately.
        palette (Optional[str]): Color palette for hue groups.
        title (str): Plot title.
        xlabel (Optional[str]): Label for the x-axis.
        ylabel (Optional[str]): Label for the y-axis.
        figsize (Tuple[int, int]): Figure size as (width, height).
    """
    plt.figure(figsize=figsize)

    if isinstance(df, pd.Series):
        df = pd.DataFrame({"value": df})
        if x is None and y is None:
            x = "value"

    if (x is None and y is None) or (x is not None and y is not None):
        raise ValueError("You must provide exactly one of the parameters: x or y.")

    sns.histplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        multiple=multiple,
        bins=bins,
        kde=kde,
        common_norm=common_norm,
        palette=palette,
    )

    plt.title(title)
    plt.xlabel(xlabel if xlabel else (x if x else "Frequency"))
    plt.ylabel(ylabel if ylabel else (y if y else "Frequency"))
    plt.show()


def plot_kde(
    df: Union[pd.DataFrame, pd.Series],
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    fill: bool = True,
    common_norm: bool = True,
    palette: Optional[str] = None,
    title: str = "KDE Plot",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot a Kernel Density Estimation (KDE) plot.

    Args:
        df (Union[pd.DataFrame, pd.Series]): Input DataFrame or Series.
        x (Optional[str]): Column name for x-axis KDE.
        y (Optional[str]): Column name for y-axis KDE.
        hue (Optional[str]): Column name for grouping (categorical).
        fill (bool): Whether to fill the area under the KDE curve.
        common_norm (bool): If True, normalize density across groups; if False, normalize separately.
        palette (Optional[str]): Color palette for hue groups.
        title (str): Plot title.
        xlabel (Optional[str]): X-axis label.
        ylabel (Optional[str]): Y-axis label.
        figsize (Tuple[int, int]): Figure size as (width, height).
    """
    plt.figure(figsize=figsize)

    if isinstance(df, pd.Series):
        df = pd.DataFrame({"value": df})
        if x is None and y is None:
            x = "value"

    if (x is None and y is None) or (x is not None and y is not None):
        raise ValueError("You must provide exactly one of the parameters: x or y.")

    sns.kdeplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        fill=fill,
        common_norm=common_norm,
        linewidth=2,
    )

    plt.title(title)
    plt.xlabel(xlabel if xlabel else (x if x else ""))
    plt.ylabel(ylabel if ylabel else (y if y else "Density"))
    plt.tight_layout()
    plt.show()


def plot_boxplot(
    df: Union[pd.DataFrame, pd.Series],
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    palette: Optional[str] = None,
    title: str = "Boxplot",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot a boxplot for numerical data.

    Args:
        df (Union[pd.DataFrame, pd.Series]): Input DataFrame or Series.
        x (Optional[str]): Column for x-axis (categorical or numeric).
        y (Optional[str]): Column for y-axis (categorical or numeric).
        hue (Optional[str]): Column name for grouping (categorical).
        palette (Optional[str]): Color palette for hue groups.
        title (str): Plot title.
        xlabel (Optional[str]): X-axis label.
        ylabel (Optional[str]): Y-axis label.
        figsize (Tuple[int, int]): Figure size as (width, height).
    """
    plt.figure(figsize=figsize)

    if isinstance(df, pd.Series):
        df = pd.DataFrame({"value": df})
        if x is None and y is None:
            y = "value"

    if x is None and y is None:
        raise ValueError("You must provide either x or y (or both for comparison).")

    sns.boxplot(data=df, x=x, y=y, hue=hue, palette=palette)

    plt.title(title)
    plt.xlabel(xlabel if xlabel else (x if x else ""))
    plt.ylabel(ylabel if ylabel else (y if y else ""))
    plt.show()


def plot_barplot(
    df: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    palette: Optional[str] = None,
    title: str = "Barplot",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot a barplot for categorical data.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        x (Optional[str]): Column for x-axis.
        y (Optional[str]): Column for y-axis.
        hue (Optional[str]): Column name for grouping (categorical).
        palette (Optional[str]): Color palette for hue groups.
        title (str): Plot title.
        xlabel (Optional[str]): X-axis label.
        ylabel (Optional[str]): Y-axis label.
        figsize (Tuple[int, int]): Figure size as (width, height).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Barplot requires a DataFrame input")
    if x is None and y is None:
        raise ValueError("You must provide at least one axis (x or y).")

    plt.figure(figsize=figsize)
    sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else (x if x else ""))
    plt.ylabel(ylabel if ylabel else (y if y else ""))
    plt.show()


def plot_countplot(
    df: Union[pd.DataFrame, pd.Series],
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    palette: Optional[str] = None,
    title: str = "Countplot",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot a countplot for categorical data.

    Args:
        df (Union[pd.DataFrame, pd.Series]): Input DataFrame or Series.
        x (Optional[str]): Column for x-axis (categorical).
        y (Optional[str]): Column for y-axis (categorical).
        hue (Optional[str]): Column name for grouping (categorical).
        palette (Optional[str]): Color palette for hue groups.
        title (str): Plot title.
        xlabel (Optional[str]): X-axis label.
        ylabel (Optional[str]): Y-axis label.
        figsize (Tuple[int, int]): Figure size as (width, height).
    """
    plt.figure(figsize=figsize)

    if isinstance(df, pd.Series):
        df = pd.DataFrame({"value": df})
        if x is None and y is None:
            x = "value"

    if x is None and y is None:
        raise ValueError("You must provide either x or y for countplot.")

    sns.countplot(data=df, x=x, y=y, hue=hue, palette=palette)

    plt.title(title)
    plt.xlabel(xlabel if xlabel else (x if x else "Category"))
    plt.ylabel(ylabel if ylabel else (y if y else "Count"))
    plt.show()
