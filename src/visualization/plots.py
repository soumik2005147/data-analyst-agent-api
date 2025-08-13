"""
Visualization utilities for data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union


# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_distribution(
    df: pd.DataFrame, 
    column: str, 
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot distribution of a numeric column.
    
    Args:
        df: pandas DataFrame
        column: Column name to plot
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(df[column].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title(f'Histogram of {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    
    # Box plot
    axes[1].boxplot(df[column].dropna())
    axes[1].set_title(f'Box Plot of {column}')
    axes[1].set_ylabel(column)
    
    plt.tight_layout()
    plt.show()


def correlation_heatmap(
    df: pd.DataFrame, 
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df: pandas DataFrame
        figsize: Figure size tuple
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric columns found for correlation analysis.")
        return
    
    plt.figure(figsize=figsize)
    correlation_matrix = numeric_df.corr()
    
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": .5}
    )
    
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(
    df: pd.DataFrame, 
    column: str, 
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot distribution of a categorical column.
    
    Args:
        df: pandas DataFrame
        column: Column name to plot
        top_n: Show only top N categories
        figsize: Figure size tuple
    """
    value_counts = df[column].value_counts()
    
    if top_n:
        value_counts = value_counts.head(top_n)
    
    plt.figure(figsize=figsize)
    value_counts.plot(kind='bar')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def interactive_scatter_plot(
    df: pd.DataFrame, 
    x: str, 
    y: str, 
    color: Optional[str] = None,
    size: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Create an interactive scatter plot using Plotly.
    
    Args:
        df: pandas DataFrame
        x: X-axis column name
        y: Y-axis column name
        color: Color grouping column name
        size: Size mapping column name
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if title is None:
        title = f'{y} vs {x}'
    
    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        color=color, 
        size=size,
        title=title,
        hover_data=df.select_dtypes(include=[np.number]).columns.tolist()
    )
    
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        font=dict(size=12)
    )
    
    return fig


def plot_time_series(
    df: pd.DataFrame, 
    date_column: str, 
    value_columns: Union[str, List[str]],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot time series data.
    
    Args:
        df: pandas DataFrame
        date_column: Date column name
        value_columns: Value column name(s)
        title: Plot title
        figsize: Figure size tuple
    """
    if isinstance(value_columns, str):
        value_columns = [value_columns]
    
    plt.figure(figsize=figsize)
    
    for col in value_columns:
        plt.plot(df[date_column], df[col], label=col, linewidth=2)
    
    plt.title(title or 'Time Series Plot')
    plt.xlabel(date_column)
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
