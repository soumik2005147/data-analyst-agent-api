"""
Feature engineering utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from typing import List, Tuple, Optional, Union


def create_datetime_features(
    df: pd.DataFrame, 
    datetime_column: str, 
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Create datetime-based features from a datetime column.
    
    Args:
        df: pandas DataFrame
        datetime_column: Name of the datetime column
        drop_original: Whether to drop the original datetime column
        
    Returns:
        DataFrame with new datetime features
    """
    df_copy = df.copy()
    
    # Convert to datetime if not already
    df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column])
    
    # Extract datetime features
    df_copy[f'{datetime_column}_year'] = df_copy[datetime_column].dt.year
    df_copy[f'{datetime_column}_month'] = df_copy[datetime_column].dt.month
    df_copy[f'{datetime_column}_day'] = df_copy[datetime_column].dt.day
    df_copy[f'{datetime_column}_dayofweek'] = df_copy[datetime_column].dt.dayofweek
    df_copy[f'{datetime_column}_hour'] = df_copy[datetime_column].dt.hour
    df_copy[f'{datetime_column}_quarter'] = df_copy[datetime_column].dt.quarter
    df_copy[f'{datetime_column}_is_weekend'] = (df_copy[datetime_column].dt.dayofweek >= 5).astype(int)
    
    if drop_original:
        df_copy.drop(columns=[datetime_column], inplace=True)
    
    return df_copy


def encode_categorical_features(
    df: pd.DataFrame, 
    categorical_columns: List[str],
    method: str = 'onehot'
) -> pd.DataFrame:
    """
    Encode categorical features.
    
    Args:
        df: pandas DataFrame
        categorical_columns: List of categorical column names
        method: Encoding method ('onehot' or 'label')
        
    Returns:
        DataFrame with encoded features
    """
    df_copy = df.copy()
    
    for col in categorical_columns:
        if col not in df_copy.columns:
            continue
            
        if method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df_copy[col], prefix=col)
            df_copy = pd.concat([df_copy.drop(columns=[col]), dummies], axis=1)
            
        elif method == 'label':
            # Label encoding
            le = LabelEncoder()
            df_copy[f'{col}_encoded'] = le.fit_transform(df_copy[col].astype(str))
            
    return df_copy


def create_binned_features(
    df: pd.DataFrame, 
    column: str, 
    n_bins: int = 5, 
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create binned features from continuous variables.
    
    Args:
        df: pandas DataFrame
        column: Column name to bin
        n_bins: Number of bins
        labels: Labels for the bins
        
    Returns:
        DataFrame with binned feature
    """
    df_copy = df.copy()
    
    if labels is None:
        labels = [f'{column}_bin_{i}' for i in range(n_bins)]
    
    df_copy[f'{column}_binned'] = pd.cut(
        df_copy[column], 
        bins=n_bins, 
        labels=labels
    )
    
    return df_copy


def scale_features(
    df: pd.DataFrame, 
    columns: List[str], 
    method: str = 'standard'
) -> Tuple[pd.DataFrame, object]:
    """
    Scale numeric features.
    
    Args:
        df: pandas DataFrame
        columns: List of column names to scale
        method: Scaling method ('standard' or 'minmax')
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    df_copy = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    # Fit and transform the specified columns
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    
    return df_copy, scaler


def select_features(
    X: pd.DataFrame, 
    y: pd.Series, 
    k: int = 10, 
    task_type: str = 'classification'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top k features using statistical tests.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        k: Number of features to select
        task_type: 'classification' or 'regression'
        
    Returns:
        Tuple of (selected features DataFrame, selected column names)
    """
    if task_type == 'classification':
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)
    
    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()].tolist()
    
    return pd.DataFrame(X_selected, columns=selected_columns), selected_columns


def create_interaction_features(
    df: pd.DataFrame, 
    column_pairs: List[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Create interaction features from column pairs.
    
    Args:
        df: pandas DataFrame
        column_pairs: List of column pairs to create interactions
        
    Returns:
        DataFrame with interaction features
    """
    df_copy = df.copy()
    
    for col1, col2 in column_pairs:
        if col1 in df_copy.columns and col2 in df_copy.columns:
            # Create interaction feature
            interaction_name = f'{col1}_{col2}_interaction'
            df_copy[interaction_name] = df_copy[col1] * df_copy[col2]
            
            # Create ratio feature if no zeros
            if (df_copy[col2] != 0).all():
                ratio_name = f'{col1}_{col2}_ratio'
                df_copy[ratio_name] = df_copy[col1] / df_copy[col2]
    
    return df_copy
