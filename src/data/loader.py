"""
Data loading and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any


def load_csv_data(
    filepath: Union[str, Path], 
    **kwargs
) -> pd.DataFrame:
    """
    Load CSV data with error handling.
    
    Args:
        filepath: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        print(f"Successfully loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        raise


def basic_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic information about the dataset.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dictionary with basic statistics
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    return info


def clean_numeric_columns(
    df: pd.DataFrame, 
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Clean numeric columns by handling missing values and outliers.
    
    Args:
        df: pandas DataFrame
        columns: List of columns to clean. If None, all numeric columns.
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_clean.columns:
            # Fill missing values with median
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            
    return df_clean


def save_processed_data(
    df: pd.DataFrame, 
    filename: str, 
    output_dir: Union[str, Path] = "data/processed"
) -> None:
    """
    Save processed data to the processed directory.
    
    Args:
        df: pandas DataFrame to save
        filename: Name of the output file
        output_dir: Output directory path
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if filename.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif filename.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
    
    print(f"Data saved to {output_path}")
