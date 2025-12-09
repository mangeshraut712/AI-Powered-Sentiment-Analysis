"""
Data Loading Utilities
=======================

This module provides functions for loading and managing the tweet emotions dataset.
"""

import pandas as pd
import os
from pathlib import Path


def get_data_path(filename='tweet_emotions.csv', data_dir='raw'):
    """
    Get the absolute path to a data file.
    
    Parameters:
    -----------
    filename : str
        Name of the data file
    data_dir : str
        Subdirectory within data/ ('raw' or 'processed')
    
    Returns:
    --------
    Path
        Absolute path to the data file
    """
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / data_dir / filename
    return data_path


def load_dataset(filepath=None, verbose=True):
    """
    Load the tweet emotions dataset.
    
    Parameters:
    -----------
    filepath : str or Path, optional
        Path to the CSV file. If None, uses default path.
    verbose : bool
        Whether to print dataset information
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    
    Examples:
    ---------
    >>> df = load_dataset()
    >>> print(df.shape)
    (40000, 3)
    """
    if filepath is None:
        filepath = get_data_path()
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    df = pd.read_csv(filepath, delimiter=',')
    
    if verbose:
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
    
    return df


def get_dataset_info(df):
    """
    Get comprehensive information about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    
    Returns:
    --------
    dict
        Dictionary containing dataset statistics
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'sentiment_counts': df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else None,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    return info


def save_processed_data(df, filename='processed_tweets.csv'):
    """
    Save processed data to the processed data directory.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed dataset
    filename : str
        Name for the output file
    """
    output_path = get_data_path(filename, data_dir='processed')
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    df = load_dataset()
    info = get_dataset_info(df)
    print("\nDataset Info:")
    for key, value in info.items():
        print(f"{key}: {value}")
