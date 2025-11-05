"""
Utility functions for the Ethereum fraud detection system.
Provides helper functions for logging, data loading, and common operations.
"""

import logging
import pickle
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from config import LOG_DIR, LOG_FORMAT, LOG_LEVEL


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Parameters
    ----------
    name : str
        Logger name (usually __name__)
    log_file : str, optional
        Log file path. If None, uses default timestamped log.
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = LOG_DIR / f"pipeline_{timestamp}.log"
    else:
        log_file_path = Path(log_file)
        
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def save_pickle(obj: Any, filepath: Path) -> None:
    """
    Save object to pickle file.
    
    Parameters
    ----------
    obj : Any
        Object to save
    filepath : Path
        Destination file path
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✅ Saved pickle to {filepath}")


def load_pickle(filepath: Path) -> Any:
    """
    Load object from pickle file.
    
    Parameters
    ----------
    filepath : Path
        Source file path
    
    Returns
    -------
    Any
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"✅ Loaded pickle from {filepath}")
    return obj


def save_json(data: Dict, filepath: Path) -> None:
    """
    Save dictionary to JSON file.
    
    Parameters
    ----------
    data : dict
        Data to save
    filepath : Path
        Destination file path
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict:
    """
    Load dictionary from JSON file.
    
    Parameters
    ----------
    filepath : Path
        Source file path
    
    Returns
    -------
    dict
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded JSON from {filepath}")
    return data


def get_timestamp() -> str:
    """
    Get current timestamp string for file naming.
    
    Returns
    -------
    str
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced dataset.
    
    Parameters
    ----------
    y : pd.Series
        Target variable
    
    Returns
    -------
    dict
        Class weights {0: weight_normal, 1: weight_fraud}
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    weight_dict = dict(zip(classes, weights))
    print(f"Class weights: {weight_dict}")
    
    return weight_dict


def get_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost (ratio of negative to positive samples).
    
    Parameters
    ----------
    y : pd.Series
        Target variable
    
    Returns
    -------
    float
        Scale pos weight ratio
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    ratio = neg_count / pos_count
    
    print(f"Class distribution: Normal={neg_count}, Fraud={pos_count}")
    print(f"Scale pos weight: {ratio:.2f}")
    
    return ratio


def memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Calculate DataFrame memory usage in MB.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    
    Returns
    -------
    float
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1024**2


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by downcasting numeric types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    pd.DataFrame
        Memory-optimized DataFrame
    """
    start_mem = memory_usage_mb(df)
    print(f"Memory usage before optimization: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = memory_usage_mb(df)
    reduction = 100 * (start_mem - end_mem) / start_mem
    print(f"Memory usage after optimization: {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that DataFrame has required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list
        List of required column names
    
    Raises
    ------
    ValueError
        If any required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✅ DataFrame validation passed ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    # Test logger
    logger = setup_logger(__name__)
    logger.info("Logger test successful")
    
    # Test timestamp
    print(f"Current timestamp: {get_timestamp()}")
