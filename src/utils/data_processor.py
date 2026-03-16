"""Core data processing and loading module.

Extracted from src/data/data_layer and src/data/feature_engineering.py.
Provides unified data access for all models.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import yaml

import pandas as pd
import numpy as np

# Adjust sys.path to ensure imports work correctly if running as script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from src.core.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class DataProcessor:
    """Core data processor for loading and basic cleaning."""

    @staticmethod
    def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file with robust error handling.
        
        Args:
            file_path: Path to the CSV file.
            
        Returns:
            pd.DataFrame: Loaded and cleaned (basic) DataFrame.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or invalid.
        """
        path = Path(file_path)
        if not path.exists():
            error_msg = f"Data file not found at: {path}. Please ensure data/data.csv exists."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError("CSV file is empty")
                
            # Basic cleaning: handle missing values
            # Using forward fill then backward fill as generic strategy for time series
            df_clean = df.ffill().bfill()
            
            # Ensure DATATIME is parsed if present
            if 'DATATIME' in df_clean.columns:
                df_clean['DATATIME'] = pd.to_datetime(df_clean['DATATIME'])
            
            logger.info(f"Successfully loaded data from {path}, shape: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error loading data from {path}: {str(e)}")
            raise

    @staticmethod
    def split_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time-series data sequentially.
        
        Args:
            df: Input DataFrame
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            
        Returns:
            (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df

def load_config(config_path: Union[str, Path] = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        # Fallback to default if not found, or raise error
        # Try finding it relative to project root
        root_path = Path(__file__).resolve().parents[2] 
        path = root_path / config_path
        
    if not path.exists():
         logger.warning(f"Config file not found at {path}, using defaults.")
         return {}
         
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
