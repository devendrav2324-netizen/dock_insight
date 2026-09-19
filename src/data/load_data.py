"""
Data loading module for CharterAI.
"""
import os
import pandas as pd
from typing import Optional
from src.utils.logging import get_logger

logger = get_logger(__name__)

def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Safely load a CSV file into a pandas DataFrame.
    Skips rows starting with '#' (our SYNTHETIC_DEMO_DATA markers).
    
    Args:
        filepath: Path to the CSV file.
        **kwargs: Additional arguments to pass to pd.read_csv.
        
    Returns:
        pd.DataFrame containing the loaded data.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Cannot load data. File not found: {filepath}")
    
    try:
        # We assume comments start with #
        df = pd.read_csv(filepath, comment='#', skipinitialspace=True, **kwargs)
        logger.info(f"Successfully loaded {len(df)} rows from {os.path.basename(filepath)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV from {filepath}: {e}")
        raise

def load_all_datasets(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all required datasets from a directory.
    
    Args:
        data_dir: Path to the directory containing the CSV files.
        
    Returns:
        Dictionary mapping dataset names to DataFrames.
    """
    datasets = {}
    expected_files = [
        "freight_rates.csv", "vessels.csv", "ports.csv", "routes.csv",
        "commodities.csv", "congestion.csv", "weather.csv",
        "economic_indicators.csv", "events.csv"
    ]
    
    for filename in expected_files:
        filepath = os.path.join(data_dir, filename)
        dataset_name = filename.split('.')[0]
        try:
            datasets[dataset_name] = load_csv(filepath)
        except Exception as e:
            logger.warning(f"Could not load dataset {dataset_name}: {e}")
            datasets[dataset_name] = pd.DataFrame()
            
    return datasets
