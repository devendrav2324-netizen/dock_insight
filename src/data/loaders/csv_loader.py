"""
DockInsights — Robust CSV Loader.

Handles comments (e.g. # SYNTHETIC_DEMO_DATA), whitespace, encodings,
and column sanitization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_csv_file(
    filepath: Union[str, Path],
    comment: str = "#",
    encoding: str = "utf-8",
    parse_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to load a single CSV file.
    """
    loader = CSVLoader()
    return loader.load(filepath, comment=comment, encoding=encoding, parse_dates=parse_dates)


class CSVLoader:
    """
    Production-grade CSV loader for maritime datasets.
    """

    def __init__(self):
        pass

    def load(
        self,
        filepath: Union[str, Path],
        comment: str = "#",
        encoding: str = "utf-8",
        parse_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load a CSV file with automatic header cleaning and comment filtering.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        try:
            df = pd.read_csv(
                path,
                comment=comment,
                encoding=encoding,
                skipinitialspace=True,
            )
        except Exception as e:
            logger.error(f"Failed to read CSV at {path}: {e}")
            raise

        # Strip whitespace from column names
        df.columns = [col.strip() for col in df.columns]

        # Strip whitespace from string/object columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
            # Replace string representations of NaN with actual NaN
            df[col] = df[col].replace({"nan": None, "None": None, "": None})

        # Optionally parse dates
        if parse_dates:
            for d_col in parse_dates:
                if d_col in df.columns:
                    df[d_col] = pd.to_datetime(df[d_col], errors="coerce")

        logger.info(f"Loaded {len(df)} rows from {path.name}")
        return df

    def load_directory(
        self,
        dir_path: Union[str, Path],
        comment: str = "#",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files in a directory into a dictionary of DataFrames.
        """
        directory = Path(dir_path)
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        datasets = {}
        for csv_file in sorted(directory.glob("*.csv")):
            key = csv_file.stem
            datasets[key] = self.load(csv_file, comment=comment)

        return datasets
