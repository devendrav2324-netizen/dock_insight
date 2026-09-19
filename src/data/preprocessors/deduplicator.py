"""
Charter-AI — Data Deduplicator.

Detects, logs, and resolves duplicate rows across composite business keys.
"""

from typing import List, Optional
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
    """Return count of duplicate rows for given subset."""
    return int(df.duplicated(subset=subset).sum())


class Deduplicator:
    """
    Deduplication engine for maritime time series and reference tables.
    """

    def deduplicate(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """
        Drop duplicate rows based on subset keys.
        keep: 'first', 'last', False.
        """
        dup_count = df.duplicated(subset=subset).sum()
        if dup_count > 0:
            logger.info(f"Removing {dup_count} duplicate rows (subset: {subset}, keep: {keep})")
            return df.drop_duplicates(subset=subset, keep=keep)
        return df
