"""
DockInsights — Database Loader.

Loads validated DataFrames into SQL database tables using SQLAlchemy ORM.
Supports transactional execution, bulk insertions, and upserts.
"""

from typing import Any, Dict, List, Type, Union
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.data.db import Base
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseLoader:
    """
    Handles bulk ingestion and persistence into relational database models.
    """

    def __init__(self, session: Union[Session, AsyncSession]):
        self.session = session

    def load_records(
        self,
        model_cls: Type[Base],
        records: List[Dict[str, Any]],
        truncate_first: bool = False,
    ) -> int:
        """
        Synchronously insert a list of dictionaries into the target model table.
        """
        if not isinstance(self.session, Session):
            raise TypeError("Synchronous Session required for load_records")

        if truncate_first:
            table_name = model_cls.__tablename__
            self.session.execute(text(f"DELETE FROM {table_name}"))
            self.session.commit()

        count = 0
        for rec in records:
            # Filter keys to only valid model columns
            valid_cols = {c.name for c in model_cls.__table__.columns}
            filtered = {k: v for k, v in rec.items() if k in valid_cols}
            instance = model_cls(**filtered)
            self.session.merge(instance)
            count += 1

        self.session.commit()
        logger.info(f"Persisted {count} records into {model_cls.__tablename__}")
        return count

    def load_dataframe(
        self,
        model_cls: Type[Base],
        df: pd.DataFrame,
        truncate_first: bool = False,
    ) -> int:
        """
        Load a pandas DataFrame directly into the target table.
        """
        records = df.to_dict(orient="records")
        return self.load_records(model_cls, records, truncate_first=truncate_first)
