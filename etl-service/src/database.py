"""SQL Server database connection and operations."""

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from contextlib import contextmanager
import structlog
from typing import Generator, Optional
from datetime import datetime

from .config import Config

logger = structlog.get_logger()


class SQLServerConnector:
    """Manages SQL Server connections and data extraction."""

    def __init__(self, config: Config):
        self.config = config
        self._engine: Optional[Engine] = None

    def connect(self) -> None:
        """Initialize database connection."""
        # Get server string (supports both "host,port" and separate host/port)
        server = self.config.get_server_string()

        connection_string = (
            f"mssql+pyodbc://{self.config.mssql_username}:{self.config.mssql_password}"
            f"@{server}"
            f"/{self.config.mssql_database}"
            f"?driver=ODBC+Driver+18+for+SQL+Server"
            f"&TrustServerCertificate=yes"
        )

        self._engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )

        logger.info("sql_server.connected", host=self.config.mssql_host)

    @contextmanager
    def get_connection(self) -> Generator:
        """Context manager for database connections."""
        if not self._engine:
            self.connect()

        conn = self._engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def get_max_updated_at(self, table_name: str) -> Optional[datetime]:
        """Get the maximum updated_at timestamp for a table."""
        with self.get_connection() as conn:
            result = conn.execute(
                text(f"SELECT MAX(updated_at) FROM {table_name}")
            ).scalar()
            return result

    def get_changed_records(
        self, table_name: str, since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch records that have changed since the last sync.

        Args:
            table_name: Name of the table to query
            since: Last sync timestamp (None for full refresh)

        Returns:
            DataFrame with changed records
        """
        if since:
            query = text(f"""
                SELECT *
                FROM {table_name}
                WHERE updated_at > :since
                ORDER BY updated_at ASC
            """)
            params = {"since": since}
        else:
            query = text(f"SELECT * FROM {table_name}")
            params = {}

        with self.get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)
            logger.info("records.fetched", table=table_name, count=len(df), since=since)
            return df

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get table schema information."""
        query = text("""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = :table_name
        """)

        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params={"table_name": table_name})

    def close(self) -> None:
        """Close all database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("sql_server.disconnected")
