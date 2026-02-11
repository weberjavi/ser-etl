"""Core synchronization logic."""

import duckdb
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import structlog
import tempfile
import os

from .config import Config
from .database import SQLServerConnector
from .s3_uploader import S3Uploader

logger = structlog.get_logger()


class SyncManager:
    """Manages the synchronization process from SQL Server to DuckDB."""

    def __init__(self, config: Config):
        self.config = config
        self.sql_connector = SQLServerConnector(config)
        self.s3_uploader = S3Uploader(config)
        self._state: Dict[str, datetime] = {}  # table -> last_sync_time

    def initialize(self) -> None:
        """Initialize connections."""
        self.sql_connector.connect()
        logger.info("sync_manager.initialized")

    def sync_table(self, table_name: str) -> int:
        """
        Sync a single table from SQL Server to staging.

        Returns:
            Number of rows synced
        """
        last_sync = self._state.get(table_name)

        # Fetch changed records
        df = self.sql_connector.get_changed_records(table_name, last_sync)

        if df.empty:
            logger.info("no_changes", table=table_name)
            return 0

        # Update state
        max_updated = df["updated_at"].max()
        self._state[table_name] = pd.to_datetime(max_updated)

        logger.info(
            "table.synced", table=table_name, rows=len(df), max_updated=max_updated
        )

        return len(df)

    def build_duckdb(self, dataframes: Dict[str, pd.DataFrame]) -> str:
        """
        Build DuckDB database from DataFrames.

        Args:
            dataframes: Dict of table_name -> DataFrame

        Returns:
            Path to created DuckDB file
        """
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_file.close()

        conn = duckdb.connect(temp_file.name)

        try:
            # Create tables and insert data
            for table_name, df in dataframes.items():
                if df.empty:
                    continue

                # Register DataFrame as a view
                conn.register(f"temp_{table_name}", df)

                # Create table with proper schema
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} AS 
                    SELECT * FROM temp_{table_name}
                """)

                # Create indexes for common query patterns
                if "id" in df.columns:
                    conn.execute(f"""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_id 
                        ON {table_name}(id)
                    """)

                if "updated_at" in df.columns:
                    conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_updated 
                        ON {table_name}(updated_at)
                    """)

                logger.info("table.created", table=table_name, rows=len(df))

            # Create helpful views for analytics
            conn.execute("""
                CREATE VIEW IF NOT EXISTS v_inventory_summary AS
                SELECT 
                    COUNT(*) as total_items,
                    SUM(quantity) as total_quantity,
                    AVG(price) as avg_price,
                    MIN(updated_at) as oldest_update,
                    MAX(updated_at) as latest_update
                FROM inventory
            """)

            conn.commit()

        finally:
            conn.close()

        return temp_file.name

    def run_sync(self) -> bool:
        """
        Run full synchronization cycle.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("sync.started")

            # Sync all tables
            dataframes = {}
            total_rows = 0

            for table_name in self.config.tables_to_sync:
                rows = self.sync_table(table_name)
                total_rows += rows

                # For incremental sync, we'd merge with existing data
                # For simplicity, this example does full table loads
                df = self.sql_connector.get_changed_records(table_name, None)
                dataframes[table_name] = df

            # Build DuckDB
            db_path = self.build_duckdb(dataframes)

            # Upload to S3
            metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "total_rows": total_rows,
                "tables": {name: len(df) for name, df in dataframes.items()},
            }

            self.s3_uploader.upload_database(db_path, metadata)

            # Cleanup old versions (keep last 24)
            self.s3_uploader.cleanup_old_versions(keep_count=24)

            # Cleanup local file
            os.unlink(db_path)

            logger.info("sync.completed", total_rows=total_rows)
            return True

        except Exception as e:
            logger.error("sync.failed", error=str(e), exc_info=True)
            return False

    def close(self) -> None:
        """Cleanup resources."""
        self.sql_connector.close()
        logger.info("sync_manager.closed")
