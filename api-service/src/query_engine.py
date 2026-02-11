"""DuckDB query execution with security and caching."""

import duckdb
import boto3
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import structlog
from functools import lru_cache
import tempfile
import os

from .config import settings

logger = structlog.get_logger()


@dataclass
class QueryResult:
    """Query execution result."""

    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time_ms: float
    cached: bool = False


class QueryValidator:
    """Validates SQL queries for security."""

    def __init__(self):
        self.blocked_patterns = [
            re.compile(rf"\b{keyword}\b", re.IGNORECASE)
            for keyword in settings.blocked_keywords
        ]
        self.allowed_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in settings.allowed_query_patterns
        ]

    def validate(self, query: str) -> bool:
        """
        Validate query is safe to execute.

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check for blocked keywords
        for pattern in self.blocked_patterns:
            if pattern.search(query):
                raise ValueError(f"Query contains blocked keyword")

        # Ensure query matches allowed patterns
        for pattern in self.allowed_patterns:
            if pattern.match(query.strip()):
                return True

        raise ValueError("Query does not match allowed patterns")


class DuckDBEngine:
    """Manages DuckDB connections and query execution."""

    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.s3_region,
        )
        self.validator = QueryValidator()
        self._db_path: Optional[str] = None
        self._last_load_time: Optional[datetime] = None
        self._db_lock = None

    def _download_database(self) -> str:
        """Download latest DuckDB from S3."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_file.close()

        try:
            self.s3_client.download_file(
                settings.s3_bucket, "current/analytics.db", temp_file.name
            )

            self._db_path = temp_file.name
            self._last_load_time = datetime.utcnow()

            logger.info("database.downloaded", path=temp_file.name)
            return temp_file.name

        except Exception as e:
            os.unlink(temp_file.name)
            raise

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection."""
        if not self._db_path or self._should_refresh():
            if self._db_path:
                try:
                    os.unlink(self._db_path)
                except:
                    pass
            self._download_database()

        return duckdb.connect(self._db_path, read_only=True)

    def _should_refresh(self) -> bool:
        """Check if database should be refreshed (older than 2 minutes)."""
        if not self._last_load_time:
            return True
        elapsed = (datetime.utcnow() - self._last_load_time).total_seconds()
        return elapsed > 120  # Refresh every 2 minutes

    def execute_query(self, query: str) -> QueryResult:
        """
        Execute validated query and return results.

        Args:
            query: SQL query string

        Returns:
            QueryResult with data and metadata
        """
        import time

        start_time = time.time()

        # Validate query
        self.validator.validate(query)

        # Execute query
        conn = self._get_connection()
        try:
            result = conn.execute(query).fetchdf()

            execution_time = (time.time() - start_time) * 1000

            return QueryResult(
                data=result.to_dict(orient="records"),
                columns=list(result.columns),
                row_count=len(result),
                execution_time_ms=execution_time,
            )

        finally:
            conn.close()

    def get_schema(self) -> Dict[str, List[str]]:
        """Get database schema information."""
        conn = self._get_connection()
        try:
            tables = conn.execute("SHOW TABLES").fetchall()
            schema = {}

            for (table_name,) in tables:
                columns = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                schema[table_name] = columns["column_name"].tolist()

            return schema

        finally:
            conn.close()

    def cleanup(self):
        """Cleanup temporary files."""
        if self._db_path and os.path.exists(self._db_path):
            try:
                os.unlink(self._db_path)
                logger.info("database.cleaned_up")
            except Exception as e:
                logger.error("cleanup.failed", error=str(e))


# Global engine instance
query_engine = DuckDBEngine()
