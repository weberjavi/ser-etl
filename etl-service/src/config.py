"""Configuration management for ETL service."""

import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Config:
    """Application configuration."""

    # SQL Server
    mssql_host: str
    mssql_port: int
    mssql_database: str
    mssql_username: str
    mssql_password: str

    # AWS S3
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_bucket: str
    s3_region: str = "us-east-1"

    # PostgreSQL (state tracking)
    postgres_url: str

    # Sync settings
    sync_interval_seconds: int = 60
    batch_size: int = 10000
    tables_to_sync: List[str] = None

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            mssql_host=os.getenv("MSSQL_HOST"),
            mssql_port=int(os.getenv("MSSQL_PORT", "1433")),
            mssql_database=os.getenv("MSSQL_DATABASE"),
            mssql_username=os.getenv("MSSQL_USERNAME"),
            mssql_password=os.getenv("MSSQL_PASSWORD"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            s3_bucket=os.getenv("S3_BUCKET"),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            postgres_url=os.getenv("DATABASE_URL"),  # Railway provides this
            sync_interval_seconds=int(os.getenv("SYNC_INTERVAL_SECONDS", "60")),
            batch_size=int(os.getenv("BATCH_SIZE", "10000")),
            tables_to_sync=os.getenv(
                "TABLES_TO_SYNC", "inventory,products,orders"
            ).split(","),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
