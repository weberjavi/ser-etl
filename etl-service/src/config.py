"""Configuration management for ETL service."""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Config:
    """Application configuration."""

    # Required fields (no defaults) - must come first
    # SQL Server
    mssql_database: str
    mssql_username: str
    mssql_password: str

    # AWS S3
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_bucket: str

    # PostgreSQL (state tracking)
    postgres_url: str

    # Optional fields (with defaults) - must come last
    # SQL Server connection - use either:
    # 1. MSSQL_SERVER="10.0.2.100,1433" (preferred for IP addresses)
    # 2. MSSQL_HOST + MSSQL_PORT separately
    mssql_server: str = ""  # Full server string like "10.0.2.100,1433"
    mssql_host: str = ""
    mssql_port: int = 1433

    s3_region: str = "us-east-1"
    sync_interval_seconds: int = 60
    batch_size: int = 10000
    tables_to_sync: List[str] = field(
        default_factory=lambda: ["inventory", "products", "orders"]
    )
    log_level: str = "INFO"

    def get_server_string(self) -> str:
        """Get SQL Server connection string (host,port format)."""
        if self.mssql_server:
            return self.mssql_server
        return f"{self.mssql_host},{self.mssql_port}"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            mssql_server=os.getenv("MSSQL_SERVER", ""),
            mssql_host=os.getenv("MSSQL_HOST", ""),
            mssql_port=int(os.getenv("MSSQL_PORT", "1433")),
            mssql_database=os.getenv("MSSQL_DATABASE"),
            mssql_username=os.getenv("MSSQL_USERNAME"),
            mssql_password=os.getenv("MSSQL_PASSWORD"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            s3_bucket=os.getenv("S3_BUCKET"),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            postgres_url=os.getenv("DATABASE_URL"),
            sync_interval_seconds=int(os.getenv("SYNC_INTERVAL_SECONDS", "60")),
            batch_size=int(os.getenv("BATCH_SIZE", "10000")),
            tables_to_sync=os.getenv(
                "TABLES_TO_SYNC", "inventory,products,orders"
            ).split(","),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
