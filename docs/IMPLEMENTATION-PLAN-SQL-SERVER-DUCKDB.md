# Production-Grade SQL Server to DuckDB Analytics Platform

## Implementation Plan

This document provides a comprehensive implementation plan for building a production-grade analytics platform that syncs data from SQL Server to DuckDB, serves it via a secure API, and supports multiple authenticated applications.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               CLIENT LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   App 1      │  │   App 2      │  │   App 3      │  │   Dashboard  │    │
│  │  (React)     │  │  (Next.js)   │  │  (Vue)       │  │  (Any)       │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼────────────────┼────────────────┼────────────────┼──────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API GATEWAY LAYER                                 │
│                        FastAPI + Gunicorn + Uvicorn                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • Auth0 JWT Validation                                                │ │
│  │  • Rate Limiting (50 req/min per user)                                 │ │
│  │  • Query Whitelist Validation                                          │ │
│  │  • CORS Configuration                                                  │ │
│  │  • Request/Response Logging                                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER (DuckDB)                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • Load from S3 on first query                                         │ │
│  │  • In-memory caching (LRU: 100MB)                                      │ │
│  │  • Connection pooling                                                  │ │
│  │  • Read-only queries only                                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
          ▲
          │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STORAGE LAYER                                   │
│                              AWS S3 (Standard)                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  s3://analytics-bucket/                                                │ │
│  │    ├── current/                                                        │ │
│  │    │   ├── analytics.db (latest DuckDB file)                          │ │
│  │    │   └── manifest.json (metadata: timestamp, checksum)              │ │
│  │    ├── history/                                                        │ │
│  │    │   ├── analytics_2024-01-15_14-30-00.db                           │ │
│  │    │   └── analytics_2024-01-15_14-31-00.db                           │ │
│  │    └── logs/                                                           │ │
│  │        └── etl_sync_2024-01-15.log                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
          ▲
          │
┌─────────────────────────────────────────────────────────────────────────────┐
│                               ETL LAYER                                      │
│                          Python + APScheduler                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • Incremental sync (1-minute intervals)                               │ │
│  │  • Change detection via `updated_at` timestamps                        │ │
│  │  • Transactional consistency                                           │ │
│  │  • Error handling with exponential backoff                             │ │
│  │  • Dead letter queue for failed syncs                                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
          ▲
          │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SOURCE LAYER                                      │
│                           SQL Server 2019+                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Tables to sync:                                                       │ │
│  │    • inventory (main stock table)                                      │ │
│  │    • products                                                          │ │
│  │    • orders                                                            │ │
│  │    • transactions                                                      │ │
│  │    • customers                                                         │ │
│  │                                                                        │ │
│  │  Required columns:                                                     │ │
│  │    • updated_at (DATETIME2, indexed)                                   │ │
│  │    • created_at (DATETIME2)                                            │ │
│  │    • id (PRIMARY KEY)                                                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Accounts & Services

1. **Auth0 Account** (Free tier)
   - URL: https://auth0.com/
   - Features needed: Machine-to-Machine (M2M) applications, APIs

2. **Railway Account** (Free tier: $5 credit/month)
   - URL: https://railway.app/
   - Features needed: Services, PostgreSQL (for state tracking), Cron jobs

3. **AWS Account** (Free tier: 12 months)
   - URL: https://aws.amazon.com/free/
   - Services needed: S3, CloudWatch (optional)

4. **SQL Server Access**
   - Connection string with SELECT permissions
   - Tables must have `updated_at` columns for incremental sync

### Development Environment

```bash
# Required tools
python --version  # 3.11+
node --version    # 18+
git --version     # 2.30+
docker --version  # For local testing
```

---

## Step 1: Project Structure Setup

```
analytics-platform/
├── etl-service/                 # Data sync service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py             # Entry point
│   │   ├── sync.py             # Sync logic
│   │   ├── database.py         # SQL Server connector
│   │   ├── s3_uploader.py      # S3 operations
│   │   ├── models.py           # Data models
│   │   └── config.py           # Configuration
│   ├── tests/
│   └── scripts/
│       └── setup_database.sql  # Helper SQL scripts
│
├── api-service/                 # Query API service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app
│   │   ├── auth.py             # Auth0 integration
│   │   ├── query_engine.py     # DuckDB query handler
│   │   ├── rate_limiter.py     # Rate limiting
│   │   ├── models.py           # Pydantic models
│   │   └── config.py           # Configuration
│   ├── tests/
│   └── gunicorn.conf.py        # Production server config
│
├── shared/                      # Shared utilities
│   ├── schemas/
│   │   └── query_whitelist.json
│   └── utils/
│       └── validators.py
│
├── infrastructure/              # Infrastructure as code
│   ├── terraform/              # (Optional) AWS resources
│   ├── railway.yaml            # Railway configuration
│   └── docker-compose.yml      # Local development
│
├── docs/                        # Documentation
│   ├── api-docs.md
│   └── deployment-guide.md
│
├── scripts/                     # Utility scripts
│   ├── deploy.sh
│   └── test.sh
│
└── README.md
```

---

## Step 2: ETL Service Implementation

### 2.1 Setup & Dependencies

Create `etl-service/requirements.txt`:

```txt
# Core dependencies
pandas==2.1.4
pyarrow==14.0.1
duckdb==0.9.2
sqlalchemy==2.0.23
pyodbc==5.0.1

# AWS
boto3==1.34.0
botocore==1.34.0

# Scheduling
APScheduler==3.10.4

# Configuration & Logging
python-dotenv==1.0.0
structlog==23.2.0

# Database (for state tracking)
psycopg2-binary==2.9.9

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
```

### 2.2 Configuration (`etl-service/src/config.py`)

```python
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
            tables_to_sync=os.getenv("TABLES_TO_SYNC", "inventory,products,orders").split(","),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
```

### 2.3 Database Connection (`etl-service/src/database.py`)

```python
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
        connection_string = (
            f"mssql+pyodbc://{self.config.mssql_username}:{self.config.mssql_password}"
            f"@{self.config.mssql_host}:{self.config.mssql_port}"
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
        self, 
        table_name: str, 
        since: Optional[datetime] = None
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
            logger.info(
                "records.fetched",
                table=table_name,
                count=len(df),
                since=since
            )
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
```

### 2.4 S3 Operations (`etl-service/src/s3_uploader.py`)

```python
"""S3 operations for DuckDB file storage."""

import boto3
from botocore.exceptions import ClientError
import json
from datetime import datetime
from typing import Dict, Any
import structlog

from .config import Config

logger = structlog.get_logger()


class S3Uploader:
    """Manages DuckDB file uploads to S3."""
    
    def __init__(self, config: Config):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.s3_region
        )
    
    def upload_database(
        self, 
        local_path: str, 
        metadata: Dict[str, Any]
    ) -> str:
        """
        Upload DuckDB file to S3 with metadata.
        
        Args:
            local_path: Path to local DuckDB file
            metadata: Sync metadata (timestamp, row counts, etc.)
            
        Returns:
            S3 key of uploaded file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Upload to current/ (overwrites)
        current_key = "current/analytics.db"
        
        try:
            self.s3_client.upload_file(
                local_path,
                self.config.s3_bucket,
                current_key,
                ExtraArgs={
                    "Metadata": {
                        "sync-timestamp": metadata["timestamp"],
                        "row-count": str(metadata.get("total_rows", 0)),
                        "version": metadata.get("version", "1.0"),
                    }
                }
            )
            
            logger.info("database.uploaded", key=current_key, bucket=self.config.s3_bucket)
            
            # Also save to history/ for rollback capability
            history_key = f"history/analytics_{timestamp}.db"
            self.s3_client.copy_object(
                Bucket=self.config.s3_bucket,
                CopySource={
                    "Bucket": self.config.s3_bucket,
                    "Key": current_key
                },
                Key=history_key
            )
            
            logger.info("database.archived", key=history_key)
            
            # Update manifest
            manifest_key = "current/manifest.json"
            manifest = {
                "version": metadata.get("version", "1.0"),
                "timestamp": metadata["timestamp"],
                "database_key": current_key,
                "metadata": metadata,
                "tables": metadata.get("tables", {})
            }
            
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2),
                ContentType="application/json"
            )
            
            logger.info("manifest.updated", key=manifest_key)
            
            return current_key
            
        except ClientError as e:
            logger.error("upload.failed", error=str(e))
            raise
    
    def get_latest_manifest(self) -> Dict[str, Any]:
        """Retrieve the latest sync manifest from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket,
                Key="current/manifest.json"
            )
            return json.loads(response['Body'].read())
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return {}
            raise
    
    def cleanup_old_versions(self, keep_count: int = 24) -> None:
        """
        Remove old database versions from history/, keeping only the most recent.
        
        Args:
            keep_count: Number of historical versions to retain
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix="history/"
            )
            
            if 'Contents' not in response:
                return
            
            # Sort by last modified (newest first)
            objects = sorted(
                response['Contents'],
                key=lambda x: x['LastModified'],
                reverse=True
            )
            
            # Delete old versions
            to_delete = objects[keep_count:]
            for obj in to_delete:
                self.s3_client.delete_object(
                    Bucket=self.config.s3_bucket,
                    Key=obj['Key']
                )
                logger.info("old_version.deleted", key=obj['Key'])
                
        except ClientError as e:
            logger.error("cleanup.failed", error=str(e))
```

### 2.5 Sync Logic (`etl-service/src/sync.py`)

```python
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
        max_updated = df['updated_at'].max()
        self._state[table_name] = pd.to_datetime(max_updated)
        
        logger.info(
            "table.synced",
            table=table_name,
            rows=len(df),
            max_updated=max_updated
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
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.db', 
            delete=False
        )
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
                if 'id' in df.columns:
                    conn.execute(f"""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_id 
                        ON {table_name}(id)
                    """)
                
                if 'updated_at' in df.columns:
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
                "tables": {
                    name: len(df) for name, df in dataframes.items()
                }
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
```

### 2.6 Main Entry Point (`etl-service/src/main.py`)

```python
"""ETL Service entry point."""

import time
import signal
import sys
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import structlog

from .config import Config
from .sync import SyncManager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class ETLLauncher:
    """Manages the ETL service lifecycle."""
    
    def __init__(self):
        self.config = Config.from_env()
        self.sync_manager: SyncManager = None
        self.scheduler: BackgroundScheduler = None
        self._shutdown = False
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            logger.info("shutdown.signal_received", signal=signum)
            self._shutdown = True
            if self.scheduler:
                self.scheduler.shutdown(wait=True)
            if self.sync_manager:
                self.sync_manager.close()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def run_sync_job(self):
        """Wrapper for sync job with error handling."""
        try:
            success = self.sync_manager.run_sync()
            if not success:
                logger.warning("sync_job.failed")
        except Exception as e:
            logger.error("sync_job.exception", error=str(e), exc_info=True)
    
    def start(self):
        """Start the ETL service."""
        logger.info(
            "etl.starting",
            interval_seconds=self.config.sync_interval_seconds,
            tables=self.config.tables_to_sync
        )
        
        # Initialize sync manager
        self.sync_manager = SyncManager(self.config)
        self.sync_manager.initialize()
        
        # Setup scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.run_sync_job,
            trigger=IntervalTrigger(seconds=self.config.sync_interval_seconds),
            id='sync_job',
            replace_existing=True,
            max_instances=1,  # Prevent overlapping runs
            coalesce=True     # Skip missed runs if behind
        )
        
        # Run initial sync immediately
        self.run_sync_job()
        
        # Start scheduler
        self.scheduler.start()
        
        logger.info("etl.started")
        
        # Keep main thread alive
        try:
            while not self._shutdown:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.scheduler.shutdown()
            self.sync_manager.close()


def main():
    """Entry point."""
    launcher = ETLLauncher()
    launcher.setup_signal_handlers()
    launcher.start()


if __name__ == "__main__":
    main()
```

### 2.7 Dockerfile (`etl-service/Dockerfile`)

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    libodbc1 \
    odbcinst \
    odbcinst1debian2 \
    libpq-dev \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver 18 for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create non-root user
RUN useradd -m -u 1000 etluser && chown -R etluser:etluser /app
USER etluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the service
CMD ["python", "-m", "src.main"]
```

---

## Step 3: API Service Implementation

### 3.1 Setup & Dependencies

Create `api-service/requirements.txt`:

```txt
# Web framework
fastapi==0.105.0
uvicorn[standard]==0.24.0
gunicorn==21.2.0

# Authentication
pyjwt==2.8.0
cryptography==41.0.8
python-jose[cryptography]==3.3.0

# Database
duckdb==0.9.2
boto3==1.34.0

# Rate limiting
slowapi==0.1.9

# Configuration
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
```

### 3.2 Configuration (`api-service/src/config.py`)

```python
"""API service configuration."""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings."""
    
    # Auth0 Configuration
    auth0_domain: str
    auth0_api_audience: str
    auth0_algorithms: List[str] = ["RS256"]
    
    # AWS S3
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_bucket: str
    s3_region: str = "us-east-1"
    
    # Rate Limiting
    rate_limit_requests: int = 50  # requests per minute
    rate_limit_window: int = 60    # seconds
    
    # CORS
    cors_origins: List[str] = ["*"]  # Restrict in production
    
    # DuckDB
    duckdb_cache_size_mb: int = 100
    duckdb_connection_pool_size: int = 10
    
    # Security
    allowed_query_patterns: List[str] = [
        "^SELECT\\s+.+\\s+FROM\\s+\\w+",  # Basic SELECT queries
        "^SELECT\\s+COUNT\\(\\*\\)",        # Count queries
    ]
    blocked_keywords: List[str] = [
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
        "ALTER", "EXEC", "EXECUTE", "UNION", "--", ";--"
    ]
    
    class Config:
        env_file = ".env"


settings = Settings()
```

### 3.3 Authentication (`api-service/src/auth.py`)

```python
"""Auth0 JWT authentication."""

import jwt
from jwt.exceptions import InvalidTokenError
import requests
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
import structlog

from .config import settings

logger = structlog.get_logger()
security = HTTPBearer()


class Auth0Validator:
    """Validates Auth0 JWT tokens."""
    
    def __init__(self):
        self.domain = settings.auth0_domain
        self.audience = settings.auth0_api_audience
        self.algorithms = settings.auth0_algorithms
        self._jwks = None
        self._jwks_url = f"https://{self.domain}/.well-known/jwks.json"
    
    def _get_jwks(self) -> Dict:
        """Fetch and cache JWKS from Auth0."""
        if not self._jwks:
            response = requests.get(self._jwks_url)
            response.raise_for_status()
            self._jwks = response.json()
        return self._jwks
    
    def _get_signing_key(self, token: str) -> str:
        """Extract signing key from token header."""
        try:
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header["kid"]
            
            jwks = self._get_jwks()
            for key in jwks["keys"]:
                if key["kid"] == kid:
                    return jwt.algorithms.RSAAlgorithm.from_jwk(key)
            
            raise HTTPException(status_code=401, detail="Unable to find signing key")
            
        except Exception as e:
            logger.error("auth.signing_key_error", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid token header")
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token and return payload.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
        """
        try:
            signing_key = self._get_signing_key(token)
            
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=f"https://{self.domain}/"
            )
            
            logger.info("auth.token_validated", user_id=payload.get("sub"))
            return payload
            
        except InvalidTokenError as e:
            logger.warning("auth.token_invalid", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error("auth.validation_error", error=str(e))
            raise HTTPException(status_code=401, detail="Token validation failed")


# Global validator instance
auth_validator = Auth0Validator()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    Dependency to validate JWT and return user info.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user_id": user["sub"]}
    """
    token = credentials.credentials
    return auth_validator.validate_token(token)


async def require_scope(required_scope: str):
    """Factory for scope-based authorization."""
    async def scope_checker(user: Dict = Depends(get_current_user)) -> Dict:
        scopes = user.get("scope", "").split()
        if required_scope not in scopes:
            logger.warning(
                "auth.scope_denied",
                user_id=user.get("sub"),
                required=required_scope,
                available=scopes
            )
            raise HTTPException(
                status_code=403, 
                detail=f"Missing required scope: {required_scope}"
            )
        return user
    return scope_checker
```

### 3.4 Query Engine (`api-service/src/query_engine.py`)

```python
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
            re.compile(rf'\b{keyword}\b', re.IGNORECASE)
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
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.s3_region
        )
        self.validator = QueryValidator()
        self._db_path: Optional[str] = None
        self._last_load_time: Optional[datetime] = None
        self._db_lock = None
    
    def _download_database(self) -> str:
        """Download latest DuckDB from S3."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()
        
        try:
            self.s3_client.download_file(
                settings.s3_bucket,
                'current/analytics.db',
                temp_file.name
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
                data=result.to_dict(orient='records'),
                columns=list(result.columns),
                row_count=len(result),
                execution_time_ms=execution_time
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
                schema[table_name] = columns['column_name'].tolist()
            
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
```

### 3.5 Rate Limiting (`api-service/src/rate_limiter.py`)

```python
"""Rate limiting implementation."""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
import redis
import os

from .config import settings


# Use in-memory storage for Railway free tier
# For production, switch to Redis
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_requests} per minute"]
)


def setup_rate_limiting(app):
    """Configure rate limiting for FastAPI app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

### 3.6 Main Application (`api-service/src/main.py`)

```python
"""FastAPI application entry point."""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.util import get_remote_address
from slowapi import Limiter
import structlog
import time

from .config import settings
from .auth import get_current_user, require_scope
from .query_engine import query_engine, QueryResult
from .rate_limiter import setup_rate_limiting

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Create app
app = FastAPI(
    title="Analytics API",
    description="Secure API for querying DuckDB analytics database",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Setup rate limiting
setup_rate_limiting(app)


@app.middleware("http")
async def logging_middleware(request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    logger.info(
        "request.completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
        client_ip=get_remote_address(request)
    )
    
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/schema")
async def get_schema(user: dict = Depends(get_current_user)):
    """
    Get database schema.
    
    Returns:
        Dictionary of table names and their columns
    """
    try:
        schema = query_engine.get_schema()
        return {"schema": schema}
    except Exception as e:
        logger.error("schema.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query")
async def execute_query(
    q: str = Query(..., description="SQL query to execute"),
    user: dict = Depends(get_current_user)
):
    """
    Execute a read-only SQL query.
    
    Args:
        q: SQL query (SELECT only)
        
    Returns:
        Query results with metadata
    """
    try:
        result: QueryResult = query_engine.execute_query(q)
        
        return {
            "success": True,
            "data": result.data,
            "columns": result.columns,
            "row_count": result.row_count,
            "execution_time_ms": result.execution_time_ms
        }
        
    except ValueError as e:
        logger.warning("query.validation_failed", error=str(e), user_id=user.get("sub"))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("query.execution_failed", error=str(e), user_id=user.get("sub"))
        raise HTTPException(status_code=500, detail="Query execution failed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    query_engine.cleanup()
    logger.info("application.shutdown")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3.7 Gunicorn Configuration (`api-service/gunicorn.conf.py`)

```python
"""Gunicorn configuration for production."""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")

# Process naming
proc_name = "analytics-api"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"

# SSL (handled by Railway/proxy)
 forwarded_allow_ips = "*"
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'ssl',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}
```

### 3.8 Dockerfile (`api-service/Dockerfile`)

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY gunicorn.conf.py .

# Create non-root user
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with Gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "src.main:app"]
```

---

## Step 4: Auth0 Setup

### 4.1 Create Auth0 Account & Application

1. **Sign up**: https://auth0.com/signup
2. **Create Application**:
   - Type: Single Page Application (for frontend apps)
   - Name: "Analytics Dashboard"
   - Allowed Callback URLs: `http://localhost:5173/callback, https://your-production-url/callback`
   - Allowed Logout URLs: `http://localhost:5173, https://your-production-url`
   - Allowed Web Origins: `http://localhost:5173, https://your-production-url`

3. **Create API**:
   - Name: "Analytics API"
   - Identifier: `https://analytics-api.yourdomain.com` (this is your `audience`)
   - Signing Algorithm: RS256
   - Permissions (scopes):
     - `read:analytics` - Read analytics data
     - `read:schema` - Read database schema

### 4.2 Configure Environment Variables

```bash
# Auth0
AUTH0_DOMAIN=your-domain.auth0.com
AUTH0_API_AUDIENCE=https://analytics-api.yourdomain.com

# For frontend apps
AUTH0_CLIENT_ID=your-spa-client-id
AUTH0_REDIRECT_URI=http://localhost:5173/callback
```

### 4.3 Frontend Integration Example

```typescript
// auth.ts - Auth0 integration for React
import { Auth0Provider, useAuth0 } from '@auth0/auth0-react';

export const authConfig = {
  domain: import.meta.env.VITE_AUTH0_DOMAIN,
  clientId: import.meta.env.VITE_AUTH0_CLIENT_ID,
  authorizationParams: {
    redirect_uri: window.location.origin + '/callback',
    audience: import.meta.env.VITE_AUTH0_AUDIENCE,
    scope: 'read:analytics read:schema'
  }
};

// Hook for API calls with authentication
export function useAnalyticsApi() {
  const { getAccessTokenSilently } = useAuth0();
  
  const fetchQuery = async (query: string) => {
    const token = await getAccessTokenSilently();
    
    const response = await fetch(
      `${import.meta.env.VITE_API_URL}/query?q=${encodeURIComponent(query)}`,
      {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      }
    );
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return response.json();
  };
  
  return { fetchQuery };
}
```

---

## Step 5: Railway Deployment

### 5.1 Create Railway Account

1. Sign up at https://railway.app/
2. Connect GitHub repository
3. Create new project

### 5.2 Setup Services

#### ETL Service

```yaml
# railway.yaml (in etl-service directory)
services:
  - name: etl-sync
    build:
      dockerfile: Dockerfile
    deploy:
      restartPolicy:
        maxRetries: 3
    env:
      - name: MSSQL_HOST
        value: ${MSSQL_HOST}
      - name: MSSQL_PORT
        value: "1433"
      - name: MSSQL_DATABASE
        value: ${MSSQL_DATABASE}
      - name: MSSQL_USERNAME
        value: ${MSSQL_USERNAME}
      - name: MSSQL_PASSWORD
        value: ${MSSQL_PASSWORD}
      - name: AWS_ACCESS_KEY_ID
        value: ${AWS_ACCESS_KEY_ID}
      - name: AWS_SECRET_ACCESS_KEY
        value: ${AWS_SECRET_ACCESS_KEY}
      - name: S3_BUCKET
        value: ${S3_BUCKET}
      - name: DATABASE_URL
        fromDatabase:
          name: etl-state
          property: connectionString
```

#### API Service

```yaml
# railway.yaml (in api-service directory)
services:
  - name: analytics-api
    build:
      dockerfile: Dockerfile
    deploy:
      healthcheck:
        path: /health
        port: 8000
    env:
      - name: AUTH0_DOMAIN
        value: ${AUTH0_DOMAIN}
      - name: AUTH0_API_AUDIENCE
        value: ${AUTH0_API_AUDIENCE}
      - name: AWS_ACCESS_KEY_ID
        value: ${AWS_ACCESS_KEY_ID}
      - name: AWS_SECRET_ACCESS_KEY
        value: ${AWS_SECRET_ACCESS_KEY}
      - name: S3_BUCKET
        value: ${S3_BUCKET}
```

### 5.3 Environment Variables

Set these in Railway dashboard for each service:

**ETL Service:**
- `MSSQL_HOST` - SQL Server hostname
- `MSSQL_DATABASE` - Database name
- `MSSQL_USERNAME` - SQL Server username
- `MSSQL_PASSWORD` - SQL Server password
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `S3_BUCKET` - S3 bucket name
- `SYNC_INTERVAL_SECONDS` - 60

**API Service:**
- `AUTH0_DOMAIN` - your-domain.auth0.com
- `AUTH0_API_AUDIENCE` - https://analytics-api.yourdomain.com
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `S3_BUCKET` - S3 bucket name

### 5.4 Deploy

```bash
# Deploy via Railway CLI (optional)
npm install -g @railway/cli
railway login
railway link
railway up
```

Or deploy via GitHub integration (recommended):
1. Push code to GitHub
2. Railway auto-deploys on push to main branch

---

## Step 6: Security Hardening

### 6.1 SQL Injection Prevention

The API already implements:
- ✅ Whitelist-based query validation
- ✅ Blocked dangerous keywords (INSERT, UPDATE, DELETE, etc.)
- ✅ Pattern matching for SELECT queries only
- ✅ Read-only DuckDB connections

Additional measures:
```python
# Add to query_engine.py

class QuerySanitizer:
    """Additional query sanitization."""
    
    @staticmethod
    def sanitize(query: str) -> str:
        """Remove potentially dangerous patterns."""
        # Remove comments
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        query = re.sub(r'--.*?$', '', query, flags=re.MULTILINE)
        
        # Remove multiple statements
        if ';' in query.rstrip('; '):
            raise ValueError("Multiple statements not allowed")
        
        return query.strip()
```

### 6.2 Network Security

1. **SQL Server**: Use VPC peering or IP whitelisting
2. **Railway**: Enable private networking between services
3. **S3**: Use bucket policies to restrict access

```json
// S3 Bucket Policy
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::YOUR_ACCOUNT:root"},
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::your-bucket/*"
    }
  ]
}
```

### 6.3 Auth0 Security

1. Enable MFA for admin users
2. Set token expiration to 1 hour
3. Enable "Refresh Token Rotation"
4. Configure "Attack Protection" (Brute Force, Breached Password)

---

## Step 7: Monitoring & Observability

### 7.1 Logging

Both services use structured JSON logging. View logs in Railway dashboard:

```bash
# Filter for errors
railway logs --service etl-sync | jq 'select(.level=="error")'

# Monitor sync performance
railway logs --service etl-sync | jq 'select(.event=="sync.completed")'
```

### 7.2 Metrics (Optional)

Add Prometheus metrics to API:

```python
# Add to api-service/src/main.py
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('api_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 7.3 Alerting

Setup alerts in Railway or use external service (PagerDuty, Opsgenie):

```python
# Add to critical error handlers
import requests

def send_alert(message: str, severity: str = "warning"):
    """Send alert to PagerDuty/Opsgenie."""
    if os.getenv("PAGERDUTY_KEY"):
        requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json={
                "routing_key": os.getenv("PAGERDUTY_KEY"),
                "event_action": "trigger",
                "payload": {
                    "summary": message,
                    "severity": severity,
                    "source": "analytics-platform"
                }
            }
        )
```

---

## Step 8: Testing Strategy

### 8.1 Unit Tests (ETL Service)

```python
# etl-service/tests/test_sync.py
import pytest
from datetime import datetime
from src.sync import SyncManager
from src.config import Config

@pytest.fixture
def config():
    return Config(
        mssql_host="localhost",
        mssql_port=1433,
        mssql_database="test",
        mssql_username="test",
        mssql_password="test",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        s3_bucket="test-bucket",
        postgres_url="postgresql://localhost/test"
    )

def test_sync_manager_initialization(config):
    manager = SyncManager(config)
    assert manager.config == config
    assert manager._state == {}

def test_query_validation():
    from src.query_engine import QueryValidator
    
    validator = QueryValidator()
    
    # Valid queries
    assert validator.validate("SELECT * FROM inventory")
    assert validator.validate("SELECT COUNT(*) FROM products")
    
    # Invalid queries
    with pytest.raises(ValueError):
        validator.validate("INSERT INTO table VALUES (1)")
    
    with pytest.raises(ValueError):
        validator.validate("DROP TABLE inventory")
```

### 8.2 Integration Tests

```python
# tests/test_integration.py
import requests
import pytest

API_URL = "http://localhost:8000"

def test_health_endpoint():
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_without_auth():
    response = requests.get(f"{API_URL}/query?q=SELECT * FROM inventory")
    assert response.status_code == 403  # Forbidden without token
```

### 8.3 Load Testing

Use Locust for load testing:

```python
# tests/locustfile.py
from locust import HttpUser, task, between

class AnalyticsUser(HttpUser):
    wait_time = between(1, 5)
    
    def on_start(self):
        # Get Auth0 token (implement token acquisition)
        self.token = get_auth0_token()
    
    @task(3)
    def query_inventory(self):
        self.client.get(
            "/query?q=SELECT * FROM inventory LIMIT 100",
            headers={"Authorization": f"Bearer {self.token}"}
        )
    
    @task(1)
    def get_schema(self):
        self.client.get(
            "/schema",
            headers={"Authorization": f"Bearer {self.token}"}
        )
```

---

## Step 9: Cost Optimization

### 9.1 Railway Free Tier Strategy

**Current Limits:**
- $5 credit/month
- 512 MB RAM per service
- 1 GB disk per service
- Sleep after inactivity (except always-on services)

**Optimizations:**

1. **Combine ETL + API** into single service if possible
2. **Use GitHub Actions** for ETL (2000 min/month free)
3. **S3 Intelligent-Tiering** for database files

### 9.2 AWS S3 Costs

**Estimated Monthly Costs:**
- Storage: $0.023/GB (minimal for analytics DB)
- Requests: $0.005 per 1,000 GET requests
- **Total**: ~$1-5/month for moderate usage

**Optimizations:**
- Lifecycle policies to archive old versions to Glacier
- Use S3 Transfer Acceleration only if needed

---

## Step 10: Operations Runbook

### 10.1 Daily Operations

```bash
# Check sync status
curl https://your-api.railway.app/health

# View recent logs
railway logs --service etl-sync --tail

# Check S3 for latest database
aws s3 ls s3://your-bucket/current/
```

### 10.2 Troubleshooting

**Issue: ETL sync failing**
```bash
# Check SQL Server connectivity
railway logs --service etl-sync | grep "sql_server"

# Verify credentials
echo $MSSQL_HOST  # Should be set

# Manual sync test
docker run --env-file .env etl-sync python -c "from src.sync import SyncManager; ..."
```

**Issue: API returning 401**
```bash
# Check Auth0 configuration
curl https://your-domain.auth0.com/.well-known/jwks.json

# Verify token validity
jwt decode --json $TOKEN
```

**Issue: Query timeout**
```bash
# Check database size
aws s3 ls s3://your-bucket/current/analytics.db

# Review slow queries in logs
railway logs --service analytics-api | grep "execution_time_ms"
```

### 10.3 Disaster Recovery

1. **Database Corruption**:
   ```bash
   # Restore from S3 history
   aws s3 cp s3://bucket/history/analytics_2024-01-15_14-30-00.db s3://bucket/current/analytics.db
   ```

2. **Auth0 Outage**:
   - Implement fallback to API keys
   - Cache JWKS locally with 24hr TTL

3. **Railway Service Down**:
   - Deploy to Fly.io as backup
   - Use Cloudflare Load Balancer

---

## Quick Start Checklist

- [ ] Create Auth0 account and configure application
- [ ] Setup AWS S3 bucket with proper permissions
- [ ] Clone repository and configure environment variables
- [ ] Run ETL service locally to test SQL Server connection
- [ ] Run API service locally and test with curl/Postman
- [ ] Deploy ETL service to Railway
- [ ] Deploy API service to Railway
- [ ] Configure Auth0 allowed origins with Railway URLs
- [ ] Build frontend app and integrate Auth0 SDK
- [ ] Test end-to-end flow
- [ ] Setup monitoring and alerting
- [ ] Document API for frontend teams

---

## Next Steps

1. **Implement caching layer** (Redis) for frequent queries
2. **Add GraphQL endpoint** for flexible querying
3. **Build admin dashboard** for monitoring sync status
4. **Implement data lineage** tracking
5. **Add support for real-time streaming** (WebSocket subscriptions)
6. **Create SDKs** for popular languages (Python, JavaScript, Go)

---

## Support & Resources

- **Auth0 Docs**: https://auth0.com/docs
- **Railway Docs**: https://docs.railway.app/
- **DuckDB Docs**: https://duckdb.org/docs/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Issue Tracking**: GitHub Issues
- **Slack Channel**: #analytics-platform-support
