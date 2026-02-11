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
    rate_limit_window: int = 60  # seconds

    # CORS
    cors_origins: List[str] = ["*"]  # Restrict in production

    # DuckDB
    duckdb_cache_size_mb: int = 100
    duckdb_connection_pool_size: int = 10

    # Security
    allowed_query_patterns: List[str] = [
        "^SELECT\\s+.+\\s+FROM\\s+\\w+",  # Basic SELECT queries
        "^SELECT\\s+COUNT\\(\\*\\)",  # Count queries
    ]
    blocked_keywords: List[str] = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "EXEC",
        "EXECUTE",
        "UNION",
        "--",
        ";--",
    ]

    class Config:
        env_file = ".env"


settings = Settings()
