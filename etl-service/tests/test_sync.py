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
        postgres_url="postgresql://localhost/test",
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
