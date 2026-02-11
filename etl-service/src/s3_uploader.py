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
            "s3",
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.s3_region,
        )

    def upload_database(self, local_path: str, metadata: Dict[str, Any]) -> str:
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
                },
            )

            logger.info(
                "database.uploaded", key=current_key, bucket=self.config.s3_bucket
            )

            # Also save to history/ for rollback capability
            history_key = f"history/analytics_{timestamp}.db"
            self.s3_client.copy_object(
                Bucket=self.config.s3_bucket,
                CopySource={"Bucket": self.config.s3_bucket, "Key": current_key},
                Key=history_key,
            )

            logger.info("database.archived", key=history_key)

            # Update manifest
            manifest_key = "current/manifest.json"
            manifest = {
                "version": metadata.get("version", "1.0"),
                "timestamp": metadata["timestamp"],
                "database_key": current_key,
                "metadata": metadata,
                "tables": metadata.get("tables", {}),
            }

            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2),
                ContentType="application/json",
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
                Bucket=self.config.s3_bucket, Key="current/manifest.json"
            )
            return json.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
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
                Bucket=self.config.s3_bucket, Prefix="history/"
            )

            if "Contents" not in response:
                return

            # Sort by last modified (newest first)
            objects = sorted(
                response["Contents"], key=lambda x: x["LastModified"], reverse=True
            )

            # Delete old versions
            to_delete = objects[keep_count:]
            for obj in to_delete:
                self.s3_client.delete_object(
                    Bucket=self.config.s3_bucket, Key=obj["Key"]
                )
                logger.info("old_version.deleted", key=obj["Key"])

        except ClientError as e:
            logger.error("cleanup.failed", error=str(e))
