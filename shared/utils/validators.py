"""Shared validation utilities."""

import re
import json
from typing import Dict, List, Optional
from pathlib import Path


def load_query_whitelist(schema_path: Optional[str] = None) -> Dict:
    """Load query whitelist schema."""
    if schema_path is None:
        schema_path = Path(__file__).parent.parent / "schemas" / "query_whitelist.json"

    with open(schema_path, "r") as f:
        return json.load(f)


class QueryValidator:
    """Validates SQL queries against whitelist."""

    def __init__(self, schema_path: Optional[str] = None):
        self.whitelist = load_query_whitelist(schema_path)
        self.blocked_patterns = [
            re.compile(rf"\b{keyword}\b", re.IGNORECASE)
            for keyword in self.whitelist.get("blocked_patterns", [])
        ]
        self.allowed_patterns = [
            (query["name"], re.compile(query["pattern"], re.IGNORECASE))
            for query in self.whitelist.get("allowed_queries", [])
        ]
        self.max_length = self.whitelist.get("max_query_length", 5000)
        self.max_execution_time = self.whitelist.get("max_execution_time_seconds", 30)

    def validate(self, query: str) -> Dict:
        """
        Validate a SQL query.

        Returns:
            Dict with 'valid' (bool) and 'error' (str) keys
        """
        # Check length
        if len(query) > self.max_length:
            return {
                "valid": False,
                "error": f"Query exceeds maximum length of {self.max_length} characters",
            }

        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(query):
                return {
                    "valid": False,
                    "error": "Query contains blocked keywords or patterns",
                }

        # Check against allowed patterns
        for name, pattern in self.allowed_patterns:
            if pattern.match(query.strip()):
                return {"valid": True, "pattern": name}

        return {"valid": False, "error": "Query does not match any allowed pattern"}


def sanitize_query(query: str) -> str:
    """Sanitize query by removing comments and extra whitespace."""
    # Remove SQL comments
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
    query = re.sub(r"--.*?$", "", query, flags=re.MULTILINE)

    # Remove extra whitespace
    query = " ".join(query.split())

    return query.strip()
