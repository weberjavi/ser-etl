#!/bin/bash

# Test script for Analytics Platform
# Usage: ./scripts/test.sh

set -e

echo "Running tests..."

# Test ETL service
echo "Testing ETL service..."
cd etl-service
python -m pytest tests/ -v --tb=short || echo "ETL tests failed"
cd ..

# Test API service
echo "Testing API service..."
cd api-service
python -m pytest tests/ -v --tb=short || echo "API tests failed"
cd ..

# Test shared utilities
echo "Testing shared utilities..."
cd shared
python -m pytest tests/ -v --tb=short || echo "Shared tests failed"
cd ..

echo "All tests completed!"
