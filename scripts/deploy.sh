#!/bin/bash

# Deploy script for Analytics Platform
# Usage: ./scripts/deploy.sh [environment]

set -e

ENVIRONMENT=${1:-production}
echo "Deploying to $ENVIRONMENT..."

# Verify environment files exist
if [ ! -f "etl-service/.env" ]; then
    echo "Error: etl-service/.env not found. Copy from .env.example and configure."
    exit 1
fi

if [ ! -f "api-service/.env" ]; then
    echo "Error: api-service/.env not found. Copy from .env.example and configure."
    exit 1
fi

# Build and deploy ETL service
echo "Building ETL service..."
cd etl-service
docker build -t etl-sync:latest .
cd ..

# Build and deploy API service
echo "Building API service..."
cd api-service
docker build -t analytics-api:latest .
cd ..

echo "Build complete!"
echo ""
echo "To deploy to Railway:"
echo "  1. Install Railway CLI: npm install -g @railway/cli"
echo "  2. Login: railway login"
echo "  3. Deploy: railway up"
echo ""
echo "To deploy locally with Docker Compose:"
echo "  docker-compose -f infrastructure/docker-compose.yml up -d"
