# Production-Grade SQL Server to DuckDB Analytics Platform

A secure, scalable analytics platform that syncs data from SQL Server to DuckDB and serves it via an authenticated API.

## Architecture

```
┌─────────────────┐      ┌──────────────┐      ┌─────────────────┐
│   SQL Server    │─────▶│  ETL Service │─────▶│       S3        │
│   (Source DB)   │      │  (Sync Data) │      │  (DuckDB Files) │
└─────────────────┘      └──────────────┘      └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Service (FastAPI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Auth0 JWT    │  │ Rate Limit   │  │ DuckDB Query Engine  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Client Applications                           │
│         (React, Next.js, Vue, Dashboards, etc.)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- AWS Account (S3 access)
- Auth0 Account
- SQL Server access

### 1. Clone & Setup

```bash
git clone <repository>
cd analytics-platform

# Setup environment files
cp etl-service/.env.example etl-service/.env
cp api-service/.env.example api-service/.env

# Edit both .env files with your configuration
```

### 2. Local Development with Docker

```bash
# Start all services
docker-compose -f infrastructure/docker-compose.yml up -d

# View logs
docker-compose -f infrastructure/docker-compose.yml logs -f

# Stop services
docker-compose -f infrastructure/docker-compose.yml down
```

### 3. Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and link project
railway login
railway link

# Deploy
railway up
```

## Project Structure

```
analytics-platform/
├── etl-service/              # Data synchronization service
│   ├── src/
│   │   ├── config.py        # Configuration management
│   │   ├── database.py      # SQL Server connector
│   │   ├── s3_uploader.py   # S3 operations
│   │   ├── sync.py          # Sync logic
│   │   └── main.py          # Entry point
│   ├── Dockerfile
│   └── requirements.txt
│
├── api-service/              # Query API service
│   ├── src/
│   │   ├── config.py        # Settings
│   │   ├── auth.py          # Auth0 integration
│   │   ├── query_engine.py  # DuckDB handler
│   │   ├── rate_limiter.py  # Rate limiting
│   │   └── main.py          # FastAPI app
│   ├── Dockerfile
│   ├── gunicorn.conf.py
│   └── requirements.txt
│
├── shared/                   # Shared utilities
│   ├── schemas/
│   │   └── query_whitelist.json
│   └── utils/
│       └── validators.py
│
├── infrastructure/           # Deployment configs
│   ├── docker-compose.yml
│   ├── railway-etl.yaml
│   └── railway-api.yaml
│
└── scripts/                  # Utility scripts
    ├── deploy.sh
    └── test.sh
```

## Configuration

### ETL Service Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MSSQL_HOST` | SQL Server hostname | Yes |
| `MSSQL_PORT` | SQL Server port (default: 1433) | No |
| `MSSQL_DATABASE` | Database name | Yes |
| `MSSQL_USERNAME` | Username | Yes |
| `MSSQL_PASSWORD` | Password | Yes |
| `AWS_ACCESS_KEY_ID` | AWS Access Key | Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key | Yes |
| `S3_BUCKET` | S3 bucket name | Yes |
| `S3_REGION` | AWS region (default: us-east-1) | No |
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `SYNC_INTERVAL_SECONDS` | Sync frequency (default: 60) | No |
| `TABLES_TO_SYNC` | Comma-separated table names | No |

### API Service Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AUTH0_DOMAIN` | Auth0 domain (e.g., yourapp.auth0.com) | Yes |
| `AUTH0_API_AUDIENCE` | API identifier | Yes |
| `AWS_ACCESS_KEY_ID` | AWS Access Key | Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key | Yes |
| `S3_BUCKET` | S3 bucket name | Yes |
| `S3_REGION` | AWS region (default: us-east-1) | No |
| `RATE_LIMIT_REQUESTS` | Max requests per minute (default: 50) | No |
| `CORS_ORIGINS` | Allowed CORS origins | No |

## API Endpoints

### Health Check
```bash
GET /health
```

### Get Database Schema
```bash
GET /schema
Authorization: Bearer <token>
```

### Execute Query
```bash
GET /query?q=SELECT%20*%20FROM%20inventory%20LIMIT%20100
Authorization: Bearer <token>
```

### Response Format
```json
{
  "success": true,
  "data": [...],
  "columns": ["id", "name", "quantity", ...],
  "row_count": 100,
  "execution_time_ms": 45.2
}
```

## Security Features

- **Auth0 JWT Authentication**: All API endpoints require valid JWT tokens
- **Rate Limiting**: 50 requests per minute per IP
- **Query Whitelisting**: Only SELECT queries allowed
- **SQL Injection Prevention**: Blocked dangerous keywords (INSERT, UPDATE, DELETE, DROP, etc.)
- **Read-Only DuckDB**: Database connections are read-only
- **CORS Protection**: Configurable allowed origins

## Sync Process

1. **ETL Service** connects to SQL Server every 60 seconds
2. Fetches changed records based on `updated_at` timestamps
3. Builds optimized DuckDB database with indexes
4. Uploads to S3 (current/ + history/)
5. Updates manifest.json with metadata
6. Cleans up old versions (keeps last 24)

## Query Validation

Allowed query patterns:
- `SELECT * FROM table`
- `SELECT columns FROM table`
- `SELECT COUNT(*) FROM table`
- Aggregate functions (SUM, AVG, MIN, MAX)
- GROUP BY queries
- WHERE clauses
- ORDER BY clauses
- LIMIT clauses

Blocked patterns:
- INSERT, UPDATE, DELETE
- DROP, CREATE, ALTER
- UNION, EXEC, EXECUTE
- Comments (--)
- Multiple statements (;)

## Monitoring & Logs

Both services use structured JSON logging:

```bash
# View ETL logs
railway logs --service etl-sync

# View API logs  
railway logs --service analytics-api

# Filter for errors
railway logs --service etl-sync | jq 'select(.level=="error")'
```

## Cost Optimization

**Railway Free Tier ($5/month credit):**
- 512 MB RAM per service
- 1 GB disk per service
- Services sleep after inactivity

**AWS S3:**
- ~$1-5/month for moderate usage
- Use S3 Intelligent-Tiering for cost savings

## Troubleshooting

**ETL sync failing:**
```bash
# Check SQL Server connectivity
railway logs --service etl-sync | grep "sql_server"

# Verify credentials
railway variables --service etl-sync
```

**API returning 401:**
```bash
# Check Auth0 configuration
curl https://your-domain.auth0.com/.well-known/jwks.json

# Verify token
jwt decode --json $TOKEN
```

**Query timeouts:**
```bash
# Check database size
aws s3 ls s3://your-bucket/current/analytics.db

# Review slow queries
railway logs --service analytics-api | grep "execution_time_ms"
```

## Development

```bash
# Run tests
./scripts/test.sh

# Deploy
./scripts/deploy.sh production
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests
5. Submit pull request

## License

MIT License - see LICENSE file for details

## Support

- Documentation: `/docs`
- Issues: GitHub Issues
- Email: support@yourdomain.com
