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
        structlog.processors.JSONRenderer(),
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
    redoc_url="/redoc",
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
        client_ip=get_remote_address(request),
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
    user: dict = Depends(get_current_user),
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
            "execution_time_ms": result.execution_time_ms,
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
