"""
Ultimate MCP Server – Comprehensive SQL Tools (v6, 2025-04-25)

A feature-rich, production-grade SQL toolkit for database interactions.

Features:
• Core functionality: Connection management, query execution, schema inspection
• Multi-dialect support: SQLite, PostgreSQL, MySQL, SQL Server, Snowflake
• Security: PII masking, prohibited statement detection, secrets management, ACL
• Performance: Connection pooling, async timeouts, cancellation, metrics
• Schema: Comprehensive schema exploration, table details, column statistics
• Export: Pandas DataFrame, Excel output
• Validation: Pandera schema validation
• Advanced: Natural language to SQL, audit trails (SEC/LP compliant)
• Secure schema drift detection with cryptographic verification

This unified module provides a complete set of tools for SQL operations with
proper error handling, metrics collection, and security safeguards.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

# SQLAlchemy imports
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

# Local imports
from ultimate_mcp_server.exceptions import ToolError, ToolInputError
from ultimate_mcp_server.services.cache import with_cache
from ultimate_mcp_server.tools.base import (
    with_error_handling,
    with_retry,
    with_tool_metrics,
)
from ultimate_mcp_server.tools.completion import generate_completion  # For NL→SQL
from ultimate_mcp_server.utils import get_logger

# Optional imports with graceful fallbacks
try:
    import boto3  # For AWS Secrets Manager
except ImportError:
    boto3 = None

try:
    import hvac  # For HashiCorp Vault
except ImportError:
    hvac = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pandera as pa
except ImportError:
    pa = None

try:
    import prometheus_client as prom
except ImportError:
    prom = None

logger = get_logger("ultimate_mcp_server.tools.sql")

# ────────────────────────────────── CONSTANTS ──────────────────────────────────

# SQL patterns that are prohibited for security reasons
PROHIBITED = re.compile(
    r"""^\s*(DROP\s+TABLE|TRUNCATE\s+TABLE|DELETE\s+FROM|
             DROP\s+DATABASE|ALTER\s+TABLE\s+\S+\s+DROP\s+|
             UPDATE\s+|INSERT\s+INTO)""",
    re.I | re.X,
)

# Global engine registry: connection_id → AsyncEngine
_DB: MutableMapping[str, AsyncEngine] = {}

# Lineage tracking for schema drift detection
_LINEAGE, _SCHEMA_VERSIONS = [], {}

# Pattern to extract table names from queries
_TABLE_RX = re.compile(r"\bfrom\s+([\w.]+)|\bjoin\s+([\w.]+)", re.I)

# ────────────────────────────── PROMETHEUS METRICS ─────────────────────────────

# Query counters and latency metrics
Q_CNT = prom.Counter("mcp_sqltool_calls", "SQL tool calls", ["fn", "db"])
Q_LAT = prom.Histogram(
    "mcp_sqltool_latency_seconds",
    "SQL latency",
    ["fn", "db"],
    buckets=(.01, .05, .1, .25, .5, 1, 2, 5, 10, 30),
)

# ───────────────────────────────── ACCESS CONTROL ─────────────────────────────

# ACL (Access Control Lists)
RESTRICTED_TABLES: set[str] = set()
RESTRICTED_COLUMNS: set[str] = set()

def update_acl(*, tables: Optional[List[str]] = None, columns: Optional[List[str]] = None) -> None:
    """Update the ACL lists for restricted tables and columns."""
    if tables is not None:
        RESTRICTED_TABLES.clear()
        RESTRICTED_TABLES.update({t.lower() for t in tables})
    if columns is not None:
        RESTRICTED_COLUMNS.clear()
        RESTRICTED_COLUMNS.update({c.lower() for c in columns})

def _check_acl(sql: str) -> None:
    """Check if SQL contains any restricted tables or columns."""
    toks = re.findall(r"[\w$]+", sql.lower())
    if RESTRICTED_TABLES.intersection(toks):
        raise ToolError("restricted table", http_status_code=403)
    if RESTRICTED_COLUMNS.intersection(toks):
        raise ToolError("restricted column", http_status_code=403)

# ───────────────────────────────── AUDIT TRAIL ───────────────────────────────

# In-memory audit log (can be extended to persistent storage)
_AUDIT_LOG: List[Dict[str, Any]] = []
_AUDIT_ID = 0

def _next_audit_id() -> str:
    """Generate the next sequential audit ID."""
    global _AUDIT_ID
    _AUDIT_ID += 1
    return f"a{_AUDIT_ID:09d}"

def _now() -> str:
    """Get current UTC timestamp in ISO format."""
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _audit(
    *,
    action: str,
    connection_id: str | None,
    sql: str | None,
    tables: List[str] | None,
    row_count: int | None,
    success: bool,
    error: str | None,
    user_id: str | None,
    session_id: str | None
) -> None:
    """Record an audit trail entry."""
    _AUDIT_LOG.append(
        dict(
            audit_id=_next_audit_id(),
            timestamp=_now(),
            user_id=user_id,
            session_id=session_id,
            connection_id=connection_id,
            action=action,
            sql=sql,
            tables=tables,
            row_count=row_count,
            success=success,
            error=error
        )
    )

@with_tool_metrics
@with_error_handling
def get_audit_log() -> Dict[str, Any]:
    """Retrieve the complete audit log."""
    return {"records": list(_AUDIT_LOG), "success": True}

@with_tool_metrics
@with_error_handling
def export_audit_log_excel() -> Dict[str, Any]:
    """Export the audit log to Excel format."""
    df = pd.DataFrame(_AUDIT_LOG)
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    df.to_excel(path, index=False, engine="xlsxwriter")
    return {"path": path, "success": True}

# ───────────────────────────── SECRETS MANAGEMENT ─────────────────────────────

@lru_cache(maxsize=64)
def _pull_secret(name: str) -> str:
    """
    Retrieve a secret from various sources (in order):
    1. AWS Secrets Manager
    2. HashiCorp Vault
    3. Environment variables
    """
    if boto3:
        try:
            # Try AWS Secrets Manager
            client = boto3.client("secretsmanager")
            return client.get_secret_value(SecretId=name)["SecretString"]
        except Exception:
            pass
    
    if hvac:
        try:
            # Try HashiCorp Vault
            url, token = os.getenv("VAULT_ADDR"), os.getenv("VAULT_TOKEN")
            if url and token:
                vault_client = hvac.Client(url=url, token=token, timeout=2)
                if vault_client.is_authenticated():
                    return vault_client.secrets.kv.read_secret_version(name)["data"]["data"]["value"]
        except Exception:
            pass
    
    # Try environment variables
    if env_val := os.getenv(name):
        return env_val
        
    raise ToolError(f"Secret '{name}' not found", http_status_code=404)

def _resolve_conn(raw: str) -> str:
    """Resolve connection string, handling secret references."""
    return _pull_secret(raw[10:]) if raw.startswith("secrets://") else raw

# ───────────────────────────── PII MASKING CONFIG ─────────────────────────────

@dataclass
class MaskRule:
    """Rule for masking sensitive data in query results."""
    rx: re.Pattern  # Regular expression to match
    repl: str | None  # Replacement (None = redact completely, callable = dynamic)

# Default masking rules
_RULES = [
    MaskRule(re.compile(r"^\d{3}-\d{2}-\d{4}$"), "***-**-XXXX"),  # SSN
    MaskRule(re.compile(r".+@.+\..+"), lambda v: v[:2] + "…@" + v.split("@")[-1]),  # Email
    # Add more rules here as needed
]

def _mask_val(v: Any) -> Any:
    """Apply masking rules to a single value."""
    if not isinstance(v, str):
        return v
    for rule in _RULES:
        if rule.rx.fullmatch(v):
            return rule.repl(v) if callable(rule.repl) else rule.repl
    return v

def _mask_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Apply masking rules to an entire row of data."""
    return {k: _mask_val(v) for k, v in row.items()}

# ───────────────────────────── CONNECTION HELPERS ─────────────────────────────

def _driver_url(conn_str: str) -> Tuple[str, str]:
    """Convert generic connection string to dialect-specific async URL."""
    # Convert file paths to sqlite URL if needed
    if "://" not in conn_str:
        # Use Path to handle platform-specific path formatting
        sqlite_path = Path(conn_str).expanduser().resolve()
        url_str = f"sqlite:///{sqlite_path}"
    else:
        url_str = conn_str
    
    # Parse URL using SQLAlchemy's make_url
    url = make_url(url_str)
    drv = url.drivername.lower()
    
    if drv.startswith("sqlite"):
        return str(url.set(drivername="sqlite+aiosqlite")), "sqlite"
    if drv.startswith("postgresql"):
        return str(url.set(drivername="postgresql+asyncpg")), "postgresql"
    if drv.startswith(("mysql", "mariadb")):
        return str(url.set(drivername="mysql+aiomysql")), "mysql"
    if drv.startswith("mssql"):
        return str(url.set(drivername="mssql+aioodbc")), "sqlserver"
    if drv.startswith("snowflake"):
        return str(url.set(drivername="snowflake+async")), "snowflake"
        
    raise ToolInputError("unsupported dialect", param_name="connection_string")

def _auto_pool(db_type: str) -> Dict[str, Any]:
    """Provide optimal connection pool settings based on database type."""
    if db_type == "sqlite":
        return {"pool_size": 1, "max_overflow": 0, "pool_pre_ping": True}
    if db_type in {"postgresql", "mysql", "sqlserver"}:
        return {
            "pool_size": 10,
            "max_overflow": 20,
            "pool_recycle": 900,
            "pool_pre_ping": True,
        }
    if db_type == "snowflake":
        return {"pool_size": 5, "max_overflow": 5, "pool_pre_ping": True}
    return {}

async def _eng(cid: str) -> AsyncEngine:
    """Get engine by connection ID, with error handling."""
    if cid not in _DB:
        raise ToolInputError("unknown connection_id", param_name="connection_id")
    return _DB[cid]

def _tables(sql: str) -> List[str]:
    """Extract table names referenced in a SQL query."""
    return list({m.group(1) or m.group(2) for m in _TABLE_RX.finditer(sql)})

# ───────────────────────────── SAFETY CHECKS ───────────────────────────────

def _check_safe(sql: str, read_only: bool = True) -> None:
    """
    Validate SQL for safety:
    1. Check against ACL restrictions
    2. Prevent prohibited statements (DROP, DELETE, etc.)
    3. Enforce read-only mode if requested
    """
    _check_acl(sql)
    if PROHIBITED.match(sql):
        raise ToolInputError("prohibited statement", param_name="query")
    if read_only and not sql.lstrip().upper().startswith(
        ("SELECT", "WITH", "SHOW", "EXPLAIN", "DESCRIBE", "PRAGMA")
    ):
        raise ToolInputError("write op in read-only mode", param_name="query")

# ───────────────────────────── CORE SQL EXECUTOR ─────────────────────────────

async def _exec(
    eng: AsyncEngine,
    sql: str,
    params: Optional[Dict[str, Any]],
    *,
    limit: Optional[int],
    fn: str,
    timeout: float = 30.0
) -> Tuple[List[str], List[Dict[str, Any]], int]:
    """
    Core async SQL executor with:
    - Timeouts and cancellation
    - Metrics collection
    - PII masking
    - Row limiting
    - Error handling
    """
    db = "sqlite" if "sqlite" in str(eng.url) else eng.url.get_backend_name()
    start = time.perf_counter()
    
    # Record metrics if prometheus is available
    if prom:
        Q_CNT.labels(fn=fn, db=db).inc()
    
    async def _run(conn: AsyncConnection):
        res = await conn.execute(text(sql), params or {})
        if not res.returns_rows:
            return [], [], 0
        rows = res.fetchmany(limit) if limit else res.fetchall()
        return list(res.keys()), [_mask_row(r._mapping) for r in rows], len(rows)
    
    try:
        async with eng.connect() as conn:
            cols, rows, cnt = await asyncio.wait_for(_run(conn), timeout=timeout)
            # Only record metrics if prometheus client is available
            if prom:
                Q_LAT.labels(fn=fn, db=db).observe(time.perf_counter() - start)
            return cols, rows, cnt
    except asyncio.TimeoutError:
        raise ToolError("query timed-out", http_status_code=504) from None
    except (ProgrammingError, OperationalError) as e:
        raise ToolError(str(e), http_status_code=400) from e
    except SQLAlchemyError as e:
        raise ToolError(str(e), http_status_code=500) from e

# ────────────────────────────── DATA EXPORT HELPERS ──────────────────────────

def _export_rows(
    cols: List[str],
    rows: List[Dict[str, Any]],
    export: str | None,
) -> Tuple[Any | None, str | None]:
    """Export query results to pandas DataFrame or Excel file."""
    if not export or pd is None:
        return None, None
        
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    
    if export == "pandas":
        return df, None
    if export == "excel":
        fd, path = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        df.to_excel(path, index=False, engine="xlsxwriter")
        return None, path
        
    raise ToolInputError("unsupported export format", param_name="export")

async def _validate_df(df: Any, schema: Any | None) -> None:
    """Validate DataFrame against Pandera schema."""
    if schema is None or pa is None:
        return
    try:
        schema.validate(df, lazy=True)
    except Exception as se:
        raise ToolError(f"validation failed: {se}", http_status_code=422) from se

# ──────────────────────────────── PUBLIC API ────────────────────────────────

# ------------------------------ CONNECTION MANAGEMENT --------------------------

@with_tool_metrics
@with_error_handling
@with_retry(max_retries=2, retry_delay=1)
async def connect_to_database(
    connection_string: str,
    *,
    connection_id: str | None = None,
    echo: bool = False,
    user_id: str | None = None,
    session_id: str | None = None,
    **opts: Any
) -> Dict[str, Any]:
    """
    Connect to a database with the provided connection string.
    
    Supports multiple dialects:
    - SQLite (file path or :memory:)
    - PostgreSQL (postgresql://)
    - MySQL/MariaDB (mysql://)
    - SQL Server (mssql://)
    - Snowflake (snowflake://)
    
    Connection strings can reference secrets with secrets:// prefix.
    
    Args:
        connection_string: Database connection string
        connection_id: Optional custom ID (generated if not provided)
        echo: Enable SQLAlchemy echo mode for debugging
        user_id: User ID for audit trail
        session_id: Session ID for audit trail
        **opts: Additional engine options passed to SQLAlchemy
        
    Returns:
        Dict with connection_id, database_type, and success flag
    """
    cid = connection_id or str(uuid.uuid4())
    if cid in _DB:
        await disconnect_from_database(cid)
        
    # Resolve secrets and get correct async driver URL
    url, db_type = _driver_url(_resolve_conn(connection_string))
    
    # Create engine with auto-tuned pool settings
    eng = create_async_engine(url, echo=echo, **{**_auto_pool(db_type), **opts})
    
    # Test connection
    await _exec(eng, "SELECT 1", None, limit=0, fn="connect", timeout=10)
    _DB[cid] = eng
    
    # Audit trail
    _audit(
        action="connect",
        connection_id=cid,
        sql=None,
        tables=None,
        row_count=None,
        success=True,
        error=None,
        user_id=user_id,
        session_id=session_id
    )
    
    return {"connection_id": cid, "database_type": db_type, "success": True}

@with_tool_metrics
@with_error_handling
async def disconnect_from_database(
    connection_id: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None
) -> Dict[str, Any]:
    """
    Disconnect from a database and release resources.
    
    Args:
        connection_id: Connection ID to disconnect
        user_id: User ID for audit trail
        session_id: Session ID for audit trail
        
    Returns:
        Dict with connection_id and success flag
    """
    ok = connection_id in _DB
    try:
        if eng := _DB.pop(connection_id, None):
            await eng.dispose()
        _audit(
            action="disconnect",
            connection_id=connection_id,
            sql=None,
            tables=None,
            row_count=None,
            success=ok,
            error=None,
            user_id=user_id,
            session_id=session_id
        )
    except Exception as e:
        _audit(
            action="disconnect",
            connection_id=connection_id,
            sql=None,
            tables=None,
            row_count=None,
            success=False,
            error=str(e),
            user_id=user_id,
            session_id=session_id
        )
        raise
        
    return {"connection_id": connection_id, "success": True}

@with_tool_metrics
@with_error_handling
@with_retry(max_retries=2, retry_delay=0.5, retry_exceptions=(OperationalError,))
async def test_connection(connection_id: str) -> Dict[str, Any]:
    """
    Test a database connection and get version information.
    
    Args:
        connection_id: Connection ID to test
        
    Returns:
        Dict with connection_id, response_time, version, and success flag
    """
    eng = await _eng(connection_id)
    t0 = time.perf_counter()
    
    # Get database version using appropriate SQL for the dialect
    vsql = "SELECT sqlite_version()" if "sqlite" in str(eng.url) else "SELECT version()"
    cols, rows, _ = await _exec(eng, vsql, None, limit=1, fn="test_connection", timeout=5)
    
    return {
        "connection_id": connection_id,
        "response_time": round(time.perf_counter() - t0, 3),
        "version": rows[0][cols[0]],
        "success": True,
    }

# ------------------------------ QUERY EXECUTION ------------------------------

@with_tool_metrics
@with_error_handling
async def execute_query(
    connection_id: str,
    query: str,
    *,
    read_only: bool = True,
    max_rows: int = 1000,
    timeout: float = 30.0,
    validate_schema: Optional[Any] = None,
    export: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a SQL query.
    
    Args:
        connection_id: Connection ID to use
        query: SQL query to execute
        read_only: If True, reject write operations
        max_rows: Maximum number of rows to return
        timeout: Query timeout in seconds
        validate_schema: Optional Pandera schema for validation
        export: Export format ("pandas" or "excel")
        user_id: User ID for audit trail
        session_id: Session ID for audit trail
        
    Returns:
        Dict with columns, rows, row_count, truncated flag, and success flag
        If export is specified, also includes dataframe or excel_path
    """
    # Check safety and extract referenced tables
    _check_safe(query, read_only)
    tables = _tables(query)
    
    # Execute query
    eng = await _eng(connection_id)
    cols, rows, cnt = await _exec(
        eng, query, None, limit=max_rows + 1, fn="execute_query", timeout=timeout
    )
    
    # Validate and export if requested
    if validate_schema:
        await _validate_df(
            pd.DataFrame(rows[:max_rows], columns=cols) if pd else None, 
            validate_schema
        )
    
    df, xls = None, None
    if export:
        df, xls = _export_rows(cols, rows[:max_rows], export)
    
    # Audit trail
    _audit(
        action="execute_query",
        connection_id=connection_id,
        sql=query,
        tables=tables,
        row_count=min(cnt, max_rows),
        success=True,
        error=None,
        user_id=user_id,
        session_id=session_id
    )
    
    return {
        "columns": cols,
        "rows": rows[:max_rows],
        "row_count": min(cnt, max_rows),
        "truncated": cnt > max_rows,
        "dataframe": df,
        "excel_path": xls,
        "success": True,
    }

@with_tool_metrics
@with_error_handling
async def execute_parameterized_query(
    connection_id: str,
    query: str,
    parameters: Dict[str, Any],
    *,
    read_only: bool = True,
    timeout: float = 30.0,
    validate_schema: Optional[Any] = None,
    export: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a parameterized SQL query with bound parameters.
    
    Args:
        connection_id: Connection ID to use
        query: SQL query with placeholders
        parameters: Dict of parameter values
        read_only: If True, reject write operations
        timeout: Query timeout in seconds
        validate_schema: Optional Pandera schema for validation
        export: Export format ("pandas" or "excel")
        user_id: User ID for audit trail
        session_id: Session ID for audit trail
        
    Returns:
        Dict with columns, rows, row_count, and success flag
        If export is specified, also includes dataframe or excel_path
    """
    # Check safety and extract referenced tables
    _check_safe(query, read_only)
    tables = _tables(query)
    
    # Execute query with parameters
    eng = await _eng(connection_id)
    cols, rows, _ = await _exec(
        eng, query, parameters, limit=None, fn="execute_param_query", timeout=timeout
    )
    
    # Validate and export if requested
    if validate_schema and pd:
        await _validate_df(pd.DataFrame(rows, columns=cols), validate_schema)
    
    df, xls = None, None
    if export:
        df, xls = _export_rows(cols, rows, export)
    
    # Audit trail
    _audit(
        action="execute_parameterized_query",
        connection_id=connection_id,
        sql=query,
        tables=tables,
        row_count=len(rows),
        success=True,
        error=None,
        user_id=user_id,
        session_id=session_id
    )
    
    return {
        "columns": cols,
        "rows": rows,
        "row_count": len(rows),
        "dataframe": df,
        "excel_path": xls,
        "success": True,
    }

@with_tool_metrics
@with_error_handling
async def execute_query_with_pagination(
    connection_id: str,
    query: str,
    *,
    page_size: int = 100,
    page_number: int = 1,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a SQL query with pagination support.
    
    Args:
        connection_id: Connection ID to use
        query: SQL query to execute
        page_size: Number of rows per page
        page_number: Page number to retrieve (1-based)
        parameters: Optional parameters for the query
        
    Returns:
        Dict with columns, rows, pagination info, and success flag
    """
    _check_safe(query, True)  # Only allow read operations
    
    # Calculate offset and add LIMIT/OFFSET
    offset = (page_number - 1) * page_size
    paginated = f"{query} LIMIT :lim OFFSET :off"
    params = dict(parameters or {}, lim=page_size + 1, off=offset)
    
    # Execute query
    eng = await _eng(connection_id)
    cols, rows, _ = await _exec(eng, paginated, params, limit=None, fn="execute_paginated")
    
    return {
        "columns": cols,
        "rows": rows[:page_size],
        "pagination": {
            "page": page_number,
            "page_size": page_size,
            "has_next_page": len(rows) > page_size,
            "has_previous_page": page_number > 1,
        },
        "success": True,
    }

# ------------------------------ SCHEMA DISCOVERY ------------------------------

@with_tool_metrics
@with_error_handling
@with_cache(ttl=3600)
async def discover_database_schema(
    connection_id: str,
    include_indexes: bool = True,
    include_foreign_keys: bool = True,
    detailed: bool = False,
    filter_schema: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive database schema introspection.
    
    Discovers:
    - Tables and their columns
    - Indexes (optional)
    - Foreign keys (optional)
    - Views and their definitions
    - Relationships between tables
    
    Args:
        connection_id: Connection ID to use
        include_indexes: Whether to include index information
        include_foreign_keys: Whether to include foreign key information
        detailed: Whether to include additional column details
        filter_schema: Optional schema name to filter by
        
    Returns:
        Dict with tables, views, relationships, and success flag
    """
    eng = await _eng(connection_id)

    async def _sync(inspector: sa.InspectionEngine) -> Dict[str, Any]:
        insp = sa.inspect(inspector)
        sch = filter_schema or (insp.default_schema_name if hasattr(insp, "default_schema_name") else None)
        tables, rels, views = [], [], []

        # Process tables
        for t in insp.get_table_names(schema=sch):
            t_info: Dict[str, Any] = {"name": t, "columns": []}
            if sch:
                t_info["schema"] = sch
                
            # Columns
            for c in insp.get_columns(t, schema=sch):
                col = {
                    "name": c["name"],
                    "type": str(c["type"]),
                    "nullable": c["nullable"],
                    "primary_key": bool(c.get("primary_key")),
                }
                if detailed:
                    col["default"] = str(c.get("default", ""))
                    col["comment"] = c.get("comment", "")
                t_info["columns"].append(col)
                
            # Indexes
            if include_indexes:
                idx = [
                    {
                        "name": i["name"],
                        "columns": i["column_names"],
                        "unique": i.get("unique", False)
                    }
                    for i in insp.get_indexes(t, schema=sch)
                ]
                t_info["indexes"] = idx
                
            # Foreign keys
            fks = insp.get_foreign_keys(t, schema=sch) if include_foreign_keys else []
            if fks:
                t_info["foreign_keys"] = []
            for fk in fks:
                fk_info = {
                    "constrained_columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"],
                }
                if fk.get("name"):
                    fk_info["name"] = fk["name"]
                t_info["foreign_keys"].append(fk_info)
                
                # Track relationships for the graph
                rels.append({
                    "source_table": t,
                    "target_table": fk["referred_table"],
                    "source_columns": fk["constrained_columns"],
                    "target_columns": fk["referred_columns"],
                })
                
            tables.append(t_info)

        # Process views
        for v in insp.get_view_names(schema=sch):
            views.append({
                "name": v,
                "schema": sch,
                "definition": insp.get_view_definition(v, schema=sch) or "",
            })

        return {
            "tables": tables,
            "views": views,
            "relationships": rels,
            "success": True
        }

    async with eng.connect() as conn:
        schema_info = await conn.run_sync(_sync)
        
        # Record schema version for drift detection
        # Use the imported hashlib module to create a schema hash
        schema_hash = hashlib.sha256(
            json.dumps(schema_info, sort_keys=True).encode()
        ).hexdigest()
        
        # Record the new schema version
        timestamp = dt.datetime.utcnow().isoformat()
        if (connection_id not in _SCHEMA_VERSIONS or 
            _SCHEMA_VERSIONS[connection_id] != schema_hash):
            _SCHEMA_VERSIONS[connection_id] = schema_hash
            _LINEAGE.append({
                "connection_id": connection_id,
                "timestamp": timestamp,
                "schema_hash": schema_hash,
                "user_id": None,
                "tables_count": len(schema_info.get("tables", [])),
                "views_count": len(schema_info.get("views", [])),
            })
        
        return schema_info

@with_tool_metrics
@with_error_handling
@with_cache(ttl=1800)
async def get_table_details(
    connection_id: str,
    table_name: str,
    schema_name: Optional[str] = None,
    include_sample_data: bool = False,
    sample_size: int = 5,
    include_statistics: bool = False,
) -> Dict[str, Any]:
    """
    Get detailed information about a specific table.
    
    Args:
        connection_id: Connection ID to use
        table_name: Table to inspect
        schema_name: Optional schema name
        include_sample_data: Whether to include sample rows
        sample_size: Number of sample rows to include
        include_statistics: Whether to include column statistics
        
    Returns:
        Dict with table details, columns, indexes, foreign keys, and more
    """
    eng = await _eng(connection_id)
    
    async def _sync(inspector: sa.InspectionEngine) -> Dict[str, Any]:
        insp = sa.inspect(inspector)
        if table_name not in insp.get_table_names(schema=schema_name):
            raise ToolInputError("table not found", param_name="table_name") from None
        cols = insp.get_columns(table_name, schema=schema_name)
        idx = insp.get_indexes(table_name, schema=schema_name)
        fks = insp.get_foreign_keys(table_name, schema=schema_name)
        return {"columns": cols, "indexes": idx, "foreign_keys": fks}

    async with eng.connect() as conn:
        # Get table metadata
        meta = await conn.run_sync(_sync)

        result = {
            "table_name": table_name,
            "columns": [
                {
                    "name": c["name"],
                    "type": str(c["type"]),
                    "nullable": c["nullable"],
                    "primary_key": bool(c.get("primary_key")),
                }
                for c in meta["columns"]
            ],
            "indexes": meta["indexes"],
            "foreign_keys": meta["foreign_keys"],
            "success": True,
        }

        # Get row count
        q = f'SELECT COUNT(*) FROM "{table_name}"'
        cnt = (await conn.execute(text(q))).scalar()
        result["row_count"] = cnt

        # Get sample data if requested
        if include_sample_data:
            cols, rows, _ = await _exec(
                eng,
                f'SELECT * FROM "{table_name}" LIMIT :n',
                {"n": sample_size},
                limit=sample_size,
                fn="table_sample",
            )
            result["sample_data"] = rows

        # Get statistics if requested
        if include_statistics:
            stats = {}
            for c in result["columns"]:
                name = c["name"]
                nulls = (await conn.execute(
                    text(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{name}" IS NULL')
                )).scalar()
                distinct = (await conn.execute(
                    text(f'SELECT COUNT(DISTINCT "{name}") FROM "{table_name}"')
                )).scalar()
                stats[name] = {"null_count": nulls, "unique_count": distinct}
            result["statistics"] = stats

        return result

@with_tool_metrics
@with_error_handling
async def find_related_tables(
    connection_id: str,
    table_name: str,
    depth: int = 1,
    schema_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Find tables related to the specified table via foreign keys.
    
    Args:
        connection_id: Connection ID to use
        table_name: Starting table
        depth: Max recursion depth (1-5)
        schema_name: Optional schema name
        
    Returns:
        Dict with relationship graph
    """
    # Limit depth for performance
    depth = max(1, min(depth, 5))
    
    # Get schema with foreign keys
    schema = await discover_database_schema(
        connection_id,
        include_indexes=False,
        include_foreign_keys=True,
        filter_schema=schema_name
    )
    
    # Build table lookup
    tables = {t["name"]: t for t in schema["tables"]}
    
    # Recursively build relationship graph
    def _recurse(t: str, d: int) -> Dict[str, Any]:
        node = {"table": t, "children": [], "parents": []}
        if d >= depth:
            return node
            
        # Add parent tables (tables this table references)
        for fk in tables[t].get("foreign_keys", []):
            node["parents"].append(_recurse(fk["referred_table"], d + 1))
            
        # Add child tables (tables that reference this table)
        for other, info in tables.items():
            for fk in info.get("foreign_keys", []):
                if fk["referred_table"] == t:
                    node["children"].append(_recurse(other, d + 1))
                    
        return node

    # Start the recursion from the requested table
    graph = _recurse(table_name, 0)
    return {"source_table": table_name, "relationships": graph, "success": True}

@with_tool_metrics
@with_error_handling
async def analyze_column_statistics(
    connection_id: str,
    table_name: str,
    column_name: str,
    histogram: bool = False,
    num_buckets: int = 10,
) -> Dict[str, Any]:
    """
    Analyze statistics for a specific column.
    
    Args:
        connection_id: Connection ID to use
        table_name: Table containing the column
        column_name: Column to analyze
        histogram: Whether to generate a histogram
        num_buckets: Number of buckets for histogram
        
    Returns:
        Dict with statistics and optional histogram
    """
    eng = await _eng(connection_id)
    q_base = f'SELECT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL'

    async with eng.connect() as conn:
        # Get basic statistics
        total = (await conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))).scalar()
        nulls = (await conn.execute(
            text(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{column_name}" IS NULL')
        )).scalar()
        distinct = (await conn.execute(
            text(f'SELECT COUNT(DISTINCT "{column_name}") FROM "{table_name}"')
        )).scalar()

        result = {
            "table": table_name,
            "column": column_name,
            "total_rows": total,
            "null_count": nulls,
            "unique_count": distinct,
            "success": True,
        }

        # Generate histogram if requested
        if histogram:
            rows = (await conn.execute(text(q_base))).fetchall()
            vals = [r[0] for r in rows]
            
            if not vals:
                result["histogram"] = []
            elif isinstance(vals[0], (int, float)):
                # Numeric histogram with evenly distributed buckets
                mn, mx = min(vals), max(vals)
                width = (mx - mn) / num_buckets if mx > mn else 1
                buckets = [
                    {
                        "range": f"{mn + i*width:.4g}-{mn + (i+1)*width:.4g}",
                        "count": 0
                    }
                    for i in range(num_buckets)
                ]
                for v in vals:
                    idx = min(int((v - mn) / width) if width > 0 else 0, num_buckets - 1)
                    buckets[idx]["count"] += 1
                result["histogram"] = buckets
            else:
                # Frequency histogram for non-numeric values
                freq: Dict[str, int] = {}
                for v in vals:
                    val_str = str(v)[:50]  # Truncate long values
                    freq[val_str] = freq.get(val_str, 0) + 1
                buckets = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:num_buckets]
                result["histogram"] = [{"value": k, "count": c} for k, c in buckets]
                
        return result

# ------------------------------ DOCUMENTATION ------------------------------

@with_tool_metrics
@with_error_handling
async def generate_database_documentation(
    connection_id: str,
    *,
    output_format: str = "markdown",
) -> Dict[str, Any]:
    """
    Generate documentation for the database schema.
    
    Args:
        connection_id: Connection ID to use
        output_format: "markdown" or "json"
        
    Returns:
        Dict with documentation in the requested format
    """
    schema = await discover_database_schema(connection_id)
    
    if output_format == "json":
        return {"documentation": schema, "format": "json", "success": True}

    # Generate markdown documentation
    lines = ["# Database Documentation", ""]
    
    for t in schema["tables"]:
        lines += [f"## {t['name']}", ""]
        lines += ["| Column | Type | Null | PK |", "|--------|------|------|----|"]
        for c in t["columns"]:
            lines.append(
                f"| {c['name']} | {c['type']} | {c['nullable']} | {c['primary_key']} |"
            )
        lines.append("")
        
    # Add views if available
    if schema.get("views"):
        lines += ["# Views", ""]
        for v in schema["views"]:
            lines += [f"## {v['name']}", ""]
            if v.get("definition"):
                lines += ["```sql", v["definition"], "```", ""]
                
    return {
        "documentation": "\n".join(lines),
        "format": "markdown",
        "success": True,
    }

# ------------------------------ ADVANCED FEATURES ------------------------------

@with_tool_metrics
@with_error_handling
async def generate_sql_from_nl(
    connection_id: str,
    natural_language_query: str,
    *,
    confidence_threshold: float = 0.60,
) -> Dict[str, Any]:
    """
    Generate SQL from natural language using LLM.
    
    Args:
        connection_id: Connection ID to use
        natural_language_query: Question in natural language
        confidence_threshold: Minimum required confidence (0-1)
        
    Returns:
        Dict with generated SQL and confidence score
    """
    eng = await _eng(connection_id)

    # Get schema fingerprint for context
    async def _schema_fingerprint(conn: AsyncConnection) -> str:
        insp = sa.inspect(conn)
        tbls = []
        for t in insp.get_table_names():
            cols = [c["name"] + ":" + str(c["type"]) for c in insp.get_columns(t)]
            tbls.append(f"{t}({','.join(cols)})")
        return "; ".join(sorted(tbls))

    async with eng.connect() as conn:
        fp = await conn.run_sync(_schema_fingerprint)

    # Generate SQL using LLM
    prompt = (
        "You are an expert SQL generator.\n"
        f"Schema:\n{fp}\n"
        f"Question:\n{natural_language_query}\n"
        'Respond with JSON {"sql":"...","confidence":0-1}'
    )
    
    llm_out = await generate_completion(prompt, max_tokens=256)
    
    try:
        data = json.loads(llm_out)
        sql = data["sql"]
        conf = float(data.get("confidence", 0))
    except Exception as e:
        raise ToolError(f"LLM returned invalid JSON: {e}", http_status_code=500) from e

    # Check confidence threshold
    if conf < confidence_threshold:
        raise ToolError(f"confidence {conf:.2f} below threshold", http_status_code=400) from None

    # Validate generated SQL
    _check_safe(sql)
    
    # Quick parse: must reference at least one known table
    if not any(t in sql for t in fp.split("(")[0::2]):
        raise ToolError("SQL references no known tables", http_status_code=400) from None

    return {"sql": sql, "confidence": conf, "success": True}