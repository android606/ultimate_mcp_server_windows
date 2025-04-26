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
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import inspect as sa_inspect

# SQLAlchemy imports
from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

# Local imports
from ultimate_mcp_server.exceptions import ToolError, ToolInputError
from ultimate_mcp_server.tools.base import (
    BaseTool,
    tool,
    with_error_handling,
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

logger = get_logger("ultimate_mcp_server.tools.sql_databases")


@lru_cache(maxsize=64)
def _pull_secret_from_sources(name: str) -> str:
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
            logger.debug(f"Secret '{name}' not found in AWS Secrets Manager.")
            pass
    
    if hvac:
        try:
            # Try HashiCorp Vault
            url, token = os.getenv("VAULT_ADDR"), os.getenv("VAULT_TOKEN")
            if url and token:
                vault_client = hvac.Client(url=url, token=token, timeout=2)
                if vault_client.is_authenticated():
                    # Assuming KV v2, adjust path if needed
                    mount_point = os.getenv("VAULT_KV_MOUNT_POINT", "secret") 
                    secret_path = name 
                    read_response = vault_client.secrets.kv.v2.read_secret_version(
                        path=secret_path, mount_point=mount_point
                    )
                    # Adjust the key extraction based on your vault structure
                    if 'data' in read_response and 'data' in read_response['data'] and 'value' in read_response['data']['data']:
                         return read_response['data']['data']['value']
                    else:
                         logger.debug(f"Secret key 'value' not found at path '{secret_path}' in Vault.")

        except Exception as e:
            logger.debug(f"Error accessing Vault for secret '{name}': {e}")
            pass
    
    # Try environment variables
    if env_val := os.getenv(name):
        return env_val
    
    # Try environment variables prefixed with MCP_SECRET_ (common pattern)
    mcp_secret_name = f"MCP_SECRET_{name.upper()}"
    if env_val := os.getenv(mcp_secret_name):
        logger.debug(f"Found secret '{name}' using prefixed env var '{mcp_secret_name}'.")
        return env_val

    raise ToolError(f"Secret '{name}' not found in any source (AWS, Vault, Env: {name}, Env: {mcp_secret_name})", http_status_code=404)


class ConnectionManager:
    """Manages database connections with automatic cleanup after inactivity."""
    
    def __init__(self, cleanup_interval_seconds=600, check_interval_seconds=60):  # 10 minutes default cleanup, 1 min check
        self.connections: Dict[str, Tuple[AsyncEngine, float]] = {}  # connection_id -> (engine, last_accessed_time)
        self.cleanup_interval = cleanup_interval_seconds
        self.check_interval = check_interval_seconds
        self._cleanup_task: Optional[asyncio.Task] = None

    def start_cleanup_task(self):
        """Starts the background cleanup task if not already running."""
        if self._cleanup_task is None or self._cleanup_task.done():
             # Ensure loop runs in the current event loop context
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop())
                logger.info("Started connection cleanup task.")
            except RuntimeError:
                logger.warning("No running event loop found, cleanup task not started.")


    async def _cleanup_loop(self):
        """Background task that cleans up inactive connections."""
        logger.debug(f"Cleanup loop started. Check interval: {self.check_interval}s, Inactivity threshold: {self.cleanup_interval}s")
        while True:
            await asyncio.sleep(self.check_interval)
            try:
                await self.cleanup_inactive_connections()
            except Exception as e:
                logger.error(f"Error during connection cleanup: {e}", exc_info=True)

    async def cleanup_inactive_connections(self):
        """Close connections that have been inactive for longer than the cleanup interval."""
        current_time = time.time()
        conn_ids_to_close = []
        
        # Use items() for safe iteration while potentially modifying dict later
        for conn_id, (_engine, last_accessed) in list(self.connections.items()):
            if current_time - last_accessed > self.cleanup_interval:
                logger.info(f"Connection {conn_id} exceeded inactivity timeout ({current_time - last_accessed:.1f}s > {self.cleanup_interval}s)")
                conn_ids_to_close.append(conn_id)
                
        closed_count = 0
        for conn_id in conn_ids_to_close:
            if await self.close_connection(conn_id):
                logger.info(f"Auto-closed inactive connection: {conn_id}")
                closed_count += 1
        if closed_count > 0:
             logger.debug(f"Closed {closed_count} inactive connections.")
        elif conn_ids_to_close:
             logger.debug(f"Attempted to close {len(conn_ids_to_close)} connections, but they were already removed.")


    async def get_connection(self, conn_id: str) -> AsyncEngine:
        """Get a connection by ID and update last accessed time."""
        if conn_id not in self.connections:
            raise ToolInputError("unknown connection_id", param_name="connection_id")
            
        engine, _ = self.connections[conn_id]
        # Update last accessed time
        self.connections[conn_id] = (engine, time.time())
        logger.debug(f"Accessed connection {conn_id}, updated last accessed time.")
        return engine
    
    async def add_connection(self, conn_id: str, engine: AsyncEngine):
        """Add a new connection to the manager."""
        if conn_id in self.connections:
            logger.warning(f"Overwriting existing connection entry for {conn_id}.")
            await self.close_connection(conn_id) # Close the old one first

        self.connections[conn_id] = (engine, time.time())
        logger.info(f"Added connection {conn_id} for URL: {str(engine.url).split('@')[0]}...") # Avoid logging credentials
        self.start_cleanup_task() # Ensure cleanup is running

    async def close_connection(self, conn_id: str) -> bool:
        """Close a specific connection and remove it from management."""
        if conn_id in self.connections:
            engine, _ = self.connections.pop(conn_id)
            logger.info(f"Closing connection {conn_id}...")
            try:
                await engine.dispose()
                logger.info(f"Connection {conn_id} disposed successfully.")
                return True
            except Exception as e:
                 logger.error(f"Error disposing engine for connection {conn_id}: {e}", exc_info=True)
                 return False # Still removed from dict, but disposal failed
        else:
             logger.warning(f"Attempted to close non-existent connection ID: {conn_id}")
             return False

    async def shutdown(self):
        """Gracefully shut down all connections and stop the cleanup task."""
        logger.info("Shutting down Connection Manager...")
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled.")
            except Exception as e:
                 logger.error(f"Error stopping cleanup task: {e}", exc_info=True)

        conn_ids = list(self.connections.keys())
        logger.info(f"Closing {len(conn_ids)} active connections...")
        for conn_id in conn_ids:
             await self.close_connection(conn_id)
        logger.info("Connection Manager shutdown complete.")


class SQLTool(BaseTool):
    """
    A comprehensive SQL toolkit for database interactions in the Ultimate MCP Server.
    
    Provides a feature-rich set of tools for working with various SQL databases including
    SQLite, PostgreSQL, MySQL, SQL Server, and Snowflake. Features include connection
    management, secure query execution, schema inspection, PII masking, and more.

    Consolidated Tools:
    - manage_database: Handles connection, disconnection, testing, and status.
    - execute_sql: Executes SQL queries, NL->SQL, parameterized queries, pagination, and exports.
    - explore_database: Performs schema discovery, table/column details, relationship finding, documentation.
    - access_audit_log: Views or exports the audit trail.
    
    Features:
    • Core functionality: Connection management, query execution, schema inspection
    • Multi-dialect support: SQLite, PostgreSQL, MySQL, SQL Server, Snowflake
    • Security: PII masking, prohibited statement detection, secrets management, ACL
    • Performance: Connection pooling, async timeouts, cancellation, metrics, auto-cleanup
    • Schema: Comprehensive schema exploration, table details, column statistics
    • Export: Pandas DataFrame, Excel, CSV output
    • Validation: Pandera schema validation (via execute_sql options)
    • Advanced: Natural language to SQL, audit trails (SEC/LP compliant)
    • Secure schema drift detection with cryptographic verification
    """
    
    tool_name = "sql"
    description = "Unified SQL tools for connection management, query execution, schema exploration, and auditing."
    
    def __init__(self, mcp_server):
        """Initialize the SQL tool.
        
        Args:
            mcp_server: MCP server instance
        """
        super().__init__(mcp_server)
        
        # SQL patterns that are prohibited for security reasons
        self.PROHIBITED = re.compile(
            r"""^\s*(DROP\s+(TABLE|DATABASE|INDEX|VIEW|FUNCTION|PROCEDURE|USER|ROLE)|
                     TRUNCATE\s+TABLE|
                     DELETE\s+FROM|
                     ALTER\s+(TABLE|DATABASE)\s+\S+\s+DROP\s+|
                     UPDATE\s+|INSERT\s+INTO(?!\s+OR\s+IGNORE)|
                     GRANT\s+|REVOKE\s+|
                     CREATE\s+USER|ALTER\s+USER|DROP\s+USER|
                     CREATE\s+ROLE|ALTER\s+ROLE|DROP\s+ROLE|
                     SHUTDOWN|REBOOT|RESTART)""", # Added more dangerous commands
            re.I | re.X,
        )
        
        # Initialize Connection Manager
        self._conn_manager = ConnectionManager()
        
        # Lineage tracking for schema drift detection
        self._LINEAGE = []
        self._SCHEMA_VERSIONS: Dict[str, str] = {} # connection_id -> schema_hash
        
        # Pattern to extract table names from queries
        self._TABLE_RX = re.compile(r"\b(?:FROM|JOIN|UPDATE|INSERT\s+INTO|DELETE\s+FROM)\s+([\w.\"$-]+)", re.I) # Improved table name capture

        # Access control lists
        self.RESTRICTED_TABLES = set()
        self.RESTRICTED_COLUMNS = set()
        
        # In-memory audit log (can be extended to persistent storage)
        self._AUDIT_LOG: List[Dict[str, Any]] = []
        self._AUDIT_ID = 0
        
        # Setup Prometheus metrics if available
        if prom:
            self.Q_CNT = prom.Counter("mcp_sqltool_calls", "SQL tool calls", ["tool", "action", "db"])
            self.Q_LAT = prom.Histogram(
                "mcp_sqltool_latency_seconds",
                "SQL latency",
                ["tool", "action", "db"],
                buckets=(.01, .05, .1, .25, .5, 1, 2, 5, 10, 30, 60), # Added 60s bucket
            )
            self.CONN_GAUGE = prom.Gauge("mcp_sqltool_active_connections", "Number of active SQL connections")
            self.CONN_GAUGE.set_function(lambda: len(self._conn_manager.connections)) # Dynamically report count
        else:
             self.Q_CNT = None
             self.Q_LAT = None
             self.CONN_GAUGE = None # Explicitly None if prom not available


        # Default masking rules
        self._RULES = [
            self.MaskRule(re.compile(r"^\d{3}-\d{2}-\d{4}$"), "***-**-XXXX"),  # SSN
            self.MaskRule(re.compile(r"(\b\d{4}-?){3}\d{4}\b"), lambda v: f"XXXX-...-{v[-4:]}"), # Credit Card basic mask
            self.MaskRule(re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"), lambda v: v[:2] + "***@" + v.split("@")[-1] if "@" in v else "***"),  # Email
            # Add more rules here as needed
        ]
    
    @dataclass
    class MaskRule:
        """Rule for masking sensitive data in query results."""
        rx: re.Pattern  # Regular expression to match
        repl: Union[str, callable] # Replacement (str=literal, callable=dynamic)

    async def shutdown(self):
        """Gracefully shuts down the SQL tool and its resources."""
        logger.info("Shutting down SQLTool...")
        await self._conn_manager.shutdown()
        logger.info("SQLTool shutdown complete.")

    # ───────────────────────────── HELPER METHODS ─────────────────────────────
    
    def _next_audit_id(self) -> str:
        """Generate the next sequential audit ID."""
        self._AUDIT_ID += 1
        return f"a{self._AUDIT_ID:09d}"
    
    def _now(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    
    def _audit(
        self,
        *,
        tool_name: str, # e.g., 'manage_database', 'execute_sql'
        action: str, # e.g., 'connect', 'query', 'schema'
        connection_id: str | None,
        sql: str | None,
        tables: List[str] | None,
        row_count: int | None,
        success: bool,
        error: str | None,
        user_id: str | None,
        session_id: str | None,
        **extra_data: Any # For additional context
    ) -> None:
        """Record an audit trail entry."""
        log_entry = dict(
            audit_id=self._next_audit_id(),
            timestamp=self._now(),
            tool_name=tool_name,
            action=action,
            user_id=user_id,
            session_id=session_id,
            connection_id=connection_id,
            sql=sql,
            tables=tables,
            row_count=row_count,
            success=success,
            error=error,
            **extra_data # Merge extra data
        )
        self._AUDIT_LOG.append(log_entry)
        # Optional: Log to logger as well for immediate visibility
        logger.info(f"Audit[{log_entry['audit_id']}]: Tool={tool_name}, Action={action}, Conn={connection_id}, Success={success}" + (f", Error={error}" if error else ""))


    def update_acl(self, *, tables: Optional[List[str]] = None, columns: Optional[List[str]] = None) -> None:
        """Update the ACL lists for restricted tables and columns."""
        if tables is not None:
            self.RESTRICTED_TABLES.clear()
            self.RESTRICTED_TABLES.update({t.lower() for t in tables})
            logger.info(f"Updated restricted tables ACL: {self.RESTRICTED_TABLES}")
        if columns is not None:
            self.RESTRICTED_COLUMNS.clear()
            self.RESTRICTED_COLUMNS.update({c.lower() for c in columns})
            logger.info(f"Updated restricted columns ACL: {self.RESTRICTED_COLUMNS}")
    
    def _check_acl(self, sql: str) -> None:
        """Check if SQL contains any restricted tables or columns."""
        # Improved tokenization to handle quoted identifiers
        toks = set(re.findall(r'[\w$"\'.]+', sql.lower())) # Extract words, allow quotes etc.
        # Normalize tokens (remove quotes, handle schema.table)
        normalized_toks = set()
        for tok in toks:
            # Remove potential quotes
            tok_norm = tok.strip('"`\'[]')
            normalized_toks.add(tok_norm)
            # Add table part if schema.table format
            if '.' in tok_norm:
                 normalized_toks.add(tok_norm.split('.')[-1])

        restricted_tables_found = self.RESTRICTED_TABLES.intersection(normalized_toks)
        if restricted_tables_found:
            logger.warning(f"ACL Violation: Restricted table(s) found in query: {restricted_tables_found}")
            raise ToolError(f"Access denied: Query involves restricted table(s): {', '.join(restricted_tables_found)}", http_status_code=403)

        restricted_columns_found = self.RESTRICTED_COLUMNS.intersection(normalized_toks)
        if restricted_columns_found:
            logger.warning(f"ACL Violation: Restricted column(s) found in query: {restricted_columns_found}")
            raise ToolError(f"Access denied: Query involves restricted column(s): {', '.join(restricted_columns_found)}", http_status_code=403)

    def _resolve_conn(self, raw: str) -> str:
        """Resolve connection string, handling secret references."""
        if raw.startswith("secrets://"):
             secret_name = raw[10:]
             logger.info(f"Resolving secret reference: '{secret_name}'")
             return _pull_secret_from_sources(secret_name)
        return raw
    
    def _mask_val(self, v: Any) -> Any:
        """Apply masking rules to a single value."""
        if not isinstance(v, str) or not v: # Skip non-strings or empty strings
            return v
        for rule in self._RULES:
            if rule.rx.fullmatch(v):
                if callable(rule.repl):
                    try:
                        return rule.repl(v)
                    except Exception as e:
                         logger.error(f"Error applying dynamic mask rule {rule.rx.pattern}: {e}")
                         return "MASKING_ERROR" # Indicate error without revealing data
                else:
                    return rule.repl
        return v # No rule matched
    
    def _mask_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Apply masking rules to an entire row of data."""
        return {k: self._mask_val(v) for k, v in row.items()}
    
    def _driver_url(self, conn_str: str) -> Tuple[str, str]:
        """Convert generic connection string to dialect-specific async URL."""
        # Handle potential file path for SQLite
        is_file_path = "://" not in conn_str and (Path(conn_str).exists() or conn_str == ":memory:")
        
        if is_file_path:
            if conn_str == ":memory:":
                url_str = "sqlite+aiosqlite:///:memory:"
                logger.info("Using in-memory SQLite database.")
            else:
                # Use Path to handle platform-specific path formatting and ensure absolute path
                sqlite_path = Path(conn_str).expanduser().resolve()
                # Ensure the directory exists for file-based DBs if it doesn't exist
                if not sqlite_path.parent.exists():
                    try:
                        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created directory for SQLite DB: {sqlite_path.parent}")
                    except OSError as e:
                        raise ToolError(f"Failed to create directory for SQLite DB '{sqlite_path.parent}': {e}", http_status_code=500) from e
                url_str = f"sqlite+aiosqlite:///{sqlite_path}"
                logger.info(f"Using SQLite database file: {sqlite_path}")
            url = make_url(url_str)
            return str(url), "sqlite"
        else:
            # Assume it's a URL
             url_str = conn_str
             try:
                url = make_url(url_str)
             except Exception as e:
                  raise ToolInputError(f"Invalid connection string format: {e}", param_name="connection_string") from e

        # Map to async drivers
        drv = url.drivername.lower()
        
        if drv.startswith("sqlite"):
            # Already handled file paths, this covers sqlite:// URLs
            return str(url.set(drivername="sqlite+aiosqlite")), "sqlite"
        if drv.startswith("postgresql") or drv == "postgres":
            return str(url.set(drivername="postgresql+asyncpg")), "postgresql"
        if drv.startswith("mysql") or drv == "mariadb":
            # Ensure PyMySQL charset handling if needed
            query = dict(url.query)
            query.setdefault('charset', 'utf8mb4')
            return str(url.set(drivername="mysql+aiomysql", query=query)), "mysql"
        if drv.startswith("mssql") or drv == "sqlserver":
            # Check for required ODBC driver, provide guidance if missing
            # This check is basic; a proper check would involve trying to load pyodbc
            odbc_driver = url.query.get("driver")
            if not odbc_driver:
                 logger.warning("MSSQL connection string lacks 'driver' parameter. Ensure a valid ODBC driver (e.g., 'ODBC Driver 17 for SQL Server') is installed and specified.")
            return str(url.set(drivername="mssql+aioodbc")), "sqlserver"
        if drv.startswith("snowflake"):
             # Snowflake connector handles async internally via snowflake-sqlalchemy
             # No drivername change needed, but specify async execution strategy
             return str(url), "snowflake" # Keep original URL

        logger.error(f"Unsupported database dialect: {drv}")
        raise ToolInputError(f"Unsupported database dialect: '{drv}'. Supported: sqlite, postgresql, mysql, mssql, snowflake", param_name="connection_string")

    
    def _auto_pool(self, db_type: str) -> Dict[str, Any]:
        """Provide sensible default connection pool settings based on database type."""
        # Default settings good for many cases
        defaults = {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_recycle": 1800, # Recycle connections every 30 mins
            "pool_pre_ping": True, # Check connection health before checkout
            "pool_timeout": 30, # Wait 30s for a connection from pool
        }
        
        if db_type == "sqlite":
            # SQLite often doesn't benefit from pooling, especially file-based or :memory:
            # aiosqlite uses StaticPool which doesn't support pool_size and max_overflow
            return {"pool_pre_ping": True}  # Minimal settings for SQLite
        if db_type in {"postgresql", "mysql", "sqlserver"}:
             # More robust defaults for server-based DBs
            return {
                "pool_size": 10,
                "max_overflow": 20,
                "pool_recycle": 900, # Recycle more often (15 mins)
                "pool_pre_ping": True,
                "pool_timeout": 30,
            }
        if db_type == "snowflake":
             # Snowflake might have different scaling characteristics
            return {"pool_size": 5, "max_overflow": 5, "pool_pre_ping": True, "pool_timeout": 60} # Allow longer wait

        logger.warning(f"Using default pool settings for unknown db_type: {db_type}")
        return defaults
    
    async def _eng(self, cid: str) -> AsyncEngine:
        """Get engine by connection ID using the ConnectionManager."""
        # This now uses the ConnectionManager's method, which also updates access time
        return await self._conn_manager.get_connection(cid)
        
    def _tables(self, sql: str) -> List[str]:
        """Extract table names referenced in a SQL query. Handles quoted identifiers."""
        # Find potential table names after FROM, JOIN, UPDATE, INSERT INTO, DELETE FROM
        matches = self._TABLE_RX.findall(sql)
        
        # Normalize: remove quotes and schema prefixes if present
        tables = set()
        for match in matches:
             # Remove leading/trailing whitespace and quotes
             table = match.strip().strip('"`\'[]')
             # If it contains a dot, take the part after the last dot
             if '.' in table:
                 table = table.split('.')[-1].strip('"`\'[]')
             if table: # Avoid adding empty strings
                 tables.add(table)
                 
        return sorted(list(tables))

    
    def _check_safe(self, sql: str, read_only: bool = True) -> None:
        """
        Validate SQL for safety:
        1. Check against ACL restrictions
        2. Prevent prohibited statements (DROP, DELETE etc.)
        3. Enforce read-only mode if requested
        """
        self._check_acl(sql) # Check ACL first

        # Check for prohibited keywords at the start of the statement (allowing CTEs)
        normalized_sql = sql.lstrip().upper()
        if normalized_sql.startswith("WITH"):
             # Find the end of the CTE definition to check the main statement
             try:
                 # Basic CTE detection - might fail for complex nested CTEs or comments
                 main_statement_start = re.search(r"\)\s*(SELECT|INSERT|UPDATE|DELETE|MERGE)", normalized_sql, re.IGNORECASE | re.DOTALL)
                 if main_statement_start:
                     check_sql_part = main_statement_start.group(0).lstrip(') \t\n\r')
                 else:
                     check_sql_part = normalized_sql # Fallback if CTE parsing fails
             except Exception:
                 check_sql_part = normalized_sql # Fallback
        else:
            check_sql_part = normalized_sql

        if self.PROHIBITED.match(check_sql_part):
            prohibited_match = self.PROHIBITED.match(check_sql_part).group(1).strip()
            logger.warning(f"Security Violation: Prohibited statement detected: {prohibited_match}")
            raise ToolInputError(f"Prohibited statement type detected: '{prohibited_match}'", param_name="query")
        
        if read_only:
            # Allow common read-only keywords, including CTEs starting SELECT
            allowed_starts = ("SELECT", "SHOW", "EXPLAIN", "DESCRIBE", "PRAGMA")
            # Check the start of the main part of the query
            is_read_query = check_sql_part.startswith(allowed_starts)
            
            if not is_read_query:
                logger.warning(f"Security Violation: Write operation attempted in read-only mode: {sql[:100]}...")
                raise ToolInputError("Write operation attempted in read-only mode", param_name="query")
    
    async def _exec(
        self,
        eng: AsyncEngine,
        sql: str,
        params: Optional[Dict[str, Any]],
        *,
        limit: Optional[int], # Row limit to fetch initially
        tool_name: str, # For metrics/logging
        action_name: str, # For metrics/logging
        timeout: float = 30.0
    ) -> Tuple[List[str], List[Dict[str, Any]], int]:
        """
        Core async SQL executor with:
        - Timeouts and cancellation
        - Metrics collection
        - PII masking (applied after fetch)
        - Row limiting (applied during fetch)
        - Error handling
        """
        db_dialect = eng.dialect.name
        start_time = time.perf_counter()
        
        # Record metrics if prometheus is available
        if self.Q_CNT:
            self.Q_CNT.labels(tool=tool_name, action=action_name, db=db_dialect).inc()
        
        cols: List[str] = []
        rows_raw: List[Any] = []
        row_count: int = 0

        async def _run(conn: AsyncConnection):
            nonlocal cols, rows_raw, row_count
            # Use SQLAlchemy text() for parameter binding
            statement = text(sql)
            try:
                res = await conn.execute(statement, params or {})
            except (ProgrammingError, OperationalError) as db_err:
                # Catch specific DB errors here for better context
                logger.error(f"Database execution error ({type(db_err).__name__}) for {tool_name}/{action_name} on {db_dialect}: {db_err}")
                raise ToolError(f"Database Error: {db_err}", http_status_code=400) from db_err
            except SQLAlchemyError as sa_err:
                 logger.error(f"SQLAlchemy error ({type(sa_err).__name__}) for {tool_name}/{action_name} on {db_dialect}: {sa_err}")
                 raise ToolError(f"SQLAlchemy Error: {sa_err}", http_status_code=500) from sa_err


            if not res.cursor or not res.cursor.description: # Check if the query returned rows (e.g., SELECT vs INSERT)
                logger.debug(f"Query did not return rows or description. Action: {action_name}")
                row_count = res.rowcount if res.rowcount >= 0 else 0 # DML statements might have rowcount
                return [], [], row_count

            cols = list(res.keys())
            
            # Fetch rows with limit if specified
            try:
                # Handle SQLite dialect differently - it might not support async fetching properly
                if db_dialect == "sqlite":
                    # For SQLite, fetch all rows and then apply limit in Python
                    rows_raw = list(res.mappings())  # Get all rows as mappings
                    if limit is not None and limit >= 0:
                        rows_raw = rows_raw[:limit]  # Apply limit in Python
                else:
                    # For other databases, use native async fetch methods
                    if limit is not None and limit >= 0:
                        rows_raw = await res.fetchmany(limit)  # Use fetchmany for limit
                    else:
                        rows_raw = await res.fetchall()  # Fetch all if no limit
                
                row_count = len(rows_raw)  # Count based on fetched rows
            except Exception as fetch_err:
                 logger.error(f"Error fetching rows for {tool_name}/{action_name}: {fetch_err}")
                 raise ToolError(f"Error fetching results: {fetch_err}", http_status_code=500) from fetch_err

            # Apply masking *after* fetching
            # For SQLite, rows are already in mapping format
            if db_dialect == "sqlite":
                masked_rows = [self._mask_row(r) for r in rows_raw]
            else:
                masked_rows = [self._mask_row(r._mapping) for r in rows_raw]
            
            return cols, masked_rows, row_count # Return masked rows

        try:
            # Establish connection and execute within timeout
            async with eng.connect() as conn:
                # Run the execution logic within asyncio.wait_for
                cols, masked_rows, cnt = await asyncio.wait_for(
                    _run(conn), 
                    timeout=timeout
                )
                
                # Record latency metric on success
                latency = time.perf_counter() - start_time
                if self.Q_LAT:
                    self.Q_LAT.labels(tool=tool_name, action=action_name, db=db_dialect).observe(latency)
                logger.debug(f"Execution successful for {tool_name}/{action_name}. Latency: {latency:.3f}s, Rows fetched: {cnt}")

                return cols, masked_rows, cnt
                
        except asyncio.TimeoutError:
            logger.warning(f"Query timeout ({timeout}s) exceeded for {tool_name}/{action_name} on {db_dialect}.")
            raise ToolError(f"Query timed out after {timeout} seconds", http_status_code=504) from None
        except ToolError: # Re-raise ToolErrors directly
            raise
        except Exception as e:
            # Catch unexpected errors during connect or execution
            logger.error(f"Unexpected error during _exec for {tool_name}/{action_name}: {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred: {e}", http_status_code=500) from e
            
    def _export_rows(
        self,
        cols: List[str],
        rows: List[Dict[str, Any]],
        export_format: str,
        export_path: Optional[str] = None
    ) -> Tuple[Any | None, str | None]:
        """Export query results to pandas DataFrame, Excel file, or CSV file."""
        if not export_format:
            return None, None
            
        export_format = export_format.lower()

        if export_format not in ["pandas", "excel", "csv"]:
             raise ToolInputError(f"Unsupported export format: '{export_format}'. Use 'pandas', 'excel', or 'csv'.", param_name="export.format")

        if pd is None:
            raise ToolError(f"Pandas library is not installed. Cannot export to '{export_format}'.", http_status_code=501) # 501 Not Implemented

        # Create DataFrame
        try:
            df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
            logger.info(f"Created DataFrame with shape {df.shape} for export.")
        except Exception as e:
            logger.error(f"Error creating Pandas DataFrame: {e}", exc_info=True)
            raise ToolError(f"Failed to create DataFrame for export: {e}", http_status_code=500) from e

        # Handle export types
        if export_format == "pandas":
            logger.debug("Returning raw Pandas DataFrame.")
            return df, None # Return the DataFrame itself

        # For file exports, determine path
        if export_path:
             path_obj = Path(export_path).resolve()
             # Ensure parent directory exists
             try:
                 path_obj.parent.mkdir(parents=True, exist_ok=True)
             except OSError as e:
                 raise ToolError(f"Cannot create directory for export path '{path_obj.parent}': {e}", http_status_code=500) from e
             final_path = str(path_obj)
             logger.info(f"Using specified export path: {final_path}")
        else:
            # Create temporary file
            suffix = ".xlsx" if export_format == "excel" else ".csv"
            try:
                fd, final_path = tempfile.mkstemp(suffix=suffix, prefix=f"mcp_export_{export_format}_")
                os.close(fd) # Close file descriptor, pandas/xlsxwriter will handle the file
                logger.info(f"Created temporary file for export: {final_path}")
            except Exception as e:
                logger.error(f"Failed to create temporary file for export: {e}", exc_info=True)
                raise ToolError(f"Failed to create temporary file: {e}", http_status_code=500) from e

        # Write to file
        try:
            if export_format == "excel":
                # Consider using 'openpyxl' if xlsxwriter is not available or causes issues
                df.to_excel(final_path, index=False, engine="xlsxwriter") 
                logger.info(f"Exported data to Excel file: {final_path}")
            elif export_format == "csv":
                df.to_csv(final_path, index=False)
                logger.info(f"Exported data to CSV file: {final_path}")
            
            return None, final_path # Return None for DataFrame, and the path

        except Exception as e:
            logger.error(f"Error exporting DataFrame to {export_format} file '{final_path}': {e}", exc_info=True)
            # Clean up temp file if creation failed
            if not export_path and Path(final_path).exists():
                 try:
                     Path(final_path).unlink()
                 except OSError:
                     logger.warning(f"Could not clean up temporary export file: {final_path}")
            raise ToolError(f"Failed to export data to {export_format}: {e}", http_status_code=500) from e

    async def _validate_df(self, df: Any, schema: Any | None) -> None:
        """Validate DataFrame against Pandera schema."""
        if schema is None:
            logger.debug("No Pandera schema provided for validation.")
            return
        if pa is None:
            logger.warning("Pandera library not installed, skipping validation.")
            # Don't raise error, maybe validation is optional
            # raise ToolError("Pandera library not installed, cannot validate schema.", http_status_code=501)
            return
        if pd is None or not isinstance(df, pd.DataFrame):
             logger.warning("Pandas DataFrame not available for validation.")
             # raise ToolError("Pandas DataFrame required for Pandera validation.", http_status_code=400)
             return

        logger.info(f"Validating DataFrame (shape {df.shape}) against provided Pandera schema.")
        try:
            # Assume schema is a Pandera SchemaModel or DataFrameSchema
            schema.validate(df, lazy=True) # lazy=True collects all errors
            logger.info("Pandera validation successful.")
            # Note: validated_df might have coerced types, but we don't use it further here.
        except pa.errors.SchemaErrors as se:
            # Provide more detailed error feedback
            error_details = se.failure_cases.to_dict(orient="records") if hasattr(se.failure_cases, 'to_dict') else str(se.failure_cases)
            error_count = len(se.failure_cases) if hasattr(se.failure_cases, '__len__') else 'multiple'
            logger.warning(f"Pandera validation failed with {error_count} errors. Details: {error_details}")
            # Combine errors into a user-friendly message
            error_msg = f"Pandera validation failed ({error_count} errors):\n"
            for err in error_details[:5]: # Show first 5 errors
                 error_msg += f"- Column '{err.get('column', 'N/A')}': {err.get('check', 'N/A')} failed for index {err.get('index', 'N/A')}. Data: {err.get('failure_case', 'N/A')}\n"
            if error_count > 5:
                 error_msg += f"... and {error_count - 5} more errors."

            raise ToolError(f"Schema validation failed: {error_msg}", http_status_code=422, validation_errors=error_details) from se
        except Exception as e:
            logger.error(f"Unexpected error during Pandera validation: {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred during schema validation: {e}", http_status_code=500) from e
            
    async def _convert_nl_to_sql(self, connection_id: str, natural_language: str, confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """Helper method to convert natural language to SQL."""
        logger.info(f"Converting NL to SQL for connection {connection_id}. Query: '{natural_language[:100]}...'")
        eng = await self._eng(connection_id) # Uses ConnectionManager
        
        # Get schema fingerprint for LLM context
        async def _get_schema_fingerprint(conn: AsyncConnection) -> str:
            logger.debug("Generating schema fingerprint for NL->SQL...")
            try:
                # Use run_sync to work with SQLAlchemy's inspect function properly
                # Get an inspector object through run_sync to ensure proper sync context
                return await conn.run_sync(lambda sync_conn: _get_schema_fingerprint_sync(sync_conn))
            except Exception as e:
                logger.error(f"Error generating schema fingerprint: {e}", exc_info=True)
                return "Error: Could not retrieve schema." # Provide error feedback

        def _get_schema_fingerprint_sync(sync_conn):
            """Synchronous helper function to get schema fingerprint using SQLAlchemy inspection"""
            try:
                # Use SQLAlchemy's inspect directly on the sync connection
                sync_inspector = sa_inspect(sync_conn)
                
                tbls = []
                schema_names = sync_inspector.get_schema_names() # Get all schemas
                default_schema = sync_inspector.default_schema_name
                schemas_to_inspect = [default_schema] + [s for s in schema_names if s != default_schema]

                for schema_name in schemas_to_inspect:
                    prefix = f"{schema_name}." if schema_name and schema_name != default_schema else ""
                    for t in sync_inspector.get_table_names(schema=schema_name):
                        try:
                            cols = sync_inspector.get_columns(t, schema=schema_name)
                            col_defs = [f"{c['name']}:{str(c['type']).split('(')[0]}" for c in cols] # Simplified type
                            tbls.append(f"{prefix}{t}({','.join(col_defs)})")
                        except Exception as col_err:
                             logger.warning(f"Could not get columns for table {prefix}{t}: {col_err}")
                             tbls.append(f"{prefix}{t}(...)") # Indicate columns couldn't be fetched

                fp = "; ".join(sorted(tbls))
                logger.debug(f"Schema fingerprint generated: {fp[:200]}...")
                if not fp: 
                    logger.warning("Schema fingerprint generation resulted in empty string.")
                    return "Error: Could not retrieve schema." # Provide error feedback
                return fp
            except Exception as e:
                logger.error(f"Error in _get_schema_fingerprint_sync: {e}", exc_info=True)
                return "Error: Could not retrieve schema."

        # Use the engine's connect context manager
        async with eng.connect() as conn:
            schema_fingerprint = await _get_schema_fingerprint(conn)

        # Generate SQL using LLM
        # Improved prompt engineering
        prompt = (
            "You are a highly specialized AI assistant that translates natural language questions into SQL queries.\n"
            "You must adhere STRICTLY to the following rules:\n"
            "1. Generate only a SINGLE, executable SQL query for the given database schema and question.\n"
            "2. Use the exact table and column names provided in the schema fingerprint.\n"
            "3. Do NOT generate any explanatory text, comments, or markdown formatting.\n"
            "4. The output MUST be a valid JSON object containing two keys: 'sql' (the generated SQL query as a string) and 'confidence' (a float between 0.0 and 1.0 indicating your confidence in the generated SQL).\n"
            "5. If the question cannot be answered from the schema or is ambiguous, set confidence to 0.0 and provide a minimal, safe query like 'SELECT 1;' in the 'sql' field.\n"
            "6. Prioritize safety: Avoid generating queries that could modify data (UPDATE, INSERT, DELETE, DROP, etc.) unless explicitly and clearly requested, and even then, be cautious.\n\n"
            f"Database Schema Fingerprint:\n```\n{schema_fingerprint}\n```\n\n"
            f"Natural Language Question:\n```\n{natural_language}\n```\n\n"
            "JSON Output:"
        )
        
        try:
             logger.debug("Sending prompt to LLM for NL->SQL conversion.")
             llm_response = await generate_completion(prompt, max_tokens=300, temperature=0.2) # More tokens, lower temp
             logger.debug(f"LLM Response received: {llm_response}")
        except Exception as llm_err:
             logger.error(f"LLM completion failed for NL->SQL: {llm_err}", exc_info=True)
             raise ToolError(f"Failed to get response from LLM: {llm_err}", http_status_code=502) from llm_err # 502 Bad Gateway

        # Parse and validate LLM response
        try:
            # Check if response is already a dictionary (direct JSON mode)
            if isinstance(llm_response, dict):
                # Handle nested JSON in OpenAI response format
                if 'text' in llm_response and isinstance(llm_response['text'], str):
                    # Extract the JSON from the 'text' field
                    try:
                        data = json.loads(llm_response['text'])
                    except json.JSONDecodeError as e:
                        # Try to find JSON block if the text isn't strictly JSON
                        json_match = re.search(r'\{.*\}', llm_response['text'], re.DOTALL)
                        if not json_match:
                            raise ValueError("No JSON object found in the LLM text response.") from e
                        data = json.loads(json_match.group(0))
                else:
                    # Use the response directly if it's not in the nested format
                    data = llm_response
            else:
                # Original handling for string responses
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON object found in the LLM response.")
                data = json.loads(json_match.group(0))
            
            if not isinstance(data, dict) or "sql" not in data or "confidence" not in data:
                raise ValueError("LLM response JSON is missing required keys ('sql', 'confidence').")

            sql = data["sql"]
            conf = float(data["confidence"])

            if not isinstance(sql, str) or not (0.0 <= conf <= 1.0):
                 raise ValueError("LLM response has invalid types for 'sql' or 'confidence'.")

            logger.info(f"LLM generated SQL with confidence {conf:.2f}: {sql[:150]}...")

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            error_detail = f"LLM returned invalid or malformed JSON: {e}. Response: '{str(llm_response)[:200]}...'"
            logger.error(error_detail)
            raise ToolError(error_detail, http_status_code=500) from e # 500 Internal Server Error (LLM failed contract)

        # Check confidence threshold
        if conf < confidence_threshold:
            low_conf_msg = f"LLM confidence ({conf:.2f}) is below the required threshold ({confidence_threshold}). NL Query: '{natural_language}'"
            logger.warning(low_conf_msg)
            # Return a specific error or the low-confidence SQL based on desired behavior
            # Option 1: Raise error
            raise ToolError(low_conf_msg, http_status_code=400, generated_sql=sql, confidence=conf) from None
            # Option 2: Return low-confidence result (caller decides)
            # return {"sql": sql, "confidence": conf, "low_confidence_warning": low_conf_msg}

        # Perform safety and basic validation on the generated SQL
        try:
            # Enforce read-only check even more strictly for generated SQL
            self._check_safe(sql, read_only=True) 
            
            # Basic structural check: does it look like SQL?
            if not sql.upper().lstrip().startswith(("SELECT", "WITH")):
                 raise ToolError("Generated query does not appear to be a valid SELECT statement.", http_status_code=400)

            # Check if it references *any* table from the fingerprint (can be improved)
            fingerprint_tables = { re.split(r'\(', tbl)[0].split('.')[-1] for tbl in schema_fingerprint.split('; ') if '(' in tbl }
            sql_tokens = set(re.findall(r'[\w$"\'.]+', sql.lower()))
            normalized_sql_tables = set()
            for tok in sql_tokens:
                tok_norm = tok.strip('"`\'[]')
                if '.' in tok_norm:
                    normalized_sql_tables.add(tok_norm.split('.')[-1])
                else:
                    normalized_sql_tables.add(tok_norm)

            if not fingerprint_tables.intersection(normalized_sql_tables) and schema_fingerprint != "Error: Could not retrieve schema.":
                # Allow queries like 'SELECT 1' if schema couldn't be fetched or is empty
                logger.warning(f"Generated SQL '{sql[:100]}...' does not seem to reference known tables from fingerprint: {fingerprint_tables}")
                # Decide whether to raise an error or allow simple queries like 'SELECT CURRENT_TIMESTAMP'
                # For now, allow it but log warning. Can be made stricter.
                # raise ToolError("Generated SQL does not reference any known tables from the database schema.", http_status_code=400)

        except ToolInputError as safety_err:
            logger.error(f"Generated SQL failed safety check: {safety_err}. SQL: {sql}")
            raise ToolError(f"Generated SQL failed validation: {safety_err}", http_status_code=400, generated_sql=sql, confidence=conf) from safety_err

        return {"sql": sql, "confidence": conf}


    # ───────────────────────────── PUBLIC API TOOLS ─────────────────────────────
    
    # ------------------------------ CONNECTION MANAGEMENT --------------------------

    @tool()
    @with_tool_metrics
    @with_error_handling
    async def manage_database(
        self,
        action: str,
        connection_string: Optional[str] = None,
        connection_id: Optional[str] = None,
        echo: bool = False, # For connect action
        user_id: Optional[str] = None, # For auditing
        session_id: Optional[str] = None, # For auditing
        **options: Any # Pass-through for create_engine or action-specific opts
    ) -> Dict[str, Any]:
        """
        Unified database connection management tool.
        
        Args:
            action: The action to perform: "connect", "disconnect", "test", or "status".
            connection_string: Database connection string or secrets:// reference. (Required for "connect").
            connection_id: An existing connection ID (Required for "disconnect", "test"). Can be provided for "connect" to suggest an ID.
            echo: Enable SQLAlchemy engine logging (For "connect" action, default: False).
            user_id: Optional user identifier for audit logging.
            session_id: Optional session identifier for audit logging.
            **options: Additional options:
                - For "connect": Passed directly to SQLAlchemy's `create_async_engine`.
                - Can include custom audit context.
                
        Returns:
            Dict with action results and metadata. Varies based on action.
            - connect: {"action": "connect", "connection_id": str, "database_type": str, "success": True}
            - disconnect: {"action": "disconnect", "connection_id": str, "success": bool}
            - test: {"action": "test", "connection_id": str, "response_time": float, "version": str, "success": True}
            - status: {"action": "status", "active_connections": int, "connections": Dict[str, Dict], "success": True}
        """
        tool_name = "manage_database"
        db_dialect = "unknown" # Default
        audit_extras = {k: v for k, v in options.items() if k not in ['echo']} # Capture extra opts for audit

        try:
            if action == "connect":
                if not connection_string:
                    raise ToolInputError("connection_string is required for the 'connect' action", param_name="connection_string")
                    
                cid = connection_id or str(uuid.uuid4())
                logger.info(f"Attempting to connect with connection_id: {cid}")

                # Resolve secrets and get dialect-specific async URL
                resolved_conn_str = self._resolve_conn(connection_string)
                url, db_type = self._driver_url(resolved_conn_str)
                db_dialect = db_type # For metrics/logging
                
                # Create engine with auto-tuned pool settings + overrides from options
                engine_opts = {**self._auto_pool(db_type), **options}
                logger.debug(f"Creating engine for {db_type} with options: { {k:v for k,v in engine_opts.items() if k != 'password'} }") # Log options except password
                
                # Explicitly handle 'connect_args' which might be needed for some drivers (e.g., SQLite flags)
                connect_args = engine_opts.pop('connect_args', {}) 

                # Handle Snowflake-specific execution strategy
                execution_options = {}
                if db_type == "snowflake":
                    execution_options["async_execution"] = True # Necessary for snowflake-sqlalchemy async

                eng = create_async_engine(
                    url, 
                    echo=echo, 
                    connect_args=connect_args, 
                    execution_options=execution_options,
                    **engine_opts # Pass remaining options
                )
                
                # Test connection immediately with a short timeout
                try:
                    # Use a simple, universal query; avoid SELECT 1 for dialects where it might fail in edge cases
                    test_sql = "SELECT CURRENT_TIMESTAMP" if db_type != 'sqlite' else 'SELECT 1' # More portable
                    await self._exec(eng, test_sql, None, limit=1, tool_name=tool_name, action_name="connect_test", timeout=15)
                    logger.info(f"Connection test successful for {cid} ({db_type}).")
                except ToolError as test_err:
                     logger.error(f"Connection test failed for {cid} ({db_type}): {test_err}")
                     # Attempt to dispose the partially created engine
                     await eng.dispose()
                     raise ToolError(f"Connection test failed: {test_err}", http_status_code=400) from test_err
                except Exception as e:
                     logger.error(f"Unexpected error during connection test for {cid} ({db_type}): {e}", exc_info=True)
                     await eng.dispose()
                     raise ToolError(f"Unexpected error during connection test: {e}", http_status_code=500) from e

                # Add successfully tested engine to connection manager
                await self._conn_manager.add_connection(cid, eng)
                
                # Audit successful connection
                self._audit(
                    tool_name=tool_name,
                    action="connect",
                    connection_id=cid,
                    sql=None,
                    tables=None,
                    row_count=None,
                    success=True,
                    error=None,
                    user_id=user_id,
                    session_id=session_id,
                    database_type=db_type,
                    echo=echo,
                    **audit_extras
                )
                
                return {
                    "action": "connect",
                    "connection_id": cid, 
                    "database_type": db_type, 
                    "success": True
                }
                
            elif action == "disconnect":
                if not connection_id:
                    raise ToolInputError("connection_id is required for the 'disconnect' action", param_name="connection_id")
                    
                logger.info(f"Attempting to disconnect connection_id: {connection_id}")
                # Retrieve dialect before closing if possible, for logging consistency
                try:
                    engine_to_close = await self._conn_manager.get_connection(connection_id)
                    db_dialect = engine_to_close.dialect.name
                except ToolInputError:
                    logger.warning(f"Disconnect requested for unknown connection_id: {connection_id}")
                    # Audit failure to find connection
                    self._audit(
                        tool_name=tool_name, action="disconnect", connection_id=connection_id, sql=None, tables=None, row_count=None,
                        success=False, error="Connection ID not found", user_id=user_id, session_id=session_id, **audit_extras
                    )
                    return {"action": "disconnect", "connection_id": connection_id, "success": False, "message": "Connection ID not found"}
                except Exception as e:
                    logger.error(f"Error retrieving engine for disconnect ({connection_id}): {e}")
                    # Audit potential error state
                    self._audit(
                        tool_name=tool_name, action="disconnect", connection_id=connection_id, sql=None, tables=None, row_count=None,
                        success=False, error=f"Error retrieving engine: {e}", user_id=user_id, session_id=session_id, **audit_extras
                    )
                    # Proceed to attempt close anyway
                    pass 

                success = await self._conn_manager.close_connection(connection_id)
                
                # Audit disconnect attempt
                self._audit(
                    tool_name=tool_name,
                    action="disconnect",
                    connection_id=connection_id,
                    sql=None,
                    tables=None,
                    row_count=None,
                    success=success,
                    error=None if success else "Failed to close or already closed",
                    user_id=user_id,
                    session_id=session_id,
                    database_type=db_dialect, # Log dialect if known
                    **audit_extras
                )
                
                return {
                    "action": "disconnect",
                    "connection_id": connection_id, 
                    "success": success
                }
                
            elif action == "test":
                if not connection_id:
                    raise ToolInputError("connection_id is required for the 'test' action", param_name="connection_id")
                
                logger.info(f"Testing connection_id: {connection_id}")
                eng = await self._eng(connection_id) # Uses manager, updates access time
                db_dialect = eng.dialect.name
                t0 = time.perf_counter()
                
                # Get database version using appropriate SQL
                if db_dialect == "sqlite":
                    vsql = "SELECT sqlite_version()"
                elif db_dialect == "snowflake":
                     vsql = "SELECT CURRENT_VERSION()" # Snowflake specific
                else:
                    vsql = "SELECT version()" # Common for Postgres, MySQL, SQL Server (often works)

                cols, rows, _ = await self._exec(eng, vsql, None, limit=1, tool_name=tool_name, action_name="test", timeout=10)
                latency = time.perf_counter() - t0

                version_info = "N/A"
                if rows and cols:
                     version_info = rows[0].get(cols[0], "N/A")

                logger.info(f"Connection test successful for {connection_id}. Version: {version_info}, Latency: {latency:.3f}s")

                # No specific audit for 'test' usually needed unless tracking usage patterns
                # self._audit(tool_name=tool_name, action="test", connection_id=connection_id, ...)
                
                return {
                    "action": "test",
                    "connection_id": connection_id,
                    "response_time_seconds": round(latency, 3),
                    "version": version_info,
                    "database_type": db_dialect,
                    "success": True
                }
                
            elif action == "status":
                logger.info("Retrieving connection status.")
                connections_info = {}
                current_time = time.time()
                # Iterate safely over a copy of items
                for conn_id, (eng, last_access) in list(self._conn_manager.connections.items()):
                    try:
                        # Sanitize URL display
                        url_display = str(eng.url)
                        parsed_url = make_url(url_display)
                        if parsed_url.password:
                            url_display = str(parsed_url.set(password="***"))

                        connections_info[conn_id] = {
                            "url_summary": url_display, # Show URL without password
                            "dialect": eng.dialect.name,
                            "last_accessed": dt.datetime.fromtimestamp(last_access).isoformat(),
                            "idle_time_seconds": round(current_time - last_access, 1),
                            # Potential future enhancements: pool stats if accessible
                            # "pool_stats": await eng.pool.status() # Needs check for async pool status method
                        }
                        db_dialect = eng.dialect.name # Capture for potential metrics later if needed
                    except Exception as status_err:
                        logger.error(f"Error retrieving status for connection {conn_id}: {status_err}")
                        connections_info[conn_id] = {"error": str(status_err)}
                        
                # No specific audit for status view needed typically
                
                return {
                    "action": "status",
                    "active_connections_count": len(connections_info),
                    "connections": connections_info,
                    "cleanup_interval_seconds": self._conn_manager.cleanup_interval,
                    "success": True
                }
                
            else:
                logger.error(f"Invalid action specified for manage_database: {action}")
                raise ToolInputError(
                    f"Unknown action: '{action}'. Valid actions: connect, disconnect, test, status", 
                    param_name="action"
                )

        except ToolInputError as tie:
            # Audit failures due to bad input
            self._audit(tool_name=tool_name, action=action, connection_id=connection_id, sql=None, tables=None, row_count=None,
                        success=False, error=str(tie), user_id=user_id, session_id=session_id, database_type=db_dialect, **audit_extras)
            raise tie # Re-raise
        except ToolError as te:
             # Audit operational failures
            self._audit(tool_name=tool_name, action=action, connection_id=connection_id, sql=None, tables=None, row_count=None,
                        success=False, error=str(te), user_id=user_id, session_id=session_id, database_type=db_dialect, **audit_extras)
            raise te # Re-raise
        except Exception as e:
            # Audit unexpected errors
            logger.error(f"Unexpected error in manage_database (action: {action}): {e}", exc_info=True)
            self._audit(tool_name=tool_name, action=action, connection_id=connection_id, sql=None, tables=None, row_count=None,
                        success=False, error=f"Unexpected error: {e}", user_id=user_id, session_id=session_id, database_type=db_dialect, **audit_extras)
            raise ToolError(f"An unexpected error occurred in manage_database: {e}", http_status_code=500) from e


    # ------------------------------ QUERY EXECUTION ------------------------------

    @tool()
    @with_tool_metrics
    @with_error_handling
    async def execute_sql(
        self,
        connection_id: str,
        query: Optional[str] = None,
        natural_language: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        pagination: Optional[Dict[str, int]] = None, # {"page": int, "page_size": int}
        read_only: bool = True,
        export: Optional[Dict[str, Any]] = None, # {"format": "pandas|excel|csv", "path": str?}
        timeout: float = 60.0, # Increased default timeout
        validate_schema: Optional[Any] = None, # Optional Pandera schema
        max_rows: Optional[int] = 1000, # Default row limit for non-paginated queries
        confidence_threshold: float = 0.6, # For NL->SQL
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **options: Any # For future extensions or custom audit data
    ) -> Dict[str, Any]:
        """
        Unified SQL query execution tool supporting:
        - Direct SQL queries
        - Parameterized queries (using :key syntax)
        - Natural language to SQL conversion (generates and executes SELECT)
        - Server-side pagination
        - Export results to Pandas DataFrame, Excel, or CSV
        - Optional result validation using Pandera schemas
        
        Args:
            connection_id: The active connection ID to use.
            query: The SQL query string to execute (required unless `natural_language` is provided).
            natural_language: A natural language question to convert into a SQL query (if `query` is not provided).
            parameters: A dictionary of parameters to bind to the query (e.g., `{"name": "value"}`).
            pagination: Settings for server-side pagination: `{"page": <page_number>, "page_size": <rows_per_page>}` (1-based page).
            read_only: If True (default), actively prevents modification statements (UPDATE, INSERT, DELETE, etc.).
            export: Settings to export results: `{"format": "pandas|excel|csv", "path": "/optional/path/to/file"}`. If path is omitted for excel/csv, a temporary file is created.
            timeout: Maximum execution time in seconds (default: 60).
            validate_schema: A Pandera DataFrameSchema or SchemaModel to validate the results against (requires Pandas/Pandera).
            max_rows: Maximum rows to return for non-paginated queries (default: 1000). Set to `None` for potentially unlimited rows (use with caution).
            confidence_threshold: Minimum confidence score (0.0-1.0) required from the LLM for NL-to-SQL conversion (default: 0.6).
            user_id: Optional user identifier for audit logging.
            session_id: Optional session identifier for audit logging.
            **options: Additional options for auditing or future features.
            
        Returns:
            A dictionary containing the results and metadata:
            - columns: List[str] - Names of the returned columns.
            - rows: List[Dict] - The result rows (list of dictionaries). Limited by `max_rows` or `pagination`.
            - row_count: int - The number of rows returned in this response.
            - truncated: bool - True if `max_rows` was reached for a non-paginated query.
            - pagination: Optional[Dict] - Info if pagination was used: `{"page": int, "page_size": int, "has_next_page": bool, "has_previous_page": bool}`.
            - dataframe: Optional[pd.DataFrame] - Included if `export={"format": "pandas"}`.
            - excel_path: Optional[str] - Included if `export={"format": "excel"}`.
            - csv_path: Optional[str] - Included if `export={"format": "csv"}`.
            - generated_sql: Optional[str] - The SQL query generated from natural language, if applicable.
            - confidence: Optional[float] - The confidence score of the generated SQL, if applicable.
            - success: bool - Always True if the tool execution finished without critical errors (validation errors might still occur).
            - validation_errors: Optional[List[Dict]] - Included if Pandera validation fails.
        """
        tool_name = "execute_sql"
        action_name = "query" # Default action for metrics/logging
        original_query = query # Keep original for logging if NL is used
        generated_sql = None
        confidence = None
        final_query: str
        final_params = parameters or {}
        result: Dict[str, Any] = {}
        tables: List[str] = []
        audit_extras = {**options} # Capture extra opts for audit

        try:
            # 1. Determine the SQL query to execute (Direct, NL->SQL)
            if natural_language and not query:
                action_name = "nl_to_sql_exec"
                logger.info(f"Received natural language query for connection {connection_id}: '{natural_language[:100]}...'")
                try:
                    nl_result = await self._convert_nl_to_sql(
                        connection_id, 
                        natural_language, 
                        confidence_threshold=confidence_threshold
                    )
                    final_query = nl_result["sql"]
                    generated_sql = final_query
                    confidence = nl_result["confidence"]
                    original_query = natural_language # Log the NL query instead
                    audit_extras["generated_sql"] = generated_sql
                    audit_extras["confidence"] = confidence
                    logger.info(f"Successfully converted NL to SQL (Confidence: {confidence:.2f}): {final_query[:150]}...")
                    # NL->SQL implies read-only, safety check already done in _convert_nl_to_sql
                    read_only = True # Ensure read-only for generated SQL
                except ToolError as nl_err:
                    # Audit NL->SQL failure
                    self._audit(tool_name=tool_name, action="nl_to_sql_fail", connection_id=connection_id, sql=natural_language, tables=None, row_count=None,
                                success=False, error=str(nl_err), user_id=user_id, session_id=session_id, **audit_extras)
                    raise nl_err # Re-raise the specific error
            elif query:
                final_query = query
                logger.info(f"Executing direct SQL query on {connection_id}: {final_query[:150]}...")
            else:
                raise ToolInputError("Either 'query' or 'natural_language' must be provided.", param_name="query/natural_language")

            # 2. Check safety (unless already checked during NL->SQL generation)
            # Safety check is critical, especially for direct queries or if NL->SQL check needs re-verification
            self._check_safe(final_query, read_only)
            
            # Extract table names *after* safety check
            tables = self._tables(final_query)
            logger.debug(f"Query targets tables: {tables}")

            # 3. Get database engine
            eng = await self._eng(connection_id) # Uses manager, updates access time

            # 4. Handle Pagination or Standard Execution
            if pagination:
                action_name = "query_paginated"
                page = pagination.get("page", 1)
                page_size = pagination.get("page_size", 100)
                
                if not isinstance(page, int) or page < 1:
                    raise ToolInputError("Pagination 'page' must be a positive integer.", param_name="pagination.page")
                if not isinstance(page_size, int) or page_size < 1:
                     raise ToolInputError("Pagination 'page_size' must be a positive integer.", param_name="pagination.page_size")

                offset = (page - 1) * page_size
                
                # Add LIMIT/OFFSET - Syntax varies slightly across dialects
                # Basic syntax works for MySQL, PostgreSQL, SQLite
                # SQL Server uses OFFSET FETCH NEXT
                db_dialect = eng.dialect.name
                if db_dialect == "sqlserver":
                    # Ensure ORDER BY exists for OFFSET FETCH (SQL Server requirement)
                    if "order by" not in final_query.lower():
                         raise ToolInputError("SQL Server pagination requires an ORDER BY clause in the query.", param_name="query")
                    paginated_query = f"{final_query} OFFSET :_page_offset ROWS FETCH NEXT :_page_size ROWS ONLY"
                elif db_dialect == 'oracle':
                     # Oracle 12c+ syntax
                     paginated_query = f"{final_query} OFFSET :_page_offset ROWS FETCH NEXT :_page_size ROWS ONLY"
                else: # Assume LIMIT/OFFSET syntax (PostgreSQL, MySQL, SQLite, Snowflake)
                     paginated_query = f"{final_query} LIMIT :_page_size OFFSET :_page_offset"

                # Parameters for pagination (fetch one extra row to check for next page)
                paginated_params = {**final_params, "_page_size": page_size + 1, "_page_offset": offset}
                
                logger.debug(f"Executing paginated query (Page: {page}, Size: {page_size}): {paginated_query}")
                
                # Execute paginated query (no internal limit needed)
                cols, rows, _ = await self._exec(
                    eng, paginated_query, paginated_params, 
                    limit=None, # Limit applied via SQL
                    tool_name=tool_name, action_name=action_name, timeout=timeout
                )
                
                has_next_page = len(rows) > page_size
                # Return only the requested page size
                returned_rows = rows[:page_size] 
                
                result = {
                    "columns": cols,
                    "rows": returned_rows,
                    "row_count": len(returned_rows),
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "has_next_page": has_next_page,
                        "has_previous_page": page > 1,
                    },
                    "truncated": False, # Not applicable in the same way for pagination
                    "success": True
                }

            else: # Standard execution (no pagination)
                action_name = "query_standard"
                # Use max_rows for limiting, fetch one extra to detect truncation
                fetch_limit = (max_rows + 1) if max_rows is not None and max_rows >= 0 else None
                
                logger.debug(f"Executing standard query (Max rows: {max_rows}): {final_query[:150]}...")

                cols, rows, fetched_count = await self._exec(
                    eng, final_query, final_params, 
                    limit=fetch_limit, 
                    tool_name=tool_name, action_name=action_name, timeout=timeout
                )
                
                truncated = fetch_limit is not None and fetched_count >= fetch_limit
                # Return only up to max_rows
                returned_rows = rows[:max_rows] if max_rows is not None and max_rows >= 0 else rows
                
                result = {
                    "columns": cols,
                    "rows": returned_rows,
                    "row_count": len(returned_rows),
                    "truncated": truncated,
                    "success": True
                }

            # Add NL->SQL info if applicable
            if generated_sql:
                 result["generated_sql"] = generated_sql
                 result["confidence"] = confidence

            # 5. Handle Validation (if schema provided)
            if validate_schema:
                 # Requires Pandas, _validate_df checks for it
                 temp_df = None
                 if pd:
                     try:
                         # Create DF from the *returned* rows for validation
                         temp_df = pd.DataFrame(result["rows"], columns=result["columns"]) if result["rows"] else pd.DataFrame(columns=result["columns"])
                     except Exception as df_err:
                          logger.error(f"Error creating DataFrame for validation: {df_err}")
                          # Decide if this should be a fatal error or just a warning
                          result['validation_status'] = f'Failed to create DataFrame: {df_err}'
                     
                     if temp_df is not None:
                         try:
                             await self._validate_df(temp_df, validate_schema)
                             result['validation_status'] = 'success'
                             logger.info("Pandera validation passed.")
                         except ToolError as val_err:
                             # Validation failed, capture details but don't stop execution
                             logger.warning(f"Pandera validation failed: {val_err}")
                             result['validation_status'] = 'failed'
                             result['validation_errors'] = getattr(val_err, 'validation_errors', str(val_err))
                             # Optionally re-raise if validation failure should halt the process
                             # raise val_err 
                 else:
                     logger.warning("Pandas not installed, skipping Pandera validation.")
                     result['validation_status'] = 'skipped (Pandas not installed)'


            # 6. Handle Export (if requested)
            export_path = None
            dataframe = None
            if export and export.get("format"):
                export_format = export["format"].lower()
                req_path = export.get("path")
                logger.info(f"Export requested: Format={export_format}, Path={req_path or 'Temporary'}")
                
                # _export_rows creates DF internally if needed and handles file I/O
                try:
                    dataframe, export_path = self._export_rows(
                        result["columns"], 
                        result["rows"], # Export the potentially truncated/paginated rows
                        export_format, 
                        req_path
                    )
                    if dataframe is not None:
                         result["dataframe"] = dataframe # For pandas export
                    if export_path:
                         # Add path based on format
                         result[f"{export_format}_path"] = export_path
                    logger.info(f"Export successful. Format: {export_format}, Path: {export_path or 'In-memory DataFrame'}")
                    audit_extras["export_format"] = export_format
                    audit_extras["export_path"] = export_path

                except (ToolError, ToolInputError) as export_err:
                     logger.error(f"Export failed: {export_err}")
                     # Decide if export failure is critical. Here, we'll just log and maybe add to result.
                     result["export_status"] = f"Failed: {export_err}"
                     # Optionally raise export_err if it should halt execution

            # 7. Audit successful execution (even if validation/export had issues)
            self._audit(
                tool_name=tool_name,
                action=action_name,
                connection_id=connection_id,
                sql=original_query if original_query else final_query, # Log original NL or the SQL
                tables=tables,
                row_count=result.get("row_count", 0),
                success=True, # Tool execution succeeded
                error=None,
                user_id=user_id,
                session_id=session_id,
                read_only=read_only,
                pagination_used=bool(pagination),
                validation_status=result.get('validation_status'),
                export_status=result.get('export_status', 'not requested'),
                **audit_extras
            )

            return result

        except ToolInputError as tie:
            # Audit failures due to bad input
            self._audit(tool_name=tool_name, action=action_name + "_fail", connection_id=connection_id, sql=original_query or query, tables=tables, row_count=0,
                        success=False, error=str(tie), user_id=user_id, session_id=session_id, **audit_extras)
            raise tie
        except ToolError as te:
            # Audit operational failures (DB errors, timeouts, safety violations, etc.)
            self._audit(tool_name=tool_name, action=action_name + "_fail", connection_id=connection_id, sql=original_query or query, tables=tables, row_count=0,
                        success=False, error=str(te), user_id=user_id, session_id=session_id, **audit_extras)
            raise te
        except Exception as e:
             # Audit unexpected errors
            logger.error(f"Unexpected error in execute_sql (action: {action_name}): {e}", exc_info=True)
            self._audit(tool_name=tool_name, action=action_name + "_fail", connection_id=connection_id, sql=original_query or query, tables=tables, row_count=0,
                        success=False, error=f"Unexpected error: {e}", user_id=user_id, session_id=session_id, **audit_extras)
            raise ToolError(f"An unexpected error occurred during SQL execution: {e}", http_status_code=500) from e


    # ------------------------------ SCHEMA DISCOVERY ------------------------------

    # Apply caching carefully to schema exploration - TTL should be configurable or based on use case
    # @with_cache(ttl=3600) # Example: Cache schema results for 1 hour
    @tool()
    @with_tool_metrics 
    @with_error_handling
    async def explore_database(
        self,
        connection_id: str,
        action: str, # "schema", "table", "column", "relationships", "documentation"
        table_name: Optional[str] = None, # Required for table, column, relationships
        column_name: Optional[str] = None, # Required for column
        schema_name: Optional[str] = None, # Optional schema filter/context
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **options: Any # Action-specific options
    ) -> Dict[str, Any]:
        """
        Unified database schema exploration and documentation tool.
        
        Args:
            connection_id: The active connection ID to use.
            action: The exploration action to perform:
                - "schema": Get comprehensive database schema (tables, views, columns, keys, indexes).
                - "table": Get detailed information for a specific table. (Requires `table_name`).
                - "column": Analyze statistics for a specific column. (Requires `table_name`, `column_name`).
                - "relationships": Find tables related to a specific table via foreign keys. (Requires `table_name`).
                - "documentation": Generate database schema documentation.
            table_name: The name of the table for 'table', 'column', and 'relationships' actions.
            column_name: The name of the column for the 'column' action.
            schema_name: Optional database schema/namespace to focus the exploration on.
            user_id: Optional user identifier for audit logging.
            session_id: Optional session identifier for audit logging.
            **options: Additional options specific to the action:
                - For "schema":
                    - `include_indexes` (bool, default: True): Include index details.
                    - `include_foreign_keys` (bool, default: True): Include foreign key details.
                    - `detailed` (bool, default: False): Include extra column details like comments, defaults.
                - For "table":
                    - `include_sample_data` (bool, default: False): Fetch sample rows.
                    - `sample_size` (int, default: 5): Number of sample rows to fetch.
                    - `include_statistics` (bool, default: False): Compute basic column stats (nulls, distinct count).
                - For "column":
                    - `histogram` (bool, default: False): Generate a frequency or numeric histogram.
                    - `num_buckets` (int, default: 10): Number of buckets for the histogram.
                - For "relationships":
                    - `depth` (int, default: 1): Max recursion depth for finding related tables (1-5).
                - For "documentation":
                    - `output_format` (str, default: "markdown"): Format for documentation ("markdown" or "json").
                    - `include_indexes`, `include_foreign_keys`: Control details in documentation.

        Returns:
            A dictionary containing the exploration results, structure depends on the 'action'. 
            All responses include `{"action": str, "success": True}` on success.
        """
        tool_name = "explore_database"
        audit_extras = {**options, "table_name": table_name, "column_name": column_name, "schema_name": schema_name} # Capture args/opts for audit
        
        try:
            logger.info(f"Exploring database for connection {connection_id}. Action: {action}, Table: {table_name}, Column: {column_name}, Schema: {schema_name}")
            eng = await self._eng(connection_id) # Uses manager, updates access time
            db_dialect = eng.dialect.name
            audit_extras['database_type'] = db_dialect

            # Use a single connection for potentially multiple inspection calls within an action
            async with eng.connect() as conn:
                
                # Define sync function for SQLAlchemy Inspector usage
                def _run_sync_inspection(inspector_target: Union[AsyncConnection, AsyncEngine], func_to_run: callable):
                    # Need to run inspection methods using run_sync
                    sync_inspector = sa_inspect(inspector_target)
                    return func_to_run(sync_inspector)

                # --- Action: schema ---
                if action == "schema":
                    include_indexes = options.get("include_indexes", True)
                    include_foreign_keys = options.get("include_foreign_keys", True)
                    detailed = options.get("detailed", False)
                    filter_schema = schema_name # Use schema_name as the filter here

                    # This function contains the logic originally in discover_database_schema's run_sync
                    def _get_full_schema(sync_conn) -> Dict[str, Any]:
                        # Create inspector from sync connection
                        insp = sa_inspect(sync_conn)
                        target_schema = filter_schema or getattr(insp, 'default_schema_name', None)
                        logger.info(f"Inspecting schema: {target_schema or 'Default'}. Detailed: {detailed}, Indexes: {include_indexes}, FKs: {include_foreign_keys}")
                        
                        tables_data, views_data, relationships = [], [], []
                        
                        try:
                            # Use inspector methods directly
                            table_names = insp.get_table_names(schema=target_schema)
                            view_names = insp.get_view_names(schema=target_schema)
                        except Exception as inspect_err:
                             logger.error(f"Error listing tables/views for schema '{target_schema}': {inspect_err}", exc_info=True)
                             raise ToolError(f"Failed to list tables/views for schema '{target_schema}': {inspect_err}", http_status_code=500) from inspect_err

                        # Process Tables
                        for tbl_name in table_names:
                            try:
                                t_info: Dict[str, Any] = {"name": tbl_name, "columns": []}
                                if target_schema: 
                                    t_info["schema"] = target_schema
                                
                                # Columns
                                columns = insp.get_columns(tbl_name, schema=target_schema)
                                for c in columns:
                                    col_info = {
                                        "name": c["name"],
                                        "type": str(c["type"]),
                                        "nullable": c["nullable"],
                                        "primary_key": bool(c.get("primary_key")),
                                    }
                                    if detailed:
                                        col_info["default"] = c.get("default", None) # Keep None if no default
                                        col_info["comment"] = c.get("comment") # None if no comment
                                        col_info["autoincrement"] = c.get("autoincrement", "auto")
                                    t_info["columns"].append(col_info)
                                    
                                # Indexes (if requested)
                                if include_indexes:
                                    try:
                                        indexes = insp.get_indexes(tbl_name, schema=target_schema)
                                        t_info["indexes"] = [
                                            {
                                                "name": i["name"],
                                                "columns": i["column_names"],
                                                "unique": i.get("unique", False),
                                                # Include dialect options if detailed? i.get('dialect_options')
                                            } for i in indexes
                                        ]
                                    except NotImplementedError:
                                        logger.warning(f"get_indexes not implemented for dialect {db_dialect}")
                                        t_info["indexes"] = [] # Indicate not available
                                    except Exception as idx_err:
                                         logger.warning(f"Could not retrieve indexes for table {tbl_name}: {idx_err}")
                                         t_info["indexes"] = []

                                # Foreign Keys (if requested)
                                if include_foreign_keys:
                                    try:
                                         fks = insp.get_foreign_keys(tbl_name, schema=target_schema)
                                         if fks: 
                                             t_info["foreign_keys"] = []
                                         for fk in fks:
                                             fk_info = {
                                                 "name": fk.get("name"), # Optional name
                                                 "constrained_columns": fk["constrained_columns"],
                                                 "referred_schema": fk.get("referred_schema"),
                                                 "referred_table": fk["referred_table"],
                                                 "referred_columns": fk["referred_columns"],
                                                 # "options": fk.get("options") # Include if detailed needed
                                             }
                                             t_info["foreign_keys"].append(fk_info)
                                             
                                             # Track relationships for the overall graph
                                             relationships.append({
                                                 "source_schema": target_schema,
                                                 "source_table": tbl_name,
                                                 "source_columns": fk["constrained_columns"],
                                                 "target_schema": fk.get("referred_schema"),
                                                 "target_table": fk["referred_table"],
                                                 "target_columns": fk["referred_columns"],
                                             })
                                    except NotImplementedError:
                                         logger.warning(f"get_foreign_keys not implemented for dialect {db_dialect}")
                                         # t_info["foreign_keys"] = [] # Indicate not available
                                    except Exception as fk_err:
                                          logger.warning(f"Could not retrieve foreign keys for table {tbl_name}: {fk_err}")
                                          # t_info["foreign_keys"] = []
                                    tables_data.append(t_info)
                            except Exception as tbl_err:
                                 logger.error(f"Failed to inspect table '{tbl_name}' in schema '{target_schema}': {tbl_err}", exc_info=True)
                                 # Optionally add a placeholder indicating the error for this table
                                 tables_data.append({"name": tbl_name, "schema": target_schema, "error": f"Failed to inspect: {tbl_err}"})

                        # Process Views
                        for view_name in view_names:
                             try:
                                 view_info = {"name": view_name, "schema": target_schema}
                                 try:
                                     view_def = insp.get_view_definition(view_name, schema=target_schema)
                                     view_info["definition"] = view_def or ""
                                 except NotImplementedError:
                                     logger.warning(f"get_view_definition not implemented for dialect {db_dialect}")
                                     view_info["definition"] = "N/A (Not Implemented by Dialect)"
                                 except Exception as view_def_err:
                                      logger.warning(f"Could not retrieve definition for view {view_name}: {view_def_err}")
                                      view_info["definition"] = "Error retrieving definition"

                                 # Try getting columns for views too (might work on some dialects)
                                 try:
                                     view_cols = insp.get_columns(view_name, schema=target_schema)
                                     view_info["columns"] = [{"name": vc["name"], "type": str(vc["type"])} for vc in view_cols]
                                 except Exception: # Ignore if getting columns for view fails
                                     pass 

                                 views_data.append(view_info)
                             except Exception as view_err:
                                 logger.error(f"Failed to inspect view '{view_name}' in schema '{target_schema}': {view_err}", exc_info=True)
                                 views_data.append({"name": view_name, "schema": target_schema, "error": f"Failed to inspect: {view_err}"})

                        schema_result = {
                            "action": "schema",
                            "database_type": db_dialect,
                            "inspected_schema": target_schema or 'Default',
                            "tables": tables_data,
                            "views": views_data,
                            "relationships": relationships,
                            "success": True
                        }

                        # Record schema version for drift detection (only if successful)
                        try:
                            schema_hash = hashlib.sha256(
                                json.dumps(schema_result, sort_keys=True, default=str).encode() # Use default=str for non-serializable types like Decimal
                            ).hexdigest()
                            
                            timestamp = self._now()
                            last_hash = self._SCHEMA_VERSIONS.get(connection_id)

                            if last_hash != schema_hash:
                                self._SCHEMA_VERSIONS[connection_id] = schema_hash
                                lineage_entry = {
                                    "connection_id": connection_id,
                                    "timestamp": timestamp,
                                    "schema_hash": schema_hash,
                                    "previous_hash": last_hash, # Track previous hash
                                    "user_id": user_id, # Capture who triggered the discovery
                                    "tables_count": len(tables_data),
                                    "views_count": len(views_data),
                                    "action_source": f"{tool_name}/{action}"
                                }
                                self._LINEAGE.append(lineage_entry)
                                logger.info(f"Schema change detected or initial capture for {connection_id}. New hash: {schema_hash[:8]}..., Previous: {last_hash[:8] if last_hash else 'None'}")
                                schema_result["schema_hash"] = schema_hash # Include hash in result
                                schema_result["schema_change_detected"] = True if last_hash else False # Indicate if it was a change
                        except Exception as hash_err:
                             logger.error(f"Error generating schema hash or recording lineage: {hash_err}", exc_info=True)
                             # Don't fail the whole operation for hashing error

                        return schema_result

                    # Run the inspection logic using run_sync on the connection directly with the function
                    result = await conn.run_sync(_get_full_schema)


                # --- Action: table ---
                elif action == "table":
                    if not table_name:
                        raise ToolInputError("`table_name` is required for the 'table' action.", param_name="table_name")
                    
                    include_sample = options.get("include_sample_data", False)
                    sample_size = int(options.get("sample_size", 5))
                    include_stats = options.get("include_statistics", False)
                    
                    if sample_size < 0: 
                        sample_size = 0
                    
                    # Function for sync inspection part
                    def _get_basic_table_meta(sync_conn) -> Dict[str, Any]:
                        """Get basic table metadata using the SQLAlchemy inspector properly"""
                        # Create inspector from the sync connection
                        insp = sa_inspect(sync_conn)
                        target_schema = schema_name or getattr(insp, 'default_schema_name', None)
                        logger.info(f"Inspecting table details: {target_schema}.{table_name}")

                        # Verify table exists first
                        try:
                             all_tables = insp.get_table_names(schema=target_schema)
                             if table_name not in all_tables:
                                 # Maybe try case-insensitive? Depends on DB. For now, exact match.
                                 logger.error(f"Table '{table_name}' not found in schema '{target_schema}'. Available tables: {all_tables}")
                                 raise ToolInputError(f"Table '{table_name}' not found in schema '{target_schema}'.", param_name="table_name")
                        except Exception as list_err:
                              logger.error(f"Error listing tables while checking for '{table_name}': {list_err}")
                              # Assume it might exist and proceed, or raise error
                              raise ToolError(f"Could not verify if table '{table_name}' exists: {list_err}", http_status_code=500) from list_err

                        cols = insp.get_columns(table_name, schema=target_schema)
                        
                        # Handle indexes
                        idx = []
                        try:
                            idx = insp.get_indexes(table_name, schema=target_schema)
                        except (NotImplementedError, Exception) as idx_err:
                            logger.warning(f"Could not get indexes for table {table_name}: {idx_err}")
                        
                        # Handle foreign keys
                        fks = []
                        try:
                            fks = insp.get_foreign_keys(table_name, schema=target_schema)
                        except (NotImplementedError, Exception) as fk_err:
                            logger.warning(f"Could not get foreign keys for table {table_name}: {fk_err}")
                        
                        # Try to get primary key constraint info
                        pk_constraint = {}
                        try:
                             pk_info = insp.get_pk_constraint(table_name, schema=target_schema)
                             if pk_info and pk_info.get('constrained_columns'):
                                 pk_constraint = {"name": pk_info.get('name'), "columns": pk_info['constrained_columns']}
                        except (NotImplementedError, Exception) as pk_err: 
                            logger.warning(f"Could not get PK constraint for {table_name}: {pk_err}")

                        # Try to get table comment/description
                        table_comment = None
                        try:
                             table_comment = insp.get_table_comment(table_name, schema=target_schema)
                        except (NotImplementedError, Exception) as cmt_err: 
                            logger.warning(f"Could not get table comment for {table_name}: {cmt_err}")

                        return {
                            "columns": cols, 
                            "indexes": idx, 
                            "foreign_keys": fks, 
                            "pk_constraint": pk_constraint,
                            "table_comment": table_comment.get("text") if table_comment else None
                        }

                    # Run sync inspection directly with the function 
                    meta = await conn.run_sync(_get_basic_table_meta)

                    # Format results nicely
                    result = {
                        "action": "table",
                        "table_name": table_name,
                        "schema_name": schema_name or getattr(meta, 'schema_name', None),
                        "comment": meta.get("table_comment"),
                        "columns": [
                            {
                                "name": c["name"],
                                "type": str(c["type"]),
                                "nullable": c["nullable"],
                                "primary_key": bool(c.get("primary_key")), # From column definition
                                "default": c.get("default"),
                                "comment": c.get("comment")
                            } for c in meta["columns"]
                        ],
                        "primary_key": meta.get("pk_constraint"), # Explicit PK constraint
                        "indexes": meta.get("indexes", []),
                        "foreign_keys": meta.get("foreign_keys", []),
                        "success": True,
                    }

                    # Get row count (Use _exec for safety, metrics, etc.)
                    # Need to handle quoting for table/schema names correctly
                    quoted_table = eng.dialect.identifier_preparer.quote(table_name)
                    if schema_name:
                        quoted_schema = eng.dialect.identifier_preparer.quote(schema_name)
                        full_table_name = f"{quoted_schema}.{quoted_table}"
                    else:
                        full_table_name = quoted_table

                    try:
                        _, count_rows, _ = await self._exec(
                            eng, f"SELECT COUNT(*) AS row_count FROM {full_table_name}", None, limit=1, 
                            tool_name=tool_name, action_name="table_count", timeout=30
                        )
                        result["row_count"] = count_rows[0]['row_count'] if count_rows else 0
                    except Exception as count_err:
                         logger.warning(f"Could not get row count for table {full_table_name}: {count_err}")
                         result["row_count"] = "Error" # Indicate error

                    # Get sample data if requested
                    if include_sample and sample_size > 0:
                        try:
                            sample_cols, sample_rows, _ = await self._exec(
                                eng,
                                f"SELECT * FROM {full_table_name} LIMIT :n",
                                {"n": sample_size},
                                limit=sample_size, # Fetch exactly sample size
                                tool_name=tool_name, action_name="table_sample", timeout=30
                            )
                            # Ensure sample_cols match result cols order if possible, though SELECT * should be fine
                            result["sample_data"] = {"columns": sample_cols, "rows": sample_rows}
                        except Exception as sample_err:
                            logger.warning(f"Could not get sample data for table {full_table_name}: {sample_err}")
                            result["sample_data"] = {"error": f"Failed to retrieve sample data: {sample_err}"}

                    # Get statistics if requested
                    if include_stats:
                        stats = {}
                        logger.debug(f"Calculating basic statistics for columns in {full_table_name}")
                        for c in result["columns"]:
                            col_name = c["name"]
                            quoted_col = eng.dialect.identifier_preparer.quote(col_name)
                            try:
                                # Null count
                                _, null_rows, _ = await self._exec(eng, f"SELECT COUNT(*) AS null_count FROM {full_table_name} WHERE {quoted_col} IS NULL", None, limit=1, tool_name=tool_name, action_name="col_stat_null", timeout=20)
                                null_count = null_rows[0]['null_count'] if null_rows else 'Error'
                                
                                # Distinct count
                                _, distinct_rows, _ = await self._exec(eng, f"SELECT COUNT(DISTINCT {quoted_col}) AS distinct_count FROM {full_table_name}", None, limit=1, tool_name=tool_name, action_name="col_stat_distinct", timeout=45) # Longer timeout for distinct
                                distinct_count = distinct_rows[0]['distinct_count'] if distinct_rows else 'Error'
                                
                                stats[col_name] = {"null_count": null_count, "distinct_count": distinct_count}
                            except Exception as stat_err:
                                logger.warning(f"Could not calculate statistics for column {col_name} in {full_table_name}: {stat_err}")
                                stats[col_name] = {"error": f"Failed: {stat_err}"}
                        result["statistics"] = stats


                # --- Action: column ---
                elif action == "column":
                    if not table_name: 
                        raise ToolInputError("`table_name` is required for 'column' action.", param_name="table_name")
                    if not column_name: 
                        raise ToolInputError("`column_name` is required for 'column' action.", param_name="column_name")
                    
                    generate_histogram = options.get("histogram", False)
                    num_buckets = int(options.get("num_buckets", 10))
                    if num_buckets < 1: 
                        num_buckets = 1

                    # Quote identifiers
                    quoted_table = eng.dialect.identifier_preparer.quote(table_name)
                    quoted_column = eng.dialect.identifier_preparer.quote(column_name)
                    if schema_name:
                        quoted_schema = eng.dialect.identifier_preparer.quote(schema_name)
                        full_table_name = f"{quoted_schema}.{quoted_table}"
                    else:
                        full_table_name = quoted_table

                    # Get basic statistics using separate queries for reliability
                    logger.info(f"Analyzing column {full_table_name}.{quoted_column}")
                    stats_data = {}
                    try:
                        # Total Rows
                        _, total_rows, _ = await self._exec(eng, f"SELECT COUNT(*) as cnt FROM {full_table_name}", None, limit=1, tool_name=tool_name, action_name="col_stat_total", timeout=30)
                        stats_data["total_rows"] = total_rows[0]['cnt'] if total_rows else 0

                        # Null Count
                        _, null_rows, _ = await self._exec(eng, f"SELECT COUNT(*) as cnt FROM {full_table_name} WHERE {quoted_column} IS NULL", None, limit=1, tool_name=tool_name, action_name="col_stat_null", timeout=30)
                        stats_data["null_count"] = null_rows[0]['cnt'] if null_rows else 0
                        stats_data["null_percentage"] = round((stats_data["null_count"] / stats_data["total_rows"]) * 100, 2) if stats_data["total_rows"] else 0

                        # Distinct Count
                        _, distinct_rows, _ = await self._exec(eng, f"SELECT COUNT(DISTINCT {quoted_column}) as cnt FROM {full_table_name}", None, limit=1, tool_name=tool_name, action_name="col_stat_distinct", timeout=60) # Longer timeout
                        stats_data["distinct_count"] = distinct_rows[0]['cnt'] if distinct_rows else 0
                        stats_data["distinct_percentage"] = round((stats_data["distinct_count"] / stats_data["total_rows"]) * 100, 2) if stats_data["total_rows"] else 0

                        # Basic numeric stats (if applicable) - requires fetching column type info first
                        # This adds complexity, maybe skip Min/Max/Avg for simplicity or add later
                        
                    except Exception as stat_err:
                         logger.error(f"Failed to get basic statistics for column {full_table_name}.{quoted_column}: {stat_err}", exc_info=True)
                         # Return partial results if possible
                         stats_data["error"] = f"Failed to retrieve some statistics: {stat_err}"


                    result = {
                        "action": "column",
                        "table_name": table_name,
                        "column_name": column_name,
                        "schema_name": schema_name,
                        "statistics": stats_data,
                        "success": True,
                    }

                    # Generate histogram if requested
                    if generate_histogram:
                        logger.debug(f"Generating histogram for {full_table_name}.{quoted_column}")
                        histogram_data = None
                        try:
                            # Fetch non-null values for histogram analysis - Limit fetch size for performance?
                            # Fetching all values can be very slow/memory intensive on large tables.
                            # Consider adding a sampling option or limit for histogram data.
                            # For now, fetch all non-null values.
                            hist_query = f'SELECT {quoted_column} FROM {full_table_name} WHERE {quoted_column} IS NOT NULL'
                            _, value_rows, _ = await self._exec(eng, hist_query, None, limit=None, tool_name=tool_name, action_name="col_hist_fetch", timeout=90) # Longer timeout for fetch
                            
                            values = [r[column_name] for r in value_rows] # Extract the single column

                            if not values:
                                histogram_data = {"type": "empty", "buckets": []}
                            else:
                                first_val = values[0]
                                is_numeric = isinstance(first_val, (int, float)) # Basic numeric check
                                # Could add date/time check: isinstance(first_val, (dt.date, dt.datetime))
                                
                                if is_numeric:
                                    # Numeric histogram (using min/max)
                                    try:
                                        min_val = min(values)
                                        max_val = max(values)
                                        
                                        if min_val == max_val: # Handle case where all values are the same
                                             buckets = [{"range": f"{min_val}", "count": len(values)}]
                                        else:
                                            bin_width = (max_val - min_val) / num_buckets
                                            # Create bucket ranges
                                            bucket_ranges = [(min_val + i * bin_width, min_val + (i + 1) * bin_width) for i in range(num_buckets)]
                                            # Ensure the last bucket includes the max value
                                            bucket_ranges[-1] = (bucket_ranges[-1][0], max_val) 
                                            
                                            buckets = [{"range": f"{r[0]:.4g} - {r[1]:.4g}", "count": 0} for r in bucket_ranges]
                                            
                                            # Assign values to buckets
                                            for v in values:
                                                # Find the bucket index
                                                # Handle edge case where v == max_val
                                                if v == max_val:
                                                    idx = num_buckets - 1
                                                else:
                                                     # Calculate relative position, avoiding division by zero if bin_width is 0
                                                     relative_pos = (v - min_val) / bin_width if bin_width > 0 else 0
                                                     idx = min(int(relative_pos), num_buckets - 1) # Clamp to max index
                                                
                                                buckets[idx]["count"] += 1
                                        histogram_data = {"type": "numeric", "min": min_val, "max": max_val, "buckets": buckets}
                                    except Exception as num_hist_err:
                                         logger.error(f"Error generating numeric histogram: {num_hist_err}", exc_info=True)
                                         histogram_data = {"error": f"Failed to generate numeric histogram: {num_hist_err}"}

                                else:
                                    # Categorical/Frequency histogram for non-numeric types
                                    try:
                                        from collections import Counter
                                        value_counts = Counter(map(str, values)) # Convert to string for counting
                                        # Get top N most frequent items
                                        top_buckets = value_counts.most_common(num_buckets)
                                        
                                        buckets_data = [{"value": str(k)[:100], "count": v} for k, v in top_buckets] # Truncate long string values
                                        other_count = len(values) - sum(b['count'] for b in buckets_data)
                                        
                                        histogram_data = {"type": "frequency", "top_n": num_buckets, "buckets": buckets_data}
                                        if other_count > 0:
                                             histogram_data['other_values_count'] = other_count

                                    except Exception as freq_hist_err:
                                         logger.error(f"Error generating frequency histogram: {freq_hist_err}", exc_info=True)
                                         histogram_data = {"error": f"Failed to generate frequency histogram: {freq_hist_err}"}
                                
                        except Exception as hist_err:
                             logger.error(f"Failed to generate histogram for column {full_table_name}.{quoted_column}: {hist_err}", exc_info=True)
                             histogram_data = {"error": f"Histogram generation failed: {hist_err}"}
                        
                        result["histogram"] = histogram_data


                # --- Action: relationships ---
                elif action == "relationships":
                    if not table_name: 
                        raise ToolInputError("`table_name` is required for 'relationships' action.", param_name="table_name")
                    
                    depth = int(options.get("depth", 1))
                    depth = max(1, min(depth, 5)) # Clamp depth 1-5
                    
                    logger.info(f"Finding relationships for table '{table_name}' (depth: {depth}, schema: {schema_name})")

                    # Fetch the full schema first, as it contains relationship info
                    # Use the 'schema' action logic internally - avoid caching here if called from relationship finder?
                    # Or rely on cache if explore_database itself is cached. Let's call it directly.
                    schema_info = await self.explore_database(
                        connection_id=connection_id,
                        action="schema",
                        schema_name=schema_name,
                        include_indexes=False, # Don't need indexes for relationships
                        include_foreign_keys=True, # Essential
                        # Pass user/session if needed for audit within the nested call?
                    )

                    if not schema_info.get("success"):
                        raise ToolError("Failed to retrieve schema information needed to find relationships.")

                    # Build lookup tables for efficient graph traversal
                    tables_by_name: Dict[str, Dict] = { t["name"]: t for t in schema_info.get("tables", []) }
                    # Also consider tables from other schemas referenced in FKs if filter_schema was used?
                    # For simplicity, currently only considers tables within the fetched schema scope.

                    if table_name not in tables_by_name:
                         raise ToolInputError(f"Starting table '{table_name}' not found in the inspected schema '{schema_name}'.", param_name="table_name")

                    # Store visited nodes to prevent infinite loops in potential cyclic relationships
                    visited_nodes = set()

                    # Recursively build relationship graph
                    def _build_relationship_graph(current_table: str, current_depth: int) -> Dict[str, Any]:
                        
                        node_id = f"{schema_name or 'default'}.{current_table}" # Unique ID across schemas
                        if current_depth >= depth or node_id in visited_nodes:
                            # Stop recursion if max depth reached or node already visited
                            return {"table": current_table, "schema": schema_name, "max_depth_reached": current_depth >= depth, "cyclic_reference": node_id in visited_nodes}

                        visited_nodes.add(node_id)

                        node_info = tables_by_name.get(current_table)
                        if not node_info:
                             # Should not happen if initial check passed, but handle defensively
                             visited_nodes.remove(node_id) # Backtrack visited
                             return {"table": current_table, "schema": schema_name, "error": "Table info not found"}

                        graph_node = {"table": current_table, "schema": schema_name, "children": [], "parents": []}
                        
                        # Find PARENTS (tables referenced BY this table via FKs)
                        for fk in node_info.get("foreign_keys", []):
                            ref_table = fk["referred_table"]
                            ref_schema = fk.get("referred_schema", schema_name) # Assume same schema if not specified
                            # Only recurse if the parent table is within our fetched schema scope
                            if ref_table in tables_by_name:
                                parent_node = _build_relationship_graph(ref_table, current_depth + 1)
                                graph_node["parents"].append({
                                    "relationship": f"{current_table}.({','.join(fk['constrained_columns'])}) -> {ref_table}.({','.join(fk['referred_columns'])})",
                                    "target": parent_node
                                })
                            else:
                                graph_node["parents"].append({
                                    "relationship": f"{current_table}.({','.join(fk['constrained_columns'])}) -> {ref_schema or '?'}.{ref_table}.({','.join(fk['referred_columns'])})",
                                    "target": {"table": ref_table, "schema": ref_schema, "outside_scope": True}
                                })

                        # Find CHILDREN (tables that reference THIS table via FKs)
                        for other_table_name, other_table_info in tables_by_name.items():
                             if other_table_name == current_table: 
                                 continue # Skip self-references here (handled in parents)
                             for fk in other_table_info.get("foreign_keys", []):
                                 if fk["referred_table"] == current_table and fk.get("referred_schema", schema_name) == schema_name:
                                     # This 'other_table' references our 'current_table'
                                     child_node = _build_relationship_graph(other_table_name, current_depth + 1)
                                     graph_node["children"].append({
                                         "relationship": f"{other_table_name}.({','.join(fk['constrained_columns'])}) -> {current_table}.({','.join(fk['referred_columns'])})",
                                         "source": child_node
                                     })
                                     
                        visited_nodes.remove(node_id) # Backtrack visited status after exploring children
                        return graph_node

                    # Start the recursion
                    relationship_graph = _build_relationship_graph(table_name, 0)
                    
                    result = {
                        "action": "relationships",
                        "source_table": table_name,
                        "schema_name": schema_name,
                        "max_depth": depth,
                        "relationship_graph": relationship_graph, 
                        "success": True
                    }


                # --- Action: documentation ---
                elif action == "documentation":
                    output_format = options.get("output_format", "markdown").lower()
                    if output_format not in ["markdown", "json"]:
                         raise ToolInputError("Invalid 'output_format'. Use 'markdown' or 'json'.", param_name="output_format")

                    # Get schema details first - reuse 'schema' action logic
                    doc_include_indexes = options.get("include_indexes", True)
                    doc_include_fks = options.get("include_foreign_keys", True)

                    logger.info(f"Generating database documentation (Format: {output_format}, Schema: {schema_name})")

                    schema_data = await self.explore_database(
                        connection_id=connection_id,
                        action="schema",
                        schema_name=schema_name,
                        include_indexes=doc_include_indexes,
                        include_foreign_keys=doc_include_fks,
                        detailed=True, # Get detailed info for docs
                        # user_id=user_id, session_id=session_id # Pass audit info if needed
                    )

                    if not schema_data.get("success"):
                         raise ToolError("Failed to retrieve schema information needed for documentation.")
                    
                    if output_format == "json":
                         # Just return the schema data itself
                         result = {
                             "action": "documentation",
                             "format": "json",
                             "documentation": schema_data, # Contains tables, views, relationships, etc.
                             "success": True
                         }
                    else: # output_format == "markdown"
                         lines = [f"# Database Documentation ({db_dialect})"]
                         db_schema_name = schema_data.get('inspected_schema', 'Default Schema')
                         lines.append(f"Schema: **{db_schema_name}**")
                         lines.append(f"Generated: {self._now()}")
                         if schema_data.get("schema_hash"):
                             lines.append(f"Schema Version (Hash): `{schema_data['schema_hash'][:12]}`")
                         lines.append("")

                         # Tables Section
                         lines.append("## Tables")
                         lines.append("")
                         tables = sorted(schema_data.get("tables", []), key=lambda x: x['name'])
                         if not tables: 
                             lines.append("*No tables found in this schema.*")

                         for t in tables:
                            if t.get('error'): # Handle tables that failed inspection
                                lines += [f"### {t['name']} (Error)", f"```\n{t['error']}\n```", ""]
                                continue

                            lines += [f"### {t['name']}", ""]
                            if t.get('comment'): 
                                lines += [f"> {t['comment']}", ""]

                            # Columns Table
                            lines += ["| Column | Type | Nullable | PK | Default | Comment |", 
                                      "|--------|------|----------|----|---------|---------|"]
                            for c in t.get("columns", []):
                                pk_flag = "✅" if c['primary_key'] else ""
                                null_flag = "✅" if c['nullable'] else ""
                                default_val = f"`{c.get('default', '')}`" if c.get('default') is not None else ""
                                comment_val = c.get('comment') or ""
                                lines.append(f"| `{c['name']}` | `{c['type']}` | {null_flag} | {pk_flag} | {default_val} | {comment_val} |")
                            lines.append("")
                            
                            # Primary Key Constraint
                            if t.get("primary_key") and t['primary_key'].get('columns'):
                                 lines.append(f"**Primary Key:** `{t['primary_key'].get('name', 'PK')}` ({', '.join(['`'+c+'`' for c in t['primary_key']['columns']])})")
                                 lines.append("")

                            # Indexes (if included)
                            if doc_include_indexes and t.get("indexes"):
                                lines.append("**Indexes:**")
                                lines.append("")
                                lines.append("| Name | Columns | Unique |")
                                lines.append("|------|---------|--------|")
                                for idx in t["indexes"]:
                                    unique_flag = "✅" if idx['unique'] else ""
                                    cols_str = ", ".join(['`'+c+'`' for c in idx['columns']])
                                    lines.append(f"| `{idx['name']}` | {cols_str} | {unique_flag} |")
                                lines.append("")

                            # Foreign Keys (if included)
                            if doc_include_fks and t.get("foreign_keys"):
                                lines.append("**Foreign Keys:**")
                                lines.append("")
                                lines.append("| Name | Column(s) | References |")
                                lines.append("|------|-----------|------------|")
                                for fk in t["foreign_keys"]:
                                     constrained_cols = ", ".join(['`'+c+'`' for c in fk['constrained_columns']])
                                     ref_table = f"`{fk.get('referred_schema', db_schema_name)}`.`{fk['referred_table']}`"
                                     ref_cols = ", ".join(['`'+c+'`' for c in fk['referred_columns']])
                                     lines.append(f"| `{fk.get('name', 'FK')}` | {constrained_cols} | {ref_table} ({ref_cols}) |")
                                lines.append("")

                         # Views Section
                         views = sorted(schema_data.get("views", []), key=lambda x: x['name'])
                         if views:
                             lines.append("## Views")
                             lines.append("")
                             for v in views:
                                if v.get('error'):
                                    lines += [f"### {v['name']} (Error)", f"```\n{v['error']}\n```", ""]
                                    continue

                                lines += [f"### {v['name']}", ""]
                                if v.get("columns"):
                                     lines.append("**Columns:** " + ", ".join([f"`{vc['name']}` ({vc['type']})" for vc in v['columns']]))
                                     lines.append("")
                                if v.get("definition") and v['definition'] != "N/A (Not Implemented by Dialect)":
                                    lines += ["**Definition:**", "```sql", v["definition"], "```", ""]
                                else:
                                     lines.append("**Definition:** *Not available or not implemented by dialect.*")
                                     lines.append("")

                         # Relationships Summary (Optional)
                         # Could add a summary of relationships if useful

                         result = {
                             "action": "documentation",
                             "format": "markdown",
                             "documentation": "\n".join(lines),
                             "success": True,
                         }

                else:
                    logger.error(f"Invalid action specified for explore_database: {action}")
                    raise ToolInputError(
                        f"Unknown action: '{action}'. Valid actions: schema, table, column, relationships, documentation", 
                        param_name="action"
                    )

                # Audit successful exploration action
                self._audit(
                    tool_name=tool_name, action=action, connection_id=connection_id, sql=None, tables=[table_name] if table_name else None, 
                    row_count=None, success=True, error=None, user_id=user_id, session_id=session_id, **audit_extras
                )

                return result # Return the constructed result dictionary

        except ToolInputError as tie:
            # Audit failures due to bad input
            self._audit(tool_name=tool_name, action=action + "_fail", connection_id=connection_id, sql=None, tables=[table_name] if table_name else None, 
                        row_count=None, success=False, error=str(tie), user_id=user_id, session_id=session_id, **audit_extras)
            raise tie
        except ToolError as te:
             # Audit operational failures
            self._audit(tool_name=tool_name, action=action + "_fail", connection_id=connection_id, sql=None, tables=[table_name] if table_name else None, 
                        row_count=None, success=False, error=str(te), user_id=user_id, session_id=session_id, **audit_extras)
            raise te
        except Exception as e:
            # Audit unexpected errors
            logger.error(f"Unexpected error in explore_database (action: {action}): {e}", exc_info=True)
            self._audit(tool_name=tool_name, action=action + "_fail", connection_id=connection_id, sql=None, tables=[table_name] if table_name else None, 
                        row_count=None, success=False, error=f"Unexpected error: {e}", user_id=user_id, session_id=session_id, **audit_extras)
            raise ToolError(f"An unexpected error occurred during database exploration: {e}", http_status_code=500) from e


    # ------------------------------ AUDIT TRAIL ------------------------------
    
    @tool()
    @with_tool_metrics
    @with_error_handling
    async def access_audit_log(
        self,
        action: str = "view", # "view" or "export"
        export_format: Optional[str] = None, # For export: "excel", "csv", "json"
        limit: Optional[int] = 100, # For view action, limit recent records
        user_id: Optional[str] = None, # Filter by user_id
        connection_id: Optional[str] = None, # Filter by connection_id
        # Add more filters? time range? success/fail?
    ) -> Dict[str, Any]:
        """
        Access and export the in-memory SQL audit log.
        
        Args:
            action: Action to perform: "view" (default) or "export".
            export_format: Required for "export" action. Supported formats: "excel", "csv", "json".
            limit: For "view" action, the maximum number of *most recent* records to return (default: 100). Use `None` for all.
            user_id: Filter records by user_id.
            connection_id: Filter records by connection_id.

        Returns:
            Dict containing audit records or export file path.
            - view: {"action": "view", "records": List[Dict], "total_records": int, "success": True}
            - export: {"action": "export", "path": str, "format": str, "record_count": int, "success": True}
        """
        tool_name = "access_audit_log"
        # No direct DB interaction, so db_dialect is N/A
        
        # Apply filters first
        filtered_log = list(self._AUDIT_LOG) # Start with a copy
        if user_id:
             filtered_log = [r for r in filtered_log if r.get('user_id') == user_id]
        if connection_id:
             filtered_log = [r for r in filtered_log if r.get('connection_id') == connection_id]
             
        total_records_in_log = len(self._AUDIT_LOG)
        filtered_record_count = len(filtered_log)
        
        if action == "view":
            # Return the most recent 'limit' records from the filtered list
            if limit is not None and limit >= 0:
                records_to_return = filtered_log[-limit:]
            else:
                records_to_return = filtered_log # Return all filtered records
                
            logger.info(f"View audit log requested. Returning {len(records_to_return)}/{filtered_record_count} filtered records (Total in log: {total_records_in_log}).")
            return {
                "action": "view",
                "records": records_to_return, 
                "filtered_record_count": filtered_record_count,
                "total_records_in_log": total_records_in_log,
                "filters_applied": {"user_id": user_id, "connection_id": connection_id},
                "success": True
            }
            
        elif action == "export":
            if not export_format:
                raise ToolInputError("`export_format` is required for the 'export' action.", param_name="export_format")
            
            export_format = export_format.lower()
            logger.info(f"Export audit log requested. Format: {export_format}. Records to export: {filtered_record_count}")

            if not filtered_log:
                 logger.warning("Audit log is empty or filtered log is empty, nothing to export.")
                 # Return success but indicate no file generated? Or raise error?
                 # Let's return success with a message.
                 return {"action": "export", "message": "No audit records found matching filters to export.", "record_count": 0, "success": True}


            # Handle JSON export
            if export_format == "json":
                try:
                    fd, path = tempfile.mkstemp(suffix=".json", prefix="mcp_audit_export_")
                    os.close(fd)
                    with open(path, "w") as f:
                        # Dump the filtered log
                        json.dump(filtered_log, f, indent=2, default=str) # Use default=str for any non-standard types
                    logger.info(f"Successfully exported {filtered_record_count} audit records to JSON: {path}")
                    return {"action": "export", "path": path, "format": "json", "record_count": filtered_record_count, "success": True}
                except Exception as e:
                     logger.error(f"Failed to export audit log to JSON: {e}", exc_info=True)
                     raise ToolError(f"Failed to export audit log to JSON: {e}", http_status_code=500) from e

            # Handle CSV/Excel export (requires pandas)
            elif export_format in ["excel", "csv"]:
                if pd is None:
                    raise ToolError(f"Pandas library is not installed, cannot export audit log to '{export_format}'.", http_status_code=501)
                
                try:
                    # Create DataFrame from the filtered log
                    df = pd.DataFrame(filtered_log)
                    
                    # Determine suffix and export function
                    suffix, writer_func, engine = (".xlsx", df.to_excel, "xlsxwriter") if export_format == "excel" else (".csv", df.to_csv, None)

                    # Create temporary file
                    fd, path = tempfile.mkstemp(suffix=suffix, prefix="mcp_audit_export_")
                    os.close(fd)

                    # Write using pandas
                    export_kwargs = {"index": False}
                    if engine: 
                        export_kwargs["engine"] = engine
                    writer_func(path, **export_kwargs)
                    
                    logger.info(f"Successfully exported {filtered_record_count} audit records to {export_format.upper()}: {path}")
                    return {"action": "export", "path": path, "format": export_format, "record_count": filtered_record_count, "success": True}

                except Exception as e:
                     logger.error(f"Failed to export audit log to {export_format}: {e}", exc_info=True)
                     # Clean up temp file if it exists
                     if 'path' in locals() and Path(path).exists():
                         try: 
                             Path(path).unlink()
                         except OSError: 
                             logger.warning(f"Could not clean up temporary export file: {path}")
                     raise ToolError(f"Failed to export audit log to {export_format}: {e}", http_status_code=500) from e
            
            else:
                raise ToolInputError(
                    f"Unsupported export format: '{export_format}'. Use 'excel', 'csv', or 'json'.",
                    param_name="export_format"
                )
        else:
            raise ToolInputError(f"Unknown action: '{action}'. Use 'view' or 'export'.", param_name="action")

