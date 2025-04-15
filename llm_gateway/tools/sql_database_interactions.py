"""SQL database interaction tools for LLM Gateway.

This module provides tools for connecting to, exploring, and querying SQL databases.
It supports both SQLite and PostgreSQL databases through SQLAlchemy.
"""

# No asyncio import needed as implicit async handling is sufficient
import json  # Used for logging complex data structures
import re
import time
import uuid  # Used for generating connection IDs
from typing import Any, Dict, List, Optional, Tuple  # Union removed, using | (PEP 604)

import sqlalchemy

# Column, ForeignKey, Index, MetaData, Table removed as they are for schema definition, not interaction
from sqlalchemy import text

# Engine removed (using AsyncEngine), URL replaced by make_url usage
from sqlalchemy.engine import make_url

# OperationalError, ProgrammingError added for specific exception handling
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError

# AsyncConnection added for type hinting
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

# func removed as text() is used for queries
from llm_gateway.constants import TaskType

# ProviderError removed as it's not relevant to tool errors
from llm_gateway.exceptions import ToolError, ToolInputError
from llm_gateway.services.cache import with_cache

# with_retry added for connection attempts
from llm_gateway.tools.base import with_error_handling, with_retry, with_tool_metrics
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.sql_database_interactions")

# --- Constants ---
PROHIBITED_STATEMENTS = [
    r"^\s*DROP\s+TABLE",  # Prevent dropping tables
    r"^\s*TRUNCATE\s+TABLE",  # Prevent emptying tables
    r"^\s*DELETE\s+FROM",  # Prevent deleting data
    r"^\s*DROP\s+DATABASE",  # Prevent dropping the entire database
    r"^\s*ALTER\s+TABLE\s+.*\s+DROP\s+",  # Prevent dropping columns
    r"^\s*UPDATE\s+",  # Prevent updating data
    r"^\s*INSERT\s+INTO",  # Prevent inserting data
]

SQLITE_PREFIX = "sqlite://"
POSTGRES_PREFIX = "postgresql+asyncpg://"
SQLITE_ASYNC_PREFIX = "sqlite+aiosqlite:///"

# --- Connection Management ---
_active_connections: Dict[str, AsyncEngine] = {}

async def _validate_and_get_engine(connection_id: str) -> AsyncEngine:
    """Get an active database engine for the given connection ID.

    Args:
        connection_id: The unique identifier for the database connection.

    Returns:
        The SQLAlchemy AsyncEngine instance.

    Raises:
        ToolInputError: If the connection does not exist or is not active.
    """
    if connection_id not in _active_connections:
        raise ToolInputError(
            f"No active connection found with ID '{connection_id}'. Use connect_to_database first.",
            param_name="connection_id",
            provided_value=connection_id
        )
    return _active_connections[connection_id]

def _is_query_safe(query: str) -> Tuple[bool, Optional[str]]:
    """Check if a SQL query is safe to execute.

    Args:
        query: The SQL query to check.

    Returns:
        A tuple with (is_safe, reason_if_unsafe).
    """
    # Check against prohibited statements
    normalized_query = query.strip().upper()
    for pattern in PROHIBITED_STATEMENTS:
        if re.search(pattern, normalized_query, re.IGNORECASE):
            return False, f"Query contains prohibited operation: {pattern}"

    # Query is considered safe
    return True, None

async def _execute_safe_query(engine: AsyncEngine, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a SQL query that has been verified as safe.

    Args:
        engine: The SQLAlchemy AsyncEngine instance.
        query: The SQL query to execute.
        parameters: Optional parameters for the query.

    Returns:
        A dictionary with query results.

    Raises:
        ToolError: If the query execution fails.
    """
    start_time = time.time()

    try:
        async with engine.begin() as conn:
            conn: AsyncConnection # Type hint for clarity
            # Start with text query
            stmt = text(query)

            # Execute the query
            if parameters:
                result = await conn.execute(stmt, parameters)
            else:
                result = await conn.execute(stmt)

            # Process results
            if result.returns_rows:
                # Convert to list of dictionaries
                columns = result.keys()
                rows = []
                # Use fetchall() which returns a list directly
                fetched_rows = result.fetchall()
                for row in fetched_rows:
                    # Use _mapping for direct dict conversion, potentially more efficient
                    rows.append(row._mapping)

                return {
                    "columns": list(columns), # Ensure it's a list
                    "rows": rows,
                    "row_count": len(rows),
                    "execution_time": time.time() - start_time,
                    "success": True
                }
            else:
                # Handle non-row-returning queries (like CREATE VIEW)
                return {
                    "rows": [],
                    "row_count": 0,
                    "execution_time": time.time() - start_time,
                    "rowcount": result.rowcount if hasattr(result, 'rowcount') else None,
                    "success": True
                }
    except ProgrammingError as e:
        error_message = f"Syntax error or access violation executing query: {str(e)}"
        logger.error(error_message, exc_info=True, query=query, params=parameters)
        raise ToolError(message=error_message, http_status_code=400) from e # Use message and http_status_code
    except OperationalError as e:
        error_message = f"Database operational error (e.g., connection issue, missing object): {str(e)}"
        logger.error(error_message, exc_info=True, query=query, params=parameters)
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"Error executing query: {str(e)}"
        logger.error(error_message, exc_info=True, query=query, params=parameters)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error during query execution: {str(e)}"
        logger.error(error_message, exc_info=True, query=query, params=parameters)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code

# --- Tool Functions ---

@with_tool_metrics
@with_error_handling
@with_cache(ttl=24 * 60 * 60)  # Cache for 24 hours
async def generate_database_documentation(
    connection_id: str,
    output_format: str = "markdown",
    include_schema: bool = True,
    include_relationships: bool = True,
    include_samples: bool = False,
    include_statistics: bool = False,
    filter_schema: Optional[str] = None,
    tables_to_include: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generates comprehensive documentation for a database.

    This tool analyzes the database structure and generates detailed documentation,
    which can be useful for understanding, onboarding, or creating technical documentation.

    Args:
        connection_id: The unique identifier of the database connection.
        output_format: Format of the output documentation. Currently supports "markdown" or "json". Default "markdown".
        include_schema: If True, includes detailed schema information. Default True.
        include_relationships: If True, includes relationship diagrams/information. Default True.
        include_samples: If True, includes sample data for each table. Default False.
        include_statistics: If True, includes basic statistics for each column. Default False.
        filter_schema: Optional schema name to filter results (for PostgreSQL). Default None (uses 'public' schema).
        tables_to_include: Optional list of specific tables to document. Default None (all tables).

    Returns:
        A dictionary containing the generated documentation:
        {
            "documentation": "# Database Documentation...", # Or JSON object if output_format="json"
            "format": "markdown" | "json",
            "database_name": "name_or_path",
            "database_type": "sqlite" | "postgresql",
            "tables_documented": 10,
            "views_documented": 2,
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists with the provided ID or output_format is invalid.
        ToolError: If documentation generation fails.
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        # Validate output format
        if output_format not in ["markdown", "json"]:
            raise ToolInputError(
                "output_format must be either 'markdown' or 'json'.",
                param_name="output_format",
                provided_value=output_format
            )

        # Determine database type and get database info
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"
        db_info = {}

        async with engine.connect() as conn:
            conn: AsyncConnection # Type hint
            if db_type == "postgresql":
                # Get database name and version
                result = await conn.execute(text("SELECT current_database(), version()"))
                db_name, version = result.fetchone()
                db_info["name"] = db_name
                db_info["version"] = version

                # Get database size
                result = await conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """))
                size = result.scalar()
                db_info["size"] = size
            else:
                # SQLite
                # Get version
                result = await conn.execute(text("SELECT sqlite_version()"))
                row = result.fetchone()
                version = row[0] if row else None # Extract the value from the row
                db_info["version"] = version

                # Extract path for SQLite from engine URL
                path = engine.url.database
                db_info["path"] = path
                db_info["name"] = path.split("/")[-1] if path else "in-memory"

                # Try to get file size
                try:
                    import os
                    if path and os.path.exists(path):
                        size_bytes = os.path.getsize(path)
                        # Convert to human-readable
                        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                            if size_bytes < 1024 or unit == 'TB':
                                db_info["size"] = f"{size_bytes:.2f} {unit}"
                                break
                            size_bytes /= 1024
                except Exception as e:
                    logger.warning(f"Could not determine SQLite database size: {e}")

        # Get schema information
        schema_data = await discover_database_schema(
            connection_id=connection_id,
            include_indexes=True,
            include_foreign_keys=include_relationships,
            detailed=True,
            filter_schema=filter_schema
        )

        # Filter tables if specified
        if tables_to_include:
            tables_to_include_set = set(tables_to_include) # More efficient lookup
            schema_data["tables"] = [table for table in schema_data["tables"] if table["name"] in tables_to_include_set]
            # Also filter relationships
            if include_relationships and "relationships" in schema_data:
                schema_data["relationships"] = [
                    rel for rel in schema_data["relationships"]
                    if rel["source_table"] in tables_to_include_set or rel["target_table"] in tables_to_include_set
                ]

        # Get sample data and statistics if requested
        tables_with_details = []

        for table in schema_data["tables"]:
            table_name = table["name"]
            schema_name = table.get("schema", filter_schema)

            table_with_details = dict(table)

            if include_samples or include_statistics:
                try:
                    table_details = await get_table_details(
                        connection_id=connection_id,
                        table_name=table_name,
                        schema_name=schema_name,
                        include_sample_data=include_samples,
                        sample_size=3 if include_samples else 0,
                        include_statistics=include_statistics
                    )

                    if include_samples and "sample_data" in table_details:
                        table_with_details["sample_data"] = table_details["sample_data"]

                    if include_statistics and "statistics" in table_details:
                        table_with_details["statistics"] = table_details["statistics"]
                except ToolError as e:
                    logger.warning(f"Could not get details for table {table_name}: {e.detail}. Skipping details.")
                    if include_samples: 
                        table_with_details["sample_data_error"] = str(e.detail)
                    if include_statistics: 
                        table_with_details["statistics_error"] = str(e.detail)


            tables_with_details.append(table_with_details)

        # Replace original tables with detailed ones
        schema_data["tables"] = tables_with_details

        # Generate documentation based on format
        if output_format == "markdown":
            # Generate markdown documentation
            doc = []

            # Database overview
            doc.append(f"# Database Documentation: {db_info.get('name', 'Database')}")
            doc.append("")
            doc.append("## Database Overview")
            doc.append("")
            doc.append(f"- **Type**: {db_type.capitalize()}")
            doc.append(f"- **Version**: {db_info.get('version', 'Unknown')}")

            if "size" in db_info:
                doc.append(f"- **Size**: {db_info['size']}")

            if db_type == "sqlite":
                doc.append(f"- **Path**: `{db_info.get('path', 'Unknown')}`") # Use code formatting for path

            doc.append(f"- **Tables**: {len(schema_data['tables'])}")
            doc.append(f"- **Views**: {len(schema_data.get('views', []))}")
            if include_relationships and schema_data.get("relationships"):
                doc.append(f"- **Relationships**: {len(schema_data['relationships'])}")

            doc.append("")

            # Table of Contents
            doc.append("## Table of Contents")
            doc.append("")
            doc.append("1. [Tables](#tables)")
            for table in schema_data["tables"]:
                sanitized_name = table["name"].replace(" ", "-").lower()
                doc.append(f"   - [{table['name']}](#table-{sanitized_name})")

            if schema_data.get("views"):
                doc.append("2. [Views](#views)")
                for view in schema_data["views"]:
                    sanitized_name = view["name"].replace(" ", "-").lower()
                    doc.append(f"   - [{view['name']}](#view-{sanitized_name})")

            if include_relationships and schema_data.get("relationships"):
                doc.append("3. [Relationships](#relationships)")

            doc.append("")

            # Tables
            doc.append("## Tables")
            doc.append("")

            for table in schema_data["tables"]:
                sanitized_name = table["name"].replace(" ", "-").lower()
                doc.append(f"<a id='table-{sanitized_name}'></a>")
                doc.append(f"### {table['name']}")
                doc.append("")

                # Table description if available
                if "comment" in table and table["comment"]:
                    doc.append(f"{table['comment']}")
                    doc.append("")

                # Row count if available (add note if error occurred)
                if "row_count" in table:
                    doc.append(f"**Estimated Row Count**: {table['row_count']}") # Note: get_table_details provides this
                    doc.append("")
                elif "statistics_error" in table:
                    doc.append(f"**Note**: Could not retrieve row count ({table['statistics_error']})")
                    doc.append("")


                # Primary Key
                pk_columns = [col['name'] for col in table.get("columns", []) if col.get("primary_key")]
                if pk_columns:
                    doc.append(f"**Primary Key**: `{', '.join(pk_columns)}`")
                    doc.append("")


                # Columns
                doc.append("#### Columns")
                doc.append("")
                doc.append("| Column | Type | Nullable | Primary Key | Description |")
                doc.append("| ------ | ---- | -------- | ----------- | ----------- |")

                for column in table.get("columns",[]):
                    nullable = "Yes" if column.get("nullable", True) else "No"
                    primary_key = "Yes" if column.get("primary_key", False) else "No"
                    comment = column.get("comment", "")
                    doc.append(f"| `{column['name']}` | `{column['type']}` | {nullable} | {primary_key} | {comment} |") # Use code formatting

                doc.append("")

                # Indexes if available
                if include_schema and "indexes" in table and table["indexes"]:
                    doc.append("#### Indexes")
                    doc.append("")
                    doc.append("| Name | Columns | Unique |")
                    doc.append("| ---- | ------- | ------ |")

                    for index in table["indexes"]:
                        columns = ", ".join([f"`{c}`" for c in index["columns"]]) # Code format columns
                        unique = "Yes" if index.get("unique", False) else "No"
                        doc.append(f"| `{index['name']}` | {columns} | {unique} |") # Code format name

                    doc.append("")

                # Foreign keys if available
                if include_relationships and "foreign_keys" in table and table["foreign_keys"]:
                    doc.append("#### Foreign Keys")
                    doc.append("")
                    doc.append("| Columns | Referenced Table | Referenced Columns |")
                    doc.append("| ------- | --------------- | ------------------ |")

                    for fk in table["foreign_keys"]:
                        columns = ", ".join([f"`{c}`" for c in fk["constrained_columns"]])
                        ref_table = f"`{fk['referred_table']}`"
                        ref_columns = ", ".join([f"`{c}`" for c in fk["referred_columns"]])
                        doc.append(f"| {columns} | {ref_table} | {ref_columns} |")

                    doc.append("")

                # Sample data if requested
                if include_samples:
                    if "sample_data" in table and table["sample_data"]:
                        doc.append("#### Sample Data (First 3 Rows)")
                        doc.append("")

                        # Create header row
                        sample = table["sample_data"]
                        if sample:
                            # Get column names from first sample
                            cols = list(sample[0].keys())
                            doc.append("| " + " | ".join([f"`{c}`" for c in cols]) + " |") # Code format header
                            doc.append("| " + " | ".join(["---" for _ in cols]) + " |")

                            # Create data rows
                            for row in sample:
                                values = []
                                for col in cols:
                                    value = str(row.get(col, ""))
                                    # Truncate very long values
                                    if len(value) > 30:
                                        value = value[:27] + "..."
                                    # Escape pipe characters in values to avoid breaking markdown table
                                    value = value.replace("|", "\\|")
                                    values.append(f"`{value}`") # Code format values for readability
                                doc.append("| " + " | ".join(values) + " |")
                        else:
                             doc.append("*No sample data retrieved or table is empty.*")

                        doc.append("")
                    elif "sample_data_error" in table:
                        doc.append("#### Sample Data")
                        doc.append("")
                        doc.append(f"*Could not retrieve sample data: {table['sample_data_error']}*")
                        doc.append("")


                # Column statistics if requested
                if include_statistics:
                     if "statistics" in table and table["statistics"]:
                        doc.append("#### Column Statistics")
                        doc.append("")

                        for column_name, stats in table.get("statistics", {}).items():
                            doc.append(f"**`{column_name}`**:") # Code format name
                            for stat_name, stat_value in stats.items():
                                if stat_name not in ["unique_count_estimated"]:
                                    # Format numbers nicely
                                    if isinstance(stat_value, (int, float)):
                                        formatted_value = f"{stat_value:,.2f}" if isinstance(stat_value, float) else f"{stat_value:,}"
                                    else:
                                         formatted_value = stat_value

                                    doc.append(f"- {stat_name.replace('_', ' ').capitalize()}: `{formatted_value}`")
                            doc.append("")
                     elif "statistics_error" in table:
                        doc.append("#### Column Statistics")
                        doc.append("")
                        doc.append(f"*Could not retrieve statistics: {table['statistics_error']}*")
                        doc.append("")

            # Views
            if schema_data.get("views"):
                doc.append("## Views")
                doc.append("")

                for view in schema_data["views"]:
                    sanitized_name = view["name"].replace(" ", "-").lower()
                    doc.append(f"<a id='view-{sanitized_name}'></a>")
                    doc.append(f"### {view['name']}")
                    doc.append("")

                    if "definition" in view and view["definition"]:
                        doc.append("#### Definition")
                        doc.append("")
                        doc.append("```sql")
                        doc.append(view["definition"].strip()) # Strip leading/trailing whitespace
                        doc.append("```")
                    else:
                        doc.append("*View definition not available.*")

                    doc.append("")

            # Relationships
            if include_relationships and schema_data.get("relationships"):
                doc.append("## Relationships")
                doc.append("")
                doc.append("| Source Table | Source Columns | Target Table | Target Columns |")
                doc.append("| ------------ | -------------- | ------------ | -------------- |")

                for rel in schema_data["relationships"]:
                    source_table = f"`{rel['source_table']}`"
                    source_cols = ", ".join([f"`{c}`" for c in rel["source_columns"]])
                    target_table = f"`{rel['target_table']}`"
                    target_cols = ", ".join([f"`{c}`" for c in rel["target_columns"]])
                    doc.append(f"| {source_table} | {source_cols} | {target_table} | {target_cols} |")
                doc.append("") # Add final newline

            # Join all lines with newlines
            documentation = "\n".join(doc)
        else:
            # JSON format - just return the structured data
            documentation = {
                "database_info": db_info,
                "schema": schema_data if include_schema else None,
                "timestamp": time.time()
            }

        logger.info(
            f"Successfully generated database documentation in {output_format} format",
            emoji_key="tool",
            connection_id=connection_id,
            format=output_format,
            tables_count=len(schema_data["tables"]),
            views_count=len(schema_data.get("views", []))
        )

        return {
            "documentation": documentation,
            "format": output_format,
            "database_name": db_info.get("name", db_info.get("path", "Unknown")),
            "database_type": db_type,
            "tables_documented": len(schema_data["tables"]),
            "views_documented": len(schema_data.get("views", [])),
            "success": True
        }
    except OperationalError as e:
        error_message = f"Database connection error during documentation generation: {str(e)}"
        logger.error(
            error_message,
            emoji_key="error",
            connection_id=connection_id,
            exc_info=True
        )
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"SQLAlchemy error during documentation generation: {str(e)}"
        logger.error(
            error_message,
            emoji_key="error",
            connection_id=connection_id,
            exc_info=True
        )
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error generating database documentation: {str(e)}"
        logger.error(
            error_message,
            emoji_key="error",
            connection_id=connection_id,
            exc_info=True
        )
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code

@with_tool_metrics
@with_error_handling
@with_retry(max_retries=2, retry_delay=1) # Use max_retries instead of attempts
async def connect_to_database(
    connection_string: str,
    connection_id: Optional[str] = None,
    connection_options: Optional[Dict[str, Any]] = None,
    echo: bool = False
) -> Dict[str, Any]:
    """Establishes a connection to a SQL database using SQLAlchemy.

    Use this tool to connect to either a SQLite or PostgreSQL database.
    For SQLite, provide a path to the database file. For PostgreSQL, provide
    a complete connection string.

    The connection is maintained for use by other SQL tools. It persists until
    explicitly disconnected or the server is restarted.

    Args:
        connection_string: The SQLAlchemy connection URL or path to SQLite file.
            - For SQLite: either 'sqlite:///path/to/database.db' or just 'path/to/database.db'
            - For PostgreSQL: 'postgresql+asyncpg://username:password@host:port/database'
        connection_id: Optional unique identifier for this connection. If not provided,
                      a UUID will be generated. Use this ID with other SQL tools.
        connection_options: Optional dictionary of SQLAlchemy connection options
                          (e.g., {"pool_size": 5, "max_overflow": 10}).
        echo: If True, enables SQLAlchemy echo mode for debugging. Default False.

    Returns:
        A dictionary containing connection information:
        {
            "connection_id": "unique-id-for-this-connection",
            "database_type": "sqlite" or "postgresql",
            "database_info": {
                "path": "path/to/file" or "database_name",
                "version": "database version" (if available)
            },
            "connection_params": {  # Sanitized connection parameters
                "host": "database host" (for PostgreSQL),
                "database": "database name" (for PostgreSQL)
            },
            "success": true
        }

    Raises:
        ToolInputError: If the connection string format is invalid.
        ToolError: If the connection fails.
    """
    start_time = time.time()

    # Process connection string
    connection_options = connection_options or {}

    # Ensure echo isn't passed twice if it's already in connection_options
    explicit_echo = echo # Store the explicit echo value
    if 'echo' in connection_options:
        explicit_echo = connection_options.pop('echo') # Prioritize value from options if present

    # Auto-generate connection ID if not provided
    if not connection_id:
        connection_id = str(uuid.uuid4())

    if connection_id in _active_connections:
         logger.warning(f"Connection ID '{connection_id}' already exists. Overwriting.")
         # Consider disconnecting the old one first if desired:
         # await disconnect_from_database(connection_id) # Requires handling potential errors

    processed_conn_string: str
    db_type: str

    try:
        # Use make_url for robust parsing and handling
        url = make_url(connection_string)

        # Handle simplification for SQLite (accept just the path)
        if url.drivername == "sqlite" and not url.database:
            # If 'sqlite:///' prefix is missing and it's just a path
            if not connection_string.startswith(SQLITE_PREFIX):
                 # Assume it's a file path, reconstruct the URL
                 url = make_url(f"{SQLITE_PREFIX}/{connection_string}")
            else:
                 # It might be 'sqlite://' which means in-memory, let that pass.
                 pass

        # Ensure async driver is used
        if url.drivername == "sqlite":
            db_type = "sqlite"
            # Ensure the async driver variant is specified
            if not url.drivername.endswith("aiosqlite"):
                 url = url._replace(drivername="sqlite+aiosqlite")

        elif url.drivername == "postgresql":
            db_type = "postgresql"
            # Ensure the asyncpg driver variant is specified
            if not url.drivername.endswith("asyncpg"):
                url = url._replace(drivername="postgresql+asyncpg")

        else:
             raise ToolInputError(
                f"Unsupported database dialect: '{url.drivername}'. Only SQLite and PostgreSQL (with asyncpg) are supported.",
                param_name="connection_string",
                provided_value=connection_string
            )

        processed_conn_string = str(url) # Get the final string representation

    except Exception as e: # Catch potential make_url errors
         raise ToolInputError(
            f"Invalid connection string format: {e}. Must be a SQLAlchemy URL or a SQLite file path.",
            param_name="connection_string",
            provided_value=connection_string
        ) from e


    try:
        # Create SQLAlchemy async engine
        engine = create_async_engine(
            processed_conn_string,
            echo=explicit_echo, # Use the determined echo value
            **connection_options
        )

        # Test connection
        async with engine.connect() as conn:
            conn: AsyncConnection # Type hint
            # Get database info
            db_info: Dict[str, Any] = {}
            sanitized_connection_params: Dict[str, Any] = {}

            if db_type == "sqlite":
                # Extract path for SQLite
                db_info["path"] = url.database if url.database else ":memory:" # Handle in-memory explicitly

                # Get SQLite version
                result = await conn.execute(text("SELECT sqlite_version()"))
                row = result.fetchone()
                version = row[0] if row else None # Extract the value from the row
                db_info["version"] = version
                sanitized_connection_params["database"] = db_info["path"]


            elif db_type == "postgresql":
                # Get PostgreSQL info
                result = await conn.execute(text("SELECT current_database(), version()"))
                db_name, version = result.fetchone()
                db_info["database"] = db_name
                db_info["version"] = version

                # Get sanitized connection parameters from URL object
                sanitized_connection_params["host"] = url.host
                sanitized_connection_params["port"] = url.port
                sanitized_connection_params["database"] = url.database
                sanitized_connection_params["username"] = url.username # Logged potentially, but not returned usually

            # Store the connection in our registry
            _active_connections[connection_id] = engine

            # Return success result
            processing_time = time.time() - start_time
            logger.success(
                f"Successfully connected to {db_type} database with ID {connection_id}",
                emoji_key="tool",
                db_type=db_type,
                connection_id=connection_id,
                time=processing_time
            )

            return {
                "connection_id": connection_id,
                "database_type": db_type,
                "database_info": db_info,
                "connection_params": sanitized_connection_params, # Return sanitized params
                "processing_time": processing_time,
                "success": True
            }

    except OperationalError as e:
        error_message = f"Database connection error: {str(e)}. Check connection string, credentials, network, and database status."
        logger.error(
            error_message,
            emoji_key="error",
            db_type=db_type,
            connection_string_used=processed_conn_string # Log the processed string
        )
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"Database setup or configuration error: {str(e)}"
        logger.error(
            error_message,
            emoji_key="error",
            db_type=db_type,
            connection_string_used=processed_conn_string
        )
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error establishing database connection: {str(e)}"
        logger.error(
            error_message,
            emoji_key="error",
            exc_info=True
        )
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code

@with_tool_metrics
@with_error_handling
async def disconnect_from_database(connection_id: str) -> Dict[str, Any]:
    """Closes a database connection and releases all resources.

    Args:
        connection_id: The unique identifier of the database connection to close.

    Returns:
        A dictionary indicating the result of the operation:
        {
            "success": true,
            "message": "Successfully disconnected from database.",
            "connection_id": "connection-id"
        }

    Raises:
        ToolInputError: If no connection exists with the provided ID.
    """
    if connection_id not in _active_connections:
        # Changed to warning as trying to disconnect non-existent is not critical input error
        logger.warning(f"Attempted to disconnect non-existent connection ID '{connection_id}'.")
        return {
            "success": True, # Indicate operation completed (as there's nothing to do)
            "message": f"Connection ID '{connection_id}' not found or already disconnected.",
            "connection_id": connection_id
        }

    try:
        engine = _active_connections[connection_id]
        await engine.dispose() # Gracefully close pool connections
        del _active_connections[connection_id]

        logger.info(
            f"Successfully disconnected from database with ID {connection_id}",
            emoji_key="tool",
            connection_id=connection_id
        )

        return {
            "success": True,
            "message": "Successfully disconnected from database.",
            "connection_id": connection_id
        }
    except Exception as e:
        # Log error but don't necessarily raise ToolError if dispose fails,
        # as the goal is cleanup. Remove from dict anyway.
        error_message = f"Error disposing engine during disconnect for {connection_id}: {str(e)}"
        logger.error(
            error_message,
            emoji_key="error",
            connection_id=connection_id,
            exc_info=True
        )
        # Ensure removal even if dispose fails
        if connection_id in _active_connections:
             del _active_connections[connection_id]
        # Return success=False to indicate potential issue during cleanup
        return {
            "success": False,
            "message": f"Error during disconnection: {error_message}",
            "connection_id": connection_id
        }


@with_tool_metrics
@with_error_handling
async def get_database_status(connection_id: str) -> Dict[str, Any]:
    """Gets the status and basic information about a database connection.

    Args:
        connection_id: The unique identifier of the database connection.

    Returns:
        A dictionary containing status information:
        {
            "connection_id": "connection-id",
            "active": true,
            "database_type": "sqlite" or "postgresql",
            "stats": {
                "tables_count": 10,
                "views_count": 2,
                "size": "10.5 MB" (if available for SQLite)
            },
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists with the provided ID.
        ToolError: If checking status fails (e.g., connection lost).
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        stats = {}
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"
        schema_name = "public" if db_type == "postgresql" else None # Define schema for PG query

        async with engine.connect() as conn:
            conn: AsyncConnection # Type hint
            # Get table and view counts
            if db_type == "postgresql":
                # Use information_schema which is standard SQL
                table_count_query = text("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = :schema AND table_type = 'BASE TABLE'
                """)
                view_count_query = text("""
                    SELECT COUNT(*) FROM information_schema.views
                    WHERE table_schema = :schema
                """)
                table_result = await conn.execute(table_count_query, {"schema": schema_name})
                stats["tables_count"] = table_result.scalar()
                view_result = await conn.execute(view_count_query, {"schema": schema_name})
                stats["views_count"] = view_result.scalar()

                # Get database size
                result = await conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """))
                size = result.scalar()
                stats["size"] = size

            else:
                # SQLite queries
                # Tables count
                result = await conn.execute(text("""
                    SELECT COUNT(*) FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """))
                stats["tables_count"] = result.scalar()

                # Views count
                result = await conn.execute(text("""
                    SELECT COUNT(*) FROM sqlite_master WHERE type='view'
                """))
                stats["views_count"] = result.scalar()

                # For SQLite, get file size if possible
                try:
                    import os
                    db_path = engine.url.database
                    if db_path and os.path.exists(db_path):
                        size_bytes = os.path.getsize(db_path)
                        # Convert to human-readable
                        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                            if size_bytes < 1024 or unit == 'TB':
                                stats["size"] = f"{size_bytes:.2f} {unit}"
                                break
                            size_bytes /= 1024
                except Exception as e:
                    logger.warning(f"Could not determine SQLite database size: {e}")

        return {
            "connection_id": connection_id,
            "active": True,
            "database_type": db_type,
            "stats": stats,
            "success": True
        }
    except OperationalError as e:
        error_message = f"Database connection error while getting status: {str(e)}"
        logger.error(
            error_message,
            emoji_key="error",
            connection_id=connection_id
        )
        # Mark connection as inactive if status check fails due to connection issue
        if connection_id in _active_connections:
            # We might not want to remove it here, but indicate failure
            pass
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"Error getting database status: {str(e)}"
        logger.error(
            error_message,
            emoji_key="error",
            connection_id=connection_id
        )
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error getting database status: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code


@with_tool_metrics
@with_error_handling
@with_cache(ttl=3600)  # Cache for 1 hour since schema rarely changes
async def discover_database_schema(
    connection_id: str,
    include_indexes: bool = True,
    include_foreign_keys: bool = True,
    detailed: bool = False,
    filter_schema: Optional[str] = None
) -> Dict[str, Any]:
    """Discovers and returns a comprehensive overview of the database schema.

    This tool provides a complete overview of the database structure, including
    tables, columns, data types, and relationships. It's essential for understanding
    an unfamiliar database.

    Args:
        connection_id: The unique identifier of the database connection.
        include_indexes: If True, includes information about indexes. Default True.
        include_foreign_keys: If True, includes information about foreign key relationships. Default True.
        detailed: If True, includes additional metadata like column default values and constraints. Default False.
        filter_schema: Optional schema name to filter results (for PostgreSQL). Default None (uses 'public' schema).

    Returns:
        A dictionary containing the database schema:
        {
            "tables": [
                {
                    "name": "table_name",
                    "schema": "schema_name" (for PostgreSQL),
                    "columns": [
                        {
                            "name": "column_name",
                            "type": "column_type",
                            "nullable": true|false,
                            "primary_key": true|false,
                            "default": "default_value" (if detailed=True),
                            "comment": "column comment" (if detailed=True and available)
                        }
                    ],
                    "indexes": [
                        {
                            "name": "index_name",
                            "columns": ["column1", "column2"],
                            "unique": true|false
                        }
                    ] (if include_indexes=True),
                    "foreign_keys": [
                        {
                            "constrained_columns": ["column_name"],
                            "referred_table": "referenced_table_name",
                            "referred_columns": ["referenced_column_name"]
                        }
                    ] (if include_foreign_keys=True),
                    "primary_key": ["column1", "column2"]
                }
            ],
            "views": [
                {
                    "name": "view_name",
                    "schema": "schema_name" (for PostgreSQL),
                    "definition": "SQL definition" (if available)
                }
            ],
            "relationships": [
                {
                    "source_table": "table_name",
                    "source_columns": ["column_name"],
                    "target_table": "referenced_table_name",
                    "target_columns": ["referenced_column_name"]
                }
            ] (if include_foreign_keys=True),
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists with the provided ID.
        ToolError: If schema discovery fails.
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"
        schema_data = {"tables": [], "views": [], "relationships": [], "success": True}

        # Default schema for PostgreSQL is 'public'
        if db_type == "postgresql" and not filter_schema:
            filter_schema = "public"

        # Use a single connection for all inspector operations
        async with engine.connect() as conn:
            conn: AsyncConnection # Type hint

            # Define sync helper functions for inspection
            def _sync_get_table_names(sync_conn):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_table_names(schema=filter_schema)

            def _sync_get_columns(sync_conn, table_name):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_columns(table_name, schema=filter_schema)

            def _sync_get_indexes(sync_conn, table_name):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_indexes(table_name, schema=filter_schema)

            def _sync_get_foreign_keys(sync_conn, table_name):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_foreign_keys(table_name, schema=filter_schema)

            def _sync_get_view_names(sync_conn):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_view_names(schema=filter_schema)

            def _sync_get_view_definition(sync_conn, view_name):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_view_definition(view_name, schema=filter_schema)


            # Get all tables using run_sync with the helper
            table_names = await conn.run_sync(_sync_get_table_names)

            for table_name in table_names:
                table_info = {
                    "name": table_name,
                    "columns": [],
                    "primary_key": [] # Will be populated from columns
                }

                if db_type == "postgresql":
                    table_info["schema"] = filter_schema or "public" # Record the schema used

                # Get columns information using run_sync
                columns = await conn.run_sync(_sync_get_columns, table_name)
                for column in columns:
                    column_info = {
                        "name": column["name"],
                        "type": str(column["type"]), # Ensure type is stringified
                        "nullable": column.get("nullable", True),
                        "primary_key": column.get("pk", False) or column.get("primary_key", False), # Check both 'pk' and 'primary_key'
                    }

                    # Add detailed info if requested
                    if detailed:
                        # Ensure default is stringified, handle None
                        column_info["default"] = str(column["default"]) if column.get("default") is not None else None
                        column_info["autoincrement"] = column.get("autoincrement", "auto") # Provide 'auto' if unknown

                        # Add comment if available (SQLAlchemy provides this via 'comment' key if dialect supports it)
                        if column.get("comment"):
                            column_info["comment"] = column["comment"]
                        # Fallback for PostgreSQL if comment isn't directly in column dict
                        elif db_type == "postgresql" and column.get("comment") is None:
                             try:
                                comment_query = text("""
                                    SELECT col_description(c.oid, a.attnum)
                                    FROM pg_class c
                                    JOIN pg_namespace n ON n.oid = c.relnamespace
                                    JOIN pg_attribute a ON a.attrelid = c.oid
                                    WHERE c.relname = :table_name
                                      AND n.nspname = :schema_name
                                      AND a.attname = :column_name
                                      AND NOT a.attisdropped
                                """)
                                result = await conn.execute(comment_query, {
                                    "table_name": table_name,
                                    "schema_name": filter_schema,
                                    "column_name": column["name"]
                                })
                                comment = result.scalar()
                                if comment:
                                    column_info["comment"] = comment
                             except Exception as e:
                                 logger.debug(f"Could not get PG column comment for {table_name}.{column['name']}: {e}")


                    table_info["columns"].append(column_info)

                    # Track primary keys
                    if column_info["primary_key"]:
                        table_info["primary_key"].append(column["name"])

                # Get table comment if available (PostgreSQL)
                if detailed and db_type == "postgresql":
                     try:
                        table_comment_query = text("""
                            SELECT obj_description(c.oid, 'pg_class')
                            FROM pg_class c
                            JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE c.relname = :table_name
                              AND n.nspname = :schema_name
                        """)
                        result = await conn.execute(table_comment_query, {
                            "table_name": table_name,
                            "schema_name": filter_schema
                        })
                        comment = result.scalar()
                        if comment:
                            table_info["comment"] = comment
                     except Exception as e:
                         logger.debug(f"Could not get PG table comment for {table_name}: {e}")

                # Get indexes if requested using run_sync
                if include_indexes:
                    try:
                        indexes = await conn.run_sync(_sync_get_indexes, table_name)
                        table_info["indexes"] = []
                        for index in indexes:
                            table_info["indexes"].append({
                                "name": index["name"],
                                "columns": index["column_names"],
                                "unique": index.get("unique", False)
                            })
                    except Exception as e:
                         logger.warning(f"Could not retrieve indexes for table {table_name}: {e}")
                         table_info["indexes"] = [] # Ensure key exists

                # Get foreign keys if requested using run_sync
                if include_foreign_keys:
                    try:
                        foreign_keys = await conn.run_sync(_sync_get_foreign_keys, table_name)
                        table_info["foreign_keys"] = []
                        for fk in foreign_keys:
                            fk_info = {
                                "name": fk.get("name"), # Include FK constraint name if available
                                "constrained_columns": fk["constrained_columns"],
                                "referred_table": fk["referred_table"],
                                "referred_columns": fk["referred_columns"]
                            }

                            # Add schema for PostgreSQL
                            if db_type == "postgresql" and "referred_schema" in fk:
                                fk_info["referred_schema"] = fk["referred_schema"]

                            table_info["foreign_keys"].append(fk_info)

                            # Add to overall relationships list
                            schema_data["relationships"].append({
                                "source_table": table_name,
                                "source_schema": filter_schema if db_type == "postgresql" else None,
                                "source_columns": fk["constrained_columns"],
                                "target_table": fk["referred_table"],
                                "target_schema": fk.get("referred_schema") if db_type == "postgresql" else None,
                                "target_columns": fk["referred_columns"],
                                "name": fk.get("name")
                            })
                    except Exception as e:
                         logger.warning(f"Could not retrieve foreign keys for table {table_name}: {e}")
                         table_info["foreign_keys"] = [] # Ensure key exists

                schema_data["tables"].append(table_info)

            # Get views using run_sync
            try:
                 view_names = await conn.run_sync(_sync_get_view_names)
                 for view_name in view_names:
                     view_info = {
                         "name": view_name,
                     }
                     if db_type == "postgresql":
                         view_info["schema"] = filter_schema or "public"

                     # Get view definition if possible using run_sync
                     try:
                         view_def = await conn.run_sync(_sync_get_view_definition, view_name)
                         view_info["definition"] = view_def
                     except NotImplementedError:
                         logger.debug(f"View definition retrieval not implemented for dialect {engine.dialect.name}")
                         # Try manual query as fallback
                         if db_type == "postgresql":
                             view_def_query = text("""
                                 SELECT pg_get_viewdef(c.oid, true) as view_def
                                 FROM pg_class c
                                 JOIN pg_namespace n ON n.oid = c.relnamespace
                                 WHERE c.relname = :view_name
                                   AND n.nspname = :schema_name
                                   AND c.relkind = 'v'
                             """)
                             params = {"view_name": view_name, "schema_name": filter_schema or "public"}
                         elif db_type == "sqlite":
                              view_def_query = text("""
                                 SELECT sql FROM sqlite_master
                                 WHERE type='view' AND name = :view_name
                             """)
                              params = {"view_name": view_name}
                         else:
                              view_def_query = None

                         if view_def_query:
                             try:
                                 result = await conn.execute(view_def_query, params)
                                 view_def = result.scalar()
                                 if view_def:
                                     view_info["definition"] = view_def
                             except Exception as e_manual:
                                 logger.warning(f"Could not get view definition for {view_name} via manual query: {e_manual}")

                     except Exception as e_insp:
                         logger.warning(f"Could not get view definition for {view_name} via inspector: {e_insp}")

                     schema_data["views"].append(view_info)
            except Exception as e:
                 logger.warning(f"Could not retrieve view names: {e}")


        logger.info(
            f"Successfully discovered schema for database: {len(schema_data['tables'])} tables, {len(schema_data['views'])} views",
            emoji_key="tool",
            connection_id=connection_id,
            schema_filter=filter_schema
        )

        return schema_data
    except OperationalError as e:
        error_message = f"Database connection error during schema discovery: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, exc_info=True)
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"Error discovering database schema: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error discovering schema: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code


@with_tool_metrics
@with_error_handling
@with_cache(ttl=3600)  # Cache for 1 hour
async def get_table_details(
    connection_id: str,
    table_name: str,
    schema_name: Optional[str] = None,
    include_sample_data: bool = False,
    sample_size: int = 5,
    include_statistics: bool = False
) -> Dict[str, Any]:
    """Gets detailed information about a specific table.

    This tool provides in-depth information about a particular table, including
    its structure, relationships, and optionally sample data and column statistics.

    Args:
        connection_id: The unique identifier of the database connection.
        table_name: The name of the table to inspect.
        schema_name: The schema name (for PostgreSQL). Default None (uses 'public' schema).
        include_sample_data: If True, includes sample data from the table. Default False.
        sample_size: Number of sample rows to include if include_sample_data is True. Default 5.
        include_statistics: If True, includes basic statistics about each column. Default False.

    Returns:
        A dictionary containing table details:
        {
            "table_name": "table_name",
            "schema_name": "schema_name" (for PostgreSQL),
            "columns": [
                {
                    "name": "column_name",
                    "type": "column_type",
                    "nullable": true|false,
                    "primary_key": true|false
                }
            ],
            "indexes": [
                {
                    "name": "index_name",
                    "columns": ["column1", "column2"],
                    "unique": true|false
                }
            ],
            "foreign_keys": [
                {
                    "constrained_columns": ["column_name"],
                    "referred_table": "referenced_table_name",
                    "referred_columns": ["referenced_column_name"]
                }
            ],
            "relationships": {
                "parent_tables": [
                    {
                        "table_name": "parent_table_name",
                        "foreign_key": {
                            "columns": ["column_name"],
                            "referenced_columns": ["parent_column_name"]
                        }
                    }
                ],
                "child_tables": [
                    {
                        "table_name": "child_table_name",
                        "foreign_key": {
                            "columns": ["child_column_name"],
                            "referenced_columns": ["column_name"]
                        }
                    }
                ]
            },
            "row_count": 1000, (Estimated count for large tables)
            "sample_data": [
                {"column1": "value1", "column2": "value2", ...},
                ...
            ] (if include_sample_data=True),
            "statistics": {
                "column1": {
                    "min": "min_value",
                    "max": "max_value",
                    "avg": "avg_value" (for numeric columns),
                    "null_count": 5,
                    "unique_count": 10 (estimated for large columns)
                },
                ...
            } (if include_statistics=True),
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists with the provided ID or the table doesn't exist.
        ToolError: If table inspection fails.
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"

        # Default schema for PostgreSQL is 'public'
        if db_type == "postgresql" and not schema_name:
            schema_name = "public"

        async with engine.connect() as conn:
            conn: AsyncConnection # Type hint

            # Define sync helper functions for inspection
            def _sync_get_table_names(sync_conn, schema):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_table_names(schema=schema)

            def _sync_get_columns(sync_conn, table_name, schema):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_columns(table_name, schema=schema)

            def _sync_get_indexes(sync_conn, table_name, schema):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_indexes(table_name, schema=schema)

            def _sync_get_foreign_keys(sync_conn, table_name, schema):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_foreign_keys(table_name, schema=schema)

            # Verify table exists using run_sync
            tables = await conn.run_sync(_sync_get_table_names, schema_name)
            if table_name not in tables:
                schema_msg = f" in schema '{schema_name}'" if schema_name else ""
                raise ToolInputError(
                    f"Table '{table_name}' does not exist{schema_msg} in the database.",
                    param_name="table_name",
                    provided_value=table_name
                )

            # Initialize result structure
            result = {
                "table_name": table_name,
                "columns": [],
                "indexes": [],
                "foreign_keys": [],
                "relationships": {
                    "parent_tables": [],
                    "child_tables": []
                },
                "success": True
            }

            if db_type == "postgresql":
                result["schema_name"] = schema_name

            # Get columns information using run_sync
            columns = await conn.run_sync(_sync_get_columns, table_name, schema_name)
            column_map = {col["name"]: col for col in columns} # For quick lookup  # noqa: F841
            for column in columns:
                column_info = {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column.get("nullable", True),
                     "primary_key": column.get("pk", False) or column.get("primary_key", False),
                    "default": str(column["default"]) if column.get("default") is not None else None,
                    "comment": column.get("comment")
                }
                result["columns"].append(column_info)

            # Get indexes using run_sync
            try:
                indexes = await conn.run_sync(_sync_get_indexes, table_name, schema_name)
                for index in indexes:
                    result["indexes"].append({
                        "name": index["name"],
                        "columns": index["column_names"],
                        "unique": index.get("unique", False)
                    })
            except Exception as e:
                logger.warning(f"Could not retrieve indexes for table {table_name}: {e}")

            # Get foreign keys (outgoing FKs from this table) using run_sync
            try:
                foreign_keys = await conn.run_sync(_sync_get_foreign_keys, table_name, schema_name)
                for fk in foreign_keys:
                    fk_info = {
                        "name": fk.get("name"),
                        "constrained_columns": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"]
                    }

                    # Add schema for PostgreSQL
                    if db_type == "postgresql" and "referred_schema" in fk:
                        fk_info["referred_schema"] = fk["referred_schema"]

                    result["foreign_keys"].append(fk_info)

                    # Add to parent relationships
                    parent_info = {
                        "table_name": fk["referred_table"],
                        "schema_name": fk.get("referred_schema") if db_type == "postgresql" else None,
                        "foreign_key": {
                            "name": fk.get("name"),
                            "columns": fk["constrained_columns"],
                            "referenced_columns": fk["referred_columns"]
                        }
                    }
                    result["relationships"]["parent_tables"].append(parent_info)
            except Exception as e:
                logger.warning(f"Could not retrieve foreign keys for table {table_name}: {e}")

            # Find child tables (tables that reference this one)
            # This uses discover_database_schema internally, which already uses run_sync
            try:
                # Pass include_details=False as we only need FK info from other tables here
                schema_info = await discover_database_schema(
                    connection_id,
                    include_indexes=False, # Not needed for this part
                    include_foreign_keys=True,
                    detailed=False, # Only need FKs
                    filter_schema=schema_name
                )
                for other_table in schema_info.get("tables", []):
                    if other_table["name"] == table_name:
                        continue
                    for fk in other_table.get("foreign_keys", []):
                         # Check if the FK refers to the current table_name
                         # And if PG, check if the referred schema matches the current table's schema
                         refers_to_current_table = (fk["referred_table"] == table_name)
                         if db_type == "postgresql":
                              refers_to_current_schema = (fk.get("referred_schema") == schema_name)
                              refers_to_current_table = refers_to_current_table and refers_to_current_schema

                         if refers_to_current_table:
                            child_info = {
                                "table_name": other_table["name"],
                                "schema_name": other_table.get("schema") if db_type == "postgresql" else None,
                                "foreign_key": {
                                    "name": fk.get("name"),
                                    "columns": fk["constrained_columns"],
                                    "referenced_columns": fk["referred_columns"] # These are columns in *our* table
                                }
                            }
                            result["relationships"]["child_tables"].append(child_info)
            except Exception as e:
                logger.warning(f"Could not determine child relationships for table {table_name}: {e}")


            # Get approximate row count
            try:
                 # Use COUNT(*) for simplicity and broad compatibility, accept potential performance hit
                 # Properly quote table and schema names
                 table_identifier = f'"{schema_name}"."{table_name}"' if db_type == "postgresql" else f'"{table_name}"'
                 count_query = text(f"SELECT COUNT(*) FROM {table_identifier}")
                 count_result = await conn.execute(count_query)
                 row_count = count_result.scalar()
                 result["row_count"] = row_count if row_count is not None else 0
            except Exception as e:
                logger.warning(f"Could not determine row count for {table_name}: {e}")
                result["row_count"] = None # Indicate count is unknown

            # Get sample data if requested
            if include_sample_data and sample_size > 0:
                try:
                    # Properly quote table and schema names
                    table_identifier = f'"{schema_name}"."{table_name}"' if db_type == "postgresql" else f'"{table_name}"'
                    sample_query = text(f"SELECT * FROM {table_identifier} LIMIT :limit")
                    sample_result = await conn.execute(sample_query, {"limit": sample_size})

                    # Convert to list of dictionaries
                    columns = sample_result.keys()
                    sample_rows = []
                    fetched_rows = sample_result.fetchall()
                    for row in fetched_rows:
                        # Use _mapping for direct dict conversion without awaiting row
                        sample_rows.append(dict(row._mapping))

                    result["sample_data"] = sample_rows
                except Exception as e:
                    logger.warning(f"Could not retrieve sample data for {table_name}: {e}")
                    result["sample_data"] = []
                    result["sample_data_error"] = str(e)

            # Get column statistics if requested
            if include_statistics:
                result["statistics"] = {}

                for column_info in result["columns"]:
                    column_name = column_info["name"]
                    column_type_str = column_info["type"].lower()

                    # Quote identifiers
                    col_identifier = f'"{column_name}"'
                    table_identifier = f'"{schema_name}"."{table_name}"' if db_type == "postgresql" else f'"{table_name}"'

                    try:
                        stats: Dict[str, Any] = {}

                        # Get null count
                        null_query = text(f"SELECT COUNT(*) FROM {table_identifier} WHERE {col_identifier} IS NULL")
                        null_result = await conn.execute(null_query)
                        null_count_row = null_result.fetchone()
                        stats["null_count"] = null_count_row[0] if null_count_row else 0

                        # Basic Numeric Stats (MIN, MAX, AVG)
                        is_numeric = any(t in column_type_str for t in ["int", "float", "double", "decimal", "numeric", "real"])
                        if is_numeric:
                            num_stats_query = text(f"""
                                SELECT MIN({col_identifier}), MAX({col_identifier}), AVG({col_identifier})
                                FROM {table_identifier} WHERE {col_identifier} IS NOT NULL
                            """)
                            num_stats_result = await conn.execute(num_stats_query)
                            num_stats_row = num_stats_result.fetchone()
                            min_val, max_val, avg_val = num_stats_row if num_stats_row else (None, None, None)
                            stats["min"] = min_val
                            stats["max"] = max_val
                            # Format avg nicely, handle potential Decimal types
                            stats["avg"] = float(avg_val) if avg_val is not None else None


                        # Basic String/Date Stats (MIN, MAX length for string, MIN/MAX for date)
                        is_stringy = any(t in column_type_str for t in ["char", "text", "string"])
                        is_datey = any(t in column_type_str for t in ["date", "time"]) # Includes timestamp

                        if is_stringy:
                             len_func = "LENGTH" # Standard SQL, works on PG and SQLite
                             str_stats_query = text(f"""
                                 SELECT MIN({len_func}({col_identifier})), MAX({len_func}({col_identifier}))
                                 FROM {table_identifier} WHERE {col_identifier} IS NOT NULL
                             """)
                             str_stats_result = await conn.execute(str_stats_query)
                             str_stats_row = str_stats_result.fetchone()
                             min_len, max_len = str_stats_row if str_stats_row else (None, None)
                             stats["min_length"] = min_len
                             stats["max_length"] = max_len
                        elif is_datey:
                             date_stats_query = text(f"""
                                 SELECT MIN({col_identifier}), MAX({col_identifier})
                                 FROM {table_identifier} WHERE {col_identifier} IS NOT NULL
                             """)
                             date_stats_result = await conn.execute(date_stats_query)
                             date_stats_row = date_stats_result.fetchone()
                             min_val, max_val = date_stats_row if date_stats_row else (None, None)
                             # Convert date/time objects to ISO strings for JSON compatibility
                             stats["min"] = min_val.isoformat() if min_val else None
                             stats["max"] = max_val.isoformat() if max_val else None


                        # Get estimate of unique values (COUNT DISTINCT)
                        # Avoid on very large tables if row_count is available and large
                        row_count = result.get("row_count")
                        unique_count = None
                        unique_count_estimated = False

                        # Only calculate if row count is known and not excessively large, or unknown
                        if row_count is None or row_count < 50000:
                            try:
                                unique_query = text(f"SELECT COUNT(DISTINCT {col_identifier}) FROM {table_identifier}")
                                unique_result = await conn.execute(unique_query)
                                unique_row = unique_result.fetchone()
                                unique_count = unique_row[0] if unique_row else 0
                            except Exception as uc_e:
                                logger.debug(f"Could not run COUNT DISTINCT on {table_name}.{column_name}: {uc_e}")
                                # Attempt PG stats estimation if applicable and count distinct failed/skipped
                                if db_type == "postgresql" and row_count is not None and row_count >= 50000:
                                     unique_count_estimated = True # Fallback to PG stats
                        elif db_type == "postgresql":
                             unique_count_estimated = True # Use PG stats for large tables

                        # Use pg_stats for estimation on large PG tables or if COUNT DISTINCT failed/skipped
                        if unique_count_estimated and db_type == "postgresql":
                            try:
                                pg_stats_query = text("""
                                    SELECT CASE WHEN n_distinct > 0 THEN n_distinct
                                                WHEN n_distinct < 0 THEN -n_distinct * :total_rows
                                                ELSE null END AS estimate
                                    FROM pg_stats
                                    WHERE schemaname = :schema_name AND tablename = :table_name AND attname = :column_name
                                """)
                                pg_stats_result = await conn.execute(pg_stats_query, {
                                    "schema_name": schema_name,
                                    "table_name": table_name,
                                    "column_name": column_name,
                                    "total_rows": row_count # Pass total rows for fraction calculation
                                })
                                pg_stats_row = pg_stats_result.fetchone()
                                estimated_unique = pg_stats_row[0] if pg_stats_row else None
                                if estimated_unique is not None:
                                     unique_count = int(estimated_unique)
                                else:
                                     unique_count_estimated = False # Failed to get estimate
                            except Exception as pgs_e:
                                logger.warning(f"Could not query pg_stats for {table_name}.{column_name}: {pgs_e}")
                                unique_count_estimated = False # Revert if query fails

                        stats["unique_count"] = unique_count
                        if unique_count_estimated and unique_count is not None:
                             stats["unique_count_estimated"] = True

                        result["statistics"][column_name] = stats
                    except Exception as e:
                        logger.warning(f"Could not compute statistics for column {table_name}.{column_name}: {e}")
                        # Store error marker instead of empty dict
                        result["statistics"][column_name] = {"error": str(e)}


            logger.info(
                f"Successfully retrieved details for table '{table_name}'",
                emoji_key=TaskType.DATABASE.value, 
                connection_id=connection_id,
                table_name=table_name,
                schema_name=schema_name
            )

            return result
    except OperationalError as e:
        error_message = f"Database connection error getting table details: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, schema_name=schema_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"Error getting table details for '{table_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, schema_name=schema_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error getting table details for '{table_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, schema_name=schema_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code


@with_tool_metrics
@with_error_handling
async def find_related_tables(
    connection_id: str,
    table_name: str,
    schema_name: Optional[str] = None,
    depth: int = 1,
    include_details: bool = False
) -> Dict[str, Any]:
    """Discovers tables that have relationships with a given table.

    This tool helps explore database relationships by finding tables connected to
    the specified table through foreign keys, either as parent or child tables,
    up to a specified depth level.

    Args:
        connection_id: The unique identifier of the database connection.
        table_name: The name of the table to find relationships for.
        schema_name: The schema name (for PostgreSQL). Default None (uses 'public' schema).
        depth: How many levels of relationships to traverse. Default 1.
                A depth of 1 returns direct relationships, 2 includes relationships
                of the related tables, etc. Max depth is 5.
        include_details: If True, includes column information for related tables. Default False.

    Returns:
        A dictionary containing the relationship graph:
        {
            "source_table": "table_name",
            "source_schema": "schema_name", # If PG
            "relationships": { # Nested structure reflects traversal depth
                 "table": "current_table",
                 "schema": "current_schema", # If PG
                 "details": { ... } # If include_details=True
                 "parents": [ { "table": "parent", "relationship": {...}, "details": {...}, "parents": [...], "children": [...] } ],
                 "children": [ { "table": "child", "relationship": {...}, "details": {...}, "parents": [...], "children": [...] } ]
            },
            "max_depth_reached": true|false,
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists with the provided ID or the table doesn't exist.
        ToolError: If relationship discovery fails.
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"

        # Default schema for PostgreSQL is 'public'
        if db_type == "postgresql" and not schema_name:
            schema_name = "public"

        # Limit depth to prevent excessive recursion
        max_depth = 5
        max_depth_reached = False
        if depth > max_depth:
            depth = max_depth
            max_depth_reached = True
            logger.warning(f"Depth limited to maximum of {max_depth} to prevent excessive recursion")
        elif depth < 1:
             depth = 1 # Minimum depth is 1


        # Use a single connection for inspector operations
        async with engine.connect() as conn:
            conn: AsyncConnection # Type hint
            # inspector = sqlalchemy.inspect(engine) # Use inspect on engine - Causes Error
            # Get inspector using the synchronous connection from run_sync
            inspector = await conn.run_sync(sqlalchemy.inspect, conn.sync_connection)

            # Verify source table exists
            # all_table_names = await conn.run_sync(inspector.get_table_names, schema=schema_name) # Old way
            all_table_names = await conn.run_sync(lambda sync_conn: sqlalchemy.inspect(sync_conn).get_table_names(schema=schema_name))
            if table_name not in all_table_names:
                 schema_msg = f" in schema '{schema_name}'" if schema_name else ""
                 raise ToolInputError(
                    f"Source table '{table_name}' does not exist{schema_msg}.",
                    param_name="table_name",
                    provided_value=table_name
                )

            # Cache schema details to avoid redundant lookups within recursion
            # Key: (schema, table), Value: {columns: [...], fks: [...]}
            schema_cache: Dict[Tuple[Optional[str], str], Dict] = {}

            async def get_cached_table_info(tbl_schema: Optional[str], tbl_name: str) -> Dict:
                """Helper to get table info, using cache."""
                cache_key = (tbl_schema, tbl_name)
                if cache_key not in schema_cache:
                    try:
                        # cols = await conn.run_sync(inspector.get_columns, tbl_name, schema=tbl_schema)
                        # fks = await conn.run_sync(inspector.get_foreign_keys, tbl_name, schema=tbl_schema)
                        cols = await conn.run_sync(lambda sync_conn: sqlalchemy.inspect(sync_conn).get_columns(tbl_name, schema=tbl_schema))
                        fks = await conn.run_sync(lambda sync_conn: sqlalchemy.inspect(sync_conn).get_foreign_keys(tbl_name, schema=tbl_schema))
                        schema_cache[cache_key] = {"columns": cols, "fks": fks}
                    except Exception as e:
                         logger.warning(f"Failed to cache schema info for {tbl_schema}.{tbl_name}: {e}")
                         schema_cache[cache_key] = {"columns": [], "fks": [], "error": str(e)}
                return schema_cache[cache_key]

            # Pre-populate cache for all tables in the target schema to find children efficiently
            logger.debug(f"Pre-caching schema info for {len(all_table_names)} tables in schema '{schema_name}'...")
            start_cache_time = time.time()
            for t_name in all_table_names:
                 await get_cached_table_info(schema_name, t_name)
            logger.debug(f"Schema pre-caching took {time.time() - start_cache_time:.2f}s")


            # Track tables processed in the current path to avoid cycles
            # Key: (schema, table), Value: current_depth
            processed_in_path: Dict[Tuple[Optional[str], str], int] = {}

            # Recursive function to find relationships
            async def find_relationships_recursive(
                curr_table: str,
                curr_schema: Optional[str],
                current_depth: int,
                max_d: int
            ) -> Optional[Dict[str, Any]]:

                if current_depth > max_d:
                    return None # Stop recursion

                process_key = (curr_schema, curr_table)

                # Cycle detection: If we encounter the same table at a shallower or equal depth in the current path
                if process_key in processed_in_path and processed_in_path[process_key] <= current_depth:
                     logger.debug(f"Cycle detected at {process_key}, depth {current_depth}. Stopping branch.")
                     return {"cycle_detected": True} # Indicate cycle

                processed_in_path[process_key] = current_depth

                node_info = {
                    "table": curr_table,
                    "parents": [],
                    "children": []
                }
                if db_type == "postgresql":
                    node_info["schema"] = curr_schema

                # Get details if requested
                table_info = await get_cached_table_info(curr_schema, curr_table)
                if "error" in table_info:
                     node_info["error"] = f"Could not retrieve schema details: {table_info['error']}"
                     # Don't recurse further if we can't get info
                     del processed_in_path[process_key] # Remove from path tracking before returning
                     return node_info


                if include_details:
                    node_info["details"] = {
                        "columns": [
                            {"name": col["name"], "type": str(col["type"]), "primary_key": col.get("pk", False) or col.get("primary_key", False)}
                            for col in table_info.get("columns", [])
                        ]
                    }

                # --- Find Parents (Outgoing FKs) ---
                for fk in table_info.get("fks", []):
                    parent_table = fk["referred_table"]
                    parent_schema = fk.get("referred_schema") if db_type == "postgresql" else None # PG specific

                    parent_rel_info = {
                        "table": parent_table,
                        "relationship": {
                            "fk_name": fk.get("name"),
                            "child_columns": fk["constrained_columns"], # Columns in 'curr_table'
                            "parent_columns": fk["referred_columns"]  # Columns in 'parent_table'
                        }
                    }
                    if db_type == "postgresql":
                        parent_rel_info["schema"] = parent_schema

                    # Recurse
                    sub_results = await find_relationships_recursive(parent_table, parent_schema, current_depth + 1, max_d)
                    if sub_results:
                        # Merge sub-results into parent_rel_info
                        parent_rel_info.update(sub_results)

                    node_info["parents"].append(parent_rel_info)


                # --- Find Children (Incoming FKs) ---
                # Iterate through all tables *in the cache* (pre-populated for the target schema)
                for (other_schema, other_table_name), other_table_info in schema_cache.items():
                     # Skip self and tables in different schemas (unless exploring cross-schema)
                     if other_table_name == curr_table and other_schema == curr_schema:
                          continue
                     # Optimization: Only check tables within the same schema for children by default
                     if curr_schema != other_schema:
                          continue
                     if "error" in other_table_info: # Skip tables we couldn't inspect
                          continue

                     for fk in other_table_info.get("fks", []):
                         # Check if FK refers to *our* current table and schema
                         refers_to_current_table = (fk["referred_table"] == curr_table)
                         refers_to_current_schema = True # Assume true for SQLite
                         if db_type == "postgresql":
                             refers_to_current_schema = (fk.get("referred_schema") == curr_schema)

                         if refers_to_current_table and refers_to_current_schema:
                             child_table = other_table_name
                             child_schema = other_schema

                             child_rel_info = {
                                "table": child_table,
                                "relationship": {
                                    "fk_name": fk.get("name"),
                                    "child_columns": fk["constrained_columns"], # Columns in 'child_table'
                                    "parent_columns": fk["referred_columns"] # Columns in 'curr_table'
                                }
                             }
                             if db_type == "postgresql":
                                 child_rel_info["schema"] = child_schema

                             # Recurse
                             sub_results = await find_relationships_recursive(child_table, child_schema, current_depth + 1, max_d)
                             if sub_results:
                                 child_rel_info.update(sub_results)

                             node_info["children"].append(child_rel_info)

                # Backtrack: remove from current path tracking
                del processed_in_path[process_key]
                return node_info

            # Start the recursive process
            relationship_graph = await find_relationships_recursive(table_name, schema_name, 1, depth)

            final_result = {
                "source_table": table_name,
                "relationships": relationship_graph if relationship_graph else {}, # Ensure it's at least an empty dict
                "max_depth_reached": max_depth_reached,
                "success": True
            }
            if db_type == "postgresql":
                final_result["source_schema"] = schema_name


            # Simple count for logging
            parent_count = len(relationship_graph.get("parents", [])) if relationship_graph else 0
            child_count = len(relationship_graph.get("children", [])) if relationship_graph else 0

            logger.info(
                f"Successfully found relationships for table '{table_name}' (depth {depth}): {parent_count} parents, {child_count} children (direct)",
                emoji_key=TaskType.DATABASE.value,
                connection_id=connection_id,
                table_name=table_name,
                schema_name=schema_name,
                depth=depth
            )

            return final_result
    except OperationalError as e:
        error_message = f"Database connection error finding related tables: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, schema_name=schema_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"Error finding related tables for '{table_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, schema_name=schema_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error finding related tables for '{table_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, schema_name=schema_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code

@with_tool_metrics
@with_error_handling
async def analyze_column_statistics(
    connection_id: str,
    table_name: str,
    column_name: str,
    schema_name: Optional[str] = None,
    include_histogram: bool = False,
    num_buckets: int = 10,
    include_unique_values: bool = False,
    max_unique_values: int = 20
) -> Dict[str, Any]:
    """Analyzes statistics for a specific column in a table.

    This tool provides statistical information about a column's data,
    helping to understand its distribution, common values, and characteristics.

    Args:
        connection_id: The unique identifier of the database connection.
        table_name: The name of the table containing the column.
        column_name: The name of the column to analyze.
        schema_name: The schema name (for PostgreSQL). Default None (uses 'public' schema).
        include_histogram: If True, includes a histogram of value frequencies. Default False.
        num_buckets: Number of buckets for histogram if numeric column or max items for categorical. Default 10.
        include_unique_values: If True, includes a list of unique values and their counts. Default False.
        max_unique_values: Maximum number of unique values to return if include_unique_values is True. Default 20.

    Returns:
        A dictionary containing column statistics:
        {
            "table_name": "table_name",
            "column_name": "column_name",
            "data_type": "column_type",
            "basic_stats": {
                "count": 1000,
                "null_count": 10,
                "null_percentage": 1.0,
                "unique_count": 50,
                "min": "min_value",
                "max": "max_value",
                "avg": "avg_value" (for numeric columns),
                "std_dev": "stddev_value" (for numeric columns, if available),
                "min_length": 1, (for string columns)
                "max_length": 50, (for string columns)
                "avg_length": 25.5 (for string columns)
            },
            "histogram": { # For numeric: range buckets; For categorical: top N value frequencies
                "type": "numeric" | "categorical",
                "buckets": [
                    {"range": "0-10", "count": 100, "percentage": 10.0}, # Numeric
                    # OR
                    {"value": "value1", "count": 50, "percentage": 5.0}, # Categorical
                    ...
                ]
            } (if include_histogram=True),
            "value_frequencies": { # Only if include_unique_values=True
                 "values": [
                    {"value": "value1", "count": 50, "percentage": 5.0},
                    ...
                 ],
                 "truncated": true, # If more unique values exist than returned
                 "total_unique_in_table": 150 # From basic_stats
            }
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists with the provided ID, or the table or column doesn't exist.
        ToolError: If statistics calculation fails.
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"

        # Default schema for PostgreSQL is 'public'
        if db_type == "postgresql" and not schema_name:
            schema_name = "public"

        async with engine.connect() as conn:
            conn: AsyncConnection  # Type hint
            
            # Get tables using run_sync with lambda
            tables = await conn.run_sync(lambda sync_conn: sqlalchemy.inspect(sync_conn).get_table_names(schema=schema_name))
            if table_name not in tables:
                schema_msg = f" in schema '{schema_name}'" if schema_name else ""
                raise ToolInputError(
                    f"Table '{table_name}' does not exist{schema_msg}.",
                    param_name="table_name",
                    provided_value=table_name
                )

            # Verify column exists and get its type
            columns = await conn.run_sync(lambda sync_conn: sqlalchemy.inspect(sync_conn).get_columns(table_name, schema=schema_name))
            column_meta = None
            for col in columns:
                if col["name"] == column_name:
                    column_meta = col
                    break

            if not column_meta:
                raise ToolInputError(
                    f"Column '{column_name}' does not exist in table '{table_name}'.",
                    param_name="column_name",
                    provided_value=column_name
                )

            column_type = str(column_meta["type"])

            # Properly quote identifiers for use in raw SQL
            table_identifier = f'"{schema_name}"."{table_name}"' if db_type == "postgresql" else f'"{table_name}"'
            col_identifier = f'"{column_name}"'

            # Initialize result structure
            result = {
                "table_name": table_name,
                "column_name": column_name,
                "data_type": column_type,
                "basic_stats": {},
                "success": True
            }
            if db_type == "postgresql":
                result["schema_name"] = schema_name

            # --- Calculate Basic Statistics ---
            basic_stats: Dict[str, Any] = {}
            total_count = None
            null_count = None
            non_null_count = None
            unique_count = None
            unique_count_estimated = False

            try:
                # Get total and null counts in one query if possible
                count_query = text(f"""
                    SELECT COUNT(*), COUNT(CASE WHEN {col_identifier} IS NULL THEN 1 END)
                    FROM {table_identifier}
                """)
                count_res = await conn.execute(count_query)
                total_count, null_count = count_res.fetchone()
                basic_stats["count"] = total_count
                basic_stats["null_count"] = null_count
                non_null_count = total_count - null_count

                if total_count > 0:
                    basic_stats["null_percentage"] = round((null_count / total_count) * 100, 2)
                else:
                    basic_stats["null_percentage"] = 0.0

                # Determine column category
                is_numeric = any(t in column_type.lower() for t in ["int", "float", "double", "decimal", "numeric", "real"])
                is_string = any(t in column_type.lower() for t in ["char", "text", "string", "varchar"])
                is_date = any(t in column_type.lower() for t in ["date", "time", "timestamp"])

                # --- Unique Count (with estimation logic) ---
                # Only calculate if non-null values exist
                if non_null_count > 0:
                    # Prefer exact count for smaller datasets
                    if total_count < 50000:
                        try:
                            unique_query = text(f"SELECT COUNT(DISTINCT {col_identifier}) FROM {table_identifier}")
                            unique_result = await conn.execute(unique_query)
                            unique_row = unique_result.fetchone()  # Removed await
                            unique_count = unique_row[0] if unique_row else 0
                        except Exception as uc_e:
                            logger.debug(f"COUNT DISTINCT failed for {table_name}.{column_name}: {uc_e}")
                            if db_type == "postgresql": 
                                unique_count_estimated = True  # Fallback to PG stats
                    # Use PG stats estimate for large PG tables
                    elif db_type == "postgresql":
                        unique_count_estimated = True
                        try:
                            pg_stats_query = text("""
                                SELECT CASE WHEN n_distinct > 0 THEN n_distinct
                                            WHEN n_distinct < 0 THEN -n_distinct * :total_rows
                                            ELSE null END AS estimate
                                FROM pg_stats
                                WHERE schemaname = :schema_name AND tablename = :table_name AND attname = :column_name
                            """)
                            pg_stats_result = await conn.execute(pg_stats_query, {
                                "schema_name": schema_name, "table_name": table_name,
                                "column_name": column_name, "total_rows": total_count
                            })
                            pg_stats_row = pg_stats_result.fetchone()  # Removed await
                            estimated_unique = pg_stats_row[0] if pg_stats_row else None
                            if estimated_unique is not None: 
                                unique_count = int(estimated_unique)
                            else: 
                                unique_count_estimated = False  # Estimate failed
                        except Exception as pgs_e:
                            logger.warning(f"Could not query pg_stats for {table_name}.{column_name}: {pgs_e}")
                            unique_count_estimated = False

                basic_stats["unique_count"] = unique_count
                if unique_count_estimated and unique_count is not None:
                    basic_stats["unique_count_estimated"] = True

                # --- Type-Specific Stats (Min, Max, Avg, Lengths, StdDev) ---
                if non_null_count > 0:
                    if is_numeric:
                        # Min, Max, Avg, StdDev
                        stats_parts = [f"MIN({col_identifier})", f"MAX({col_identifier})", f"AVG({col_identifier})"]
                        # StdDev might not be available everywhere (e.g., older SQLite)
                        stddev_func = "STDDEV_POP" if db_type == "postgresql" else "STDEV"  # STDEV in newer SQLite? Check dialect
                        try:
                            # Test if stddev function exists with a simple query
                            await conn.execute(text(f"SELECT {stddev_func}(1)"))
                            stats_parts.append(f"{stddev_func}({col_identifier})")
                            has_stddev = True
                        except (ProgrammingError, OperationalError):
                            logger.debug(f"{stddev_func} not available for {db_type} or column type.")
                            has_stddev = False

                        num_stats_query = text(f"""
                            SELECT {', '.join(stats_parts)} FROM {table_identifier} WHERE {col_identifier} IS NOT NULL
                        """)
                        num_stats_result = await conn.execute(num_stats_query)
                        num_stats_row = num_stats_result.fetchone()  # Removed await
                        min_val, max_val, avg_val = num_stats_row if num_stats_row else (None, None, None)
                        basic_stats["min"] = min_val
                        basic_stats["max"] = max_val
                        basic_stats["avg"] = float(avg_val) if avg_val is not None else None
                        if has_stddev:
                            basic_stats["std_dev"] = float(num_stats_row[3]) if num_stats_row[3] is not None else None

                    elif is_string:
                        # Min/Max/Avg Length
                        len_func = "LENGTH"
                        str_stats_query = text(f"""
                            SELECT MIN({len_func}({col_identifier})), MAX({len_func}({col_identifier})), AVG({len_func}({col_identifier}))
                            FROM {table_identifier} WHERE {col_identifier} IS NOT NULL AND {col_identifier} != ''
                        """)  # Exclude empty strings from avg/min length
                        str_stats_result = await conn.execute(str_stats_query)
                        str_stats_row = str_stats_result.fetchone()  # Removed await
                        min_len, max_len, avg_len = str_stats_row if str_stats_row else (None, None, None)
                        basic_stats["min_length"] = min_len
                        basic_stats["max_length"] = max_len
                        basic_stats["avg_length"] = float(avg_len) if avg_len is not None else None

                    elif is_date:
                        # Min/Max Date
                        date_stats_query = text(f"""
                            SELECT MIN({col_identifier}), MAX({col_identifier})
                            FROM {table_identifier} WHERE {col_identifier} IS NOT NULL
                        """)
                        date_stats_result = await conn.execute(date_stats_query)
                        date_stats_row = date_stats_result.fetchone()  # Removed await
                        min_val, max_val = date_stats_row if date_stats_row else (None, None)
                        basic_stats["min"] = min_val.isoformat() if min_val else None
                        basic_stats["max"] = max_val.isoformat() if max_val else None

                result["basic_stats"] = basic_stats

            except (OperationalError, ProgrammingError) as e:
                logger.error(f"Error calculating basic stats for {table_name}.{column_name}: {e}", exc_info=True)
                result["basic_stats"]["error"] = f"Failed to calculate basic stats: {e}"
                # Stop further analysis if basic stats failed
                return result
            except Exception as e:
                logger.error(f"Unexpected error during basic stats for {table_name}.{column_name}: {e}", exc_info=True)
                result["basic_stats"]["error"] = f"Unexpected error during basic stats: {e}"
                return result

            # --- Histogram Calculation ---
            if include_histogram and non_null_count > 0:
                histogram = {"buckets": []}
                try:
                    if is_numeric and basic_stats.get("min") is not None and basic_stats.get("max") is not None:
                        min_val = basic_stats["min"]
                        max_val = basic_stats["max"]
                        histogram["type"] = "numeric"

                        if min_val == max_val:  # Handle case where all values are the same
                            histogram["buckets"].append({
                                "range": f"{min_val}",
                                "count": non_null_count,
                                "percentage": 100.0
                            })
                        else:
                            # Use database's width_bucket if available (PostgreSQL)
                            if db_type == "postgresql":
                                # Need num_buckets + 1 for range edges
                                bucket_query = text(f"""
                                    SELECT width_bucket({col_identifier}, :min_val, :max_val, :num_buckets) AS bucket, COUNT(*)
                                    FROM {table_identifier}
                                    WHERE {col_identifier} IS NOT NULL
                                    GROUP BY bucket ORDER BY bucket
                                """)
                                # Add small epsilon to max_val for width_bucket inclusivity
                                epsilon = (max_val - min_val) * 0.00001 if max_val != min_val else 0.001
                                bucket_res = await conn.execute(bucket_query, {
                                    "min_val": min_val, "max_val": max_val + epsilon, "num_buckets": num_buckets
                                })
                                bucket_rows = bucket_res.fetchall()  # No await needed
                                db_buckets = {b: c for b, c in bucket_rows}

                                bucket_width = (max_val - min_val) / num_buckets
                                for i in range(1, num_buckets + 1):
                                    b_count = db_buckets.get(i, 0)
                                    b_start = min_val + (i - 1) * bucket_width
                                    b_end = min_val + i * bucket_width
                                    histogram["buckets"].append({
                                        "range": f"{b_start:.4g} - {b_end:.4g}",
                                        "count": b_count,
                                        "percentage": round((b_count / non_null_count) * 100, 2) if non_null_count else 0
                                    })

                            else:  # Manual bucketing for SQLite (less efficient)
                                bucket_width = (max_val - min_val) / num_buckets
                                cases = []
                                for i in range(num_buckets):
                                    b_start = min_val + i * bucket_width
                                    b_end = min_val + (i + 1) * bucket_width
                                    # Handle inclusivity carefully
                                    lower_bound_op = ">=" if i == 0 else ">"
                                    upper_bound_op = "<="
                                    # Last bucket includes max value
                                    if i == num_buckets - 1: 
                                        upper_bound_op = "<="
                                    cases.append(f"SUM(CASE WHEN {col_identifier} {lower_bound_op} {b_start} AND {col_identifier} {upper_bound_op} {b_end} THEN 1 ELSE 0 END)")

                                manual_bucket_query = text(f"SELECT {', '.join(cases)} FROM {table_identifier} WHERE {col_identifier} IS NOT NULL")
                                manual_bucket_res = await conn.execute(manual_bucket_query)
                                counts = manual_bucket_res.fetchone()  # No await needed

                                for i in range(num_buckets):
                                    b_count = counts[i] if counts and counts[i] is not None else 0
                                    b_start = min_val + i * bucket_width
                                    b_end = min_val + (i + 1) * bucket_width
                                    histogram["buckets"].append({
                                        "range": f"{b_start:.4g} - {b_end:.4g}",
                                        "count": b_count,
                                        "percentage": round((b_count / non_null_count) * 100, 2) if non_null_count else 0
                                    })

                    elif is_string or is_date:  # Categorical histogram (top N frequencies)
                        histogram["type"] = "categorical"
                        freq_query = text(f"""
                            SELECT {col_identifier}, COUNT(*) as count
                            FROM {table_identifier}
                            WHERE {col_identifier} IS NOT NULL
                            GROUP BY {col_identifier}
                            ORDER BY count DESC
                            LIMIT :limit
                        """)
                        freq_res = await conn.execute(freq_query, {"limit": num_buckets})

                        fetched_values = freq_res.fetchall()  # Removed await
                        for value, count in fetched_values:
                            display_value = str(value.isoformat()) if is_date and hasattr(value, 'isoformat') else str(value)
                            if len(display_value) > 50: 
                                display_value = display_value[:47] + "..."
                            histogram["buckets"].append({
                                "value": display_value,
                                "count": count,
                                "percentage": round((count / non_null_count) * 100, 2) if non_null_count else 0
                            })
                    else:
                        histogram["notes"] = "Histogram not applicable for this data type."

                    result["histogram"] = histogram

                except Exception as e:
                    logger.warning(f"Histogram calculation failed for {table_name}.{column_name}: {e}")
                    result["histogram"] = {"error": f"Failed to calculate histogram: {e}"}

            # --- Value Frequencies ---
            if include_unique_values and non_null_count > 0 and unique_count is not None:
                value_freqs = {"values": []}
                try:
                    # Only fetch if unique count is reasonably small or if estimate is used
                    fetch_limit = max_unique_values
                    # If exact unique count is known and <= limit, fetch that many
                    if not basic_stats.get("unique_count_estimated") and unique_count <= max_unique_values:
                        fetch_limit = unique_count

                    freq_query = text(f"""
                        SELECT {col_identifier}, COUNT(*) as count
                        FROM {table_identifier}
                        WHERE {col_identifier} IS NOT NULL
                        GROUP BY {col_identifier}
                        ORDER BY count DESC
                        LIMIT :limit
                    """)
                    freq_res = await conn.execute(freq_query, {"limit": fetch_limit})
                    fetched_values = freq_res.fetchall()  # Removed await

                    for value, count in fetched_values:
                        display_value = str(value.isoformat()) if is_date and hasattr(value, 'isoformat') else str(value)
                        if len(display_value) > 100: 
                            display_value = display_value[:97] + "..."  # Allow longer values here
                        value_freqs["values"].append({
                            "value": display_value,
                            "count": count,
                            "percentage": round((count / non_null_count) * 100, 2) if non_null_count else 0
                        })

                    value_freqs["truncated"] = unique_count > fetch_limit
                    value_freqs["total_unique_in_table"] = unique_count
                    result["value_frequencies"] = value_freqs

                except Exception as e:
                    logger.warning(f"Value frequency calculation failed for {table_name}.{column_name}: {e}")
                    result["value_frequencies"] = {"error": f"Failed to get value frequencies: {e}"}
            elif include_unique_values:
                result["value_frequencies"] = {"notes": "Skipped due to zero non-null values or unknown unique count."}

            logger.info(
                f"Successfully analyzed statistics for column '{column_name}' in table '{table_name}'",
                emoji_key=TaskType.DATABASE.value,
                connection_id=connection_id,
                table_name=table_name,
                column_name=column_name
            )

            return result

    except OperationalError as e:
        error_message = f"Database connection error analyzing column stats: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, column_name=column_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=503) from e
    except ProgrammingError as e:
        error_message = f"Syntax error or invalid identifier analyzing column stats '{table_name}.{column_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, column_name=column_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=400) from e
    except SQLAlchemyError as e:
        error_message = f"Error analyzing column statistics for '{table_name}.{column_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, column_name=column_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e
    except Exception as e:
        error_message = f"Unexpected error analyzing column statistics for '{table_name}.{column_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, column_name=column_name, exc_info=True)
        # Log potentially complex intermediate results using json.dumps
        logger.debug(f"Intermediate state on error: {json.dumps(result, default=str)}")
        raise ToolError(message=error_message, http_status_code=500) from e


@with_tool_metrics
@with_error_handling
async def execute_query(
    connection_id: str,
    query: str,
    read_only: bool = True,
    max_rows: int = 1000
) -> Dict[str, Any]:
    """Executes a SQL query and returns the results.

    This tool allows execution of SQL queries against the connected database.
    By default, it's restricted to read-only queries for safety. Set read_only=False
    with extreme caution, as this can allow potentially destructive operations.

    Args:
        connection_id: The unique identifier of the database connection.
        query: The SQL query to execute.
        read_only: If True (default), only allows SELECT and similar read-only operations.
                  If False, allows potentially destructive operations (excluding DROP TABLE, etc.).
        max_rows: Maximum number of rows to return. Default 1000. Set to 0 or negative for no limit (use with caution).

    Returns:
        A dictionary containing query results:
        {
            "columns": ["column1", "column2", ...],
            "rows": [
                {"column1": "value1", "column2": "value2", ...},
                ...
            ],
            "row_count": 100, # Number of rows returned in this result
            "truncated": false, # True if more rows exist in DB than were returned
            "execution_time": 0.25, # Seconds
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists with the provided ID, or if the query is invalid or unsafe.
        ToolError: If query execution fails.
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        # Validate query safety first
        is_safe, reason = _is_query_safe(query)
        if not is_safe:
            raise ToolInputError(
                f"Unsafe query denied: {reason}",
                param_name="query",
                provided_value=query
            )

        # Additional check for read-only mode
        if read_only:
            normalized_query = query.strip().upper()
            # Allow CTEs (WITH) before SELECT
            if not (normalized_query.startswith('SELECT') or
                    normalized_query.startswith('WITH') or
                    normalized_query.startswith('SHOW') or    # Common in MySQL/PG
                    normalized_query.startswith('EXPLAIN') or # Common in many DBs
                    normalized_query.startswith('DESCRIBE') or # Common in MySQL/others
                    normalized_query.startswith('PRAGMA')): # SQLite specific
                raise ToolInputError(
                    "Only SELECT, WITH (... SELECT), SHOW, EXPLAIN, DESCRIBE, and PRAGMA statements are allowed in read-only mode.",
                    param_name="query",
                    provided_value=query
                )

        # Execute query
        start_time = time.time()

        async with engine.connect() as conn:
            conn: AsyncConnection # Type hint
            # Execute the query using text()
            stmt = text(query)
            result_proxy = await conn.execute(stmt)

            # Process results
            if result_proxy.returns_rows:
                # Get column names
                columns = list(result_proxy.keys()) # Ensure list

                # Fetch rows (up to max_rows + 1 to check for truncation)
                # Handle max_rows <= 0 meaning fetch all (use fetchall directly)
                fetch_limit = max_rows + 1 if max_rows > 0 else -1 # Use -1 for fetchall indication

                if fetch_limit > 0:
                     raw_rows = result_proxy.fetchmany(fetch_limit)
                     truncated = len(raw_rows) > max_rows
                     rows_to_process = raw_rows[:max_rows]
                else:
                     raw_rows = result_proxy.fetchall()
                     truncated = False
                     rows_to_process = raw_rows


                # Convert to list of dictionaries using mapping
                processed_rows = [row._mapping for row in rows_to_process]

                execution_time = time.time() - start_time

                logger.info(
                    f"Successfully executed query on connection {connection_id}",
                    emoji_key="tool",
                    connection_id=connection_id,
                    row_count_returned=len(processed_rows),
                    truncated=truncated,
                    time=execution_time
                )

                return {
                    "columns": columns,
                    "rows": processed_rows,
                    "row_count": len(processed_rows),
                    "truncated": truncated,
                    "execution_time": execution_time,
                    "success": True
                }
            else:
                # Non-row-returning queries (like CREATE VIEW or INDEX if read_only=False)
                execution_time = time.time() - start_time
                affected_rows = result_proxy.rowcount if hasattr(result_proxy, 'rowcount') else None # e.g., for INSERT, UPDATE

                logger.info(
                    f"Successfully executed non-row-returning query on connection {connection_id}",
                    emoji_key="tool",
                    connection_id=connection_id,
                    affected_rows=affected_rows,
                    time=execution_time
                )

                return {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "affected_rows": affected_rows,
                    "truncated": False,
                    "execution_time": execution_time,
                    "success": True
                }
    except ProgrammingError as e:
        error_message = f"Syntax error or access violation executing query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, exc_info=True)
        raise ToolError(message=error_message, http_status_code=400) from e # Use message and http_status_code
    except OperationalError as e:
        error_message = f"Database operational error executing query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, exc_info=True)
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"Error executing query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error executing query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code


@with_tool_metrics
@with_error_handling
async def execute_parameterized_query(
    connection_id: str,
    query: str,
    parameters: Dict[str, Any],
    read_only: bool = True,
    max_rows: int = 1000
) -> Dict[str, Any]:
    """Executes a SQL query with parameter binding for safety.

    This tool allows execution of SQL queries with parameter binding, which is
    safer than string concatenation for handling user input or variable data.
    Parameters are specified as :param_name in the query and provided as a dict.

    Args:
        connection_id: The unique identifier of the database connection.
        query: The SQL query to execute with parameter placeholders (e.g., "SELECT * FROM users WHERE id = :user_id").
        parameters: Dictionary mapping parameter names to values (e.g., {"user_id": 123}).
        read_only: If True (default), only allows SELECT and similar read-only operations.
                  If False, allows potentially destructive operations (excluding DROP TABLE, etc.).
        max_rows: Maximum number of rows to return. Default 1000. Set to 0 or negative for no limit (use with caution).

    Returns:
        A dictionary containing query results:
        {
            "columns": ["column1", "column2", ...],
            "rows": [
                {"column1": "value1", "column2": "value2", ...},
                ...
            ],
            "row_count": 100, # Number of rows returned
            "truncated": false, # True if more rows exist than were returned
            "execution_time": 0.25, # Seconds
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists, query invalid/unsafe, or parameters invalid.
        ToolError: If query execution fails.
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        # Validate query safety first
        is_safe, reason = _is_query_safe(query)
        if not is_safe:
            raise ToolInputError(
                f"Unsafe query denied: {reason}",
                param_name="query",
                provided_value=query
            )

        # Additional check for read-only mode
        if read_only:
            normalized_query = query.strip().upper()
            # Allow CTEs (WITH) before SELECT
            if not (normalized_query.startswith('SELECT') or
                    normalized_query.startswith('WITH') or
                    normalized_query.startswith('SHOW') or
                    normalized_query.startswith('EXPLAIN') or
                    normalized_query.startswith('DESCRIBE') or
                    normalized_query.startswith('PRAGMA')):
                raise ToolInputError(
                    "Only SELECT, WITH (... SELECT), SHOW, EXPLAIN, DESCRIBE, and PRAGMA statements are allowed in read-only mode.",
                    param_name="query",
                    provided_value=query
                )

        # Check parameters format
        if not isinstance(parameters, dict):
            raise ToolInputError(
                "Parameters must be provided as a dictionary.",
                param_name="parameters",
                provided_value=parameters
            )

        # Execute query
        start_time = time.time()

        async with engine.connect() as conn:
            conn: AsyncConnection # Type hint
            # Execute the query with parameters using text()
            stmt = text(query)
            result_proxy = await conn.execute(stmt, parameters)

            # Process results
            if result_proxy.returns_rows:
                 # Get column names
                columns = list(result_proxy.keys()) # Ensure list

                # Fetch rows (up to max_rows + 1 to check for truncation)
                fetch_limit = max_rows + 1 if max_rows > 0 else -1

                if fetch_limit > 0:
                     raw_rows = result_proxy.fetchmany(fetch_limit)
                     truncated = len(raw_rows) > max_rows
                     rows_to_process = raw_rows[:max_rows]
                else:
                     raw_rows = result_proxy.fetchall()
                     truncated = False
                     rows_to_process = raw_rows

                # Convert to list of dictionaries using mapping
                processed_rows = [row._mapping for row in rows_to_process]

                execution_time = time.time() - start_time

                logger.info(
                    f"Successfully executed parameterized query on connection {connection_id}",
                    emoji_key="tool",
                    connection_id=connection_id,
                    row_count_returned=len(processed_rows),
                    truncated=truncated,
                    time=execution_time
                )

                return {
                    "columns": columns,
                    "rows": processed_rows,
                    "row_count": len(processed_rows),
                    "truncated": truncated,
                    "execution_time": execution_time,
                    "success": True
                }
            else:
                # Non-row-returning queries
                execution_time = time.time() - start_time
                affected_rows = result_proxy.rowcount if hasattr(result_proxy, 'rowcount') else None

                logger.info(
                    f"Successfully executed non-row-returning parameterized query on connection {connection_id}",
                    emoji_key="tool",
                    connection_id=connection_id,
                    affected_rows=affected_rows,
                    time=execution_time
                )

                return {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "affected_rows": affected_rows,
                    "truncated": False,
                    "execution_time": execution_time,
                    "success": True
                }
    except ProgrammingError as e:
        # Could be bad syntax OR issues with parameters/binding
        error_message = f"Syntax error or parameter binding issue executing query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, params=parameters, exc_info=True)
        raise ToolError(message=error_message, http_status_code=400) from e # Use message and http_status_code
    except OperationalError as e:
        error_message = f"Database operational error executing parameterized query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, params=parameters, exc_info=True)
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e: # Catch other SQLAlchemy errors like IntegrityError etc.
        error_message = f"Error executing parameterized query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, params=parameters, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error executing parameterized query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, params=parameters, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code


@with_tool_metrics
@with_error_handling
async def create_database_view(
    connection_id: str,
    view_name: str,
    query: str,
    schema_name: Optional[str] = None,
    replace_if_exists: bool = False
) -> Dict[str, Any]:
    """Creates a database view from a query.

    This tool creates a view in the database based on a SELECT query. Views are useful
    for creating logical abstractions of complex queries or joining multiple tables.

    The operation is safe by default (fails if the view already exists). Set replace_if_exists=True
    to replace any existing view with the same name.

    Args:
        connection_id: The unique identifier of the database connection.
        view_name: The name for the new view.
        query: The SELECT query that defines the view.
        schema_name: The schema in which to create the view (PostgreSQL only). Default None (uses 'public' schema).
        replace_if_exists: If True, replaces the view if it already exists. Default False.

    Returns:
        A dictionary containing the result of the operation:
        {
            "view_name": "view_name",
            "schema_name": "schema_name" (for PostgreSQL),
            "definition": "SELECT query used to define the view",
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists, view name/query invalid, or query not SELECT.
        ToolError: If view creation fails.
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        # Validate that the query is a SELECT or WITH...SELECT statement
        normalized_query = query.strip().upper()
        if not (normalized_query.startswith('SELECT') or normalized_query.startswith('WITH')):
            raise ToolInputError(
                "Views can only be created from SELECT or WITH (... SELECT) queries.",
                param_name="query",
                provided_value=query
            )

        # Validate view name (basic check)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", view_name):
             raise ToolInputError(
                 f"Invalid view name: '{view_name}'. Must start with a letter or underscore, followed by letters, numbers, or underscores.",
                 param_name="view_name",
                 provided_value=view_name
             )

        # Determine database type
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"

        # Default schema for PostgreSQL is 'public'
        if db_type == "postgresql" and not schema_name:
            schema_name = "public"

        # Validate schema name if provided (basic check)
        if schema_name and not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", schema_name):
             raise ToolInputError(
                 f"Invalid schema name: '{schema_name}'.",
                 param_name="schema_name",
                 provided_value=schema_name
             )

        # Construct CREATE VIEW statement with proper quoting
        # Use sqlalchemy identifier preparation for safety? Less readable for simple case. Use manual quoting.
        quoted_view_name = f'"{view_name}"'
        if db_type == "postgresql":
            quoted_schema_name = f'"{schema_name}"'
            full_view_name = f"{quoted_schema_name}.{quoted_view_name}"
            create_prefix = "CREATE OR REPLACE VIEW" if replace_if_exists else "CREATE VIEW"
            view_sql = f"{create_prefix} {full_view_name} AS \n{query}" # Add newline for readability
        else:
            # SQLite: Handle replacement manually if needed
            full_view_name = quoted_view_name
            if replace_if_exists:
                # Execute DROP and CREATE in separate steps within the transaction
                drop_sql = f"DROP VIEW IF EXISTS {full_view_name}"
                create_sql = f"CREATE VIEW {full_view_name} AS \n{query}"
                view_sql = [drop_sql, create_sql] # List of statements for SQLite replacement
            else:
                view_sql = f"CREATE VIEW {full_view_name} AS \n{query}"


        # Execute the CREATE VIEW statement within a transaction
        retrieved_view_def = None
        async with engine.begin() as conn: # Use begin for implicit transaction
            conn: AsyncConnection # Type hint
            if isinstance(view_sql, list): # SQLite replace logic
                 for sql_step in view_sql:
                      await conn.execute(text(sql_step))
            else: # Single statement for PG or SQLite create
                 await conn.execute(text(view_sql))

            # Get the view definition to confirm (best effort)
            inspector = sqlalchemy.inspect(engine)
            try:
                 retrieved_view_def = await conn.run_sync(inspector.get_view_definition, view_name, schema=schema_name)
            except Exception as e_def:
                 logger.warning(f"Could not retrieve view definition for '{view_name}' after creation: {e_def}")


        logger.info(
            f"Successfully created{' or replaced' if replace_if_exists else ''} view '{view_name}'",
            emoji_key="tool",
            connection_id=connection_id,
            view_name=view_name,
            schema_name=schema_name
        )

        result = {
            "view_name": view_name,
            # Return original query as definition if retrieval failed
            "definition": retrieved_view_def or query,
            "success": True
        }

        if db_type == "postgresql":
            result["schema_name"] = schema_name

        return result
    except ProgrammingError as e:
        # Common for "view already exists" if replace_if_exists=False, or syntax errors
        error_message = f"Error creating view '{view_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, view_name=view_name, exc_info=True)
        # Check if it's an "already exists" error
        if "already exists" in str(e).lower() and not replace_if_exists:
            raise ToolInputError(
                f"View '{view_name}' already exists. Set replace_if_exists=True to overwrite.",
                param_name="view_name", provided_value=view_name
            ) from e
        else:
             raise ToolError(message=error_message, http_status_code=400) from e # Likely syntax error # Use message and http_status_code
    except OperationalError as e:
        error_message = f"Database operational error creating view '{view_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, view_name=view_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"Error creating view '{view_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, view_name=view_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error creating view '{view_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, view_name=view_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code


@with_tool_metrics
@with_error_handling
async def create_database_index(
    connection_id: str,
    table_name: str,
    column_names: List[str],
    index_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    unique: bool = False,
    concurrent: bool = False # Note: CONCURRENTLY is PG specific
) -> Dict[str, Any]:
    """Creates an index on a table to improve query performance.

    This tool creates an index on one or more columns of a table.
    Indexes are useful for optimizing queries that filter, sort, or join on the indexed columns.

    Args:
        connection_id: The unique identifier of the database connection.
        table_name: The name of the table to create the index on.
        column_names: A list of column names to include in the index.
        index_name: Optional name for the index. If not provided, a name will be generated
                   (e.g., idx_tablename_col1_col2). Max length may apply.
        schema_name: The schema containing the table (PostgreSQL only). Default None (uses 'public' schema).
        unique: If True, creates a unique index that enforces uniqueness of the indexed columns. Default False.
        concurrent: If True, uses CREATE INDEX CONCURRENTLY (PostgreSQL only) to avoid
                    locking the table during creation. Requires non-transactional execution. Default False.

    Returns:
        A dictionary containing the result of the operation:
        {
            "index_name": "index_name",
            "table_name": "table_name",
            "schema_name": "schema_name" (for PostgreSQL),
            "columns": ["column1", "column2", ...],
            "unique": true|false,
            "concurrent": true|false (PostgreSQL only),
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists, or if the parameters are invalid (e.g., bad names, empty columns).
        ToolError: If index creation fails (e.g., duplicate index name, invalid column, permissions).
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        # Validate parameters
        if not column_names or not isinstance(column_names, list) or not all(isinstance(c, str) for c in column_names):
            raise ToolInputError(
                "column_names must be a non-empty list of strings.",
                param_name="column_names",
                provided_value=column_names
            )
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
             raise ToolInputError(f"Invalid table name: '{table_name}'.", param_name="table_name", provided_value=table_name)
        for col in column_names:
             if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", col):
                  raise ToolInputError(f"Invalid column name: '{col}'.", param_name="column_names", provided_value=column_names)


        # Determine database type
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"

        # Default schema for PostgreSQL is 'public'
        if db_type == "postgresql" and not schema_name:
            schema_name = "public"
        if schema_name and not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", schema_name):
             raise ToolInputError(f"Invalid schema name: '{schema_name}'.", param_name="schema_name", provided_value=schema_name)


        # Generate index name if not provided, ensure reasonable length
        if not index_name:
            columns_part = "_".join(col.lower() for col in column_names[:3]) # Max 3 cols in name
            if len(column_names) > 3:
                columns_part += "_etc"
            base_name = f"idx_{table_name}_{columns_part}"
            # Max identifier length varies (e.g., 63 for PG), truncate safely
            index_name = base_name[:60] # Truncate to be safe across DBs
        elif not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", index_name):
             raise ToolInputError(f"Invalid index name: '{index_name}'.", param_name="index_name", provided_value=index_name)


        # Construct CREATE INDEX statement with proper quoting
        quoted_index_name = f'"{index_name}"'
        quoted_table_name = f'"{table_name}"'
        quoted_columns = [f'"{col}"' for col in column_names]
        columns_sql = f"({', '.join(quoted_columns)})"

        if db_type == "postgresql":
            quoted_schema_name = f'"{schema_name}"'
            full_table_name = f"{quoted_schema_name}.{quoted_table_name}"
            unique_str = "UNIQUE " if unique else ""
            # Handle CONCURRENTLY - cannot be run inside transaction block
            concurrent_str = "CONCURRENTLY " if concurrent else ""
            if concurrent and unique:
                 raise ToolInputError("Cannot create a UNIQUE index CONCURRENTLY in PostgreSQL.", param_name="concurrent/unique")

            index_sql = f"CREATE {unique_str}INDEX {concurrent_str}{quoted_index_name} ON {full_table_name} {columns_sql}"
        else:
            # SQLite syntax
            if concurrent:
                 logger.warning("CONCURRENTLY flag is ignored for SQLite.")
                 concurrent = False # Reset flag as it's not used
            full_table_name = quoted_table_name
            unique_str = "UNIQUE " if unique else ""
            index_sql = f"CREATE {unique_str}INDEX {quoted_index_name} ON {full_table_name} {columns_sql}"


        # Execute the CREATE INDEX statement
        # PG CONCURRENTLY must run outside a transaction
        if db_type == "postgresql" and concurrent:
             async with engine.connect() as conn:
                  conn: AsyncConnection
                  # Need to set isolation level for concurrent index creation
                  await conn.execution_options(isolation_level="AUTOCOMMIT")
                  await conn.execute(text(index_sql))
        else:
             # Run within a transaction for atomicity (default)
             async with engine.begin() as conn:
                  conn: AsyncConnection
                  await conn.execute(text(index_sql))

        logger.info(
            f"Successfully created{' UNIQUE' if unique else ''} index '{index_name}' on table '{table_name}'",
            emoji_key=TaskType.DATABASE.value,
            connection_id=connection_id,
            table_name=table_name,
            schema_name=schema_name,
            index_name=index_name,
            concurrent=concurrent
        )

        result = {
            "index_name": index_name,
            "table_name": table_name,
            "columns": column_names,
            "unique": unique,
            "success": True
        }

        if db_type == "postgresql":
            result["schema_name"] = schema_name
            result["concurrent"] = concurrent

        return result
    except ProgrammingError as e:
        # Common for "index already exists", "column does not exist", permission errors
        error_message = f"Error creating index '{index_name}' on '{table_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, index_name=index_name, exc_info=True)
        # Provide more specific feedback if possible
        if "already exists" in str(e).lower():
            raise ToolError(message=f"Index '{index_name}' already exists.", http_status_code=409) from e # Conflict # Use message and http_status_code
        elif "does not exist" in str(e).lower():
             raise ToolInputError(f"Table or column specified for index '{index_name}' does not exist.", param_name="table/columns") from e
        else:
            raise ToolError(message=error_message, http_status_code=400) from e # Bad Request (syntax, permissions etc) # Use message and http_status_code
    except OperationalError as e:
        error_message = f"Database operational error creating index '{index_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, index_name=index_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=503) from e # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"Error creating index '{index_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, index_name=index_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error creating index '{index_name}': {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, table_name=table_name, index_name=index_name, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code

@with_tool_metrics
@with_error_handling
@with_retry(max_retries=2, retry_delay=0.5, retry_exceptions=(OperationalError,)) # Use max_retries instead of attempts
async def test_connection(connection_id: str) -> Dict[str, Any]:
    """Tests the database connection by executing a simple query.

    Use this tool to verify that a connection is still active and properly functioning,
    especially after periods of inactivity.

    Args:
        connection_id: The unique identifier of the database connection to test.

    Returns:
        A dictionary containing the test results:
        {
            "connection_id": "connection-id",
            "active": true,
            "database_type": "sqlite" or "postgresql",
            "response_time": 0.05, # Time in seconds for the test query to complete
            "version": "Database version info",
            "success": true
        }

    Raises:
        ToolInputError: If no connection exists with the provided ID.
        ToolError: If the test query fails, indicating connection issues.
    """
    start_time = time.time()
    engine = await _validate_and_get_engine(connection_id)

    try:
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"
        version = None

        # Perform a simple, fast, read-only test query
        async with engine.connect() as conn:
            conn: AsyncConnection # Type hint
            if db_type == "postgresql":
                # SELECT 1 is very standard and fast
                test_query = text("SELECT 1")
                version_query = text("SELECT version()")
            else: # SQLite
                test_query = text("SELECT 1")
                version_query = text("SELECT sqlite_version()")

            await conn.execute(test_query) # Execute simple query first
            version_result = await conn.execute(version_query) # Then get version
            version = version_result.scalar()

        response_time = time.time() - start_time

        logger.info(
            f"Connection test successful for {connection_id}",
            emoji_key="tool",
            connection_id=connection_id,
            response_time=response_time
        )

        return {
            "connection_id": connection_id,
            "active": True,
            "database_type": db_type,
            "response_time": response_time,
            "version": version,
            "success": True
        }
    except OperationalError as e:
        error_message = f"Connection test failed for {connection_id}: Database operational error - {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id)
        # Don't remove connection here, allow retry or explicit disconnect
        raise ToolError(message=error_message, http_status_code=503) from e # Service Unavailable # Use message and http_status_code
    except SQLAlchemyError as e:
        error_message = f"Connection test failed for {connection_id}: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code
    except Exception as e:
        error_message = f"Unexpected error during connection test for {connection_id}: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, exc_info=True)
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code


@with_tool_metrics
@with_error_handling
async def execute_transaction(
    connection_id: str,
    queries: List[str],
    parameters: Optional[List[Optional[Dict[str, Any]]]] = None, # Allow None for individual query params
    read_only: bool = True
) -> Dict[str, Any]:
    """Executes multiple SQL queries in a single atomic transaction.

    This tool allows you to run multiple SQL queries as a single atomic transaction,
    ensuring that either all queries succeed or none do (rollback on failure).
    This is useful for operations that need to maintain data consistency.

    Args:
        connection_id: The unique identifier of the database connection.
        queries: A list of SQL queries to execute in order.
        parameters: Optional list of parameter dictionaries, one per query.
                   If provided, must have the same length as queries. Use None or {}
                   for queries without parameters.
        read_only: If True (default), only allows read queries like SELECT, WITH, SHOW etc.
                  If False, allows other operations (except explicitly prohibited ones like DROP TABLE).

    Returns:
        A dictionary containing the results of the transaction:
        {
            "results": [
                {
                    "query_index": 0,
                    "returns_rows": true,
                    "columns": ["col1", ...],
                    "rows": [{"col1": "value1", ...}, ...],
                    "row_count": 5, # Rows returned by this specific query
                    "affected_rows": null # Or number if DML
                },
                {
                    "query_index": 1,
                    "returns_rows": false,
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "affected_rows": 1
                },
                ...
            ],
            "execution_time": 0.15, # Seconds for the whole transaction
            "success": true
        }

    Raises:
        ToolInputError: If connection ID invalid, queries/params invalid or unsafe.
        ToolError: If the transaction fails (rollback occurs).
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        # Validate queries list
        if not queries or not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
            raise ToolInputError(
                "queries must be a non-empty list of SQL query strings.",
                param_name="queries",
                provided_value=queries
            )

        # Validate parameters list if provided
        if parameters:
            if not isinstance(parameters, list):
                raise ToolInputError(
                    "parameters must be a list.",
                    param_name="parameters",
                    provided_value=parameters
                )
            if len(parameters) != len(queries):
                raise ToolInputError(
                    f"parameters list length ({len(parameters)}) must match queries list length ({len(queries)}).",
                    param_name="parameters",
                    provided_value=parameters
                )
             # Check individual parameter dictionaries (allow None or dict)
            for i, p in enumerate(parameters):
                 if p is not None and not isinstance(p, dict):
                     raise ToolInputError(
                         f"Element at index {i} in parameters list must be a dictionary or None.",
                         param_name=f"parameters[{i}]",
                         provided_value=p
                     )
        else:
            # If no parameters list is given, create a list of None for each query
            parameters = [None] * len(queries)


        # Check each query for safety and read-only compliance
        for i, query in enumerate(queries):
            is_safe, reason = _is_query_safe(query)
            if not is_safe:
                # Check if the reason is INSERT/UPDATE and read_only is False
                is_write_op = re.search(r"^\s*(INSERT|UPDATE)\s", query.strip().upper(), re.IGNORECASE)
                if is_write_op and not read_only:
                    # Allow INSERT/UPDATE if explicitly not read-only
                    pass # Continue to the next query
                else:
                    # Otherwise, raise the error (either prohibited op, or write op in read_only mode)
                    raise ToolInputError(
                        f"Unsafe query denied in transaction at position {i}: {reason}",
                        param_name=f"queries[{i}]",
                        provided_value=query
                    )

            # Additional check for read-only mode (redundant if _is_query_safe blocks writes, but kept for clarity)
            if read_only:
                normalized_query = query.strip().upper()
                if not (normalized_query.startswith('SELECT') or
                        normalized_query.startswith('WITH') or
                        normalized_query.startswith('SHOW') or
                        normalized_query.startswith('EXPLAIN') or
                        normalized_query.startswith('DESCRIBE') or
                        normalized_query.startswith('PRAGMA')):
                    raise ToolInputError(
                        f"Only read operations allowed in read-only transaction. Query at index {i} is restricted.",
                        param_name=f"queries[{i}]",
                        provided_value=query
                    )

        # Execute transaction using engine.begin() for automatic rollback on error
        start_time = time.time()
        results = []

        async with engine.begin() as conn: # Starts transaction, commits on success, rolls back on error
            conn: AsyncConnection # Type hint
            for i, (query, params) in enumerate(zip(queries, parameters, strict=False)):
                # Use empty dict if params is None for execute call
                current_params = params or {}
                result_proxy = await conn.execute(text(query), current_params)

                # Process result for this query
                if result_proxy.returns_rows:
                    columns = list(result_proxy.keys())
                    # Fetch all rows for this query within the transaction
                    raw_rows = result_proxy.fetchall()
                    processed_rows = [row._mapping for row in raw_rows]

                    results.append({
                        "query_index": i,
                        "returns_rows": True,
                        "columns": columns,
                        "rows": processed_rows,
                        "row_count": len(processed_rows),
                        "affected_rows": None # Not applicable for SELECT typically
                    })
                else:
                    # Non-row-returning query (e.g., INSERT, UPDATE if read_only=False)
                    affected_rows = result_proxy.rowcount if hasattr(result_proxy, 'rowcount') else None
                    results.append({
                        "query_index": i,
                        "returns_rows": False,
                        "columns": [],
                        "rows": [],
                        "row_count": 0,
                        "affected_rows": affected_rows
                    })

        # If we reach here, the transaction committed successfully
        execution_time = time.time() - start_time

        logger.info(
            f"Successfully executed transaction with {len(queries)} queries",
            emoji_key="tool",
            connection_id=connection_id,
            query_count=len(queries),
            time=execution_time
        )

        return {
            "results": results,
            "execution_time": execution_time,
            "success": True
        }
    except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
        # Catch errors that would cause rollback within engine.begin()
        # ProgrammingError includes syntax errors, access violations etc.
        # OperationalError includes connection drops, deadlocks etc.
        error_type = type(e).__name__
        error_message = f"Transaction failed and rolled back due to {error_type}: {str(e)}"
        logger.error(
            error_message,
            emoji_key="error",
            connection_id=connection_id,
            failed_query_index=i if 'i' in locals() else 'unknown', # Log which query might have failed
            exc_info=True
        )
        # Log the queries/params involved if helpful (potentially large)
        try:
             logger.debug(f"Transaction details on failure: queries={json.dumps(queries)}, params={json.dumps(parameters, default=str)}")
        except Exception:
             logger.debug("Could not serialize transaction details for logging.")

        # Determine appropriate status code
        status_code = 400 if isinstance(e, ProgrammingError) else (503 if isinstance(e, OperationalError) else 500)
        raise ToolError(message=error_message, http_status_code=status_code) from e # Use message and http_status_code
    except Exception as e: # Catch unexpected errors outside SQLAlchemy
        error_message = f"Unexpected error during transaction execution: {str(e)}"
        logger.error(
            error_message,
            emoji_key="error",
            connection_id=connection_id,
            exc_info=True
        )
        raise ToolError(message=error_message, http_status_code=500) from e # Use message and http_status_code


@with_tool_metrics
@with_error_handling
async def execute_query_with_pagination(
    connection_id: str,
    query: str,
    page_size: int = 100,
    page_number: int = 1,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Executes a SQL query with pagination for large result sets.

    This tool allows execution of SQL queries with results divided into pages,
    making it suitable for handling large result sets efficiently. It only supports
    read-only SELECT or WITH (... SELECT) queries.

    Args:
        connection_id: The unique identifier of the database connection.
        query: The SQL query to execute (must be SELECT or WITH...SELECT).
        page_size: Number of rows per page. Default 100. Max 1000.
        page_number: Page number to retrieve (1-based). Default 1.
        parameters: Optional dictionary of query parameters for parameterized queries.

    Returns:
        A dictionary containing paginated query results:
        {
            "columns": ["column1", "column2", ...],
            "rows": [
                {"column1": "value1", "column2": "value2", ...},
                ...
            ],
            "row_count": 100, # Rows returned on this page
            "pagination": {
                "page": 1,
                "page_size": 100,
                "total_pages": 5, # Estimated if total_rows is estimated
                "total_rows": 470, # May be estimated for performance on large tables
                "total_rows_estimated": true|false,
                "has_next_page": true,
                "has_previous_page": false
            },
            "execution_time": 0.25, # Seconds
            "success": true
        }

    Raises:
        ToolInputError: If connection ID invalid, query invalid/not SELECT, or pagination params invalid.
        ToolError: If query execution fails.
    """
    engine = await _validate_and_get_engine(connection_id)

    try:
        # Normalize query - ensure it's a SELECT or WITH...SELECT
        normalized_query = query.strip().upper()
        if not (normalized_query.startswith('SELECT') or normalized_query.startswith('WITH')):
            raise ToolInputError(
                "Only SELECT or WITH (... SELECT) statements are allowed with pagination.",
                param_name="query",
                provided_value=query
            )

        # Validate pagination parameters
        if not isinstance(page_size, int) or page_size < 1:
            raise ToolInputError("Page size must be a positive integer.", param_name="page_size", provided_value=page_size)
        # Add a reasonable upper limit to page size
        max_page_size = 1000
        if page_size > max_page_size:
            logger.warning(f"Requested page_size {page_size} exceeds maximum of {max_page_size}. Clamping.")
            page_size = max_page_size

        if not isinstance(page_number, int) or page_number < 1:
            raise ToolInputError("Page number must be a positive integer (1-based).", param_name="page_number", provided_value=page_number)

        # Initialize parameters dictionary if None
        query_parameters = parameters or {}
        if not isinstance(query_parameters, dict):
             raise ToolInputError("Parameters must be a dictionary or None.", param_name="parameters", provided_value=parameters)


        # Determine database type
        db_type = "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"  # noqa: F841

        # Execute query
        start_time = time.time()
        total_rows = 0
        total_rows_estimated = False

        async with engine.connect() as conn:
            conn: AsyncConnection # Type hint

            # --- Get Total Row Count ---
            # Construct count query based on the original query
            # Removing ORDER BY from the count query can improve performance but might be complex to parse reliably.
            # Simpler approach: wrap the original query.
            count_query_sql = f"SELECT COUNT(*) FROM ({query}) AS _pagination_count_subquery"
            try:
                count_stmt = text(count_query_sql)
                # Time limit the count query to avoid blocking on very large tables
                # Note: `timeout()` is not directly supported by execute. Need external mechanism if required.
                # Consider alternative estimation for very large tables if this is too slow.
                count_result = await conn.execute(count_stmt, query_parameters)
                total_rows = count_result.scalar()
                if total_rows is None: 
                    total_rows = 0 # Ensure it's an int
            except (OperationalError, ProgrammingError, SQLAlchemyError) as count_e:
                logger.warning(f"Could not execute exact count query due to: {count_e}. Results will be paginated without total count.")
                # We can still proceed with pagination, but won't know total pages/rows
                total_rows = None # Indicate unknown count
                total_rows_estimated = True # Mark as estimate (unknown)
            except Exception as count_e:
                 logger.error(f"Unexpected error executing count query: {count_e}", exc_info=True)
                 total_rows = None
                 total_rows_estimated = True


            # --- Calculate Pagination Details ---
            offset = (page_number - 1) * page_size
            has_previous_page = page_number > 1
            total_pages = None
            has_next_page = None # We determine this by fetching one extra row

            if total_rows is not None:
                 total_pages = max(1, (total_rows + page_size - 1) // page_size)
                 # We still check has_next_page by fetching extra row for robustness
                 # has_next_page = page_number < total_pages


            # --- Construct and Execute Paginated Query ---
            # Append LIMIT and OFFSET clauses. Syntax varies slightly.
            # Standard SQL (PostgreSQL, newer SQLite) uses LIMIT/OFFSET
            # Older DBs might use different syntax, but we only support PG/SQLite here.
            paginated_query_sql = f"{query} LIMIT :_page_size OFFSET :_offset"

            # Combine original params with pagination params
            execution_params = dict(query_parameters)
            execution_params["_page_size"] = page_size + 1 # Fetch one extra row to check for next page
            execution_params["_offset"] = offset

            paginated_stmt = text(paginated_query_sql)
            paginated_result_proxy = await conn.execute(paginated_stmt, execution_params)

            # Get column names
            columns = list(paginated_result_proxy.keys())

            # Fetch rows for this page (+1)
            raw_rows = paginated_result_proxy.fetchall()

            # Determine if there's a next page based on the extra row fetched
            has_next_page = len(raw_rows) > page_size
            # Get the actual rows for the current page
            rows_to_process = raw_rows[:page_size]

            # Convert to list of dictionaries
            processed_rows = [row._mapping for row in rows_to_process]

            execution_time = time.time() - start_time

            logger.info(
                f"Successfully executed paginated query on connection {connection_id}",
                emoji_key="tool",
                connection_id=connection_id,
                page=page_number,
                page_size=page_size,
                rows_returned=len(processed_rows),
                has_next=has_next_page,
                total_rows_known=total_rows is not None,
                time=execution_time
            )

            return {
                "columns": columns,
                "rows": processed_rows,
                "row_count": len(processed_rows),
                "pagination": {
                    "page": page_number,
                    "page_size": page_size,
                    "total_pages": total_pages, # May be None if count failed
                    "total_rows": total_rows,   # May be None if count failed
                    "total_rows_estimated": total_rows_estimated or (total_rows is None),
                    "has_next_page": has_next_page,
                    "has_previous_page": has_previous_page
                },
                "execution_time": execution_time,
                "success": True
            }
    except ProgrammingError as e:
        error_message = f"Syntax error or parameter binding issue executing paginated query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, params=parameters, exc_info=True)
        raise ToolError(message=error_message, http_status_code=400) from e # Use message and http_status_code
    except OperationalError as e:
        error_message = f"Database operational error executing paginated query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, params=parameters, exc_info=True)
        raise ToolError(status_code=503, detail=error_message) from e
    except SQLAlchemyError as e:
        error_message = f"Error executing paginated query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, params=parameters, exc_info=True)
        raise ToolError(http_status_code=500, message=error_message) from e # Corrected param name, Use message
    except Exception as e:
        error_message = f"Unexpected error executing paginated query: {str(e)}"
        logger.error(error_message, emoji_key="error", connection_id=connection_id, query=query, params=parameters, exc_info=True)
        # Log potentially complex intermediate results using json.dumps
        logger.debug(f"Paginated query state on error: query={query}, params={json.dumps(parameters, default=str)}")
        raise ToolError(http_status_code=500, message=error_message) from e # Corrected param name, Use message
