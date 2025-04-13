#!/usr/bin/env python
"""Demo script showcasing the SQL database interactions tools."""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Rich imports for nice UI
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from llm_gateway.tools.sql_database_interactions import (
    analyze_column_statistics,
    connect_to_database,
    create_database_index,
    create_database_view,
    disconnect_from_database,
    discover_database_schema,
    execute_parameterized_query,
    execute_query,
    execute_query_with_pagination,
    execute_transaction,
    find_related_tables,
    generate_database_documentation,
    get_database_status,
    get_table_details,
    test_connection,
)
from llm_gateway.utils import get_logger

# Initialize Rich console and logger
console = Console()
logger = get_logger("example.sql_database_interactions")

# --- Demo Configuration ---
# SQLite in-memory database for demonstration
DEFAULT_CONNECTION_STRING = "sqlite:///demo_database.db"
# You can replace with a more complex connection string like:
# "postgresql://username:password@localhost:5432/demo_db"

# --- Helper Functions for Demo Data Setup ---

async def setup_demo_database(connection_id: str) -> None:
    """Set up demo database with sample tables and data."""
    logger.info("Setting up demo database with sample tables and data", emoji_key="db")
    
    # Create sample tables with relationships
    setup_queries = [
        """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            signup_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT CHECK(status IN ('active', 'inactive', 'pending')) DEFAULT 'pending'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            price DECIMAL(10,2) NOT NULL,
            category TEXT,
            in_stock BOOLEAN DEFAULT 1
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_amount DECIMAL(10,2) NOT NULL,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            price_per_unit DECIMAL(10,2) NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
        """
    ]
    
    # Insert sample data
    sample_data_queries = [
        """
        INSERT INTO customers (name, email, status)
        VALUES 
            ('Alice Johnson', 'alice@example.com', 'active'),
            ('Bob Smith', 'bob@example.com', 'active'),
            ('Charlie Davis', 'charlie@example.com', 'inactive'),
            ('Diana Miller', 'diana@example.com', 'active'),
            ('Edward Wilson', 'edward@example.com', 'pending')
        """,
        """
        INSERT INTO products (name, description, price, category, in_stock)
        VALUES
            ('Laptop Pro', 'High-performance laptop', 1299.99, 'Electronics', 1),
            ('Smartphone X', 'Latest smartphone model', 799.99, 'Electronics', 1),
            ('Wireless Headphones', 'Noise-cancelling headphones', 199.99, 'Audio', 1),
            ('Coffee Maker', 'Programmable coffee machine', 89.99, 'Kitchen', 0),
            ('Fitness Tracker', 'Waterproof fitness band', 49.99, 'Wearables', 1),
            ('Desk Lamp', 'Adjustable LED lamp', 29.99, 'Home', 1),
            ('Ergonomic Chair', 'Office chair with lumbar support', 249.99, 'Furniture', 0)
        """,
        """
        INSERT INTO orders (customer_id, total_amount, status)
        VALUES
            (1, 1499.98, 'completed'),
            (2, 89.99, 'processing'),
            (1, 249.99, 'completed'),
            (3, 1099.98, 'completed'),
            (4, 49.99, 'processing')
        """,
        """
        INSERT INTO order_items (order_id, product_id, quantity, price_per_unit)
        VALUES
            (1, 1, 1, 1299.99),
            (1, 3, 1, 199.99),
            (2, 4, 1, 89.99),
            (3, 7, 1, 249.99),
            (4, 2, 1, 799.99),
            (4, 5, 6, 49.99),
            (5, 5, 1, 49.99)
        """
    ]
    
    # Use the execute_transaction tool to run all setup queries
    try:
        transaction_result = await execute_transaction(
            connection_id=connection_id,
            queries=setup_queries,
            read_only=False
        )
        
        if transaction_result.get("success"):
            logger.success("Successfully created sample tables", emoji_key="success")
            
            # Insert sample data in a separate transaction
            data_result = await execute_transaction(
                connection_id=connection_id,
                queries=sample_data_queries,
                read_only=False
            )
            
            if data_result.get("success"):
                logger.success("Successfully inserted sample data", emoji_key="success")
            else:
                logger.error(f"Failed to insert sample data: {data_result.get('error')}", emoji_key="error")
        else:
            logger.error(f"Failed to create sample tables: {transaction_result.get('error')}", emoji_key="error")
    except Exception as e:
        logger.error(f"Error setting up demo database: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error setting up demo database:[/bold red] {escape(str(e))}")

def display_query_result(title: str, result: Dict[str, Any], include_stats: bool = True) -> None:
    """Display query result with consistent formatting."""
    console.print(Rule(f"[bold blue]{escape(title)}[/bold blue]"))
    
    if not result.get("success", False):
        console.print(Panel(
            f"[bold red]Error:[/bold red] {escape(result.get('error', 'Unknown error'))}",
            title="Query Failed",
            border_style="red",
            expand=False
        ))
        return
    
    # Handle different result formats
    rows = result.get("rows", [])
    if not rows:
        console.print("[yellow]No results returned[/yellow]")
        return
    
    # Create table for displaying results
    table = Table(title=f"Results ({len(rows)} rows)", box=None, show_header=True)
    
    # Add columns based on first row keys
    if rows and isinstance(rows[0], dict):
        for column in rows[0].keys():
            table.add_column(column, style="cyan")
        
        # Add data rows
        for row in rows:
            table.add_row(*[str(value) for value in row.values()])
    else:
        # Handle list results
        table.add_column("Result", style="cyan")
        for row in rows:
            table.add_row(str(row))
    
    console.print(table)
    
    # Display statistics if available and requested
    if include_stats:
        stats_table = Table(title="Query Stats", show_header=False, box=None)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        for key, value in result.items():
            if key in ["rows", "success", "error"]:
                continue
            if key == "execution_time":
                stats_table.add_row(key, f"{value:.4f}s")
            elif key == "row_count":
                stats_table.add_row(key, str(value))
            elif key == "query":
                # Truncate long queries
                truncated = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                stats_table.add_row(key, truncated)
        
        console.print(stats_table)
    console.print()

# --- Demo Functions ---

async def connection_demo() -> str:
    """Demonstrate database connection and status checking."""
    console.print(Rule("[bold blue]Database Connection Demo[/bold blue]"))
    logger.info("Starting database connection demo", emoji_key="start")
    
    connection_id = None
    
    try:
        # Connect to the database
        connection_result = await connect_to_database(
            connection_string=DEFAULT_CONNECTION_STRING,
            connection_options={"echo": True},
            echo=True
        )
        
        if connection_result.get("success"):
            connection_id = connection_result.get("connection_id")
            logger.success(f"Successfully connected to database with ID: {connection_id}", emoji_key="success")
            
            # Display connection details in a panel
            console.print(Panel(
                f"Connection ID: [bold green]{escape(connection_id)}[/bold green]\n"
                f"Database Type: [blue]{escape(connection_result.get('database_type', 'Unknown'))}[/blue]\n"
                f"Connection String: [dim]{escape(DEFAULT_CONNECTION_STRING)}[/dim]",
                title="Database Connection",
                border_style="green",
                expand=False
            ))
            
            # Test the connection
            test_result = await test_connection(connection_id=connection_id)
            
            if test_result.get("success"):
                console.print("[green]Connection test succeeded.[/green]")
            else:
                console.print(f"[bold red]Connection test failed:[/bold red] {escape(test_result.get('error', 'Unknown error'))}")
            
            # Get connection status
            status_result = await get_database_status(connection_id=connection_id)
            
            if status_result.get("success"):
                status_table = Table(title="Database Status", box=None)
                status_table.add_column("Metric", style="cyan")
                status_table.add_column("Value", style="white")
                
                for key, value in status_result.items():
                    if key in ["success", "error", "connection_id"]:
                        continue
                    status_table.add_row(key, str(value))
                
                console.print(status_table)
            else:
                console.print(f"[bold red]Failed to get database status:[/bold red] {escape(status_result.get('error', 'Unknown error'))}")
                
        else:
            logger.error(f"Failed to connect to database: {connection_result.get('error')}", emoji_key="error")
            console.print(f"[bold red]Connection failed:[/bold red] {escape(connection_result.get('error', 'Unknown error'))}")
    
    except Exception as e:
        logger.error(f"Error in connection demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error in connection demo:[/bold red] {escape(str(e))}")
    
    console.print()
    return connection_id

async def schema_discovery_demo(connection_id: str) -> None:
    """Demonstrate database schema discovery."""
    console.print(Rule("[bold blue]Schema Discovery Demo[/bold blue]"))
    logger.info("Starting schema discovery demo", emoji_key="start")
    
    try:
        # Discover database schema
        logger.info("Discovering database schema", emoji_key="db")
        schema_result = await discover_database_schema(
            connection_id=connection_id,
            include_indexes=True,
            include_foreign_keys=True,
            detailed=True
        )
        
        if schema_result.get("success"):
            schema_data = schema_result.get("schema", {})
            tables = schema_data.get("tables", [])
            
            if tables:
                console.print(f"[green]Successfully discovered schema with {len(tables)} tables[/green]")
                
                # Display tables in a table
                tables_table = Table(title="Database Tables", box=None)
                tables_table.add_column("Table Name", style="cyan")
                tables_table.add_column("# Columns", style="white")
                tables_table.add_column("# Indexes", style="white")
                tables_table.add_column("# Foreign Keys", style="white")
                
                for table in tables:
                    table_name = table.get("name", "Unknown")
                    columns = table.get("columns", [])
                    indexes = table.get("indexes", [])
                    foreign_keys = table.get("foreign_keys", [])
                    
                    tables_table.add_row(
                        table_name,
                        str(len(columns)),
                        str(len(indexes)),
                        str(len(foreign_keys))
                    )
                
                console.print(tables_table)
                
                # For a specific table, show detailed information
                if len(tables) > 0:
                    sample_table = tables[0].get("name", "")
                    await table_details_demo(connection_id, sample_table)
            else:
                console.print("[yellow]No tables found in database[/yellow]")
        else:
            logger.error(f"Failed to discover schema: {schema_result.get('error')}", emoji_key="error")
            console.print(f"[bold red]Schema discovery failed:[/bold red] {escape(schema_result.get('error', 'Unknown error'))}")
    
    except Exception as e:
        logger.error(f"Error in schema discovery demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error in schema discovery demo:[/bold red] {escape(str(e))}")
    
    console.print()

async def table_details_demo(connection_id: str, table_name: str) -> None:
    """Demonstrate getting table details and relationships."""
    console.print(Rule(f"[bold blue]Table Details Demo: {escape(table_name)}[/bold blue]"))
    logger.info(f"Getting details for table: {table_name}", emoji_key="db")
    
    try:
        # Get table details with sample data
        table_result = await get_table_details(
            connection_id=connection_id,
            table_name=table_name,
            include_sample_data=True,
            sample_size=3,
            include_statistics=True
        )
        
        if table_result.get("success"):
            console.print(f"[green]Successfully retrieved details for table: {escape(table_name)}[/green]")
            
            # Display columns
            columns = table_result.get("columns", [])
            if columns:
                columns_table = Table(title=f"Columns for {escape(table_name)}", box=None)
                columns_table.add_column("Column Name", style="cyan")
                columns_table.add_column("Type", style="white")
                columns_table.add_column("Nullable", style="white")
                columns_table.add_column("Primary Key", style="white")
                columns_table.add_column("Default", style="white")
                
                for column in columns:
                    columns_table.add_row(
                        column.get("name", "Unknown"),
                        column.get("type", "Unknown"),
                        str(column.get("nullable", False)),
                        str(column.get("primary_key", False)),
                        str(column.get("default", "None"))
                    )
                
                console.print(columns_table)
            
            # Display sample data
            sample_data = table_result.get("sample_data", [])
            if sample_data:
                console.print(f"[bold]Sample Data from {escape(table_name)}:[/bold]")
                
                sample_table = Table(box=None)
                # Add columns based on first row keys
                if sample_data and isinstance(sample_data[0], dict):
                    for column in sample_data[0].keys():
                        sample_table.add_column(column, style="dim cyan")
                    
                    # Add data rows
                    for row in sample_data:
                        sample_table.add_row(*[str(value) for value in row.values()])
                
                console.print(sample_table)
            
            # Show related tables
            logger.info(f"Finding tables related to {table_name}", emoji_key="db")
            relations_result = await find_related_tables(
                connection_id=connection_id,
                table_name=table_name,
                depth=1,
                include_details=True
            )
            
            if relations_result.get("success"):
                related_tables = relations_result.get("related_tables", {})
                if related_tables:
                    console.print("[bold]Related Tables:[/bold]")
                    
                    # Prepare structured display of relationships
                    relations_panel_content = []
                    for rel_table, rel_info in related_tables.items():
                        relationship_type = rel_info.get("relationship_type", "Unknown")
                        via_columns = rel_info.get("via_columns", [])
                        via_str = ", ".join([f"{src} -> {tgt}" for src, tgt in via_columns]) if via_columns else "Unknown"
                        
                        relations_panel_content.append(
                            f"[cyan]{escape(rel_table)}[/cyan] ({escape(relationship_type)})\n"
                            f"  Via: {escape(via_str)}"
                        )
                    
                    console.print(Panel(
                        "\n".join(relations_panel_content),
                        title=f"Relationships for {escape(table_name)}",
                        border_style="blue",
                        expand=False
                    ))
                else:
                    console.print(f"[yellow]No related tables found for {escape(table_name)}[/yellow]")
            else:
                console.print(f"[yellow]Failed to get related tables: {escape(relations_result.get('error', 'Unknown error'))}[/yellow]")
            
            # Analyze a sample column
            if columns:
                sample_column = columns[0].get("name", "")
                await column_statistics_demo(connection_id, table_name, sample_column)
        else:
            logger.error(f"Failed to get table details: {table_result.get('error')}", emoji_key="error")
            console.print(f"[bold red]Failed to get table details:[/bold red] {escape(table_result.get('error', 'Unknown error'))}")
    
    except Exception as e:
        logger.error(f"Error in table details demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error in table details demo:[/bold red] {escape(str(e))}")
    
    console.print()

async def column_statistics_demo(connection_id: str, table_name: str, column_name: str) -> None:
    """Demonstrate column statistics analysis."""
    console.print(Rule(f"[bold blue]Column Statistics Demo: {escape(column_name)}[/bold blue]"))
    logger.info(f"Analyzing statistics for column {column_name} in table {table_name}", emoji_key="processing")
    
    try:
        stats_result = await analyze_column_statistics(
            connection_id=connection_id,
            table_name=table_name,
            column_name=column_name,
            include_histogram=True,
            num_buckets=5,
            include_unique_values=True,
            max_unique_values=10
        )
        
        if stats_result.get("success"):
            console.print(f"[green]Successfully analyzed statistics for {escape(column_name)}[/green]")
            
            # Display basic statistics
            basic_stats = stats_result.get("basic_statistics", {})
            if basic_stats:
                stats_table = Table(title=f"Statistics for {escape(column_name)}", box=None)
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="white")
                
                # Add rows for each statistic
                for key, value in basic_stats.items():
                    if key not in ["column_name", "table_name"]:
                        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                        stats_table.add_row(key.replace("_", " ").title(), formatted_value)
                
                console.print(stats_table)
            
            # Display histogram if available
            histogram = stats_result.get("histogram", {})
            if histogram and "buckets" in histogram:
                console.print("[bold]Value Distribution:[/bold]")
                
                buckets = histogram.get("buckets", [])
                for bucket in buckets:
                    bucket_range = bucket.get("range", "")
                    count = bucket.get("count", 0)
                    # Create a visual representation of the distribution
                    bar_length = min(count, 40)  # Limit max bar length
                    bar = "█" * bar_length
                    console.print(f"{escape(bucket_range)}: {bar} ({count})")
            
            # Display unique values if available
            unique_values = stats_result.get("unique_values", [])
            if unique_values:
                console.print(f"[bold]Top Unique Values ({len(unique_values)}):[/bold]")
                
                unique_table = Table(box=None)
                unique_table.add_column("Value", style="cyan")
                unique_table.add_column("Count", style="white")
                unique_table.add_column("Percentage", style="white")
                
                for value_info in unique_values:
                    value = value_info.get("value", "")
                    count = value_info.get("count", 0)
                    percentage = value_info.get("percentage", 0)
                    unique_table.add_row(
                        str(value),
                        str(count),
                        f"{percentage:.2f}%"
                    )
                
                console.print(unique_table)
        else:
            logger.error(f"Failed to analyze column statistics: {stats_result.get('error')}", emoji_key="error")
            console.print(f"[bold red]Failed to analyze column statistics:[/bold red] {escape(stats_result.get('error', 'Unknown error'))}")
    
    except Exception as e:
        logger.error(f"Error in column statistics demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error in column statistics demo:[/bold red] {escape(str(e))}")
    
    console.print()

async def query_execution_demo(connection_id: str) -> None:
    """Demonstrate query execution capabilities."""
    console.print(Rule("[bold blue]Query Execution Demo[/bold blue]"))
    logger.info("Demonstrating query execution", emoji_key="start")
    
    try:
        # Simple SELECT query
        simple_query = "SELECT * FROM customers WHERE status = 'active'"
        logger.info(f"Executing simple query: {simple_query}", emoji_key="db")
        
        query_result = await execute_query(
            connection_id=connection_id,
            query=simple_query,
            read_only=True
        )
        
        display_query_result("Simple Query Results", query_result)
        
        # Parameterized query
        param_query = "SELECT * FROM products WHERE category = :category AND price < :max_price"
        params = {
            "category": "Electronics",
            "max_price": 1000.00
        }
        
        logger.info(f"Executing parameterized query with params: {params}", emoji_key="db")
        param_result = await execute_parameterized_query(
            connection_id=connection_id,
            query=param_query,
            parameters=params,
            read_only=True
        )
        
        display_query_result("Parameterized Query Results", param_result)
        
        # Pagination query
        pagination_query = "SELECT * FROM products ORDER BY price DESC"
        logger.info("Executing query with pagination", emoji_key="db")
        
        pagination_result = await execute_query_with_pagination(
            connection_id=connection_id,
            query=pagination_query,
            page_size=3,
            page_number=1  # First page
        )
        
        display_query_result("Paginated Query Results (Page 1)", pagination_result)
        
        # Advanced JOIN query
        join_query = """
        SELECT c.name AS customer_name, o.order_id, o.order_date, o.total_amount, o.status
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        ORDER BY o.order_date DESC
        """
        
        logger.info("Executing JOIN query", emoji_key="db")
        join_result = await execute_query(
            connection_id=connection_id,
            query=join_query,
            read_only=True
        )
        
        display_query_result("JOIN Query Results", join_result)
        
        # Aggregate query
        aggregate_query = """
        SELECT 
            category,
            COUNT(*) AS product_count,
            AVG(price) AS avg_price,
            SUM(price) AS total_value
        FROM products
        GROUP BY category
        ORDER BY product_count DESC
        """
        
        logger.info("Executing aggregate query", emoji_key="db")
        aggregate_result = await execute_query(
            connection_id=connection_id,
            query=aggregate_query,
            read_only=True
        )
        
        display_query_result("Aggregate Query Results", aggregate_result)
    
    except Exception as e:
        logger.error(f"Error in query execution demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error in query execution demo:[/bold red] {escape(str(e))}")
    
    console.print()

async def database_objects_demo(connection_id: str) -> None:
    """Demonstrate creating database objects like views and indexes."""
    console.print(Rule("[bold blue]Database Objects Demo[/bold blue]"))
    logger.info("Demonstrating database object creation", emoji_key="start")
    
    try:
        # Create a view
        view_name = "active_customers_view"
        view_query = "SELECT customer_id, name, email FROM customers WHERE status = 'active'"
        
        logger.info(f"Creating view: {view_name}", emoji_key="db")
        view_result = await create_database_view(
            connection_id=connection_id,
            view_name=view_name,
            query=view_query,
            replace_if_exists=True
        )
        
        if view_result.get("success"):
            console.print(f"[green]Successfully created view: {escape(view_name)}[/green]")
            
            # Query the new view
            view_query_result = await execute_query(
                connection_id=connection_id,
                query=f"SELECT * FROM {view_name}"
            )
            
            display_query_result(f"Query Results from {view_name}", view_query_result)
        else:
            logger.error(f"Failed to create view: {view_result.get('error')}", emoji_key="error")
            console.print(f"[bold red]Failed to create view:[/bold red] {escape(view_result.get('error', 'Unknown error'))}")
        
        # Create an index
        index_name = "products_category_idx"
        table_name = "products"
        column_names = ["category"]
        
        logger.info(f"Creating index: {index_name} on {table_name}({', '.join(column_names)})", emoji_key="db")
        index_result = await create_database_index(
            connection_id=connection_id,
            table_name=table_name,
            column_names=column_names,
            index_name=index_name,
            unique=False
        )
        
        if index_result.get("success"):
            console.print(f"[green]Successfully created index: {escape(index_name)}[/green]")
            
            # Check schema to confirm index creation
            schema_result = await discover_database_schema(
                connection_id=connection_id,
                include_indexes=True
            )
            
            if schema_result.get("success"):
                schema = schema_result.get("schema", {})
                tables = schema.get("tables", [])
                
                # Find our table and check its indexes
                for table in tables:
                    if table.get("name") == table_name:
                        indexes = table.get("indexes", [])
                        if indexes:
                            console.print(f"[bold]Indexes on {escape(table_name)}:[/bold]")
                            
                            indexes_table = Table(box=None)
                            indexes_table.add_column("Index Name", style="cyan")
                            indexes_table.add_column("Columns", style="white")
                            indexes_table.add_column("Unique", style="white")
                            
                            for idx in indexes:
                                idx_name = idx.get("name", "Unknown")
                                idx_columns = ", ".join(idx.get("columns", []))
                                idx_unique = "Yes" if idx.get("unique", False) else "No"
                                
                                indexes_table.add_row(
                                    idx_name,
                                    idx_columns,
                                    idx_unique
                                )
                            
                            console.print(indexes_table)
                        break
        else:
            logger.error(f"Failed to create index: {index_result.get('error')}", emoji_key="error")
            console.print(f"[bold red]Failed to create index:[/bold red] {escape(index_result.get('error', 'Unknown error'))}")
    
    except Exception as e:
        logger.error(f"Error in database objects demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error in database objects demo:[/bold red] {escape(str(e))}")
    
    console.print()

async def documentation_demo(connection_id: str) -> None:
    """Demonstrate database documentation generation."""
    console.print(Rule("[bold blue]Database Documentation Demo[/bold blue]"))
    logger.info("Generating database documentation", emoji_key="start")
    
    try:
        doc_result = await generate_database_documentation(
            connection_id=connection_id,
            output_format="markdown",
            include_schema=True,
            include_relationships=True,
            include_samples=True
        )
        
        if doc_result.get("success"):
            console.print("[green]Successfully generated database documentation[/green]")
            
            # Display documentation in a syntax-highlighted panel
            documentation = doc_result.get("documentation", "")
            console.print(Panel(
                Syntax(
                    documentation, 
                    "markdown", 
                    theme="default", 
                    line_numbers=True,
                    word_wrap=True
                ),
                title="Database Documentation (Markdown)",
                border_style="green",
                expand=True,
                width=100
            ))
        else:
            logger.error(f"Failed to generate documentation: {doc_result.get('error')}", emoji_key="error")
            console.print(f"[bold red]Failed to generate documentation:[/bold red] {escape(doc_result.get('error', 'Unknown error'))}")
    
    except Exception as e:
        logger.error(f"Error in documentation demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error in documentation demo:[/bold red] {escape(str(e))}")
    
    console.print()

async def transaction_demo(connection_id: str) -> None:
    """Demonstrate executing transactions."""
    console.print(Rule("[bold blue]Transaction Demo[/bold blue]"))
    logger.info("Demonstrating database transactions", emoji_key="start")
    
    try:
        # Prepare transaction queries
        transaction_queries = [
            "INSERT INTO products (name, description, price, category, in_stock) VALUES ('Smart Watch', 'Fitness tracking smartwatch', 159.99, 'Wearables', 1)",
            "UPDATE products SET price = price * 0.9 WHERE category = 'Wearables'",
            "SELECT * FROM products WHERE category = 'Wearables'"
        ]
        
        logger.info(f"Executing transaction with {len(transaction_queries)} queries", emoji_key="db")
        transaction_result = await execute_transaction(
            connection_id=connection_id,
            queries=transaction_queries,
            read_only=False
        )
        
        if transaction_result.get("success"):
            console.print("[green]Transaction executed successfully[/green]")
            
            # Display results from each query in the transaction
            results = transaction_result.get("results", [])
            for i, result in enumerate(results):
                if "rows" in result:
                    console.print(f"[bold]Query {i+1} Results:[/bold]")
                    
                    if result.get("rows"):
                        rows = result.get("rows", [])
                        if isinstance(rows[0], dict):
                            result_table = Table(box=None)
                            
                            # Add columns based on first row
                            for col in rows[0].keys():
                                result_table.add_column(col, style="cyan")
                            
                            # Add rows
                            for row in rows:
                                result_table.add_row(*[str(val) for val in row.values()])
                            
                            console.print(result_table)
                    else:
                        console.print("[yellow]No rows returned[/yellow]")
                else:
                    affected = result.get("affected_rows", 0)
                    console.print(f"Query {i+1}: Affected {affected} rows")
            
            # Display transaction summary
            console.print(Panel(
                f"Total Queries: [bold]{len(transaction_queries)}[/bold]\n"
                f"Execution Time: [bold]{transaction_result.get('execution_time', 0):.4f}s[/bold]\n"
                f"Read-Only: [bold]{'Yes' if transaction_result.get('read_only', True) else 'No'}[/bold]",
                title="Transaction Summary",
                border_style="green",
                expand=False
            ))
        else:
            logger.error(f"Transaction failed: {transaction_result.get('error')}", emoji_key="error")
            console.print(f"[bold red]Transaction failed:[/bold red] {escape(transaction_result.get('error', 'Unknown error'))}")
    
    except Exception as e:
        logger.error(f"Error in transaction demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error in transaction demo:[/bold red] {escape(str(e))}")
    
    console.print()

async def cleanup_demo(connection_id: str) -> None:
    """Demonstrate disconnecting from the database."""
    console.print(Rule("[bold blue]Database Cleanup and Disconnection[/bold blue]"))
    logger.info("Disconnecting from database", emoji_key="start")
    
    try:
        # Disconnect from the database
        disconnect_result = await disconnect_from_database(connection_id=connection_id)
        
        if disconnect_result.get("success"):
            logger.success(f"Successfully disconnected from database (ID: {connection_id})", emoji_key="success")
            console.print("[green]Successfully disconnected from database[/green]")
        else:
            logger.error(f"Failed to disconnect: {disconnect_result.get('error')}", emoji_key="error")
            console.print(f"[bold red]Failed to disconnect:[/bold red] {escape(disconnect_result.get('error', 'Unknown error'))}")
    
    except Exception as e:
        logger.error(f"Error in cleanup demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error in cleanup demo:[/bold red] {escape(str(e))}")
    
    console.print()

async def main() -> int:
    """Run all SQL database interaction demonstrations."""
    console.print(Rule("[bold magenta]SQL Database Interactions Demo Starting[/bold magenta]"))
    
    try:
        # Connect to database
        connection_id = await connection_demo()
        
        if not connection_id:
            logger.critical("Failed to establish database connection. Cannot proceed with demos.", emoji_key="critical")
            return 1
        
        # Setup demo database
        await setup_demo_database(connection_id)
        
        # Run all demos in sequence
        await schema_discovery_demo(connection_id)
        await query_execution_demo(connection_id)
        await database_objects_demo(connection_id)
        await transaction_demo(connection_id)
        await documentation_demo(connection_id)
        
        # Cleanup and disconnect
        await cleanup_demo(connection_id)
        
    except Exception as e:
        logger.critical(f"SQL database demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"[bold red]Critical Demo Error:[/bold red] {escape(str(e))}")
        return 1
    
    logger.success("SQL Database Interactions Demo Finished Successfully!", emoji_key="complete")
    console.print(Rule("[bold magenta]SQL Database Interactions Demo Complete[/bold magenta]"))
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 