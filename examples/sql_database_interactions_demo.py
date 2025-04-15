#!/usr/bin/env python
"""Demo script showcasing the SQL database interactions tools."""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import namedtuple # Import namedtuple

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Rich imports for nice UI
from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.traceback import Traceback
import rich.traceback # Import the module itself
from rich.tree import Tree

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
from llm_gateway.exceptions import ToolError, ToolInputError # Import specific exceptions
from llm_gateway.utils import get_logger
from llm_gateway.utils.display import CostTracker # Import CostTracker

# Initialize Rich console and logger
console = Console()
logger = get_logger("example.sql_database_interactions")

# Create a simple structure for cost tracking from dict (tokens might be missing)
TrackableResult = namedtuple("TrackableResult", ["cost", "input_tokens", "output_tokens", "provider", "model", "processing_time"])

# --- Demo Configuration ---
# SQLite in-memory database for demonstration
DEFAULT_CONNECTION_STRING = "sqlite:///demo_database.db"
# You can replace with a more complex connection string like:
# "postgresql://username:password@localhost:5432/demo_db"

# Install rich tracebacks for better error display
rich.traceback.install(show_locals=False, width=console.width) # Call install from the module

# --- Helper Functions for Demo Data Setup ---

async def setup_demo_database(connection_id: str) -> None:
    """Set up demo database with sample tables and data."""
    logger.info("Setting up demo database...", emoji_key="db")
    
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
    
    sample_data_queries = [
        # Customers (More records, varied statuses and sign-up dates)
        """
        INSERT OR IGNORE INTO customers (customer_id, name, email, status, signup_date) VALUES
            (1, 'Alice Johnson', 'alice.j@example.com', 'active', '2023-01-15 10:30:00'),
            (2, 'Bob Smith', 'bob.smith@example.net', 'active', '2023-02-20 14:00:00'),
            (3, 'Charlie Davis', 'charlie.d@example.org', 'inactive', '2023-03-10 09:15:00'),
            (4, 'Diana Miller', 'diana.m@example.com', 'active', '2023-04-05 11:00:00'),
            (5, 'Ethan Garcia', 'ethan.g@sample.net', 'pending', '2023-05-25 16:45:00'),
            (6, 'Fiona Rodriguez', 'fiona.r@example.com', 'active', '2023-06-12 08:00:00'),
            (7, 'George Wilson', 'george.w@example.org', 'active', '2023-07-18 13:20:00'),
            (8, 'Hannah Martinez', 'hannah.m@sample.com', 'inactive', '2023-08-01 10:00:00'),
            (9, 'Ian Anderson', 'ian.a@example.net', 'active', '2023-09-14 17:30:00'),
            (10, 'Julia Thomas', 'julia.t@example.com', 'active', '2023-10-22 12:10:00'),
            (11, 'Kevin Jackson', 'kevin.j@sample.org', 'pending', '2023-11-05 19:05:00'),
            (12, 'Laura White', 'laura.w@example.com', 'active', '2023-12-01 11:55:00'),
            (13, 'Mason Harris', 'mason.h@example.net', 'active', '2024-01-10 15:25:00'),
            (14, 'Nora Martin', 'nora.m@sample.com', 'active', '2024-02-19 09:40:00'),
            (15, 'Oscar Thompson', 'oscar.t@example.org', 'inactive', '2024-03-03 14:50:00'),
            (16, 'Penelope Lee', 'penelope.l@example.com', 'active', '2024-04-11 18:00:00'),
            (17, 'Quentin Walker', 'quentin.w@sample.net', 'active', '2024-05-02 10:15:00'),
            (18, 'Rachel Hall', 'rachel.h@example.com', 'pending', '2024-06-07 16:30:00'),
            (19, 'Steven Allen', 'steven.a@example.org', 'active', '2024-07-15 11:05:00'),
            (20, 'Tara Young', 'tara.y@sample.com', 'active', '2024-08-01 13:45:00');
        """, 
        # Products (More variety, categories, stock status)
        """
        INSERT OR IGNORE INTO products (product_id, name, description, price, category, in_stock) VALUES
            (1, 'Laptop Pro X', 'High-performance laptop with 16GB RAM, 1TB SSD', 1499.99, 'Electronics', 1),
            (2, 'Smartphone Z', 'Latest flagship smartphone with advanced camera', 999.99, 'Electronics', 1),
            (3, 'Wireless Earbuds Plus', 'Noise-cancelling earbuds with long battery life', 179.99, 'Audio', 1),
            (4, 'Smart Coffee Maker', 'WiFi-enabled programmable coffee machine', 119.99, 'Kitchen Appliances', 0),
            (5, 'Fitness Tracker Elite', 'Waterproof fitness band with GPS and heart rate monitor', 79.99, 'Wearables', 1),
            (6, 'LED Desk Lamp Pro', 'Adjustable brightness and color temperature LED lamp', 45.99, 'Home Office', 1),
            (7, 'Ergonomic Mesh Chair', 'Office chair with lumbar support and breathable mesh', 349.99, 'Furniture', 0),
            (8, '4K Monitor 27"', '27-inch IPS monitor with 4K resolution', 399.99, 'Electronics', 1),
            (9, 'Bluetooth Speaker Mini', 'Portable waterproof Bluetooth speaker', 59.99, 'Audio', 1),
            (10, 'Robot Vacuum Cleaner', 'Automated vacuum with mapping technology', 299.99, 'Home Appliances', 1),
            (11, 'Electric Kettle Go', '1.7L fast-boil electric kettle', 34.99, 'Kitchen Appliances', 1),
            (12, 'Yoga Mat Premium', 'Eco-friendly non-slip yoga mat', 29.99, 'Fitness', 1),
            (13, 'Gaming Mouse RGB', 'High-precision gaming mouse with customizable RGB', 69.99, 'Gaming', 1),
            (14, 'External SSD 2TB', 'Portable solid-state drive with USB-C', 189.99, 'Storage', 1),
            (15, 'Air Fryer XL', '5.8-quart large capacity air fryer', 99.99, 'Kitchen Appliances', 0),
            (16, 'Bookshelf Modern', '5-tier modern style bookshelf', 129.99, 'Furniture', 1),
            (17, 'Smartwatch Fit', 'Smartwatch with health tracking features', 199.99, 'Wearables', 1),
            (18, 'Mechanical Keyboard TKL', 'Tenkeyless mechanical keyboard with tactile switches', 119.99, 'Gaming', 1),
            (19, 'Noise Cancelling Headphones Pro', 'Over-ear ANC headphones with superior sound', 349.99, 'Audio', 1),
            (20, 'Webcam HD 1080p', 'Full HD webcam with built-in microphone', 49.99, 'Home Office', 1),
            (21, 'Blender PowerMax', 'High-speed blender for smoothies and more', 89.99, 'Kitchen Appliances', 1),
            (22, 'Standing Desk Converter', 'Adjustable height desk riser', 159.99, 'Home Office', 1),
            (23, 'Dumbbell Set Adjustable', 'Adjustable weight dumbbell set (up to 50 lbs)', 199.99, 'Fitness', 0),
            (24, 'Portable Projector Mini', 'Compact LED projector for home entertainment', 149.99, 'Electronics', 1),
            (25, 'Travel Backpack 40L', 'Durable carry-on size travel backpack', 79.99, 'Travel Gear', 1);
        """,
        """
        INSERT OR IGNORE INTO orders (customer_id, total_amount, status)
        VALUES
            (1, 1499.98, 'completed'),
            (2, 89.99, 'processing'),
            (1, 249.99, 'completed'),
            (3, 1099.98, 'completed'),
            (4, 49.99, 'processing'),
            (5, 600.00, 'pending'),
            (6, 50.00, 'shipped')
        """
    ]
    
    with console.status("[bold green]Setting up demo database...", spinner="dots") as status:
        try:
            status.update("Creating tables...")
            transaction_result = await execute_transaction(
                connection_id=connection_id,
                queries=setup_queries,
                read_only=False
            )
            
            if transaction_result.get("success"):
                logger.success("Successfully created sample tables.", emoji_key="heavy_check_mark")
                console.print(Panel("[green]:heavy_check_mark: Sample tables created successfully.", padding=(0, 1), border_style="green"))
                
                status.update("Inserting sample data...")
                data_result = await execute_transaction(
                    connection_id=connection_id,
                    queries=sample_data_queries,
                    read_only=False
                )
                
                if data_result.get("success"):
                    logger.success("Successfully inserted sample data.", emoji_key="heavy_check_mark")
                    console.print(Panel("[green]:heavy_check_mark: Sample data inserted successfully.", padding=(0, 1), border_style="green"))
                else:
                    error_msg = data_result.get("error", "Unknown error")
                    logger.error(f"Failed to insert sample data: {error_msg}", emoji_key="x")
                    console.print(Panel(f"[bold red]:x: Error inserting sample data:[/]\n{escape(error_msg)}", padding=(1, 2), border_style="red", title="Setup Error"))
            else:
                error_msg = transaction_result.get("error", "Unknown error")
                logger.error(f"Failed to create sample tables: {error_msg}", emoji_key="x")
                console.print(Panel(f"[bold red]:x: Error creating sample tables:[/]\n{escape(error_msg)}", padding=(1, 2), border_style="red", title="Setup Error"))
        
        except (ToolError, ToolInputError) as e:
            logger.error(f"Tool error during database setup: {e}", emoji_key="x", exc_info=True)
            console.print(Panel(f"[bold red]:x: Tool Error during setup:[/]\n{escape(str(e))}", padding=(1, 2), border_style="red", title="Setup Failed"))
        except Exception as e:
            logger.error(f"Unexpected error setting up demo database: {e}", emoji_key="x", exc_info=True)
            console.print(f"[bold red]:x: Unexpected Error during setup:[/]\n{escape(str(e))}")
            console.print(Traceback.from_exception(type(e), e, e.__traceback__))

def display_query_result(title: str, result: Dict[str, Any], query_str: Optional[str] = None, show_stats: bool = True) -> None:
    """Display query result with enhanced Rich formatting."""
    console.print(Rule(f"[bold cyan]{escape(title)}[/bold cyan]"))

    if query_str:
        console.print(Panel(
            Syntax(query_str.strip(), "sql", theme="default", line_numbers=False, word_wrap=True),
            title="Executed Query",
            border_style="blue",
            padding=(1, 2)
        ))
    
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        console.print(Panel(
            f"[bold red]:x: Query Execution Failed:[/]\n{escape(error_msg)}",
            title="Error",
            border_style="red",
            padding=(1, 2),
            expand=False
        ))
        return
    
    rows = result.get("rows", [])
    columns = result.get("columns", [])
    row_count = result.get("row_count", len(rows)) # Use provided count if available

    # Pagination Info
    pagination_info = result.get("pagination")
    
    if not rows and not pagination_info: # Handle cases where no rows returned but not due to pagination
        console.print(Panel("[yellow]No results returned for this query.", padding=(0, 1), border_style="yellow"))
        return
    elif not rows and pagination_info:
        console.print(Panel(f"[yellow]No results returned on page {pagination_info.get('page', '?')}.", padding=(0,1), border_style="yellow"))
        # Still show pagination stats below

    # Create table for displaying results
    if rows:
        table_title = f"Results ({row_count} row{'s' if row_count != 1 else ''} returned)"
        if pagination_info:
             table_title += f" - Page {pagination_info.get('page', '?')}"
        
        table = Table(title=table_title, box=box.ROUNDED, show_header=True, padding=(0, 1), border_style="bright_blue")
        
        # Add columns based on results or first row keys
        col_names = columns if columns else (rows[0].keys() if rows and isinstance(rows[0], dict) else ["Result"])
        
        for name in col_names:
            # Basic justification heuristic
            justify = "right" if any(k in name.lower() for k in ['id', 'count', 'price', 'amount', 'quantity', 'total']) else "left"
            style = "cyan" if justify == "left" else "magenta"
            table.add_column(name, style=style, justify=justify, header_style=f"bold {style}")
        
        # Add data rows
        if rows:
            for row in rows:
                if hasattr(row, '_mapping'): # Handles SQLAlchemy RowMapping
                    row_dict = row._mapping
                    table.add_row(*[escape(str(row_dict.get(col_name, ''))) for col_name in col_names])
                elif isinstance(row, dict): # Handles plain dictionaries
                    table.add_row(*[escape(str(row.get(col_name, ''))) for col_name in col_names])
                else: # Handle single value results (less common)
                    table.add_row(escape(str(row)))
        
        console.print(table)
    
    # Display statistics and pagination if available and requested
    if show_stats:
        stats_items = {}
        if "execution_time" in result:
            stats_items["Execution Time"] = f"{result['execution_time']:.4f}s"
        if "affected_rows" in result and result["affected_rows"] is not None:
             stats_items["Affected Rows"] = str(result["affected_rows"])
        if "truncated" in result:
             stats_items["Truncated"] = "[bold yellow]Yes[/]" if result["truncated"] else "[dim]No[/]"

        if stats_items or pagination_info:
            stats_table = Table(title="Query Info", show_header=False, box=box.SQUARE, padding=(0, 1), border_style="dim")
            stats_table.add_column("Metric", style="cyan", justify="right")
            stats_table.add_column("Value", style="white")

            for key, value in stats_items.items():
                stats_table.add_row(key, value)

            if pagination_info:
                 stats_table.add_row("---", "---") # Separator
                 stats_table.add_row("[bold]Pagination[/]", "")
                 stats_table.add_row("Page", str(pagination_info.get("page")))
                 stats_table.add_row("Page Size", str(pagination_info.get("page_size")))
                 total_rows_str = str(pagination_info.get("total_rows"))
                 if pagination_info.get("total_rows_estimated"):
                     total_rows_str += " (estimated)"
                 stats_table.add_row("Total Rows", total_rows_str)
                 stats_table.add_row("Total Pages", str(pagination_info.get("total_pages", "?")))
                 stats_table.add_row("Has Previous", "[green]:heavy_check_mark:[/]" if pagination_info.get("has_previous_page") else "[dim]:x:[/]")
                 stats_table.add_row("Has Next", "[green]:heavy_check_mark:[/]" if pagination_info.get("has_next_page") else "[dim]:x:[/]")

            console.print(stats_table)

    console.print() # Add spacing

# --- Demo Functions ---

async def connection_demo() -> Optional[str]:
    """Demonstrate database connection and status checking."""
    console.print(Rule("[bold green]1. Database Connection Demo[/bold green]", style="green"))
    logger.info("Starting database connection demo", emoji_key="link")
    
    connection_id = None
    
    with console.status("[bold cyan]Attempting connection...", spinner="earth"):
        try:
            connection_result = await connect_to_database(
                connection_string=DEFAULT_CONNECTION_STRING,
                # connection_options={"echo": True}, # Can be noisy, disable for demo clarity
                echo=False # Disable SQLAlchemy logging for cleaner output
            )
            
            if connection_result.get("success"):
                connection_id = connection_result.get("connection_id")
                logger.success(f"Successfully connected to database with ID: {connection_id}", emoji_key="heavy_check_mark")
                
                db_type = connection_result.get('database_type', 'Unknown')
                db_info = connection_result.get('database_info', {})
                version = db_info.get('version', 'N/A')
                db_path = db_info.get('path', db_info.get('database', 'N/A')) # Show path/name

                console.print(Panel(
                    f"Connection ID: [bold cyan]{escape(connection_id)}[/]\n"
                    f"Database Type: [blue]{escape(db_type)}[/]\n"
                    f"Version: [dim]{escape(version)}[/]\n"
                    f"Database: [dim]{escape(db_path)}[/]",
                    title="[bold green]:link: Connected[/]",
                    border_style="green",
                    padding=(1, 2),
                    expand=False
                ))
                
                # Test the connection
                console.print("[cyan]Testing connection health...[/]")
                test_result = await test_connection(connection_id=connection_id)
                if test_result.get("success"):
                    resp_time = test_result.get('response_time', 0)
                    console.print(Panel(f"[green]:heavy_check_mark: Connection test OK (response time: {resp_time:.4f}s)", border_style="green", padding=(0, 1)))
                else:
                    console.print(Panel(f"[bold red]:x: Connection test failed:[/]\n {escape(test_result.get('error', 'Unknown error'))}", border_style="red", padding=(1, 2)))
                
                # Get connection status
                console.print("[cyan]Fetching database status...[/]")
                status_result = await get_database_status(connection_id=connection_id)
                if status_result.get("success"):
                    status_table = Table(title="Database Status", box=box.HEAVY, show_header=False, padding=(0, 1), border_style="blue")
                    status_table.add_column("Metric", style="cyan", justify="right")
                    status_table.add_column("Value", style="white")
                    
                    stats = status_result.get("stats", {})
                    status_table.add_row("Active", "[bold green]Yes[/]")
                    status_table.add_row("DB Type", str(status_result.get("database_type")))
                    status_table.add_row("Tables Count", str(stats.get("tables_count", "N/A")))
                    status_table.add_row("Views Count", str(stats.get("views_count", "N/A")))
                    if "size" in stats:
                        status_table.add_row("Size", str(stats.get("size")))
                    
                    console.print(status_table)
                else:
                    console.print(Panel(f"[bold red]:x: Failed to get database status:[/]\n {escape(status_result.get('error', 'Unknown error'))}", border_style="red", padding=(1, 2)))
                    
            else:
                error_msg = connection_result.get('error', 'Unknown error')
                logger.error(f"Failed to connect to database: {error_msg}", emoji_key="x")
                console.print(Panel(f"[bold red]:x: Connection failed:[/]\n {escape(error_msg)}", border_style="red", padding=(1, 2)))
        
        except (ToolError, ToolInputError) as e:
            logger.error(f"Tool error in connection demo: {e}", emoji_key="x", exc_info=True)
            console.print(Panel(f"[bold red]:x: Connection Tool Error:[/]\n{escape(str(e))}", border_style="red", padding=(1, 2)))
        except Exception as e:
            logger.error(f"Unexpected error in connection demo: {e}", emoji_key="x", exc_info=True)
            console.print(f"[bold red]:x: Unexpected Error in Connection Demo:[/]\n{escape(str(e))}")
            # Let the main handler print the traceback
            raise # Re-raise for main handler
    
    console.print() # Spacing
    return connection_id

async def schema_discovery_demo(connection_id: str) -> None:
    """Demonstrate database schema discovery using a Rich Tree."""
    console.print(Rule("[bold green]2. Schema Discovery Demo[/bold green]", style="green"))
    logger.info("Starting schema discovery demo", emoji_key="mag")
    
    with console.status("[bold cyan]Discovering database schema...", spinner="dots"):
        try:
            schema_result = await discover_database_schema(
                connection_id=connection_id,
                include_indexes=True,
                include_foreign_keys=True,
                detailed=True # Get details for the tree
            )
            
            if schema_result.get("success"):
                tables = schema_result.get("tables", [])
                views = schema_result.get("views", [])
                logger.success(f"Schema discovered: {len(tables)} tables, {len(views)} views.", emoji_key="heavy_check_mark")
                
                if not tables and not views:
                    console.print(Panel("[yellow]Schema discovered, but no tables or views found.", padding=(0, 1), border_style="yellow"))
                    return

                tree = Tree(
                    f"[bold bright_blue]:database: Database Schema ({len(tables)} Tables, {len(views)} Views)[/]",
                    guide_style="bright_blue"
                )

                # Add Tables branch
                if tables:
                    tables_branch = tree.add("[bold cyan]:page_facing_up: Tables[/]")
                    for table in tables:
                        table_name = table.get("name", "Unknown")
                        table_node = tables_branch.add(f"[cyan]{escape(table_name)}[/]")
                        
                        # Columns
                        cols = table.get("columns", [])
                        if cols:
                            cols_branch = table_node.add("[bold yellow]:heavy_minus_sign: Columns[/]")
                            for col in cols:
                                col_name = col.get("name", "?")
                                col_type = col.get("type", "?")
                                is_pk = col.get("primary_key", False)
                                is_nullable = col.get("nullable", True)
                                pk_str = " [bold magenta](PK)[/]" if is_pk else ""
                                null_str = "" if is_nullable else " [dim]NOT NULL[/]"
                                comment = f" [dim]// {escape(col.get('comment', ''))}[/]" if col.get('comment') else ""
                                cols_branch.add(f"[yellow]{escape(col_name)}[/]: {escape(col_type)}{pk_str}{null_str}{comment}")

                        # Indexes
                        idxs = table.get("indexes", [])
                        if idxs:
                            idxs_branch = table_node.add("[bold green]:key: Indexes[/]")
                            for idx in idxs:
                                idx_name = idx.get("name", "?")
                                idx_cols = ', '.join(idx.get("columns", []))
                                is_unique = idx.get("unique", False)
                                unique_str = " [bold](UNIQUE)[/]" if is_unique else ""
                                idxs_branch.add(f"[green]{escape(idx_name)}[/] on ({escape(idx_cols)}){unique_str}")

                        # Foreign Keys (Outgoing)
                        fks = table.get("foreign_keys", [])
                        if fks:
                             fks_branch = table_node.add("[bold blue]:link: Foreign Keys (references)[/]")
                             for fk in fks:
                                 ref_table = fk.get("referred_table", "?")
                                 con_cols = ', '.join(fk.get("constrained_columns", []))
                                 ref_cols = ', '.join(fk.get("referred_columns", []))
                                 fks_branch.add(f"[blue]({escape(con_cols)})[/] -> [cyan]{escape(ref_table)}[/]({escape(ref_cols)})")

                # Add Views branch
                if views:
                    views_branch = tree.add("[bold magenta]:scroll: Views[/]")
                    for view in views:
                        view_name = view.get("name", "Unknown")
                        views_branch.add(f"[magenta]{escape(view_name)}[/]")
                        # Could add definition preview here if desired

                console.print(Panel(tree, title="Schema Overview", border_style="bright_blue", padding=(1, 2)))

                # Show detailed information for the first table as an example
                if tables:
                    sample_table_name = tables[0].get("name")
                    if sample_table_name:
                        await table_details_demo(connection_id, sample_table_name)
            else:
                error_msg = schema_result.get('error', 'Unknown error')
                logger.error(f"Failed to discover schema: {error_msg}", emoji_key="x")
                console.print(Panel(f"[bold red]:x: Schema discovery failed:[/]\n {escape(error_msg)}", border_style="red", padding=(1, 2)))
        
        except (ToolError, ToolInputError) as e:
            logger.error(f"Tool error during schema discovery: {e}", emoji_key="x", exc_info=True)
            console.print(Panel(f"[bold red]:x: Schema Discovery Tool Error:[/]\n{escape(str(e))}", border_style="red", padding=(1, 2)))
        except Exception as e:
            logger.error(f"Unexpected error in schema discovery demo: {e}", emoji_key="x", exc_info=True)
            console.print(f"[bold red]:x: Unexpected Error in Schema Discovery Demo:[/]\n{escape(str(e))}")
            raise

    console.print() # Spacing


async def table_details_demo(connection_id: str, table_name: str) -> None:
    """Demonstrate getting table details and relationships."""
    console.print(Rule(f"[bold green]3. Table Details & Relationships: [cyan]{escape(table_name)}[/cyan][/bold green]", style="green"))
    logger.info(f"Getting details for table: {table_name}", emoji_key="page_facing_up")
    
    # Removed console.status context manager to prevent nested Live displays
    try:
        table_result = await get_table_details(
            connection_id=connection_id,
            table_name=table_name,
            include_sample_data=True,
            sample_size=3,
            include_statistics=True # Get stats for column demo later
        )
        
        if table_result.get("success"):
            logger.success(f"Successfully retrieved details for table: {table_name}", emoji_key="heavy_check_mark")
            console.print(Panel(f"[green]:heavy_check_mark: Details retrieved for [cyan]{escape(table_name)}[/].", border_style="green", padding=(0, 1)))

            details_tree = Tree(f"[bold cyan]{escape(table_name)}[/] Details", guide_style="cyan")

            # Display columns
            columns = table_result.get("columns", [])
            if columns:
                cols_branch = details_tree.add("[bold yellow]:heavy_minus_sign: Columns[/]")
                cols_table = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0,1), border_style="yellow")
                cols_table.add_column("Name", style="yellow", header_style="bold yellow")
                cols_table.add_column("Type", style="white")
                cols_table.add_column("Nullable", style="dim")
                cols_table.add_column("PK", style="magenta")
                cols_table.add_column("Default", style="dim")
                
                for column in columns:
                    cols_table.add_row(
                        escape(column.get("name", "?")),
                        escape(column.get("type", "?")),
                        ":heavy_check_mark:" if column.get("nullable", False) else ":x:",
                        "[bold magenta]:key:[/]" if column.get("primary_key", False) else "",
                        escape(str(column.get("default", "")))
                    )
                cols_branch.add(cols_table)
            
            # Display sample data
            sample_data = table_result.get("sample_data", [])
            if sample_data:
                sample_branch = details_tree.add("[bold green]:eyes: Sample Data (first 3 rows)[/]")
                if sample_data and isinstance(sample_data[0], dict):
                    sample_table = Table(box=box.ROUNDED, show_header=True, padding=(0, 1), border_style="green")
                    for col_name in sample_data[0].keys():
                        sample_table.add_column(col_name, style="dim cyan", header_style="bold cyan")
                    for row_data in sample_data:
                        sample_table.add_row(*[escape(str(v)) for v in row_data.values()])
                    sample_branch.add(sample_table)
                elif "sample_data_error" in table_result:
                     sample_branch.add(f"[red]Error fetching samples: {escape(table_result['sample_data_error'])}")
                else:
                     sample_branch.add("[dim]No sample data available or table empty.[/]")

            # Display statistics (brief summary, full details in column stats demo)
            statistics = table_result.get("statistics")
            if statistics:
                 stats_branch = details_tree.add("[bold magenta]:bar_chart: Column Statistics (Summary)[/]")
                 stats_text = []
                 for col_name, stats_data in statistics.items():
                     if isinstance(stats_data, dict) and "error" not in stats_data:
                         count = stats_data.get('count', 'N/A')
                         nulls = stats_data.get('null_count', 'N/A')
                         uniques = stats_data.get('unique_count', 'N/A')
                         est = " (est.)" if stats_data.get('unique_count_estimated') else ""
                         stats_text.append(f"[magenta]{escape(col_name)}[/]: [dim]Nulls:[/]{nulls} [dim]Uniques:[/]{uniques}{est}")
                     elif isinstance(stats_data, dict) and "error" in stats_data:
                          stats_text.append(f"[magenta]{escape(col_name)}[/]: [red]Error fetching stats[/]")
                 stats_branch.add("\\n".join(stats_text))
            elif "statistics_error" in table_result:
                 stats_branch = details_tree.add("[bold red]:bar_chart: Column Statistics[/]")
                 stats_branch.add(f"[red]Error fetching stats: {escape(table_result['statistics_error'])}")


            console.print(details_tree)

            # Show related tables using the dedicated tool
            await find_related_tables_demo(connection_id, table_name)
            
            # Analyze a sample column's statistics in detail
            if columns:
                # Pick a potentially interesting column (e.g., not just PK id)
                sample_column = columns[1].get("name", "") if len(columns) > 1 else columns[0].get("name", "")
                if sample_column:
                     await column_statistics_demo(connection_id, table_name, sample_column)
        else:
            error_msg = table_result.get('error', 'Unknown error')
            logger.error(f"Failed to get table details for {table_name}: {error_msg}", emoji_key="x")
            console.print(Panel(f"[bold red]:x: Failed to get table details for '{escape(table_name)}':[/]\n {escape(error_msg)}", border_style="red", padding=(1, 2)))
    
    except (ToolError, ToolInputError) as e:
        logger.error(f"Tool error getting table details: {e}", emoji_key="x", exc_info=True)
        console.print(Panel(f"[bold red]:x: Table Details Tool Error:[/]\n{escape(str(e))}", border_style="red", padding=(1, 2)))
    except Exception as e:
        logger.error(f"Unexpected error in table details demo: {e}", emoji_key="x", exc_info=True)
        console.print(f"[bold red]:x: Unexpected Error in Table Details Demo:[/]\n{escape(str(e))}")
        raise
    
    console.print()

async def find_related_tables_demo(connection_id: str, table_name: str) -> None:
    """Demonstrates finding related tables using a Tree structure."""
    # This is now called from table_details_demo, so no separate rule
    logger.info(f"Finding tables related to {table_name}", emoji_key="link")
    
    # with console.status(f"[bold cyan]Finding relationships for '{escape(table_name)}'...", spinner="arc"):
    try:
        relations_result = await find_related_tables(
            connection_id=connection_id,
            table_name=table_name,
            depth=1, # Keep depth 1 for clarity in demo
            include_details=False # Details not needed for this tree
        )
        
        if relations_result.get("success"):
            graph = relations_result.get("relationships")
            if graph and (graph.get("parents") or graph.get("children")):
                logger.success(f"Found relationships for table: {table_name}", emoji_key="heavy_check_mark")

                rel_tree = Tree(f"[bold blue]:link: Relationships for [cyan]{escape(table_name)}[/][/]", guide_style="blue")
                
                parents = graph.get("parents", [])
                if parents:
                    parent_branch = rel_tree.add("[bold green]:arrow_up: References (Parents)[/]")
                    for parent in parents:
                         p_table = parent.get("table", "?")
                         fk_info = parent.get("relationship", {})
                         child_cols = ', '.join(fk_info.get('child_columns', []))
                         parent_cols = ', '.join(fk_info.get('parent_columns', []))
                         parent_branch.add(f"[blue]({escape(child_cols)})[/] -> [green]{escape(p_table)}[/]({escape(parent_cols)})")

                children = graph.get("children", [])
                if children:
                    child_branch = rel_tree.add("[bold magenta]:arrow_down: Referenced By (Children)[/]")
                    for child in children:
                         c_table = child.get("table", "?")
                         fk_info = child.get("relationship", {})
                         child_cols = ', '.join(fk_info.get('child_columns', []))
                         parent_cols = ', '.join(fk_info.get('parent_columns', []))
                         child_branch.add(f"[magenta]{escape(c_table)}[/]({escape(child_cols)}) -> [blue]({escape(parent_cols)})[/]")

                console.print(Panel(rel_tree, title="Table Relationships", border_style="blue", padding=(1, 2)))

            else:
                logger.info(f"No direct relationships found for {table_name}.", emoji_key="information_source")
                console.print(Panel(f"[yellow]No direct relationships found for '{escape(table_name)}'.", border_style="yellow", padding=(0, 1)))
        else:
            error_msg = relations_result.get('error', 'Unknown error')
            logger.warning(f"Failed to get related tables: {error_msg}", emoji_key="warning")
            console.print(Panel(f"[yellow]:warning: Could not get related tables:[/]\n {escape(error_msg)}", border_style="yellow", padding=(1, 2)))

    except (ToolError, ToolInputError) as e:
         logger.error(f"Tool error finding relationships: {e}", emoji_key="x", exc_info=True)
         console.print(Panel(f"[bold red]:x: Find Relationships Tool Error:[/]\n{escape(str(e))}", border_style="red", padding=(1, 2)))
    # except Exception as e:
    #      logger.error(f"Unexpected error finding relationships: {e}", emoji_key="x", exc_info=True)
    #      console.print(f"[bold red]:x: Unexpected Error Finding Relationships:[/]\n{escape(str(e))}")
    #      # Don't raise here as it's part of table details demo

    console.print()


async def column_statistics_demo(connection_id: str, table_name: str, column_name: str) -> None:
    """Demonstrate column statistics analysis with Rich elements."""
    # Called from table_details_demo, so no separate rule
    logger.info(f"Analyzing statistics for column {table_name}.{column_name}", emoji_key="bar_chart")
    
    # with console.status(f"[bold cyan]Analyzing column [yellow]{escape(column_name)}[/]...", spinner="clock"):
    try:
        stats_result = await analyze_column_statistics(
            connection_id=connection_id,
            table_name=table_name,
            column_name=column_name,
            include_histogram=True,
            num_buckets=8, # Adjust bucket count
            include_unique_values=True,
            max_unique_values=10
        )
        
        if stats_result.get("success"):
            logger.success(f"Successfully analyzed stats for {table_name}.{column_name}", emoji_key="heavy_check_mark")
            console.print(Panel(f"[green]:bar_chart: Statistics for [yellow]{escape(column_name)}[/] ([i]{stats_result.get('data_type', '?')}[/i])", padding=(1, 2), border_style="green", title=f"Column Analysis: {escape(column_name)}"))

            stats_tree = Tree(f"[bold yellow]{escape(column_name)}[/] Statistics", guide_style="yellow")

            # Display basic statistics
            basic_stats = stats_result.get("basic_stats", {})
            if basic_stats and "error" not in basic_stats:
                basic_branch = stats_tree.add("[bold]:clipboard: Basic Stats[/]")
                stats_table = Table(box=box.MINIMAL, show_header=False, padding=(0, 1))
                stats_table.add_column("Metric", style="cyan", justify="right")
                stats_table.add_column("Value", style="white")
                
                for key, value in basic_stats.items():
                    if value is not None:
                        formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                        if key == "unique_count_estimated" and value is True: continue # Don't show the flag itself
                        if key == "unique_count" and basic_stats.get("unique_count_estimated"):
                             formatted_value += " [dim](est.)[/]"
                        
                        stats_table.add_row(key.replace("_", " ").title(), formatted_value)
                basic_branch.add(stats_table)
            elif basic_stats and "error" in basic_stats:
                 basic_branch = stats_tree.add("[bold red]:clipboard: Basic Stats[/]")
                 basic_branch.add(f"[red]Error: {escape(basic_stats['error'])}")

            # Display histogram if available
            histogram = stats_result.get("histogram", {})
            if histogram and "error" not in histogram:
                hist_branch = stats_tree.add("[bold]:chart_with_upwards_trend: Value Distribution[/]")
                buckets = histogram.get("buckets", [])
                hist_type = histogram.get("type", "unknown")

                if buckets:
                    # Use Progress for a visual bar chart
                    progress = Progress(
                        TextColumn("[cyan]{task.fields[label]}", justify="right"),
                        BarColumn(bar_width=40),
                        TextColumn("[magenta]{task.fields[count]} ({task.percentage:>3.1f}%)")
                    )
                    with progress: # Use context manager
                         for bucket in buckets:
                             label = bucket.get("range") or bucket.get("value", "?")
                             count = bucket.get("count", 0)
                             percentage = bucket.get("percentage", 0)
                             # Truncate long labels
                             if len(str(label)) > 25: label = str(label)[:22] + "..."
                             progress.add_task("", total=100, completed=percentage, label=escape(str(label)), count=count)
                    hist_branch.add(progress)
                elif "notes" in histogram:
                     hist_branch.add(f"[dim]{histogram['notes']}")
                else:
                     hist_branch.add("[dim]No histogram data available.[/]")
            elif histogram and "error" in histogram:
                 hist_branch = stats_tree.add("[bold red]:chart_with_upwards_trend: Value Distribution[/]")
                 hist_branch.add(f"[red]Error: {escape(histogram['error'])}")


            # Display unique values if available
            value_freqs = stats_result.get("value_frequencies", {})
            if value_freqs and "error" not in value_freqs:
                 unique_branch = stats_tree.add("[bold]:keycaps_10: Frequent Values[/]")
                 values = value_freqs.get("values", [])
                 if values:
                     unique_table = Table(box=box.ROUNDED, show_header=True, padding=(0, 1), border_style="dim")
                     unique_table.add_column("Value", style="cyan", header_style="bold cyan")
                     unique_table.add_column("Count", style="magenta", justify="right", header_style="bold magenta")
                     unique_table.add_column("%", style="white", justify="right", header_style="bold white")
                     
                     for value_info in values:
                         value = value_info.get("value", "")
                         count = value_info.get("count", 0)
                         percentage = value_info.get("percentage", 0)
                         unique_table.add_row(
                             escape(str(value)),
                             str(count),
                             f"{percentage:.1f}%"
                         )
                     unique_branch.add(unique_table)
                     if value_freqs.get("truncated"):
                          total_unique = value_freqs.get("total_unique_in_table", "?")
                          unique_branch.add(f"[dim](Showing top {len(values)} of {total_unique} total unique values)[/]")
                 elif "notes" in value_freqs:
                      unique_branch.add(f"[dim]{value_freqs['notes']}")
                 else:
                      unique_branch.add("[dim]No unique value data available.[/]")

            elif value_freqs and "error" in value_freqs:
                 unique_branch = stats_tree.add("[bold red]:keycaps_10: Frequent Values[/]")
                 unique_branch.add(f"[red]Error: {escape(value_freqs['error'])}")

            console.print(stats_tree)

        else:
            error_msg = stats_result.get('error', 'Unknown error')
            logger.error(f"Failed to analyze column statistics for {table_name}.{column_name}: {error_msg}", emoji_key="x")
            console.print(Panel(f"[bold red]:x: Failed to analyze column '{escape(column_name)}':[/]\n {escape(error_msg)}", border_style="red", padding=(1, 2)))
    
    except (ToolError, ToolInputError) as e:
        logger.error(f"Tool error analyzing column stats: {e}", emoji_key="x", exc_info=True)
        console.print(Panel(f"[bold red]:x: Column Stats Tool Error:[/]\n{escape(str(e))}", border_style="red", padding=(1, 2)))
    # except Exception as e:
    #     logger.error(f"Unexpected error in column statistics demo: {e}", emoji_key="x", exc_info=True)
    #     console.print(f"[bold red]:x: Unexpected Error in Column Stats Demo:[/]\n{escape(str(e))}")
    #     # Don't raise here

    console.print()

async def query_execution_demo(connection_id: str) -> None:
    """Demonstrate query execution capabilities."""
    console.print(Rule("[bold green]4. Query Execution Demo[/bold green]", style="green"))
    logger.info("Demonstrating query execution", emoji_key="computer")

    try:
        # Simple SELECT query
        simple_query = "SELECT customer_id, name, email, status FROM customers WHERE status = 'active'"
        logger.info(f"Executing simple query...", emoji_key="db")
        with console.status("[cyan]Running simple query...[/]"):
            query_result = await execute_query(
                connection_id=connection_id,
                query=simple_query,
                read_only=True
            )
        display_query_result("Simple Query: Active Customers", query_result, query_str=simple_query)

        # Parameterized query
        param_query = "SELECT product_id, name, price FROM products WHERE category = :category AND price < :max_price ORDER BY price DESC"
        params = {"category": "Electronics", "max_price": 1000.00}
        logger.info(f"Executing parameterized query with params: {params}", emoji_key="db")
        with console.status("[cyan]Running parameterized query...[/]"):
             param_result = await execute_parameterized_query(
                 connection_id=connection_id,
                 query=param_query,
                 parameters=params,
                 read_only=True
             )
        display_query_result("Parameterized Query: Electronics < $1000", param_result, query_str=param_query)

        # Pagination query
        pagination_query = "SELECT product_id, name, category, price FROM products ORDER BY price DESC"
        logger.info("Executing query with pagination (Page 1)", emoji_key="db")
        with console.status("[cyan]Running paginated query (Page 1)...[/]"):
             pagination_result_p1 = await execute_query_with_pagination(
                 connection_id=connection_id,
                 query=pagination_query,
                 page_size=3,
                 page_number=1
             )
        display_query_result("Paginated Query: Products by Price (Page 1)", pagination_result_p1, query_str=pagination_query)

        logger.info("Executing query with pagination (Page 2)", emoji_key="db")
        with console.status("[cyan]Running paginated query (Page 2)...[/]"):
             pagination_result_p2 = await execute_query_with_pagination(
                 connection_id=connection_id,
                 query=pagination_query,
                 page_size=3,
                 page_number=2
             )
        # Don't show query string again
        display_query_result("Paginated Query: Products by Price (Page 2)", pagination_result_p2)

        # Advanced JOIN query
        join_query = """
        SELECT c.name AS customer_name, o.order_id, o.order_date, o.total_amount, o.status
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE c.status = 'active'
        ORDER BY o.order_date DESC
        """
        logger.info(f"Executing aggregate query...", emoji_key="db")
        with console.status("[cyan]Running aggregate query...[/]"):
            aggregate_result = await execute_query(
                connection_id=connection_id,
                query=join_query,
                read_only=True
            )
        display_query_result("Aggregate Query: Product Category Summary", aggregate_result, query_str=join_query)

    except (ToolError, ToolInputError) as e:
        logger.error(f"Tool error during query execution demo: {e}", emoji_key="x", exc_info=True)
        console.print(Panel(f"[bold red]:x: Query Execution Tool Error:[/]\\n{escape(str(e))}", border_style="red", padding=(1, 2)))
    except Exception as e:
        logger.error(f"Unexpected error in query execution demo: {e}", emoji_key="x", exc_info=True)
        console.print(f"[bold red]:x: Unexpected Error in Query Execution Demo:[/]\\n{escape(str(e))}")
        # Let the main handler print the traceback
        raise # Re-raise unexpected errors

    console.print() # Spacing after the demo section

async def transaction_demo(connection_id: str) -> None:
    """Demonstrate executing transactions."""
    console.print(Rule("[bold green]7. Transaction Demo[/bold green]", style="green"))
    logger.info("Demonstrating database transactions", emoji_key="twisted_rightwards_arrows")

    # Prepare transaction queries (example: add product, update related, select updated)
    transaction_queries = [
        "INSERT INTO products (name, description, price, category, in_stock) VALUES ('Smart Thermostat', 'WiFi enabled thermostat', 129.99, 'Home', 1)", # Query 0
        "UPDATE products SET price = price * 1.05 WHERE category = 'Home'", # Query 1
        "SELECT product_id, name, price FROM products WHERE category = 'Home'" # Query 2
    ]
    params = [None, None, None] # No parameters needed for these queries

    try: # L770: Ensure this try has corresponding except blocks
        logger.info(f"Executing transaction with {len(transaction_queries)} queries...", emoji_key="db")
        with console.status("[cyan]Executing transaction...", spinner="arrow3"):
             transaction_result = await execute_transaction(
                 connection_id=connection_id,
                 queries=transaction_queries,
                 parameters=params, # Pass parameters list
                 read_only=False # Allow INSERT/UPDATE
             )

        if transaction_result.get("success"):
            logger.success("Transaction executed successfully.", emoji_key="heavy_check_mark")

            results = transaction_result.get("results", [])
            exec_time = transaction_result.get('execution_time', 0)

            # Build content for the main panel showing overall success and then details of each query
            # Wrap the initial success message in its own simple Panel
            success_message_panel = Panel(Text.from_markup(f"[green]:heavy_check_mark: Transaction committed successfully ({len(results)} queries in {exec_time:.4f}s).[/]"), padding=(0,1), border_style="dim")
            main_panel_content = [success_message_panel]

            for i, query_result in enumerate(results):
                 # Content for the panel of this specific query result
                 query_panel_content = []
                 original_query = transaction_queries[i]
                 # Add the SQL query syntax highlighted
                 query_panel_content.append(Syntax(original_query.strip(), "sql", theme="default", line_numbers=False))

                 if query_result.get("returns_rows"):
                     # Use query_result to get rows, not undefined 'rows'
                     current_rows = query_result.get("rows", [])
                     row_count = query_result.get("row_count", len(current_rows))
                     if current_rows:
                         # Create the results table, using the correct variable 'res_table'
                         res_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), title=f"{row_count} Row(s)")
                         # Get columns from query_result or infer from the first row if necessary
                         cols = query_result.get("columns", current_rows[0].keys() if current_rows and isinstance(current_rows[0], dict) else [])
                         for col in cols: res_table.add_column(col, style="cyan")
                         # Populate the table rows
                         for row in current_rows:
                             if hasattr(row, '_mapping'):
                                  res_table.add_row(*[escape(str(v)) for v in row._mapping.values()])
                             elif isinstance(row, dict):
                                  res_table.add_row(*[escape(str(v)) for v in row.values()])
                             else:
                                  res_table.add_row(escape(str(row))) # Fallback for non-dict rows
                         query_panel_content.append(res_table) # Add table to panel content
                     else:
                         query_panel_content.append(Text.from_markup("[yellow]No rows returned.[/]"))
                 else:
                     # Handle non-row returning queries (like INSERT, UPDATE)
                     affected = query_result.get("affected_rows")
                     query_panel_content.append(Text.from_markup(f"Affected Rows: [bold magenta]{affected if affected is not None else 'N/A'}[/]"))

                 # Create the panel for this query's results
                 query_panel = Panel(
                      Group(*query_panel_content),  # Use Group to combine multiple renderables
                      title=f"Query {i+1} Result",
                      border_style="blue",
                      padding=(1,2)
                 )
                 # Add this query's panel to the main panel's content list
                 main_panel_content.append(query_panel)

            # Print the main panel containing all query result panels
            # Print each panel individually
            for panel_item in main_panel_content:
                console.print(panel_item)

        else:
            # Handle transaction failure
            error_msg = transaction_result.get('error', 'Unknown error')
            logger.error(f"Transaction failed and rolled back: {error_msg}", emoji_key="x")
            console.print(Panel(f"[bold red]:x: Transaction Failed (Rolled Back):[/]\\n {escape(error_msg)}", border_style="red", padding=(1, 2)))

    except (ToolError, ToolInputError) as e:
        logger.error(f"Tool error during transaction demo: {e}", emoji_key="x", exc_info=True)
        console.print(Panel(f"[bold red]:x: Transaction Tool Error:[/]\\n{escape(str(e))}", border_style="red", padding=(1, 2)))
    except Exception as e:
        logger.error(f"Unexpected error in transaction demo: {e}", emoji_key="x", exc_info=True)
        console.print(f"[bold red]:x: Unexpected Error in Transaction Demo:[/]\\n{escape(str(e))}")
        raise # Re-raise unexpected errors to be caught by main handler

    console.print() # Spacing after the demo section

async def documentation_demo(connection_id: str, tracker: CostTracker) -> None:
    """Demonstrate database documentation generation."""
    console.print(Rule("[bold green]8. Database Documentation Generation[/bold green]", style="green"))
    logger.info("Starting database documentation generation demo", emoji_key="doc")

    with console.status("[bold cyan]Generating database documentation (Markdown)...", spinner="monkey"):
        try:
            doc_result = await generate_database_documentation(
                connection_id=connection_id,
                output_format="markdown",
                include_schema=True,
                include_relationships=True,
                include_samples=True, # Include samples in doc
                include_statistics=True # Include stats in doc
            )

            if doc_result.get("success"):
                logger.success("Successfully generated database documentation.", emoji_key="heavy_check_mark")
                console.print(Panel("[green]:heavy_check_mark: Database documentation generated successfully.[/]", border_style="green", padding=(0, 1)))

                doc_content = doc_result.get("documentation", "")
                
                # Track cost
                if isinstance(doc_result, dict) and "cost" in doc_result and "provider" in doc_result and "model" in doc_result:
                    try:
                        trackable = TrackableResult(
                            cost=doc_result.get("cost", 0.0),
                            input_tokens=doc_result.get("tokens", {}).get("input", 0),
                            output_tokens=doc_result.get("tokens", {}).get("output", 0),
                            provider=doc_result.get("provider", "unknown"),
                            model=doc_result.get("model", "doc_generator"),
                            processing_time=doc_result.get("processing_time", 0.0)
                        )
                        tracker.add_call(trackable)
                    except Exception as track_err:
                        logger.warning(f"Could not track documentation generation cost: {track_err}", exc_info=False)

                console.print(Panel(
                    Syntax(doc_content, "markdown", theme="default", line_numbers=False, word_wrap=True),
                    title="Database Documentation (Markdown Output)",
                    border_style="magenta",
                    padding=1,
                    expand=True # Let it expand to console width
                ))
            else:
                error_msg = doc_result.get('error', 'Unknown error')
                logger.error(f"Failed to generate documentation: {error_msg}", emoji_key="x")
                console.print(Panel(f"[bold red]:x: Failed to generate documentation:[/]\n {escape(error_msg)}", border_style="red", padding=(1, 2)))

        except (ToolError, ToolInputError) as e:
            logger.error(f"Tool error during documentation demo: {e}", emoji_key="x", exc_info=True)
            console.print(Panel(f"[bold red]:x: Documentation Tool Error:[/]\n{escape(str(e))}", border_style="red", padding=(1, 2)))
        except Exception as e:
            logger.error(f"Unexpected error in documentation demo: {e}", emoji_key="x", exc_info=True)
            console.print(f"[bold red]:x: Unexpected Error in Documentation Demo:[/]\n{escape(str(e))}")
            raise

    console.print()

async def cleanup_demo(connection_id: str) -> None:
    """Demonstrate disconnecting from the database."""
    console.print(Rule("[bold green]Database Cleanup and Disconnection[/bold green]", style="green"))
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
    """Run the SQL database interactions demo."""
    console.print(Rule("[bold magenta]SQL Database Interactions Demo Starting[/bold magenta]"))
    connection_id: Optional[str] = None
    exit_code: int = 0 # Initialize exit code
    tracker = CostTracker() # Instantiate tracker

    try:
        # Setup demo database
        connection_id = await connection_demo()

        # Note: tracker is only passed to documentation_demo
        if connection_id:
            await setup_demo_database(connection_id)
            await schema_discovery_demo(connection_id)
            await table_details_demo(connection_id, "customers")
            await find_related_tables_demo(connection_id, "orders")
            await column_statistics_demo(connection_id, "products", "price")
            await query_execution_demo(connection_id)
            await transaction_demo(connection_id)
            await documentation_demo(connection_id, tracker) # Pass tracker
            
            # Final cleanup (will be called again in finally, but good to try here)
            await cleanup_demo(connection_id)
        else:
            logger.error("Skipping subsequent demos due to connection failure.", emoji_key="skip")
            exit_code = 1 # Indicate failure

    except Exception as e:
        logger.critical("SQL Demo failed with unexpected error", emoji_key="critical", exc_info=True)
        console.print("[bold red]CRITICAL ERROR[/bold red]")
        console.print(Traceback.from_exception(type(e), e, e.__traceback__))
        exit_code = 1
    finally:
        # Ensure disconnection even if errors occurred (if connection_id was obtained)
        if connection_id:
            await cleanup_demo(connection_id) # Try cleanup again
            
        # Display cost summary
        tracker.display_summary(console)

    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 