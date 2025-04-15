#!/usr/bin/env python
"""Test script for transaction demo."""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from llm_gateway.exceptions import ToolError, ToolInputError
from llm_gateway.tools.sql_database_interactions import (
    connect_to_database,
    execute_transaction,
)

# Initialize Console
console = Console()

async def test_transaction_demo():
    """Test just the transaction part."""
    console.print("Testing transaction demo...")
    
    # Connect to database
    connection_result = await connect_to_database("sqlite:///demo_database.db")
    if not connection_result.get("success"):
        console.print(f"[red]Failed to connect: {connection_result.get('error', 'Unknown error')}[/]")
        return 1
    
    connection_id = connection_result.get("connection_id")
    console.print(f"[green]Connected with ID: {connection_id}[/]")
    
    # Simple transaction to test Panel rendering
    transaction_queries = [
        "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)",
        "INSERT INTO test_table (name) VALUES ('Test value')",
        "SELECT * FROM test_table"
    ]
    
    try:
        transaction_result = await execute_transaction(
            connection_id=connection_id,
            queries=transaction_queries,
            read_only=False
        )
        
        if transaction_result.get("success"):
            console.print("[green]Transaction succeeded[/]")
            
            results = transaction_result.get("results", [])
            exec_time = transaction_result.get('execution_time', 0)
            
            # Success message panel
            success_message_panel = Panel(
                Text.from_markup(f"[green]Transaction committed successfully ({len(results)} queries in {exec_time:.4f}s).[/]"), 
                padding=(0,1), 
                border_style="dim"
            )
            main_panel_content = [success_message_panel]
            
            # Create panels for each query result
            for i, query_result in enumerate(results):
                query_panel_content = []
                original_query = transaction_queries[i]
                query_panel_content.append(Syntax(original_query.strip(), "sql", theme="default", line_numbers=False))
                
                if query_result.get("returns_rows"):
                    current_rows = query_result.get("rows", [])
                    row_count = query_result.get("row_count", len(current_rows))
                    if current_rows:
                        res_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), title=f"{row_count} Row(s)")
                        cols = query_result.get("columns", current_rows[0].keys() if current_rows and isinstance(current_rows[0], dict) else [])
                        for col in cols:
                            res_table.add_column(col, style="cyan")
                        for row in current_rows:
                            if hasattr(row, '_mapping'):
                                res_table.add_row(*[escape(str(v)) for v in row._mapping.values()])
                            elif isinstance(row, dict):
                                res_table.add_row(*[escape(str(v)) for v in row.values()])
                            else:
                                res_table.add_row(escape(str(row)))
                        query_panel_content.append(res_table)
                    else:
                        query_panel_content.append(Text.from_markup("[yellow]No rows returned.[/]"))
                else:
                    affected = query_result.get("affected_rows")
                    query_panel_content.append(Text.from_markup(f"Affected Rows: [bold magenta]{affected if affected is not None else 'N/A'}[/]"))
                
                # Create the panel for this query's results - using Group to fix the original issue
                query_panel = Panel(
                    Group(*query_panel_content),  # Using Group to combine multiple renderables
                    title=f"Query {i+1} Result",
                    border_style="blue",
                    padding=(1,2)
                )
                
                main_panel_content.append(query_panel)
            
            # Print each panel individually
            for panel_item in main_panel_content:
                console.print(panel_item)
                
        else:
            error_msg = transaction_result.get('error', 'Unknown error')
            console.print(f"[red]Transaction failed: {error_msg}[/]")
            return 1
            
    except (ToolError, ToolInputError) as e:
        console.print(f"[red]Tool error: {e}[/]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/]")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(test_transaction_demo())
    sys.exit(exit_code) 