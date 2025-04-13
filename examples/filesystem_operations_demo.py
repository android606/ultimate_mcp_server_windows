#!/usr/bin/env python
"""Filesystem operations demo for LLM Gateway MCP Server.

This example demonstrates the secure filesystem operations tools for
common file and directory manipulation tasks, with robust security
controls that restrict operations to allowed directories.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from llm_gateway.tools.filesystem import (
    create_directory,
    directory_tree,
    edit_file,
    get_file_info,
    list_allowed_directories,
    list_directory,
    move_file,
    read_file,
    read_multiple_files,
    search_files,
    write_file,
)
from llm_gateway.utils import get_logger
from llm_gateway.utils.logging.console import console

# Initialize logger
logger = get_logger("example.filesystem")

# Initialize FastMCP server
mcp = FastMCP("Filesystem Demo")

# Register all tools with MCP
mcp.tool()(read_file)
mcp.tool()(read_multiple_files)
mcp.tool()(write_file)
mcp.tool()(edit_file)
mcp.tool()(create_directory)
mcp.tool()(list_directory)
mcp.tool()(directory_tree)
mcp.tool()(move_file)
mcp.tool()(search_files)
mcp.tool()(get_file_info)
mcp.tool()(list_allowed_directories)

# Create a temporary directory structure for the demo
DEMO_ROOT = None

async def safe_tool_call(tool_name, args):
    """Helper function to safely call a tool and handle errors.
    
    Args:
        tool_name: The name of the tool to call
        args: Arguments to pass to the tool
        
    Returns:
        Dictionary with success status and result or error
    """
    try:
        result = await mcp.call_tool(tool_name, args)
        
        # Basic error checking
        if isinstance(result, dict) and result.get("error"):
            logger.error(f"Tool {tool_name} returned error: {result['error']}", emoji_key="error")
            return {"success": False, "error": result["error"]}
        
        return {"success": True, "result": result}
        
    except Exception as e:
        logger.error(f"Exception calling {tool_name}: {e}", emoji_key="error", exc_info=True)
        return {"success": False, "error": str(e)}


async def setup_demo_environment():
    """Create a temporary directory structure for the demo."""
    global DEMO_ROOT
    
    logger.info("Setting up demo environment...", emoji_key="setup")
    
    # Create a temporary directory
    DEMO_ROOT = Path(tempfile.mkdtemp(prefix="llm_gateway_fs_demo_"))
    
    # Create subdirectories
    project_dirs = [
        DEMO_ROOT / "docs",
        DEMO_ROOT / "src" / "utils",
        DEMO_ROOT / "data",
        DEMO_ROOT / "config",
    ]
    
    for directory in project_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create some sample files
    sample_files = {
        DEMO_ROOT / "README.md": """# Project Demo

This is a demonstration project for testing the secure filesystem operations.

## Features

- File reading and writing
- Directory manipulation
- File searching capabilities
- Metadata retrieval

## Security

All operations are restricted to allowed directories for safety.""",

        DEMO_ROOT / "src" / "main.py": """#!/usr/bin/env python
'''Main entry point for the demo application.'''
import sys
from pathlib import Path

def main():
    '''Main function to run the application.'''
    print("Hello from the demo application!")
    
    # Get configuration
    config = get_config()
    print(f"Running with debug mode: {config['debug']}")
    
    return 0

def get_config():
    '''Get application configuration.'''
    return {
        "debug": True,
        "log_level": "INFO",
        "max_connections": 10
    }

if __name__ == "__main__":
    sys.exit(main())
""",

        DEMO_ROOT / "src" / "utils" / "helpers.py": """'''Helper utilities for the application.'''

def format_message(message, level="info"):
    '''Format a message with level prefix.'''
    return f"[{level.upper()}] {message}"

class DataProcessor:
    '''Process application data.'''
    
    def __init__(self, data_source):
        self.data_source = data_source
        
    def process(self):
        '''Process the data.'''
        return f"Processed {self.data_source}"
""",

        DEMO_ROOT / "docs" / "api.md": """# API Documentation

## Endpoints

### GET /api/v1/status

Returns the current system status.

### POST /api/v1/data

Submit data for processing.

## Authentication

All API calls require an authorization token.
""",

        DEMO_ROOT / "config" / "settings.json": """{
    "appName": "Demo Application",
    "version": "1.0.0",
    "debug": false,
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "demo_db"
    },
    "logging": {
        "level": "info",
        "file": "app.log"
    }
}"""
    }
    
    # Write the sample files
    for file_path, content in sample_files.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    
    logger.success(f"Demo environment set up at: {DEMO_ROOT}", emoji_key="success")
    console.print(Panel(
        f"Created demo project at [cyan]{DEMO_ROOT}[/cyan]\n"
        f"Created [bold]{len(project_dirs)}[/bold] directories and [bold]{len(sample_files)}[/bold] files",
        title="Demo Environment",
        border_style="green"
    ))


async def cleanup_demo_environment():
    """Remove the temporary directory structure."""
    global DEMO_ROOT
    
    if DEMO_ROOT and DEMO_ROOT.exists():
        import shutil
        shutil.rmtree(DEMO_ROOT)
        logger.info(f"Cleaned up demo directory: {DEMO_ROOT}", emoji_key="cleanup")


async def demonstrate_file_reading():
    """Demonstrate file reading operations."""
    console.print(Rule("[bold cyan]File Reading Operations[/bold cyan]"))
    logger.info("Demonstrating file reading operations...", emoji_key="file")
    
    # --- Read Single File ---
    readme_path = str(DEMO_ROOT / "README.md")
    logger.info(f"Reading file: {readme_path}", emoji_key="read")
    
    read_result = await safe_tool_call("read_file", {"path": readme_path})
    
    if read_result["success"]:
        file_content = read_result["result"]["content"]
        file_size = read_result["result"]["size"]
        
        # Display the file content
        console.print(Panel(
            Syntax(file_content, "markdown", theme="monokai", line_numbers=True),
            title=f"[bold]README.md[/bold] ({file_size} bytes)",
            border_style="blue"
        ))
    else:
        console.print(f"[bold red]Error reading file:[/bold red] {read_result['error']}")
    
    # --- Read Multiple Files ---
    logger.info("Reading multiple files at once...", emoji_key="read")
    
    paths = [
        str(DEMO_ROOT / "README.md"),
        str(DEMO_ROOT / "config" / "settings.json"),
        str(DEMO_ROOT / "docs" / "api.md")
    ]
    
    multi_read_result = await safe_tool_call("read_multiple_files", {"paths": paths})
    
    if multi_read_result["success"]:
        results = multi_read_result["result"]
        
        # Create a table to display the results
        table = Table(title="Multiple Files Read Results")
        table.add_column("File", style="cyan")
        table.add_column("Size", style="magenta")
        table.add_column("Status", style="green")
        
        success_count = results.get("succeeded", 0)
        failed_count = results.get("failed", 0)
        
        for file_result in results.get("files", []):
            path = file_result.get("path", "Unknown")
            file_name = os.path.basename(path)
            size = file_result.get("size", "N/A")
            status = "[green]Success[/green]" if file_result.get("success", False) else f"[red]Failed: {file_result.get('error', 'Unknown error')}[/red]"
            
            table.add_row(file_name, str(size), status)
        
        console.print(table)
        console.print(f"[bold]Summary:[/bold] Successfully read [green]{success_count}[/green] files, [red]{failed_count}[/red] failed")
    else:
        console.print(f"[bold red]Error reading multiple files:[/bold red] {multi_read_result['error']}")


async def demonstrate_file_writing():
    """Demonstrate file writing operations."""
    console.print(Rule("[bold cyan]File Writing Operations[/bold cyan]"))
    logger.info("Demonstrating file writing operations...", emoji_key="file")
    
    # --- Write New File ---
    new_file_path = str(DEMO_ROOT / "data" / "report.md")
    
    logger.info(f"Writing new file: {new_file_path}", emoji_key="write")
    
    file_content = """# Analysis Report

## Summary
This report contains the analysis of project performance metrics.

## Key Findings
1. Response time improved by 15%
2. Error rate decreased to 0.5%
3. User satisfaction score: 4.8/5.0

## Recommendations
- Continue monitoring performance
- Implement suggested optimizations
- Schedule follow-up review next quarter
"""
    
    write_result = await safe_tool_call("write_file", {
        "path": new_file_path,
        "content": file_content
    })
    
    if write_result["success"]:
        console.print(Panel(
            f"Successfully wrote [cyan]{os.path.basename(new_file_path)}[/cyan]\n"
            f"Size: [yellow]{write_result['result']['size']}[/yellow] bytes",
            title="File Writing Result",
            border_style="green"
        ))
        
        # Display the content for verification
        console.print(Panel(
            Syntax(file_content, "markdown", theme="monokai"),
            title="Written Content",
            border_style="dim green"
        ))
    else:
        console.print(f"[bold red]Error writing file:[/bold red] {write_result['error']}")


async def demonstrate_file_editing():
    """Demonstrate file editing operations."""
    console.print(Rule("[bold cyan]File Editing Operations[/bold cyan]"))
    logger.info("Demonstrating file editing operations...", emoji_key="file")
    
    # --- Edit File ---
    target_file = str(DEMO_ROOT / "src" / "main.py")
    
    logger.info(f"Editing file: {target_file}", emoji_key="edit")
    
    # First read the file to display before/after
    read_before = await safe_tool_call("read_file", {"path": target_file})
    
    if not read_before["success"]:
        console.print(f"[bold red]Error reading file for edit:[/bold red] {read_before['error']}")
        return
    
    console.print(Panel(
        Syntax(read_before["result"]["content"], "python", theme="monokai", line_numbers=True),
        title="Original File Content",
        border_style="blue"
    ))
    
    # Perform edits
    edits = [
        {
            "oldText": "def get_config():\n    '''Get application configuration.'''\n    return {\n        \"debug\": True,\n        \"log_level\": \"INFO\",\n        \"max_connections\": 10\n    }",
            "newText": "def get_config():\n    '''Get application configuration.'''\n    return {\n        \"debug\": False,  # Changed to False for production\n        \"log_level\": \"WARNING\",  # Increased log level\n        \"max_connections\": 50,  # Increased connection limit\n        \"timeout\": 30  # Added timeout parameter\n    }"
        },
        {
            "oldText": "print(\"Hello from the demo application!\")",
            "newText": "print(\"Hello from the improved demo application!\")\n    print(\"Version 2.0\")"
        }
    ]
    
    # First show a dry run
    logger.info("Performing dry run edit...", emoji_key="edit")
    
    dry_run_result = await safe_tool_call("edit_file", {
        "path": target_file,
        "edits": edits,
        "dry_run": True
    })
    
    if dry_run_result["success"]:
        console.print(Panel(
            Syntax(dry_run_result["result"]["diff"], "diff", theme="monokai"),
            title="Diff Preview (Dry Run)",
            border_style="yellow"
        ))
    else:
        console.print(f"[bold red]Error in dry run edit:[/bold red] {dry_run_result['error']}")
        return
    
    # Now apply the edits
    logger.info("Applying edits...", emoji_key="edit")
    
    edit_result = await safe_tool_call("edit_file", {
        "path": target_file,
        "edits": edits,
        "dry_run": False
    })
    
    if edit_result["success"]:
        # Read the file again to show the result
        read_after = await safe_tool_call("read_file", {"path": target_file})
        
        if read_after["success"]:
            console.print(Panel(
                Syntax(read_after["result"]["content"], "python", theme="monokai", line_numbers=True),
                title="Updated File Content",
                border_style="green"
            ))
        else:
            console.print(f"[bold red]Error reading updated file:[/bold red] {read_after['error']}")
    else:
        console.print(f"[bold red]Error applying edits:[/bold red] {edit_result['error']}")


async def demonstrate_directory_operations():
    """Demonstrate directory operations."""
    console.print(Rule("[bold cyan]Directory Operations[/bold cyan]"))
    logger.info("Demonstrating directory operations...", emoji_key="directory")
    
    # --- Create Directory ---
    new_dir_path = str(DEMO_ROOT / "logs" / "debug")
    
    logger.info(f"Creating directory: {new_dir_path}", emoji_key="mkdir")
    
    mkdir_result = await safe_tool_call("create_directory", {"path": new_dir_path})
    
    if mkdir_result["success"]:
        created = mkdir_result["result"].get("created", False)
        status = "[green]Created new directory[/green]" if created else "[yellow]Directory already existed[/yellow]"
        console.print(f"Directory operation: {status}")
    else:
        console.print(f"[bold red]Error creating directory:[/bold red] {mkdir_result['error']}")
    
    # --- List Directory ---
    dir_to_list = str(DEMO_ROOT)
    
    logger.info(f"Listing directory: {dir_to_list}", emoji_key="ls")
    
    list_result = await safe_tool_call("list_directory", {"path": dir_to_list})
    
    if list_result["success"]:
        entries = list_result["result"].get("entries", [])
        
        table = Table(title=f"Contents of {os.path.basename(dir_to_list)}")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Size", style="yellow")
        
        for entry in entries:
            name = entry.get("name", "Unknown")
            entry_type = entry.get("type", "Unknown")
            size = entry.get("size", "N/A") if entry_type == "file" else ""
            
            icon = "📁" if entry_type == "directory" else "📄"
            table.add_row(f"{icon} {name}", entry_type, str(size))
        
        console.print(table)
    else:
        console.print(f"[bold red]Error listing directory:[/bold red] {list_result['error']}")
    
    # --- Directory Tree ---
    logger.info(f"Generating directory tree for: {dir_to_list}", emoji_key="tree")
    
    tree_result = await safe_tool_call("directory_tree", {
        "path": dir_to_list,
        "max_depth": 2
    })
    
    if tree_result["success"]:
        tree_data = tree_result["result"].get("tree", [])
        
        def format_tree_entry(entry, indent=0):
            """Format a tree entry for display."""
            name = entry.get("name", "Unknown")
            entry_type = entry.get("type", "Unknown")
            
            if entry_type == "directory":
                line = f"{'  ' * indent}📁 [bold cyan]{name}/[/bold cyan]"
                children = entry.get("children", [])
                return line + "\n" + "\n".join(format_tree_entry(child, indent + 1) for child in children)
            else:
                size = entry.get("size", "")
                size_str = f" ([dim]{size} bytes[/dim])" if size else ""
                return f"{'  ' * indent}📄 [green]{name}[/green]{size_str}"
        
        tree_text = "\n".join(format_tree_entry(entry) for entry in tree_data)
        
        console.print(Panel(
            tree_text,
            title="Directory Tree",
            border_style="blue",
            expand=False
        ))
    else:
        console.print(f"[bold red]Error generating directory tree:[/bold red] {tree_result['error']}")


async def demonstrate_file_operations():
    """Demonstrate file operations like move and search."""
    console.print(Rule("[bold cyan]File Operations[/bold cyan]"))
    logger.info("Demonstrating file operations...", emoji_key="file")
    
    # --- Move File ---
    source_path = str(DEMO_ROOT / "docs" / "api.md")
    dest_path = str(DEMO_ROOT / "docs" / "api_v1.md")
    
    logger.info(f"Moving file: {source_path} -> {dest_path}", emoji_key="move")
    
    move_result = await safe_tool_call("move_file", {
        "source": source_path,
        "destination": dest_path
    })
    
    if move_result["success"]:
        console.print(Panel(
            f"Moved: [cyan]{os.path.basename(source_path)}[/cyan] → [green]{os.path.basename(dest_path)}[/green]",
            title="File Move Operation",
            border_style="green"
        ))
    else:
        console.print(f"[bold red]Error moving file:[/bold red] {move_result['error']}")
    
    # --- Search Files ---
    search_dir = str(DEMO_ROOT)
    search_pattern = "config"
    
    logger.info(f"Searching for files matching pattern: '{search_pattern}'", emoji_key="search")
    
    search_result = await safe_tool_call("search_files", {
        "path": search_dir,
        "pattern": search_pattern
    })
    
    if search_result["success"]:
        matches = search_result["result"].get("matches", [])
        
        if matches:
            console.print(Panel(
                "\n".join(f"[cyan]{os.path.relpath(match, DEMO_ROOT)}[/cyan]" for match in matches),
                title=f"Search Results for '{search_pattern}'",
                border_style="magenta",
                expand=False
            ))
        else:
            console.print(f"[yellow]No matches found for '{search_pattern}'[/yellow]")
    else:
        console.print(f"[bold red]Error searching files:[/bold red] {search_result['error']}")
    
    # --- Get File Info ---
    file_path = str(DEMO_ROOT / "config" / "settings.json")
    
    logger.info(f"Getting file information for: {file_path}", emoji_key="info")
    
    info_result = await safe_tool_call("get_file_info", {"path": file_path})
    
    if info_result["success"]:
        file_info = info_result["result"]
        
        # Create a table to display file information
        table = Table(title=f"File Information: {os.path.basename(file_path)}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")
        
        for key, value in file_info.items():
            if key != "success":  # Skip the success flag
                table.add_row(key, str(value))
        
        console.print(table)
    else:
        console.print(f"[bold red]Error getting file info:[/bold red] {info_result['error']}")


async def demonstrate_security_features():
    """Demonstrate security features of the filesystem tools."""
    console.print(Rule("[bold cyan]Security Features[/bold cyan]"))
    logger.info("Demonstrating security features...", emoji_key="security")
    
    # --- List Allowed Directories ---
    logger.info("Listing allowed directories...", emoji_key="security")
    
    allowed_dirs_result = await safe_tool_call("list_allowed_directories", {})
    
    if allowed_dirs_result["success"]:
        allowed_dirs = allowed_dirs_result["result"].get("directories", [])
        
        if allowed_dirs:
            console.print(Panel(
                "\n".join(f"[green]{directory}[/green]" for directory in allowed_dirs),
                title="Allowed Directories",
                border_style="blue",
                expand=False
            ))
        else:
            console.print("[yellow]No allowed directories configured.[/yellow]")
            console.print("For this demo, the temporary directory is automatically allowed.")
    else:
        console.print(f"[bold red]Error listing allowed directories:[/bold red] {allowed_dirs_result['error']}")
    
    # --- Try to Access Outside Directory ---
    outside_path = "/etc/passwd" if sys.platform != "win32" else "C:\\Windows\\System32\\drivers\\etc\\hosts"
    
    logger.info(f"Attempting to access file outside allowed directories: {outside_path}", emoji_key="security")
    console.print(f"Attempting to read: [red]{outside_path}[/red] (should fail)")
    
    outside_result = await safe_tool_call("read_file", {"path": outside_path})
    
    if not outside_result["success"]:
        console.print(Panel(
            f"[green]Security check passed![/green] Access was denied as expected.\n\n"
            f"Error: [dim]{outside_result['error']}[/dim]",
            title="Security Verification",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold red]Security warning![/bold red] Access to outside directory was allowed!",
            title="Security Verification Failed",
            border_style="red"
        ))


async def main():
    """Run the filesystem operations demonstration."""
    try:
        console.print(Rule("[bold blue]Secure Filesystem Operations Demo[/bold blue]"))
        logger.info("Starting filesystem operations demonstration", emoji_key="start")
        
        # Set up the demo environment
        await setup_demo_environment()
        
        # Display info message
        console.print(Panel(
            "This demo showcases the secure filesystem operations provided by LLM Gateway.\n"
            "All operations are restricted to allowed directories for security.",
            title="About This Demo",
            border_style="cyan"
        ))
        
        # Run the demonstrations
        await demonstrate_file_reading()
        console.print()  # Add space between sections
        
        await demonstrate_file_writing()
        console.print()
        
        await demonstrate_file_editing()
        console.print()
        
        await demonstrate_directory_operations()
        console.print()
        
        await demonstrate_file_operations()
        console.print()
        
        await demonstrate_security_features()
        
        # Finish
        logger.success("Filesystem Operations Demo completed successfully!", emoji_key="complete")
        console.print(Rule("[bold green]Demo Complete[/bold green]"))
        
        return 0
        
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"[bold red]Critical Error:[/bold red] {escape(str(e))}")
        return 1
        
    finally:
        # Clean up the demo environment
        await cleanup_demo_environment()


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 