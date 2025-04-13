#!/usr/bin/env python
"""Filesystem operations demo for LLM Gateway Tools.

This example demonstrates the secure asynchronous filesystem operations tools,
covering file/directory manipulation, searching, metadata retrieval, and
security features like allowed directory restrictions and deletion protection.
"""
import argparse
import asyncio
import os
import platform
import shutil  # Keep sync shutil for cleanup simplicity
import sys
import tempfile
import time
from pathlib import Path

# --- Configuration ---
# Add project root to path for imports when running as script
# Adjust this path if your script location relative to the project root differs
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if not (PROJECT_ROOT / "llm_gateway").is_dir():
        # Fallback if running from a different structure
        PROJECT_ROOT = Path(__file__).resolve().parent
        if not (PROJECT_ROOT / "llm_gateway").is_dir():
             print("Error: Could not reliably determine project root. Make sure llm_gateway is importable.", file=sys.stderr)
             sys.exit(1)
    sys.path.insert(0, str(PROJECT_ROOT))
    # --- Important: Configure Allowed Directories ---
    # For this demo, we will dynamically allow the temporary directory.
    # In a real application, this would be set via llm_gateway.config
    # We'll simulate this by setting an environment variable *before* importing the tools.
    # Create a temporary directory *first*
    DEMO_TEMP_DIR = tempfile.mkdtemp(prefix="llm_gateway_fs_demo_")
    os.environ["LLM_GATEWAY__FILESYSTEM__ALLOWED_DIRECTORIES"] = f'["{DEMO_TEMP_DIR}"]'
    # Force config reload if it was cached previously (might be needed in complex setups)
    os.environ["LLM_GATEWAY_FORCE_CONFIG_RELOAD"] = "true"
    print(f"INFO: Temporarily allowing access to: {DEMO_TEMP_DIR}")
except Exception as e:
    print(f"Error during initial setup: {e}", file=sys.stderr)
    sys.exit(1)


from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

# Now import the tools and other gateway components
# We must import *after* setting the environment variable for allowed dirs
try:
    from llm_gateway.exceptions import ProtectionTriggeredError, ToolError, ToolInputError
    from llm_gateway.tools.filesystem import (
        create_directory,
        delete_path,
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
    from llm_gateway.utils.logging.console import console  # Use the shared console
    # Assume FastMCP or a similar mechanism is available for calling tools
    # For this demo, we'll call the functions directly after setup
except ImportError as e:
     print(f"Import Error: {e}. Please ensure all dependencies are installed and the script is run from the correct location relative to the project.", file=sys.stderr)
     sys.exit(1)


# Initialize logger
logger = get_logger("example.filesystem")

def parse_arguments():
    """Parse command line arguments for the demo."""
    parser = argparse.ArgumentParser(
        description="Filesystem Operations Demo for LLM Gateway Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Available demos:
  all           - Run all demos (default)
  read          - File reading operations
  write         - File writing and editing operations
  directory     - Directory operations (create, list, tree)
  move_delete   - Move, delete, search & info operations
  security      - Security features demo
"""
    )
    
    parser.add_argument('demo', nargs='?', default='all',
                        choices=['all', 'read', 'write', 'directory', 'move_delete', 'security'],
                        help='Specific demo to run (default: all)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Increase output verbosity')
    
    parser.add_argument('--rich-tree', action='store_true', 
                        help='Use enhanced rich tree visualization for directory trees')
    
    return parser.parse_args()

# --- Demo Setup ---
# DEMO_ROOT is the base *within* the allowed temporary directory
DEMO_ROOT = Path(DEMO_TEMP_DIR) / "demo_project"
BULK_FILES_COUNT = 110 # Number of files to create for deletion protection demo (>100)

async def safe_tool_call(tool_func, args_dict, description=""):
    """Helper function to safely call a tool function and display results/errors."""
    tool_name = tool_func.__name__
    call_desc = description or f"Calling [bold magenta]{tool_name}[/bold magenta]"
    args_str = ", ".join(f"{k}=[yellow]{v!r}[/yellow]" for k, v in args_dict.items())
    console.print(Panel(f"{call_desc}\nArgs: {args_str}", title="Tool Call", border_style="blue", expand=False))

    start_time = time.monotonic()
    try:
        # Directly await the function since we imported them
        result = await tool_func(**args_dict)
        duration = time.monotonic() - start_time

        # The tools now often return rich dictionaries, potentially formatted by create_tool_response
        # or directly structured for success/failure indication.
        is_error = isinstance(result, dict) and (result.get("isError") or not result.get("success", True))
        protection_triggered = isinstance(result, dict) and result.get("protectionTriggered")

        if protection_triggered:
             console.print(Panel(
                 f"[bold yellow]🛡️ Protection Triggered![/bold yellow]\n"
                 f"Message: {result.get('message', 'No message')}\n"
                 f"Context: {result.get('context', {})}",
                 title=f"Result: {tool_name} (Blocked)",
                 border_style="yellow",
                 subtitle=f"Duration: {duration:.3f}s"
             ))
             # Return a structure indicating failure due to protection
             return {"success": False, "protection_triggered": True, "result": result}
        elif is_error:
            error_msg = result.get("message", str(result.get("content", "Unknown error")))
            context = result.get("context", None)
            console.print(Panel(
                f"[bold red]Error[/bold red]\n"
                f"Message: {escape(error_msg)}\n"
                f"Context: {context if context else 'N/A'}",
                title=f"Result: {tool_name} (Failed)",
                border_style="red",
                subtitle=f"Duration: {duration:.3f}s"
            ))
            return {"success": False, "error": error_msg, "context": context, "result": result}
        else:
            # Successful result - display nicely
            # Try to format common result structures
            output_content = ""
            if isinstance(result, dict):
                 # Common success patterns
                 if "message" in result:
                      output_content += f"Message: [green]{escape(result['message'])}[/green]\n"
                 if "path" in result:
                      output_content += f"Path: [cyan]{escape(str(result['path']))}[/cyan]\n"
                 if "size" in result:
                      output_content += f"Size: [yellow]{result['size']}[/yellow] bytes\n"
                 if "created" in result:
                      output_content += f"Created: {'Yes' if result['created'] else 'No (already existed)'}\n"
                 if "diff" in result:
                       output_content += f"Diff:\n{Syntax(result['diff'], 'diff', theme='monokai', background_color='default')}\n"
                 if "matches" in result and "pattern" in result:
                       output_content += f"Search Matches ({len(result['matches'])}):\n"
                       output_content += "\n".join(f"- [cyan]{escape(os.path.relpath(m, DEMO_ROOT))}[/cyan]" for m in result['matches'][:20])
                       if len(result['matches']) > 20: 
                            output_content += "\n- ... (more matches)"
                       if "warnings" in result: 
                            output_content += "\n[yellow]Warnings:[/yellow]\n" + "\n".join(f"- {escape(w)}" for w in result['warnings']) + "\n"
                 elif "entries" in result and "path" in result: # list_directory
                       output_content += f"Directory Listing for [cyan]{escape(str(result['path']))}[/cyan]:\n"
                       table = Table(show_header=True, header_style="bold magenta", box=None)
                       table.add_column("Name", style="cyan")
                       table.add_column("Type", style="green")
                       table.add_column("Info", style="yellow")
                       for entry in result.get('entries', []):
                            name = entry.get('name', '?')
                            etype = entry.get('type', 'unknown')
                            info_str = ""
                            if etype == 'file' and 'size' in entry: 
                                 info_str += f"{entry['size']} bytes"
                            elif etype == 'symlink' and 'symlink_target' in entry: 
                                 info_str += f"-> {entry['symlink_target']}"
                            if 'error' in entry: 
                                 info_str += f" [red](Error: {entry['error']})[/red]"
                            icon = "📄" if etype == "file" else "📁" if etype == "directory" else "🔗" if etype=="symlink" else "❓"
                            table.add_row(f"{icon} {escape(name)}", etype, info_str)
                       from io import StringIO
                       string_io = StringIO()
                       console_capture = console.__class__(file=string_io, force_terminal=True, color_system="truecolor")
                       console_capture.print(table)
                       output_content += string_io.getvalue()
                       if "warnings" in result: 
                            output_content += "\n[yellow]Warnings:[/yellow]\n" + "\n".join(f"- {escape(w)}" for w in result['warnings']) + "\n"
                 elif "tree" in result and "path" in result: # directory_tree
                       output_content += f"Directory Tree for [cyan]{escape(str(result['path']))}[/cyan]:\n"
                       rich_tree = Tree(f"📁 [bold cyan]{escape(os.path.basename(result['path']))}[/bold cyan]")
                       def build_rich_tree(parent_node, children):
                           for item in children:
                                name = item.get("name", "?")
                                item_type = item.get("type", "unknown")
                                info = ""
                                if "size" in item: 
                                     size_bytes = item['size']
                                     # Improved size formatting
                                     if size_bytes < 1024:
                                          info += f" ({size_bytes}b)"
                                     elif size_bytes < 1024 * 1024:
                                          info += f" ({size_bytes/1024:.1f}KB)"
                                     else:
                                          info += f" ({size_bytes/(1024*1024):.1f}MB)"
                                if "target" in item: 
                                     info += f" → {item['target']}"
                                if "error" in item: 
                                     info += f" [red](Error: {item['error']})[/red]"

                                if item_type == "directory":
                                    node = parent_node.add(f"📁 [bold cyan]{escape(name)}[/bold cyan]{info}")
                                    if "children" in item: 
                                         build_rich_tree(node, item["children"])
                                elif item_type == "file":
                                     # Guess file type for better icon
                                     icon = "📄"
                                     ext = os.path.splitext(name)[1].lower()
                                     if ext in ['.jpg', '.png', '.gif', '.bmp', '.jpeg', '.svg']:
                                          icon = "🖼️"
                                     elif ext in ['.mp3', '.wav', '.ogg', '.flac']:
                                          icon = "🎵"
                                     elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                                          icon = "🎬"
                                     elif ext in ['.py', '.js', '.java', '.c', '.cpp', '.go', '.rs']:
                                          icon = "📜"
                                     elif ext in ['.json', '.xml', '.yaml', '.yml']:
                                          icon = "📋"
                                     elif ext in ['.zip', '.tar', '.gz', '.7z', '.rar']:
                                          icon = "📦"
                                     elif ext in ['.md', '.txt', '.doc', '.docx', '.pdf']:
                                          icon = "📝"
                                     
                                     parent_node.add(f"{icon} [green]{escape(name)}[/green]{info}")
                                elif item_type == "symlink":
                                     parent_node.add(f"🔗 [magenta]{escape(name)}[/magenta]{info}")
                                elif item_type == "info":
                                     parent_node.add(f"ℹ️ [dim]{escape(name)}[/dim]")
                                elif item_type == "error":
                                     parent_node.add(f"❌ [red]{escape(name)}[/red]")
                                else:
                                     parent_node.add(f"❓ [yellow]{escape(name)}[/yellow]{info}")
                       build_rich_tree(rich_tree, result["tree"])
                       from io import StringIO
                       string_io = StringIO()
                       console_capture = console.__class__(file=string_io, force_terminal=True, color_system="truecolor")
                       console_capture.print(rich_tree)
                       output_content += string_io.getvalue()
                 elif "directories" in result and "count" in result: # list_allowed_directories
                        output_content += f"Allowed Directories ({result['count']}):\n"
                        output_content += "\n".join(f"- [green]{escape(d)}[/green]" for d in result['directories']) + "\n"
                 elif "files" in result and "succeeded" in result: # read_multiple_files
                        output_content += f"Read Results: [green]{result['succeeded']} succeeded[/green], [red]{result['failed']} failed[/red]\n"
                        for file_res in result['files']:
                            path_str = escape(str(file_res.get('path', 'N/A')))
                            if file_res.get('success'):
                                size_info = f" ({file_res.get('size', 'N/A')}b)" if 'size' in file_res else ""
                                content_preview = escape(str(file_res.get('content', ''))[:100]) + "..." if len(str(file_res.get('content', ''))) > 100 else escape(str(file_res.get('content', '')))
                                output_content += f"- [green]Success[/green]: [cyan]{path_str}[/cyan]{size_info}\n  Content: '{content_preview}'\n"
                            else:
                                output_content += f"- [red]Failed[/red]: [cyan]{path_str}[/cyan]\n  Error: {escape(str(file_res.get('error', 'Unknown')))}\n"
                 elif "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0 and "text" in result["content"][0]:
                      # Generic MCP content block display
                      output_content += "Content:\n" + "\n".join([escape(block.get("text","")) for block in result["content"] if block.get("type")=="text"]) + "\n"
                 elif "name" in result and "modified" in result: # get_file_info
                      output_content += f"File Info for [cyan]{escape(result['name'])}[/cyan]:\n"
                      info_table = Table(show_header=False, box=None)
                      info_table.add_column("Property", style="blue")
                      info_table.add_column("Value", style="yellow")
                      skip_keys = {"success", "message", "path"} # Already shown or redundant
                      for k, v in result.items():
                           if k not in skip_keys:
                               info_table.add_row(escape(k), escape(str(v)))
                      from io import StringIO
                      string_io = StringIO()
                      console_capture = console.__class__(file=string_io, force_terminal=True, color_system="truecolor")
                      console_capture.print(info_table)
                      output_content += string_io.getvalue()
                 else:
                      # Fallback: Dump the dict (excluding potentially large 'content')
                      output_content += "Result Data:\n" + escape(str({k:v for k,v in result.items() if k != 'content'})) + "\n"
            else:
                 # Non-dict result? Should be rare with new structure.
                 output_content = escape(str(result))

            console.print(Panel(
                 output_content,
                 title=f"Result: {tool_name} (Success)",
                 border_style="green",
                 subtitle=f"Duration: {duration:.3f}s"
            ))
            return {"success": True, "result": result}

    except ProtectionTriggeredError as pte:
         duration = time.monotonic() - start_time
         logger.warning(f"Protection triggered calling {tool_name}: {pte}", emoji_key="security")
         console.print(Panel(
             f"[bold yellow]🛡️ Protection Triggered![/bold yellow]\n"
             f"Message: {escape(str(pte))}\n"
             f"Context: {pte.context}",
             title=f"Result: {tool_name} (Blocked)",
             border_style="yellow",
             subtitle=f"Duration: {duration:.3f}s"
         ))
         return {"success": False, "protection_triggered": True, "error": str(pte), "context": pte.context}
    except (ToolInputError, ToolError) as tool_err:
         duration = time.monotonic() - start_time
         logger.error(f"Tool Error calling {tool_name}: {tool_err}", emoji_key="error", details=getattr(tool_err, 'context', None))
         console.print(Panel(
             f"[bold red]{type(tool_err).__name__}[/bold red]\n"
             f"Message: {escape(str(tool_err))}\n"
             f"Context: {getattr(tool_err, 'context', 'N/A')}",
             title=f"Result: {tool_name} (Failed)",
             border_style="red",
             subtitle=f"Duration: {duration:.3f}s"
         ))
         return {"success": False, "error": str(tool_err), "context": getattr(tool_err, 'context', None)}
    except Exception as e:
        duration = time.monotonic() - start_time
        logger.critical(f"Unexpected Exception calling {tool_name}: {e}", emoji_key="critical", exc_info=True)
        console.print(Panel(
            f"[bold red]Unexpected Error ({type(e).__name__})[/bold red]\n"
            f"{escape(str(e))}",
            title=f"Result: {tool_name} (Critical Failure)",
            border_style="red",
            subtitle=f"Duration: {duration:.3f}s"
        ))
        return {"success": False, "error": f"Unexpected: {str(e)}"}

async def setup_demo_environment():
    """Create a temporary directory structure for the demo."""
    logger.info("Setting up demo environment...", emoji_key="setup")
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    project_dirs = [
        DEMO_ROOT / "docs",
        DEMO_ROOT / "src" / "utils",
        DEMO_ROOT / "data",
        DEMO_ROOT / "config",
        DEMO_ROOT / "tests",
        DEMO_ROOT / ".hidden_dir",
        DEMO_ROOT / "bulk_files" # For deletion protection demo
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

# A line with different whitespace for editing demo
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
        # TODO: Implement actual processing
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
}""",
        DEMO_ROOT / "data" / "sample.csv": "ID,Value,Category\n1,10.5,A\n2,15.2,B\n3,9.8,A",
        DEMO_ROOT / "tests" / "test_helpers.py": """import pytest
from src.utils.helpers import format_message

def test_format_message():
    assert format_message("Test", "debug") == "[DEBUG] Test"
""",
        DEMO_ROOT / ".gitignore": "*.log\n*.tmp\n.hidden_dir/\n",
        DEMO_ROOT / "temp.log": "Log file content - should be excluded by search patterns.",
        # Add a file with potentially non-UTF8 data (simulated)
        DEMO_ROOT / "data" / "binary_data.bin": b'\x80\x02\x95\n\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x04data\x94\x8c\x06binary\x94s.'
    }

    for file_path, content in sample_files.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            file_path.write_text(content, encoding='utf-8')
        else:
            file_path.write_bytes(content)

    # Create bulk files for deletion protection test
    bulk_dir = DEMO_ROOT / "bulk_files"
    for i in range(BULK_FILES_COUNT):
        # Introduce slight variations in timestamps and extensions
        ext = ".txt" if i % 3 == 0 else ".tmp" if i % 3 == 1 else ".dat"
        fpath = bulk_dir / f"file_{i:03d}{ext}"
        fpath.write_text(f"Content for file {i}")
        # Vary modification times slightly (may not be precise enough on all FS)
        if i % 5 == 0:
             await asyncio.sleep(0.001) # Small delay
             os.utime(fpath, (time.time() - i * 60, time.time() - i * 60)) # Set past mtime

    # Create a symlink (if supported)
    SYMLINK_PATH = DEMO_ROOT / "link_to_src"
    TARGET_PATH = DEMO_ROOT / "src"
    try:
        # Check if symlinks are supported (e.g., Windows needs admin rights or dev mode)
        can_symlink = hasattr(os, "symlink")
        if platform.system() == "Windows":
            # Basic check, might not be perfect
            try:
                os.symlink("NUL", DEMO_ROOT / "test_link_nul", target_is_directory=False)
                (DEMO_ROOT / "test_link_nul").unlink()
            except (OSError, AttributeError, NotImplementedError):
                 can_symlink = False
                 logger.warning("Symlink creation might not be supported or permitted on this system. Skipping symlink tests.", emoji_key="warning")

        if can_symlink:
            os.symlink(TARGET_PATH, SYMLINK_PATH, target_is_directory=True)
            logger.info(f"Created symlink: {SYMLINK_PATH} -> {TARGET_PATH}", emoji_key="link")
        else:
             SYMLINK_PATH = None # Indicate symlink wasn't created
    except OSError as e:
        logger.warning(f"Could not create symlink ({SYMLINK_PATH} -> {TARGET_PATH}): {e}. Skipping symlink tests.", emoji_key="warning")
        SYMLINK_PATH = None # Indicate symlink wasn't created
    except Exception as e:
        logger.error(f"Unexpected error creating symlink: {e}", emoji_key="error", exc_info=True)
        SYMLINK_PATH = None

    logger.success(f"Demo environment set up at: {DEMO_ROOT}", emoji_key="success")
    console.print(Panel(
        f"Created demo project within [cyan]{DEMO_ROOT.parent}[/cyan] at [cyan]{DEMO_ROOT.name}[/cyan]\n"
        f"Created [bold]{len(project_dirs)}[/bold] directories and [bold]{len(sample_files)}[/bold] files.\n"
        f"Created [bold]{BULK_FILES_COUNT}[/bold] files in 'bulk_files/' for deletion test.\n"
        f"Symlink created: {'Yes' if SYMLINK_PATH else 'No'}",
        title="Demo Environment Ready",
        border_style="green",
        expand=False
    ))
    return SYMLINK_PATH

async def cleanup_demo_environment():
    """Remove the temporary directory structure using standard shutil."""
    global DEMO_TEMP_DIR
    if DEMO_TEMP_DIR and Path(DEMO_TEMP_DIR).exists():
        try:
            # Use synchronous shutil for cleanup simplicity
            shutil.rmtree(DEMO_TEMP_DIR)
            logger.info(f"Cleaned up demo directory: {DEMO_TEMP_DIR}", emoji_key="cleanup")
            console.print(f"Cleaned up demo directory: [dim]{DEMO_TEMP_DIR}[/dim]")
        except Exception as e:
            logger.error(f"Error during cleanup of {DEMO_TEMP_DIR}: {e}", emoji_key="error")
            console.print(f"[bold red]Error cleaning up demo directory {DEMO_TEMP_DIR}: {e}[/bold red]")
    DEMO_TEMP_DIR = None


async def demonstrate_file_reading(symlink_path):
    """Demonstrate file reading operations."""
    console.print(Rule("[bold cyan]1. File Reading Operations[/bold cyan]", style="cyan"))
    logger.info("Demonstrating file reading operations...", emoji_key="file")

    # --- Read Single File (Text) ---
    readme_path = str(DEMO_ROOT / "README.md")
    await safe_tool_call(read_file, {"path": readme_path}, description="Reading a text file (README.md)")

    # --- Read Single File (JSON) ---
    settings_path = str(DEMO_ROOT / "config" / "settings.json")
    await safe_tool_call(read_file, {"path": settings_path}, description="Reading a JSON file (settings.json)")

    # --- Read Single File (Simulated Binary) ---
    binary_path = str(DEMO_ROOT / "data" / "binary_data.bin")
    await safe_tool_call(read_file, {"path": binary_path}, description="Reading a binary file (expecting hex preview)")

    # --- Read Non-Existent File ---
    non_existent_path = str(DEMO_ROOT / "non_existent.txt")
    await safe_tool_call(read_file, {"path": non_existent_path}, description="Attempting to read a non-existent file (should fail)")

    # --- Read a Directory (should fail) ---
    dir_path = str(DEMO_ROOT / "src")
    await safe_tool_call(read_file, {"path": dir_path}, description="Attempting to read a directory as a file (should fail)")

    # --- Read Multiple Files (Success and Failure Mix) ---
    paths_to_read = [
        str(DEMO_ROOT / "README.md"),
        str(DEMO_ROOT / "src" / "main.py"),
        str(DEMO_ROOT / "non_existent.txt"), # This one will fail
        str(DEMO_ROOT / "config" / "settings.json"),
        str(DEMO_ROOT / "src") # Reading a directory will also fail here
    ]
    await safe_tool_call(read_multiple_files, {"paths": paths_to_read}, description="Reading multiple files (including some that will fail)")

    # --- Read file via Symlink (if created) ---
    if symlink_path:
         # Reading a file within the linked directory
         linked_file_path = str(symlink_path / "main.py")
         await safe_tool_call(read_file, {"path": linked_file_path}, description=f"Reading a file via symlink ({os.path.basename(symlink_path)}/main.py)")

async def demonstrate_file_writing_editing():
    """Demonstrate file writing and editing operations."""
    console.print(Rule("[bold cyan]2. File Writing & Editing Operations[/bold cyan]", style="cyan"))
    logger.info("Demonstrating file writing and editing operations...", emoji_key="file")

    # --- Write New File ---
    new_file_path = str(DEMO_ROOT / "data" / "report.md")
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
    await safe_tool_call(write_file, {"path": new_file_path, "content": file_content}, description="Writing a new file (report.md)")

    # --- Overwrite Existing File ---
    overwrite_content = "# Analysis Report (V2)\n\nReport updated."
    await safe_tool_call(write_file, {"path": new_file_path, "content": overwrite_content}, description="Overwriting the existing file (report.md)")
    # Verify overwrite
    await safe_tool_call(read_file, {"path": new_file_path}, description="Reading the overwritten file to verify")

    # --- Attempt to Write to a Directory (should fail) ---
    await safe_tool_call(write_file, {"path": str(DEMO_ROOT / "src"), "content": "test"}, description="Attempting to write over a directory (should fail)")

    # --- Edit File (main.py) ---
    target_edit_file = str(DEMO_ROOT / "src" / "main.py")

    # Edits including one requiring whitespace-insensitive fallback
    edits = [
        {
            "oldText": 'print("Hello from the demo application!")', # Exact match
            "newText": 'print("Hello from the UPDATED demo application!")\n    logger.info("App started")'
        },
        {
            # This uses different leading whitespace than the original file
            "oldText": "def main():\n    '''Main function to run the application.'''",
            # Expected fallback behavior: find based on stripped lines, replace using original indentation
            "newText": "def main():\n    '''The primary execution function.''' # Docstring updated"
        },
         {
             "oldText": '    return {\n        "debug": True,\n        "log_level": "INFO",\n        "max_connections": 10\n    }',
             "newText": '    return {\n        "debug": False, # Changed to False\n        "log_level": "WARNING",\n        "max_connections": 25 # Increased limit\n    }'
         }
    ]

    await safe_tool_call(read_file, {"path": target_edit_file}, description="Reading main.py before editing")

    # Edit with Dry Run
    await safe_tool_call(edit_file, {"path": target_edit_file, "edits": edits, "dry_run": True}, description="Editing main.py (Dry Run - showing diff)")

    # Apply Edits for Real
    await safe_tool_call(edit_file, {"path": target_edit_file, "edits": edits, "dry_run": False}, description="Applying edits to main.py")

    # Verify Edits
    await safe_tool_call(read_file, {"path": target_edit_file}, description="Reading main.py after editing")

    # --- Edit with Non-Existent Old Text (should fail) ---
    failed_edit = [{"oldText": "This text does not exist in the file", "newText": "Replacement"}]
    await safe_tool_call(edit_file, {"path": target_edit_file, "edits": failed_edit}, description="Attempting edit with non-existent 'oldText' (should fail)")


async def demonstrate_directory_operations(symlink_path, use_rich_tree=False):
    """Demonstrate directory creation, listing, and tree view."""
    console.print(Rule("[bold cyan]3. Directory Operations[/bold cyan]", style="cyan"))
    logger.info("Demonstrating directory operations...", emoji_key="directory")

    # --- Create Directory ---
    new_dir_path = str(DEMO_ROOT / "logs" / "debug")
    await safe_tool_call(create_directory, {"path": new_dir_path}, description="Creating a new nested directory (logs/debug)")

    # --- Create Directory (already exists) ---
    await safe_tool_call(create_directory, {"path": new_dir_path}, description="Attempting to create the same directory again (idempotent)")

    # --- Attempt to Create Directory over a File (should fail) ---
    file_path_for_dir = str(DEMO_ROOT / "README.md")
    await safe_tool_call(create_directory, {"path": file_path_for_dir}, description="Attempting to create directory over an existing file (README.md - should fail)")

    # --- List Directory (Root) ---
    await safe_tool_call(list_directory, {"path": str(DEMO_ROOT)}, description=f"Listing contents of demo root ({DEMO_ROOT.name})")

    # --- List Directory (Subdir) ---
    await safe_tool_call(list_directory, {"path": str(DEMO_ROOT / "src")}, description="Listing contents of subdirectory (src)")

    # --- List Directory (via Symlink, if created) ---
    if symlink_path:
         await safe_tool_call(list_directory, {"path": str(symlink_path)}, description=f"Listing contents via symlink ({os.path.basename(symlink_path)})")

    # --- List Non-Existent Directory (should fail) ---
    await safe_tool_call(list_directory, {"path": str(DEMO_ROOT / "no_such_dir")}, description="Attempting to list non-existent directory (should fail)")

    # --- Enhanced visualization for directory tree if requested ---
    if use_rich_tree:
        # Create a rich Tree for visual representation
        console.print("\n[bold cyan]Enhanced Directory Tree Visualization[/bold cyan]")
        
        root_tree = Tree(f"📁 [bold cyan]{DEMO_ROOT.name}[/bold cyan]")
        
        # Helper function to build tree recursively
        async def build_visual_tree(path, tree_node, depth=0, max_depth=3):
            if depth >= max_depth:
                tree_node.add("📁 [dim]...(max depth reached)[/dim]")
                return
                
            try:
                items = sorted(os.listdir(path))
                for item in items:
                    item_path = os.path.join(path, item)
                    
                    # Skip hidden files/dirs for cleaner visualization
                    if item.startswith('.') and item != '.gitignore':
                        continue
                        
                    try:
                        is_dir = os.path.isdir(item_path)
                        is_link = os.path.islink(item_path)
                        
                        if is_link:
                            target = os.readlink(item_path)
                            link_node = tree_node.add(f"🔗 [magenta]{item}[/magenta] → {target}")  # noqa: F841
                        elif is_dir:
                            dir_node = tree_node.add(f"📁 [bold cyan]{item}[/bold cyan]")
                            await build_visual_tree(item_path, dir_node, depth + 1, max_depth)
                        else:
                            size = os.path.getsize(item_path)
                            size_str = f"({size:,} bytes)" if size < 10000 else f"({size/1024:.1f} KB)"
                            tree_node.add(f"📄 [green]{item}[/green] {size_str}")
                    except OSError as e:
                        tree_node.add(f"❌ [red]{item} - Error: {str(e)}[/red]")
            except OSError as e:
                tree_node.add(f"❌ [red]Error listing directory: {str(e)}[/red]")
        
        # Build and display the tree
        await build_visual_tree(DEMO_ROOT, root_tree, max_depth=3)
        console.print(root_tree)
        console.print()

    # --- Directory Tree (Default Depth) ---
    await safe_tool_call(directory_tree, {"path": str(DEMO_ROOT)}, description="Generating directory tree for demo root (default depth)")

    # --- Directory Tree (Specific Depth) ---
    await safe_tool_call(directory_tree, {"path": str(DEMO_ROOT), "max_depth": 1}, description="Generating directory tree (max_depth=1)")

    # --- Directory Tree (Include Size) ---
    await safe_tool_call(directory_tree, {"path": str(DEMO_ROOT), "max_depth": 2, "include_size": True}, description="Generating directory tree (max_depth=2, include_size=True)")

    # --- Directory Tree (via Symlink, if created) ---
    if symlink_path:
         await safe_tool_call(directory_tree, {"path": str(symlink_path), "max_depth": 1}, description=f"Generating directory tree via symlink ({os.path.basename(symlink_path)}, max_depth=1)")

async def demonstrate_move_delete_search(symlink_path):
    """Demonstrate file/directory moving, deletion, searching, and info retrieval."""
    console.print(Rule("[bold cyan]4. Move, Delete, Search & Info Operations[/bold cyan]", style="cyan"))
    logger.info("Demonstrating move, delete, search, info operations...", emoji_key="file")

    # --- Get File Info (File) ---
    settings_json_path = str(DEMO_ROOT / "config" / "settings.json")
    await safe_tool_call(get_file_info, {"path": settings_json_path}, description="Getting file info for settings.json")

    # --- Get File Info (Directory) ---
    src_dir_path = str(DEMO_ROOT / "src")
    await safe_tool_call(get_file_info, {"path": src_dir_path}, description="Getting file info for src directory")

    # --- Get File Info (Symlink, if created) ---
    if symlink_path:
        await safe_tool_call(get_file_info, {"path": str(symlink_path)}, description=f"Getting file info for symlink ({os.path.basename(symlink_path)}) - uses lstat")

    # --- Search Files (Name Match, Case Insensitive) ---
    await safe_tool_call(search_files, {"path": str(DEMO_ROOT), "pattern": "readme"}, description="Searching for 'readme' (case insensitive)")

    # --- Search Files (Name Match, Case Sensitive) ---
    await safe_tool_call(search_files, {"path": str(DEMO_ROOT), "pattern": "README", "case_sensitive": True}, description="Searching for 'README' (case sensitive)")

    # --- Search Files (With Exclusions) ---
    await safe_tool_call(search_files,
                         {"path": str(DEMO_ROOT), "pattern": ".py", "exclude_patterns": ["*/test*", ".hidden_dir/*"]},
                         description="Searching for '*.py', excluding tests and hidden dir")

    # --- Search Files (Content Search) ---
    await safe_tool_call(search_files,
                         {"path": str(DEMO_ROOT), "pattern": "localhost", "search_content": True},
                         description="Searching for content 'localhost' inside files")

    # --- Search Files (Content Search, Case Sensitive) ---
    await safe_tool_call(search_files,
                         {"path": str(DEMO_ROOT), "pattern": "DataProcessor", "search_content": True, "case_sensitive": True},
                         description="Searching for content 'DataProcessor' (case sensitive)")

    # --- Search Files (No Matches) ---
    await safe_tool_call(search_files, {"path": str(DEMO_ROOT), "pattern": "xyz_no_match_xyz"}, description="Searching for pattern guaranteed not to match")

    # --- Move File ---
    source_move_path = str(DEMO_ROOT / "data" / "sample.csv")
    dest_move_path = str(DEMO_ROOT / "data" / "renamed_sample.csv")
    await safe_tool_call(move_file, {"source": source_move_path, "destination": dest_move_path}, description="Moving (renaming) sample.csv")
    # Verify move by trying to get info on new path
    await safe_tool_call(get_file_info, {"path": dest_move_path}, description="Verifying move by getting info on new path")

    # --- Move File (Overwrite) ---
    # First create a file to be overwritten
    overwrite_target_path = str(DEMO_ROOT / "data" / "overwrite_me.txt")
    await safe_tool_call(write_file, {"path": overwrite_target_path, "content": "Original content"}, description="Creating file to be overwritten")
    # Now move onto it with overwrite=True
    await safe_tool_call(move_file,
                         {"source": dest_move_path, "destination": overwrite_target_path, "overwrite": True},
                         description="Moving renamed_sample.csv onto overwrite_me.txt (overwrite=True)")
    # Verify overwrite
    await safe_tool_call(get_file_info, {"path": overwrite_target_path}, description="Verifying overwrite by getting info")

    # --- Move Directory ---
    source_dir_move = str(DEMO_ROOT / "tests")
    dest_dir_move = str(DEMO_ROOT / "tests_moved")
    await safe_tool_call(move_file, {"source": source_dir_move, "destination": dest_dir_move}, description="Moving the 'tests' directory")
    # Verify move
    await safe_tool_call(list_directory, {"path": dest_dir_move}, description="Verifying directory move by listing new path")

    # --- Attempt Move (Destination Exists, No Overwrite - should fail) ---
    await safe_tool_call(move_file,
                         {"source": str(DEMO_ROOT / "README.md"), "destination": str(DEMO_ROOT / "config" / "settings.json")},
                         description="Attempting to move README.md onto settings.json (no overwrite - should fail)")

    # --- Delete File ---
    file_to_delete = str(DEMO_ROOT / "temp.log")
    await safe_tool_call(get_file_info, {"path": file_to_delete}, description="Checking temp.log exists before deleting")
    await safe_tool_call(delete_path, {"path": file_to_delete}, description="Deleting single file (temp.log)")
    await safe_tool_call(get_file_info, {"path": file_to_delete}, description="Verifying temp.log deletion (should fail)")

    # --- Delete Symlink (if created) ---
    if symlink_path:
        await safe_tool_call(get_file_info, {"path": str(symlink_path)}, description=f"Checking symlink {os.path.basename(symlink_path)} exists before deleting")
        await safe_tool_call(delete_path, {"path": str(symlink_path)}, description=f"Deleting symlink ({os.path.basename(symlink_path)})")
        await safe_tool_call(get_file_info, {"path": str(symlink_path)}, description="Verifying symlink deletion (should fail)")

    # --- Delete Empty Directory ---
    empty_dir_to_delete = str(DEMO_ROOT / "logs" / "debug") # Created earlier, should be empty
    await safe_tool_call(get_file_info, {"path": empty_dir_to_delete}, description="Checking logs/debug exists before deleting")
    await safe_tool_call(delete_path, {"path": empty_dir_to_delete}, description="Deleting empty directory (logs/debug)")
    await safe_tool_call(get_file_info, {"path": empty_dir_to_delete}, description="Verifying empty directory deletion (should fail)")

    # --- Delete Directory with Content (Testing Deletion Protection) ---
    bulk_dir_path = str(DEMO_ROOT / "bulk_files")
    console.print(Panel(
        f"Attempting to delete directory '{os.path.basename(bulk_dir_path)}' which contains {BULK_FILES_COUNT} files.\n"
        "This will trigger the deletion protection check (heuristics based on file count, timestamps, types).\n"
        "Whether it blocks depends on the config thresholds and calculated variances.",
        title="🛡️ Testing Deletion Protection 🛡️", border_style="yellow"
    ))
    # This call might raise ProtectionTriggeredError, which safe_tool_call will catch and display
    await safe_tool_call(delete_path, {"path": bulk_dir_path}, description=f"Deleting directory with {BULK_FILES_COUNT} files (bulk_files)")
    # Check if it was actually deleted or blocked by protection
    await safe_tool_call(get_file_info, {"path": bulk_dir_path}, description="Checking if bulk_files directory still exists after delete attempt")


async def demonstrate_security_features():
    """Demonstrate security features like allowed directories."""
    console.print(Rule("[bold cyan]5. Security Features[/bold cyan]", style="cyan"))
    logger.info("Demonstrating security features...", emoji_key="security")

    # --- List Allowed Directories ---
    # This reads from the config (which we set via env var for the demo)
    await safe_tool_call(list_allowed_directories, {}, description="Listing configured allowed directories")
    console.print(f"[dim]Note: For this demo, only the temporary directory [cyan]{DEMO_TEMP_DIR}[/cyan] was allowed via environment variable.[/dim]")

    # --- Try to Access Standard System Root (should fail) ---
    # Choose a path guaranteed outside the temp allowed dir
    outside_path_root = "/" if platform.system() != "Windows" else "C:\\"
    console.print(f"\nAttempting operation outside allowed directory: [red]Listing '{outside_path_root}'[/red]")
    await safe_tool_call(list_directory, {"path": outside_path_root}, description=f"Attempting to list root directory '{outside_path_root}' (should fail)")

    # --- Try to Access Specific Sensitive File (should fail) ---
    outside_path_file = "/etc/passwd" if platform.system() != "Windows" else "C:\\Windows\\System32\\drivers\\etc\\hosts"
    console.print(f"\nAttempting operation outside allowed directory: [red]Reading '{outside_path_file}'[/red]")
    await safe_tool_call(read_file, {"path": outside_path_file}, description=f"Attempting to read sensitive file '{outside_path_file}' (should fail)")

    # --- Try to use '..' to escape (should fail due to normalization) ---
    escape_path = str(DEMO_ROOT / ".." / "..") # Attempt to go above the allowed temp dir
    # Note: validate_path normalizes this, so it might resolve to something unexpected but still potentially outside
    # Or, more likely, the normalized path check against allowed dirs will fail.
    console.print(f"\nAttempting operation using '..' to potentially escape: [red]Listing '{escape_path}'[/red]")
    await safe_tool_call(list_directory, {"path": escape_path}, description=f"Attempting to list path using '..' ('{escape_path}')")

    console.print(Panel(
        "Security checks demonstrated:\n"
        "1. Operations are confined to the `allowed_directories`.\n"
        "2. Accessing paths outside these directories is denied.\n"
        "3. Path normalization prevents trivial directory traversal escapes (`..`).\n"
        "4. Symlink targets are also validated against `allowed_directories` (implicitly tested via symlink operations).\n"
        "5. Deletion protection provides a safety net against accidental bulk deletions (demonstrated earlier).",
        title="Security Summary", border_style="green", expand=False
    ))


async def main():
    """Run the filesystem operations demonstration."""
    global DEMO_TEMP_DIR # Make sure main knows about this path
    symlink_path = None
    exit_code = 0
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        console.print(Rule("[bold blue]Secure Filesystem Operations Demo[/bold blue]", style="white"))
        logger.info("Starting filesystem operations demonstration", emoji_key="start")

        # Display available options if running all demos
        if args.demo == 'all':
            console.print(Panel(
                "This demo includes multiple sections showcasing different filesystem operations.\n"
                "You can run individual sections using the following commands:\n\n"
                "[yellow]python filesystem_operations_demo.py read[/yellow] - File reading operations\n"
                "[yellow]python filesystem_operations_demo.py write[/yellow] - File writing and editing operations\n"
                "[yellow]python filesystem_operations_demo.py directory[/yellow] - Directory operations\n"
                "[yellow]python filesystem_operations_demo.py move_delete[/yellow] - Move, delete, search & info operations\n"
                "[yellow]python filesystem_operations_demo.py security[/yellow] - Security features demo\n\n"
                "Add [yellow]--rich-tree[/yellow] for enhanced directory visualization!",
                title="Demo Options", 
                border_style="cyan",
                expand=False
            ))

        # Display info message
        console.print(Panel(
            "This demo showcases the secure asynchronous filesystem tools.\n"
            f"A temporary directory ([cyan]{DEMO_TEMP_DIR}[/cyan]) has been created and automatically configured as the ONLY allowed directory for this demo's operations.",
            title="About This Demo",
            border_style="cyan"
        ))

        # Set up the demo environment *inside* the allowed temp dir
        symlink_path = await setup_demo_environment()
        
        # Run the selected demonstration(s)
        if args.demo == 'all' or args.demo == 'read':
            await demonstrate_file_reading(symlink_path)
            console.print()

        if args.demo == 'all' or args.demo == 'write':
            await demonstrate_file_writing_editing()
            console.print()

        if args.demo == 'all' or args.demo == 'directory':
            await demonstrate_directory_operations(symlink_path, use_rich_tree=args.rich_tree)
            console.print()

        if args.demo == 'all' or args.demo == 'move_delete':
            await demonstrate_move_delete_search(symlink_path)
            console.print()

        if args.demo == 'all' or args.demo == 'security':
            await demonstrate_security_features()

        logger.success(f"Filesystem Operations Demo(s) completed: {args.demo}", emoji_key="complete")
        console.print(Rule("[bold green]Demo Complete[/bold green]", style="green"))

    except Exception as e:
        logger.critical(f"Demo crashed unexpectedly: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"\n[bold red]CRITICAL ERROR:[/bold red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        exit_code = 1

    finally:
        # Clean up the demo environment
        console.print(Rule("Cleanup", style="dim"))
        await cleanup_demo_environment()

    return exit_code

if __name__ == "__main__":
    # Basic check for asyncio policy on Windows if needed
    # if sys.platform == "win32" and sys.version_info >= (3, 8):
    #     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run the demo
    final_exit_code = asyncio.run(main())
    sys.exit(final_exit_code)