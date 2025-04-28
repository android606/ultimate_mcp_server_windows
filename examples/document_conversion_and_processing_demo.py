#!/usr/bin/env python
"""
DETAILED Demonstration script for the ENHANCED DocumentProcessingTool in Ultimate MCP Server,
showcasing integrated OCR, analysis, conversion, and batch capabilities with extensive examples.
"""

import asyncio
import base64
import datetime as dt
import os
import sys
import traceback  # Added for more detailed error printing if needed
import warnings  # Added for warning control
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

# Filter Docling-related deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling_core")
warnings.filterwarnings("ignore", message="Could not parse formula with MathML")

# Add project root to path for imports when running as script
# Adjust this relative path if your script structure is different
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    print(f"INFO: Added {_PROJECT_ROOT} to sys.path")

# Rich imports for enhanced terminal UI
from rich import box, get_console  # noqa: E402
from rich.console import Group  # noqa: E402
from rich.layout import Layout  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    FileSizeColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.rule import Rule  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402
from rich.traceback import install as install_rich_traceback  # noqa: E402

# --- Global Constants ---
# Maximum number of lines to display for any content
MAX_DISPLAY_LINES = 50  # Used to truncate all displayed content

# --- Attempt to import required MCP Server components ---
try:
    # Assuming standard MCP Server structure
    from ultimate_mcp_server.core.server import Gateway
    from ultimate_mcp_server.exceptions import ToolError, ToolInputError

    # Assuming the combined tool is now in this location:
    from ultimate_mcp_server.tools.document_conversion_and_processing import DocumentProcessingTool
    from ultimate_mcp_server.utils import get_logger
    from ultimate_mcp_server.utils.display import CostTracker  # Import CostTracker

    MCP_COMPONENTS_LOADED = True
except ImportError as e:
    MCP_COMPONENTS_LOADED = False
    _IMPORT_ERROR_MSG = str(e)
    # We'll handle this error gracefully in the main function

# Initialize Rich console and logger
# Use get_console() to avoid potential issues in non-terminal environments
console = get_console()
logger = get_logger("demo.doc_proc_tool_detailed")  # Renamed logger

# Install rich tracebacks for better error display during development/debugging
install_rich_traceback(show_locals=True, width=console.width, extra_lines=2)  # Show more context

# --- Configuration ---
# Use a more robust way to find the examples directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Suggest placing samples in a dedicated subdirectory next to the script
DEFAULT_SAMPLE_DIR = SCRIPT_DIR / "sample"
# URLs for sample files (using stable, known content)
# Digital PDF: "Attention Is All You Need" paper
DEFAULT_SAMPLE_PDF_URL = "https://arxiv.org/pdf/1706.03762.pdf"
# Image: sample PNG image for OCR demo
DEFAULT_SAMPLE_IMAGE_URL = "https://raw.githubusercontent.com/IBM/MAX-OCR/refs/heads/master/samples/chap4_summary.png"
# HTML: Wikipedia page on Transformers (relevant to the PDF)
SAMPLE_HTML_URL = "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)"
DOWNLOADED_FILES_DIR = DEFAULT_SAMPLE_DIR / "downloaded"  # Subdirectory for downloads

# Configuration from environment variables
# Example: export USE_GPU=false export LOG_LEVEL=DEBUG
USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "3"))
# Default accelerator device - will be checked/updated later based on tool availability
ACCELERATOR_DEVICE = "cuda" if USE_GPU else "cpu"
# Allow skipping downloads for testing purposes
SKIP_DOWNLOADS = os.environ.get("SKIP_DOWNLOADS", "false").lower() == "true"
# Set log level from environment
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Define result types for type hints
ResultData = Dict[str, Any]
OperationResult = Tuple[bool, ResultData]  # (success_flag, result_dict)
FileResult = Optional[Path]  # Path if successful download/find, None otherwise

# --- Demo Helper Functions (Expanded and Verbose) ---


def create_demo_layout() -> Layout:
    """Create a Rich layout for the demo UI (can be enhanced later)."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=5),  # Slightly larger header
        Layout(name="body", ratio=1),
        Layout(name="footer", size=1),
    )
    layout["footer"].update("[dim]Document Processing Tool Demo Footer[/]")
    return layout


def timestamp_str(short: bool = False) -> str:
    """Return a formatted timestamp string."""
    now = dt.datetime.now()
    if short:
        return f"[dim]{now.strftime('%H:%M:%S')}[/]"
    return f"[dim]{now.strftime('%Y-%m-%d %H:%M:%S')}[/]"


def truncate_text_by_lines(text: str, max_lines: int = 300) -> str:
    """
    Truncates text to show first half of max_lines, an indicator line, then last half of max_lines.
    
    Args:
        text: The input text to truncate
        max_lines: Maximum number of lines to show (excluding the indicator line)
        
    Returns:
        Truncated text with "[...TRUNCATED...]" indicator in the middle
    """
    if not text:
        return ""
        
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text  # No truncation needed
        
    half_lines = max_lines // 2
    return "\n".join(lines[:half_lines] + ["[...TRUNCATED...]"] + lines[-half_lines:])


def format_value_for_display(key: str, value: Any, detail_level: int = 1) -> Any:
    """
    Format specific values for better display in tables/panels.
    detail_level: 0=minimal, 1=standard, 2=verbose
    """
    if value is None:
        return "[dim]None[/]"
    if isinstance(value, bool):
        return "[green]Yes[/]" if value else "[red]No[/]"
    if isinstance(value, float):
        return f"{value:.3f}"
    if key.lower().endswith("time"):
        return f"[green]{float(value):.3f}s[/]"

    if isinstance(value, list):
        if not value:
            return "[dim]Empty List[/]"
        list_len = len(value)
        preview_count = 3 if detail_level < 2 else 5
        suffix = f" [dim]... ({list_len} items total)[/]" if list_len > preview_count else ""
        # Show more detail for lists if requested
        if detail_level >= 1:
            previews = []
            for item in value[:preview_count]:
                if isinstance(item, dict):  # Special preview for dicts in list
                    item_keys = list(item.keys())[:3]
                    item_suffix = f", ...({len(item)} keys)" if len(item) > 3 else ""
                    previews.append(f"{{{', '.join(map(repr, item_keys))}{item_suffix}}}")
                elif isinstance(item, str):
                    # Truncate long strings in list previews
                    item_preview = escape(truncate_text_by_lines(item, 5)[:60]) + ("..." if len(item) > 60 else "")
                    previews.append(f"'{item_preview}'")
                else:
                    previews.append(repr(item))
            return f"[{', '.join(previews)}]{suffix}"
        else:  # Minimal detail
            return f"[List with {list_len} items]"

    if isinstance(value, dict):
        if not value:
            return "[dim]Empty Dict[/]"
        dict_len = len(value)
        preview_count = 4 if detail_level < 2 else 8
        preview_keys = list(value.keys())[:preview_count]
        suffix = f" [dim]... ({dict_len} keys total)[/]" if dict_len > preview_count else ""
        # Show more detail for dicts if requested
        if detail_level >= 1:
            items_preview = []
            for k in preview_keys:
                v = value[k]
                v_preview = format_value_for_display(
                    k, v, detail_level=0
                )  # Minimal preview for nested values
                items_preview.append(f"{repr(k)}: {v_preview}")
            return f"{{{'; '.join(items_preview)}}}{suffix}"
        else:  # Minimal detail
            return f"[Dict with {dict_len} keys]"

    if isinstance(value, str):
        str_len = len(value)
        preview_len = 300 if detail_level < 2 else 600
        
        # Ensure string values are truncated by line count first
        value = truncate_text_by_lines(value, MAX_DISPLAY_LINES // 10)  # Use a smaller limit for inline display
        
        # Then apply character limit
        if len(value) > preview_len:
            return escape(value[:preview_len]) + f"[dim]... (truncated, {str_len} chars total)[/]"
        return escape(value)  # Escape non-truncated strings too

    # Default for other types
    return escape(str(value))


def display_result(title: str, result: ResultData, display_options: Optional[Dict] = None) -> None:
    """
    Display operation result with enhanced formatting using Rich.
    Includes more verbose output and better handling of nested structures.
    """
    display_options = display_options or {}
    start_time = dt.datetime.now()
    
    # Check if title is already a Text object
    if isinstance(title, Text):
        title_display = title
    else:
        title_display = escape(title)
    
    console.print(Rule(f"[bold cyan]{title_display}[/] {timestamp_str()}", style="cyan"))

    success = result.get("success", False)
    detail_level = display_options.get("detail_level", 1)  # 0, 1, 2
    hide_keys_set = set(
        display_options.get("hide_keys", ["success", "raw_llm_response", "raw_text"])
    )
    display_keys = display_options.get("display_keys")  # If None, show all non-hidden

    # --- Summary Panel ---
    summary_panel_content = Text()
    summary_panel_content.append(
        Text.from_markup(f"Status: {'[bold green]Success[/]' if success else '[bold red]Failed[/]'}\n") # Use from_markup here too for consistency
    )
    if not success:
        error_code = result.get("error_code", "N/A")
        error_msg = result.get("error", "Unknown error")
        # These appends are simple strings, probably fine, but from_markup is safer:
        summary_panel_content.append(Text.from_markup(f"Error Code: [yellow]{escape(error_code)}[/]\n"))
        summary_panel_content.append(Text.from_markup(f"Message: [red]{escape(error_msg)}[/]\n"))
        console.print(
            Panel(
                summary_panel_content, title="Operation Status", border_style="red", padding=(1, 2)
            )
        )
        return # Stop display if failed

    top_level_info = {
        "processing_time": "Processing Time",
        "extraction_strategy_used": "Strategy Used",
        "output_format": "Output Format",
        "was_html": "Input Detected as HTML",
    }
    for key, display_name in top_level_info.items():
        if key in result and key not in hide_keys_set:
            value_str = format_value_for_display(key, result[key], detail_level=0)
            summary_panel_content.append(Text.from_markup(f"{display_name}: [blue]{value_str}[/]\n"))

    console.print(
        Panel(
            summary_panel_content, title="Operation Summary", border_style="green", padding=(1, 2)
        )
    )

    # --- Details Section ---
    details_to_display = {}
    for key, value in result.items():
        if key in hide_keys_set or key in top_level_info:
            continue
        if display_keys and key not in display_keys:
            continue
        details_to_display[key] = value

    if not details_to_display:
        console.print(Text.from_markup("[dim]No further details requested or available.[/]"))
        console.print()
        return

    console.print(Rule("Details", style="dim"))

    for key, value in details_to_display.items():
        key_title = key.replace("_", " ").title()
        panel_border = "blue"
        panel_content: Any = None

        # --- Specific Key Formatting ---
        is_content_key = key.lower() in [
            "content",
            "markdown_text",
            "optimized_markdown",
            "text",
            "summary",
            "first_table_preview",
        ]
        format_type = display_options.get("format_key", {}).get(
            key, "markdown" if "markdown" in key else "text"
        )

        if is_content_key and isinstance(value, str):
            if not value:
                panel_content = "[dim]Empty Content[/]"
            else:
                # Always enforce the global max line limit for any content display
                truncated_value = truncate_text_by_lines(value, MAX_DISPLAY_LINES)
                if format_type == "markdown":
                    panel_content = Markdown(truncated_value)
                else:
                    panel_content = Syntax(
                        truncated_value,
                        format_type,
                        theme="paraiso-dark",
                        line_numbers=False,
                        word_wrap=True,
                        background_color="default",
                    )  # Use syntax for text too
            panel_border = "green" if format_type == "markdown" else "white"
            console.print(
                Panel(
                    panel_content,
                    title=key_title,
                    border_style=panel_border,
                    padding=(1, 2),
                    expand=False,
                )
            )

        elif key.lower() == "chunks" and isinstance(value, list):
            chunk_table = Table(
                title=f"Chunk Preview (Total: {len(value)})", box=box.MINIMAL, show_header=True
            )
            chunk_table.add_column("#", style="cyan")
            chunk_table.add_column("Preview (First 80 chars)", style="white")
            chunk_table.add_column("Length", style="green")
            limit = 5 if detail_level < 2 else 10
            for i, chunk in enumerate(value[:limit], 1):
                # Truncate each chunk preview if needed
                chunk_str = truncate_text_by_lines(str(chunk), MAX_DISPLAY_LINES // 10)  # Smaller limit for previews
                chunk_table.add_row(str(i), escape(chunk_str[:80]) + "...", str(len(str(chunk))))
            if len(value) > limit:
                chunk_table.add_row("...", f"[dim]{len(value) - limit} more...[/]", "")
            console.print(Panel(chunk_table, title=key_title, border_style="blue"))

        elif key.lower() == "qa_pairs" and isinstance(value, list):
            qa_text = Text()
            limit = 3 if detail_level < 2 else 5
            for i, qa in enumerate(value[:limit], 1):
                qa_text.append(f"{i}. Q: ", style="bold cyan")
                qa_text.append(escape(truncate_text_by_lines(qa.get("question", ""), MAX_DISPLAY_LINES // 10)) + "\n")
                qa_text.append("   A: ", style="green")
                qa_text.append(escape(truncate_text_by_lines(qa.get("answer", ""), MAX_DISPLAY_LINES // 10)) + "\n\n")
            if len(value) > limit:
                qa_text.append(f"[dim]... {len(value) - limit} more Q&A pairs ...[/]")
            console.print(Panel(qa_text, title=key_title, border_style="blue"))

        elif isinstance(value, dict):  # General Dict Handling (for metadata, metrics, risks, etc.)
            dict_table = Table(title="Contents", box=box.MINIMAL, show_header=False, expand=False)
            dict_table.add_column("SubKey", style="magenta", justify="right", no_wrap=True)
            dict_table.add_column("SubValue", style="white")
            item_count = 0
            for k, v in value.items():
                dict_table.add_row(
                    escape(str(k)), format_value_for_display(k, v, detail_level=detail_level)
                )
                item_count += 1
                if detail_level == 0 and item_count >= 5:  # Limit display in low detail
                    dict_table.add_row("[dim]...[/]", f"[dim]({len(value)} total items)[/]")
                    break
            panel_content = dict_table
            panel_border = (
                "magenta" if "quality" in key.lower() or "metrics" in key.lower() else "blue"
            )
            console.print(
                Panel(panel_content, title=key_title, border_style=panel_border, padding=(1, 1))
            )

        elif isinstance(value, list):  # General List Handling
            list_panel_content = []
            limit = 5 if detail_level < 2 else 10
            list_panel_content.append(Text.from_markup(f"[cyan]Total Items:[/] {len(value)}"))
            for i, item in enumerate(value[:limit]):
                # Truncate long items if they're strings
                item_display = item
                if isinstance(item, str):
                    item_display = truncate_text_by_lines(item, MAX_DISPLAY_LINES // 10)
                list_panel_content.append(
                    f"[magenta]{i + 1}.[/] {format_value_for_display(f'{key}[{i}]', item_display, detail_level=detail_level - 1)}"
                )
            if len(value) > limit:
                list_panel_content.append(Text.from_markup(f"[dim]... {len(value) - limit} more items ...[/]"))
            console.print(Panel(Group(*list_panel_content), title=key_title, border_style="blue"))

        else:  # Fallback for simple types not handled above
            # For any string value, ensure it's truncated
            if isinstance(value, str):
                value = truncate_text_by_lines(value, MAX_DISPLAY_LINES)
            console.print(
                f"[bold cyan]{key_title}:[/] {format_value_for_display(key, value, detail_level=detail_level)}"
            )

    end_time = dt.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    console.print(Text.from_markup(f"[dim]Result details displayed in {elapsed:.3f}s[/]"))
    console.print()  # Add spacing


async def download_file_with_progress(url: str, output_path: Path, description: str, progress: Optional[Progress] = None) -> FileResult:
    """Download a file with a detailed progress bar."""
    if output_path.exists() and output_path.stat().st_size > 1000:  # Check size > 1KB
        logger.info(f"Using existing file: {output_path}")
        console.print(Text.from_markup(f"[dim]Using existing file: [blue underline]{output_path.name}[/][/]"))
        return output_path
    if SKIP_DOWNLOADS:
        console.print(
            f"[yellow]Skipping download for {description} due to SKIP_DOWNLOADS setting.[/]"
        )
        return None

    console.print(f"Attempting to download [bold]{description}[/] from [underline]{url}[/]...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=60.0
        ) as client:  # Longer timeout
            async with client.stream("GET", url) as response:
                if response.status_code == 404:
                    logger.error(f"File not found (404) at {url}")
                    console.print(f"[red]Error: File not found (404) for {description}.[/]")
                    return None
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                task_description = f"Downloading {description}..."

                # Check if we're using an external Progress object
                using_external_progress = progress is not None
                
                # Create a local progress object if none was provided
                if not using_external_progress:
                    progress = Progress(
                        TextColumn("[bold blue]{task.description}", justify="right"),
                        BarColumn(bar_width=None),
                        "[progress.percentage]{task.percentage:>3.1f}%",
                        "•",
                        TransferSpeedColumn(),
                        "•",
                        FileSizeColumn(),  # Shows completed/total size
                        "•",
                        TimeRemainingColumn(),
                        console=console,
                    )
                
                try:
                    # Only start the progress if it's our local one
                    if not using_external_progress:
                        progress.start()
                    
                    download_task = progress.add_task(task_description, total=total_size)
                    bytes_downloaded = 0
                    with open(output_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            bytes_written = len(chunk)
                            bytes_downloaded += bytes_written
                            progress.update(download_task, advance=bytes_written)
                    # Ensure completion if size was unknown or inaccurate
                    progress.update(
                        download_task,
                        completed=max(bytes_downloaded, total_size),
                        description=f"Downloaded {description}",
                    )
                finally:
                    # Only stop the progress if it's our local one
                    if not using_external_progress:
                        progress.stop()

        logger.info(f"Successfully downloaded {description} to {output_path}")
        console.print(
            Text.from_markup(f"[green]✓ Downloaded {description} to [blue underline]{output_path.name}[/][/]")
        )
        return output_path
    except httpx.RequestError as e:
        logger.error(f"Network error downloading {description} from {url}: {e}")
        console.print(
            Text.from_markup(f"[red]Network Error downloading {description}: {type(e).__name__}. Check connection or URL.[/]")
        )
        return None
    except Exception as e:
        logger.error(f"Failed to download {description} from {url}: {e}", exc_info=True)
        console.print(
            Text.from_markup(f"[red]Error downloading {description}: {type(e).__name__} - {e}[/]")
        )
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        return None


async def safe_tool_call(
    operation_name: str, tool_func: callable, *args, tracker: Optional[CostTracker] = None, **kwargs
) -> OperationResult:
    """Safely call a tool function, handling exceptions and logging."""
    console.print(
        Text.from_markup(f"\n[cyan]Calling Tool:[/][bold] {escape(operation_name)}[/] {timestamp_str(short=True)}")
    )
    display_options = kwargs.pop("display_options", {})  # Extract display options

    # Log arguments carefully, avoiding excessive length
    log_args_repr = {}
    MAX_ARG_LEN = 100  # Max length for logging arg values
    for k, v in kwargs.items():
        if isinstance(v, (str, bytes)) and len(v) > MAX_ARG_LEN:
            log_args_repr[k] = f"{type(v).__name__}(len={len(v)})"
        elif isinstance(v, (list, dict)) and len(v) > 10:  # Abbreviate large lists/dicts
            log_args_repr[k] = f"{type(v).__name__}(len={len(v)})"
        else:
            log_args_repr[k] = repr(v)  # Use repr for others
    logger.debug(f"Executing {operation_name} with kwargs: {log_args_repr}")

    try:
        result = await tool_func(*args, **kwargs)

        if not isinstance(result, dict):
            logger.error(
                Text.from_markup(f"Tool '{operation_name}' returned non-dict type: {type(result)}. Value: {str(result)[:150]}")
            )
            return False, {
                "success": False,
                "error": Text.from_markup(f"Tool returned unexpected type: {type(result).__name__}"),
                "error_code": "INTERNAL_ERROR",
                "_display_options": display_options,
            }

        # Track cost if tracker is provided and result contains cost information
        if tracker is not None and result.get("success", False):
            # Check for various cost-related fields that might be present
            if hasattr(result, 'cost') or 'cost' in result:
                tracker.add_call(result)
            elif 'llm_cost' in result:
                # Create a compatible object for cost tracking
                from collections import namedtuple
                TrackableResult = namedtuple("TrackableResult", ["cost", "input_tokens", "output_tokens", "provider", "model", "processing_time"])
                trackable = TrackableResult(
                    cost=result.get("llm_cost", 0.0),
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                    provider=result.get("provider", "unknown"),
                    model=result.get("model", "document_processing"),
                    processing_time=result.get("processing_time", 0.0)
                )
                tracker.add_call(trackable)

        result["_display_options"] = display_options  # Pass options for display func
        logger.debug(f"Tool '{operation_name}' completed successfully.")
        return True, result
    except ToolInputError as e:
        logger.warning(f"Input error for {operation_name}: {e}")  # Warning for input errors
        return False, {
            "success": False,
            "error": str(e),
            "error_code": e.error_code,  # Use error_code, not code
            "_display_options": display_options,
        }
    except ToolError as e:
        logger.error(f"Tool error during {operation_name}: {e}")
        return False, {
            "success": False,
            "error": str(e),
            "error_code": e.error_code,  # Use error_code, not code
            "_display_options": display_options,
        }
    except Exception as e:
        logger.error(f"Unexpected error during {operation_name}: {e}", exc_info=True)
        # Include traceback info in the error message for unexpected errors
        tb_str = traceback.format_exc(limit=1)  # Get brief traceback
        return False, {
            "success": False,
            "error": f"{type(e).__name__}: {e}\n{tb_str}",
            "error_type": type(e).__name__,
            "error_code": "UNEXPECTED_ERROR",
            "_display_options": display_options,
        }


# --- Demo Sections (Expanded) ---


async def demo_section_1_conversion_ocr(
    doc_tool: DocumentProcessingTool, sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate convert_document with various strategies and OCR, showing more permutations."""
    console.print(Rule("[bold green]Demo 1: Document Conversion & OCR[/]", style="green"))
    logger.info("Starting Demo Section 1: Conversion & OCR")

    pdf_digital = sample_files.get("pdf_digital")
    image_file = sample_files.get("image")
    buffett_pdf = sample_files.get("buffett_pdf") 
    backprop_pdf = sample_files.get("backprop_pdf")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")
    
    pdf_files_to_process = [pdf for pdf in [pdf_digital, buffett_pdf, backprop_pdf] if pdf is not None]
    
    if not pdf_files_to_process:
        console.print("[yellow]Skipping Demo 1: Need at least one sample PDF or Image.[/]")
        return

    # Function to generate output path
    def get_output_path(input_file: Path, format_name: str, strategy: str, output_format: str) -> str:
        """Generate standardized output path for conversions"""
        base_name = input_file.stem
        return str(conversion_outputs_dir / f"{base_name}_{strategy}_{format_name}.{output_format}")

    # --- Digital PDF ---
    pdf_files_to_process = [pdf for pdf in [pdf_digital, buffett_pdf, backprop_pdf] if pdf is not None]
    
    for pdf_file in pdf_files_to_process:
        console.print(
            Panel(Text.from_markup(f"Processing PDF: [cyan]{pdf_file.name}[/]"), border_style="blue")
        )
        console.print("Demonstrating conversion strategies suitable for PDFs...")

        # 1a: Direct Text Strategy (Raw Text Output)
        output_path = get_output_path(pdf_file, "direct", "text", "txt")
        success, result = await safe_tool_call(
            f"{pdf_file.name} -> Text (Direct Text)",
            doc_tool.convert_document,
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="text",
            extraction_strategy="direct_text",
            enhance_with_llm=False,
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{pdf_file.name} -> Text (Direct Text)",
                result,
                display_options={"format_key": {"content": "text"}},
            )

        # 1b: Direct Text Strategy (Markdown Output, requires enhance=True)
        output_path = get_output_path(pdf_file, "direct", "enhanced_md", "md")
        success, result = await safe_tool_call(
            f"{pdf_file.name} -> MD (Direct Text + Enhance)",
            doc_tool.convert_document,
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="markdown",
            extraction_strategy="direct_text",
            enhance_with_llm=True,  # Enhance needed to get MD
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{pdf_file.name} -> MD (Direct Text + Enhance)",
                result,
                display_options={"format_key": {"content": "markdown"}},
            )

        # 1c: Docling Strategy (Markdown Output, Layout Aware)
        if doc_tool._docling_available:
            output_path = get_output_path(pdf_file, "docling", "md", "md")
            success, result = await safe_tool_call(
                f"{pdf_file.name} -> MD (Docling)",
                doc_tool.convert_document,
                tracker=tracker,
                document_path=str(pdf_file),
                output_format="markdown",
                extraction_strategy="docling",
                accelerator_device=ACCELERATOR_DEVICE,
                save_to_file=True,
                output_path=output_path,
            )
            if success:
                display_result(
                    f"{pdf_file.name} -> MD (Docling)",
                    result,
                    display_options={"format_key": {"content": "markdown"}},
                )

            # 1d: Docling Strategy (HTML Output)
            output_path = get_output_path(pdf_file, "docling", "html", "html")
            success, result = await safe_tool_call(
                f"{pdf_file.name} -> HTML (Docling)",
                doc_tool.convert_document,
                tracker=tracker,
                document_path=str(pdf_file),
                output_format="html",
                extraction_strategy="docling",
                save_to_file=True,
                output_path=output_path,
            )
            if success:
                display_result(
                    f"{pdf_file.name} -> HTML (Docling)",
                    result,
                    display_options={"format_key": {"content": "html"}},
                )
        else:
            console.print("[yellow]Docling unavailable, skipping Docling conversions.[/]")

    # --- OCR on PDF (Using Digital PDF as input for demonstration) ---
    for pdf_file in pdf_files_to_process:
        console.print(
            Panel(
                f"Processing PDF with OCR Strategy: [cyan]{pdf_file.name}[/]",
                border_style="blue",
            )
        )
        console.print("Demonstrating OCR strategies (may be slow)...")

        # 1e: OCR Strategy (Raw Text, No Enhance)
        output_path = get_output_path(pdf_file, "ocr", "raw_text", "txt")
        success, result = await safe_tool_call(
            f"{pdf_file.name} -> Text (OCR Raw)",
            doc_tool.convert_document,
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="text",
            extraction_strategy="ocr",
            enhance_with_llm=False,
            ocr_options={"language": "eng", "dpi": 150},  # Low DPI for speed
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{pdf_file.name} -> Text (OCR Raw)",
                result,
                display_options={"format_key": {"content": "text"}, "detail_level": 0},
            )  # Less detail for raw

        # 1f: OCR Strategy (Markdown, Enhanced, Quality Assess)
        output_path = get_output_path(pdf_file, "ocr", "enhanced_md", "md")
        success, result = await safe_tool_call(
            f"{pdf_file.name} -> MD (OCR + Enhance + Quality)",
            doc_tool.convert_document,
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="markdown",
            extraction_strategy="ocr",
            enhance_with_llm=True,
            ocr_options={
                "language": "eng",
                "assess_quality": True,
                "remove_headers": False,
                "dpi": 200,
            },
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{pdf_file.name} -> MD (OCR + Enhance + Quality)",
                result,
                display_options={"format_key": {"content": "markdown"}},
            )

        # 1g: Hybrid Strategy (Default - should use 'direct' for digital, 'ocr' for scanned)
        console.print("Testing Hybrid Strategy (Behavior depends on input PDF type)...")
        output_path = get_output_path(pdf_file, "hybrid", "text", "txt")
        success, result = await safe_tool_call(
            f"{pdf_file.name} -> Text (Hybrid Strategy)",
            doc_tool.convert_document,
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="text",
            extraction_strategy="hybrid_direct_ocr",
            enhance_with_llm=True,  # Enable enhancement if OCR path taken
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{pdf_file.name} -> Text (Hybrid + Enhance)",
                result,
                display_options={"format_key": {"content": "text"}},
            )

    # --- Image Conversion (Using convert_document) ---
    if image_file:
        console.print(
            Panel(
                f"Processing Image via convert_document: [cyan]{image_file.name}[/]",
                border_style="blue",
            )
        )
        # 1h: Convert Image to Markdown (Enhancement enabled by default)
        output_path = get_output_path(image_file, "convert_doc", "md", "md")
        success, result = await safe_tool_call(
            "Image -> MD (Convert Doc)",
            doc_tool.convert_document,
            tracker=tracker,
            document_path=str(image_file),
            output_format="markdown",
            # strategy="ocr" # Inferred for images
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                "Image -> MD (via convert_document)",
                result,
                display_options={"format_key": {"content": "markdown"}},
            )

    # --- Conversion from Bytes ---
    if pdf_digital:
        console.print(Panel("Processing PDF from Bytes Data using OCR", border_style="blue"))
        try:
            pdf_bytes = pdf_digital.read_bytes()
            output_path = get_output_path(pdf_digital, "bytes", "ocr_text", "txt")
            success, result = await safe_tool_call(
                "PDF Bytes -> Text (OCR)",
                doc_tool.convert_document,
                tracker=tracker,
                document_data=pdf_bytes,
                output_format="text",
                extraction_strategy="ocr",
                enhance_with_llm=False,  # Get raw OCR from bytes
                ocr_options={"dpi": 150},
                save_to_file=True,
                output_path=output_path,
            )
            if success:
                display_result(
                    "PDF Bytes -> Text (OCR Raw)",
                    result,
                    display_options={"format_key": {"content": "text"}, "detail_level": 0},
                )
        except Exception as e:
            console.print(f"[red]Error processing PDF bytes with OCR: {e}[/]")


async def demo_section_2_dedicated_ocr(
    doc_tool: DocumentProcessingTool, sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate the dedicated ocr_image tool with more options."""
    console.print(Rule(Text.from_markup("[bold green]Demo 2: Dedicated Image OCR Tool[/]", style="green")))
    logger.info("Starting Demo Section 2: Dedicated Image OCR Tool")

    image_file = sample_files.get("image")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")
    
    if not image_file:
        console.print(Text.from_markup("[yellow]Skipping Demo 2: Sample image not available.[/]"))
        return

    # Function to generate output path
    def get_output_path(base_name: str, method: str, output_format: str) -> str:
        """Generate standardized output path for OCR outputs"""
        return str(conversion_outputs_dir / f"{base_name}_ocr_{method}.{output_format}")

    console.print(
        Panel(
            Text.from_markup(f"Processing Image with ocr_image Tool: [cyan]{image_file.name}[/]"), border_style="blue"
        )
    )

    # 2a: OCR Image from Path (Default: Enhance=True, Output=Markdown)
    output_path = get_output_path(image_file.stem, "default", "md")
    success, result = await safe_tool_call(
        "OCR Image (Path, Defaults)",
        doc_tool.ocr_image,
        tracker=tracker,
        image_path=str(image_file),
        # Uses default ocr_options, enhance=True, output_format="markdown"
    )
    if success:
        # Save the content to file since ocr_image doesn't have save_to_file parameter
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.get("content", ""))
            console.print(Text.from_markup(f"[green]✓ Saved OCR output to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving OCR output: {e}[/]"))
            
        display_result(
            "OCR Image (Path, Defaults)",
            result,
            display_options={"format_key": {"content": "markdown"}},
        )

    # 2b: OCR Image from Path (Raw Text Output, Specific Preprocessing)
    output_path = get_output_path(image_file.stem, "raw_preprocessing", "txt")
    success, result = await safe_tool_call(
        "OCR Image (Path, Raw Text, Preprocessing)",
        doc_tool.ocr_image,
        tracker=tracker,
        image_path=str(image_file),
        output_format="text",
        enhance_with_llm=False,
        ocr_options={
            "language": "eng",
            "preprocessing": {
                "threshold": "adaptive",
                "denoise": True,
                "deskew": False,
            },  # Custom preprocessing
        },
    )
    if success:
        # Save the content to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.get("content", ""))
            console.print(Text.from_markup(f"[green]✓ Saved OCR output to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving OCR output: {e}[/]"))
            
        display_result(
            "OCR Image (Path, Raw Text, Preprocessing)",
            result,
            display_options={"format_key": {"content": "text"}},
        )
        # Optionally show the raw text panel again if different from previous
        if result.get("content"):
            console.print(
                Panel(
                    escape(result["content"][:1000]) + "...",
                    title="Raw OCR Text Output (After Preprocessing)",
                )
            )

    # 2c: OCR Image from Base64 Data (Enhance=True, Quality Assess)
    try:
        console.print(Panel("Processing Image from Base64 Data", border_style="blue"))
        img_bytes = image_file.read_bytes()
        # Ensure correct padding for base64
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        # console.print(f"[dim]Base64 Preview (first 60 chars): {img_base64[:60]}...[/]")

        output_path = get_output_path(image_file.stem, "base64_enhanced", "md")
        success, result = await safe_tool_call(
            "OCR Image (Base64, Enhance, Quality)",
            doc_tool.ocr_image,
            tracker=tracker,
            image_data=img_base64,
            output_format="markdown",
            enhance_with_llm=True,
            ocr_options={"assess_quality": True},  # Assess quality
        )
        if success:
            # Save the content to file
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.get("content", ""))
                console.print(Text.from_markup(f"[green]✓ Saved OCR output to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving OCR output: {e}[/]"))
                
            display_result(
                "OCR Image (Base64, Enhance, Quality)",
                result,
                display_options={"format_key": {"content": "markdown"}},
            )

    except Exception as e:
        console.print(Text.from_markup(f"[red]Failed to process image from Base64: {type(e).__name__} - {e}[/]"))



async def demo_section_3_enhance_text(doc_tool: DocumentProcessingTool, sample_files: Dict[str, Path], tracker: CostTracker) -> None:
    """Demonstrate enhancing existing noisy text with various options."""
    console.print(Rule("[bold green]Demo 3: Enhance Existing OCR Text[/]", style="green"))
    logger.info("Starting Demo Section 3: Enhance OCR Text")
    
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")
    
    # More complex noisy text example
    noisy_text = """
    INVOlCE # 12345 - ACME C0rp.
    Date: Octobor 25, 2O23

    Billed To: Example Inc. , 123 Main St . Anytown USA

    Itemm Descriptiom                 Quantlty    Unlt Price    Tota1
    -----------------------------------------------------------------
    Wldget Modell A                     lO          $ I5.0O      $l5O.OO
    Gadgett Type B                      5           $ 25.5O      $l27.5O
    Assembly Srvlce                   2 hrs       $ 75.OO      $l5O.OO
    -----------------------------------------------------------------
                                        Subtota1 :             $427.5O
                                        Tax (8%) :             $ 34.2O
                                        TOTAL    :             $461.7O

    Notes: Payment due ln 3O days. Thank you for yuor buslness!

    Page I / l - Confidential Document"""
    console.print(Panel("Original Noisy Text:", border_style="yellow"))
    # Apply truncation to noisy_text before displaying
    truncated_noisy_text = truncate_text_by_lines(noisy_text, 300)
    console.print(Syntax(truncated_noisy_text, "text", theme="default", line_numbers=True))

    # Function to generate output path
    def get_output_path(base_name: str, format_name: str) -> str:
        """Generate standardized output path for enhancement outputs"""
        return str(conversion_outputs_dir / f"{base_name}.{format_name}")

    # Save the noisy text for reference
    noisy_text_path = get_output_path("sample_noisy_text", "txt")
    try:
        with open(noisy_text_path, 'w', encoding='utf-8') as f:
            f.write(truncated_noisy_text)
        console.print(Text.from_markup(f"[green]✓ Saved noisy text to: [blue underline]{noisy_text_path}[/][/]"))
    except Exception as e:
        console.print(Text.from_markup(f"[red]Error saving noisy text: {e}[/]"))

    # 3a: Enhance to Markdown (Remove Headers, Assess Quality)
    output_path = get_output_path("enhanced_noisy_text_markdown", "md")
    success, result = await safe_tool_call(
        "Enhance -> MD (Rm Headers, Quality)",
        doc_tool.enhance_ocr_text,
        tracker=tracker,
        text=truncated_noisy_text,
        output_format="markdown",
        enhancement_options={"remove_headers": True, "assess_quality": True, "detect_tables": True},
    )
    if success:
        # Save the enhanced content
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.get("content", ""))
            console.print(Text.from_markup(f"[green]✓ Saved enhanced markdown to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving enhanced markdown: {e}[/]"))
            
        display_result(
            "Enhance -> Markdown (Remove Headers, Assess Quality)",
            result,
            display_options={"format_key": {"content": "markdown"}},
        )

    # 3b: Enhance to Plain Text (Keep Headers, Clean Only)
    output_path = get_output_path("enhanced_noisy_text_plain", "txt")
    success, result = await safe_tool_call(
        "Enhance -> Text (Keep Headers)",
        doc_tool.enhance_ocr_text,
        tracker=tracker,
        text=truncated_noisy_text,
        output_format="text",
        enhancement_options={"remove_headers": False, "clean_only": True},
    )
    if success:
        # Save the enhanced content
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.get("content", ""))
            console.print(Text.from_markup(f"[green]✓ Saved enhanced text to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving enhanced text: {e}[/]"))
            
        display_result(
            "Enhance -> Text (Keep Headers, Clean Only)",
            result,
            display_options={"format_key": {"content": "text"}},
        )

    # 3c: Enhance to Markdown with Table Detection
    output_path = get_output_path("enhanced_noisy_text_tables", "md")
    success, result = await safe_tool_call(
        "Enhance -> MD (Table Detection)",
        doc_tool.enhance_ocr_text,
        tracker=tracker,
        text=truncated_noisy_text,
        output_format="markdown",
        enhancement_options={"detect_tables": True, "table_markdown": True},
    )
    if success:
        # Save the enhanced content
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.get("content", ""))
            console.print(Text.from_markup(f"[green]✓ Saved enhanced markdown with tables to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving enhanced markdown with tables: {e}[/]"))
            
        display_result(
            "Enhance -> Markdown (Table Detection)",
            result,
            display_options={"format_key": {"content": "markdown"}},
        )

    # 3d: Enhance with Custom LLM Prompt
    output_path = get_output_path("enhanced_noisy_text_custom", "md")
    success, result = await safe_tool_call(
        "Enhance -> MD (Custom LLM Prompt)",
        doc_tool.enhance_ocr_text,
        tracker=tracker,
        text=truncated_noisy_text,
        output_format="markdown",
        enhancement_options={
            "custom_llm_prompt": "This is an invoice from ACME Corp. Please convert to clean, well-formatted markdown, preserving the table structure and making sure all numbers are correct."
        },
    )
    if success:
        # Save the enhanced content
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.get("content", ""))
            console.print(Text.from_markup(f"[green]✓ Saved enhanced markdown with custom prompt to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving enhanced markdown with custom prompt: {e}[/]"))
            
        display_result(
            "Enhance -> Markdown (Custom LLM Prompt)",
            result,
            display_options={"format_key": {"content": "markdown"}},
        )


async def demo_section_4_html_markdown(
    doc_tool: DocumentProcessingTool, sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate HTML processing and Markdown utilities extensively."""
    console.print(Rule(Text.from_markup("[bold green]Demo 4: HTML & Markdown Processing[/]", style="green")))
    logger.info("Starting Demo Section 4: HTML & Markdown Processing")

    html_file = sample_files.get("html")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")
    
    if not html_file:
        console.print(Text.from_markup("[yellow]Skipping Demo 4: Sample HTML not downloaded/available.[/]"))
        return

    # Function to generate output path
    def get_output_path(base_name: str, method: str, format_name: str) -> str:
        """Generate standardized output path for HTML/MD conversions"""
        return str(conversion_outputs_dir / f"{base_name}_{method}.{format_name}")

    console.print(Panel(Text.from_markup(f"Processing HTML File: [cyan]{html_file.name}[/]"), border_style="blue"))
    try:
        html_content = html_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        console.print(Text.from_markup(f"[red]Error reading HTML file {html_file}: {e}[/]"))
        return

    # --- clean_and_format_text_as_markdown ---
    console.print(Rule(Text.from_markup("HTML to Markdown Conversion", style="dim")))

    # 4a: Auto Extraction (Default)
    output_path = get_output_path(html_file.stem, "auto_extract", "md")
    success, result_auto = await safe_tool_call(
        "HTML -> MD (Auto Extract)",
        doc_tool.clean_and_format_text_as_markdown,
        tracker=tracker,
        text=html_content,
        extraction_method="auto",
        preserve_tables=True,
    )
    if success:
        # Save the markdown content
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_auto.get("markdown_text", ""))
            console.print(Text.from_markup(f"[green]✓ Saved auto-extracted markdown to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving markdown: {e}[/]"))
            
        display_result(
            "HTML -> MD (Auto Extract)",
            result_auto,
            display_options={"format_key": {"markdown_text": "markdown"}},
        )

    # 4b: Readability Extraction
    output_path = get_output_path(html_file.stem, "readability", "md")
    success, result_read = await safe_tool_call(
        "HTML -> MD (Readability Extract)",
        doc_tool.clean_and_format_text_as_markdown,
        tracker=tracker,
        text=html_content,
        extraction_method="readability",
        preserve_tables=False,  # Don't keep tables
    )
    if success:
        # Save the markdown content
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_read.get("markdown_text", ""))
            console.print(Text.from_markup(f"[green]✓ Saved readability-extracted markdown to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving markdown: {e}[/]"))
            
        display_result(
            "HTML -> MD (Readability Extract, No Tables)",
            result_read,
            display_options={"format_key": {"markdown_text": "markdown"}},
        )

    # 4c: Trafilatura Extraction
    output_path = get_output_path(html_file.stem, "trafilatura", "md")
    success, result_traf = await safe_tool_call(
        "HTML -> MD (Trafilatura Extract)",
        doc_tool.clean_and_format_text_as_markdown,
        tracker=tracker,
        text=html_content,
        extraction_method="trafilatura",
        preserve_tables=True,
    )
    if success:
        # Save the markdown content
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_traf.get("markdown_text", ""))
            console.print(Text.from_markup(f"[green]✓ Saved trafilatura-extracted markdown to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving markdown: {e}[/]"))
            
        display_result(
            "HTML -> MD (Trafilatura Extract)",
            result_traf,
            display_options={"format_key": {"markdown_text": "markdown"}},
        )

    # 4d: No Extraction (Full HTML Conversion)
    output_path = get_output_path(html_file.stem, "no_extract", "md")
    success, result_none = await safe_tool_call(
        "HTML -> MD (No Extract)",
        doc_tool.clean_and_format_text_as_markdown,
        tracker=tracker,
        text=html_content,
        extraction_method="none",
        preserve_tables=True,
        preserve_links=False,  # Keep tables, remove links
    )
    if success:
        # Save the markdown content
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_none.get("markdown_text", ""))
            console.print(Text.from_markup(f"[green]✓ Saved full HTML conversion to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving markdown: {e}[/]"))
            
        display_result(
            "HTML -> MD (No Extract, No Links)",
            result_none,
            display_options={"format_key": {"markdown_text": "markdown"}},
        )

    # --- optimize_markdown_formatting ---
    console.print(Rule("Markdown Optimization", style="dim"))
    markdown_to_optimize = (
        result_auto.get("markdown_text")
        if result_auto and result_auto.get("success") else """# Sample MD\n\nThis is a    paragraph with extra spaces.\n\n* Item1\n* Item 2\n\n## Subheading\n\nAnother para. Link: [Example ] (http://example.com)\n\n```\ndef hello():\n  print("hi")\n```\n\n"""
    )
    if not markdown_to_optimize:
        console.print("[yellow]Cannot run optimize demo as previous MD conversion failed.[/]")
    else:
        console.print(Panel("Original Markdown for Optimization:", border_style="yellow"))
        # Apply truncation
        truncated_md = truncate_text_by_lines(markdown_to_optimize, 300)
        console.print(Syntax(truncated_md, "markdown", theme="default", line_numbers=True))
        
        # Save original markdown for reference
        orig_md_path = get_output_path("original_markdown", "for_optimization", "md")
        try:
            with open(orig_md_path, 'w', encoding='utf-8') as f:
                f.write(truncated_md)
            console.print(Text.from_markup(f"[green]✓ Saved original markdown to: [blue underline]{orig_md_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving original markdown: {e}[/]"))

        # 4e: Optimize with specific fixes and wrapping
        output_path = get_output_path("optimized_markdown", "normalized", "md")
        success, result_opt1 = await safe_tool_call(
            "Optimize MD (Normalize, Fix, Wrap)",
            doc_tool.optimize_markdown_formatting,
            tracker=tracker,
            markdown=truncated_md,
            normalize_headings=True,
            fix_lists=True,
            fix_links=True,
            add_line_breaks=True,
            max_line_length=80,
        )
        if success:
            # Save the optimized markdown
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result_opt1.get("optimized_markdown", ""))
                console.print(Text.from_markup(f"[green]✓ Saved normalized markdown to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving optimized markdown: {e}[/]"))
                
            display_result(
                "Optimize MD (Normalize, Fix, Wrap)",
                result_opt1,
                display_options={"format_key": {"optimized_markdown": "markdown"}},
            )
            
            console.print(Text.from_markup("[yellow]No tables found by Docling.[/]"))

        # 6b: Extract as JSON
        success, result_json = await safe_tool_call(
            "Extract Tables (JSON)",
            doc_tool.extract_tables,
            tracker=tracker,
            document_path=str(html_file),
            table_mode="json",
        )
        if success and result_json.get("tables"):
            display_result(
                "Extract Tables (JSON)",
                result_json,
                display_keys=["tables"],
                display_options={"detail_level": 1},
            )  # Show more structure for JSON

        # 6c: Extract as Pandas DataFrame (if available)
        if doc_tool._pandas_available:
            success, result_pd = await safe_tool_call(
                "Extract Tables (Pandas)",
                doc_tool.extract_tables,
                tracker=tracker,
                document_path=str(html_file),
                table_mode="pandas",
            )
            if success and result_pd.get("tables"):
                display_result(
                    "Extract Tables (Pandas)",
                    result_pd,
                    display_keys=["tables"],
                    display_options={"detail_level": 0},
                )
                # Cannot directly display DataFrame easily, show preview info
                first_df = result_pd["tables"][0]
                console.print(
                    Panel(
                        f"First DataFrame Info:\nShape: {first_df.shape}\nColumns: {list(first_df.columns)}",
                        title="First DataFrame Preview",
                    )
                )
        else:
            console.print(Text.from_markup("[yellow]Pandas unavailable, skipping Pandas table extraction.[/]"))

    # 4f: Optimize in Compact Mode
    output_path = get_output_path("optimized_markdown", "compact", "md")
    success, result_opt2 = await safe_tool_call(
        "Optimize MD (Compact Mode)",
        doc_tool.optimize_markdown_formatting,
        tracker=tracker,
        markdown=truncated_md,
        compact_mode=True,
        add_line_breaks=False,  # Compact, don't force extra breaks
    )
    if success:
        # Save the optimized markdown
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_opt2.get("optimized_markdown", ""))
            console.print(Text.from_markup(f"[green]✓ Saved compact markdown to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving optimized markdown: {e}[/]"))
            
        display_result(
            "Optimize MD (Compact Mode)",
            result_opt2,
            display_options={"format_key": {"optimized_markdown": "markdown"}},
        )

    # --- detect_content_type ---
    console.print(Rule("Content Type Detection", style="dim"))

    # 4g: Detect HTML Type
    success, result_detect_html = await safe_tool_call(
        "Detect Type (HTML)", doc_tool.detect_content_type, text=html_content[:6000], tracker=tracker
    )  # Test larger sample
    if success:
        display_result("Detect Type (HTML)", result_detect_html)

    # 4h: Detect Markdown Type (using optimized markdown if available)
    md_for_detect = (
        result_opt1.get("optimized_markdown")
        if result_opt1 and result_opt1.get("success")
        else markdown_to_optimize
    )
    if md_for_detect:
        success, result_detect_md = await safe_tool_call(
            "Detect Type (Markdown)", doc_tool.detect_content_type, text=md_for_detect[:6000], tracker=tracker
        )
        if success:
            display_result("Detect Type (Markdown)", result_detect_md)

    # 4i: Detect Code Type
    sample_code = """
def process_data(data: list) -> dict:
    # Process the incoming data list
    results = {}
    for i, item in enumerate(data):
        if item is not None: # Check for None
             results[f"item_{i}"] = item * item # Square numbers
    return results

class MyClass:
    def __init__(self, value):
        self.value = value # Store value
    """
    success, result_detect_code = await safe_tool_call(
        "Detect Type (Python Code)", doc_tool.detect_content_type, text=sample_code, tracker=tracker
    )
    if success:
        display_result("Detect Type (Python Code)", result_detect_code)

    # 4j: Detect Plain Text Type
    sample_plain = "This is a simple paragraph of plain text. It doesn't contain any significant markup like HTML tags or markdown syntax. Just regular sentences."
    success, result_detect_plain = await safe_tool_call(
        "Detect Type (Plain Text)", doc_tool.detect_content_type, text=sample_plain, tracker=tracker
    )
    if success:
        display_result("Detect Type (Plain Text)", result_detect_plain)

    # 4f: Extract tables from HTML content
    output_path = get_output_path(html_file.stem, "extracted_tables", "csv")
    success, html_tables_result = await safe_tool_call(
        "HTML -> Tables (Extract)",
        doc_tool.extract_tables,
        tracker=tracker,
        document_path=str(html_file),
        table_mode="csv",
        output_dir=str(conversion_outputs_dir / "html_tables")
    )
    
    if success and html_tables_result.get("tables"):
        # Save extracted tables
        try:
            # Save a copy of the first table for reference
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_tables_result["tables"][0])
            console.print(Text.from_markup(f"[green]✓ Saved extracted HTML table to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving HTML table: {e}[/]"))
            
        # Display the table extraction results
        display_result(
            "HTML -> Tables (Extract)",
            html_tables_result,
            display_keys=["tables", "saved_files"],
            display_options={"detail_level": 0},
        )
        # Show a preview of the first table
        console.print(
            Panel(
                escape(html_tables_result["tables"][0][:500]) + "...", 
                title="First HTML Table Preview (CSV)"
            )
        )
    elif success:
        console.print(Text.from_markup("[yellow]No tables found in HTML content.[/]"))

    # 4g: Optimize in Compact Mode
    output_path = get_output_path("optimized_markdown", "compact", "md")
    success, result_opt2 = await safe_tool_call(
        "Optimize MD (Compact Mode)",
        doc_tool.optimize_markdown_formatting,
        tracker=tracker,
        markdown=truncated_md,
        compact_mode=True,
        add_line_breaks=False,  # Compact, don't force extra breaks
    )
    if success:
        # Save the optimized markdown
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_opt2.get("optimized_markdown", ""))
            console.print(Text.from_markup(f"[green]✓ Saved compact markdown to: [blue underline]{output_path}[/][/]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error saving optimized markdown: {e}[/]"))
            
        display_result(
            "Optimize MD (Compact Mode)",
            result_opt2,
            display_options={"format_key": {"optimized_markdown": "markdown"}},
        )

    # --- detect_content_type ---
    console.print(Rule("Content Type Detection", style="dim"))

    # 4h: Detect HTML Type
    success, result_detect_html = await safe_tool_call(
        "Detect Type (HTML)", doc_tool.detect_content_type, text=html_content[:6000], tracker=tracker
    )  # Test larger sample
    if success:
        display_result("Detect Type (HTML)", result_detect_html)

    # 4i: Detect Markdown Type (using optimized markdown if available)
    md_for_detect = (
        result_opt1.get("optimized_markdown")
        if result_opt1 and result_opt1.get("success")
        else markdown_to_optimize
    )
    if md_for_detect:
        success, result_detect_md = await safe_tool_call(
            "Detect Type (Markdown)", doc_tool.detect_content_type, text=md_for_detect[:6000], tracker=tracker
        )
        if success:
            display_result("Detect Type (Markdown)", result_detect_md)

    # 4j: Detect Code Type
    sample_code = """
def process_data(data: list) -> dict:
    # Process the incoming data list
    results = {}
    for i, item in enumerate(data):
        if item is not None: # Check for None
             results[f"item_{i}"] = item * item # Square numbers
    return results

class MyClass:
    def __init__(self, value):
        self.value = value # Store value
    """
    success, result_detect_code = await safe_tool_call(
        "Detect Type (Python Code)", doc_tool.detect_content_type, text=sample_code, tracker=tracker
    )
    if success:
        display_result("Detect Type (Python Code)", result_detect_code)

    # 4k: Detect Plain Text Type
    sample_plain = "This is a simple paragraph of plain text. It doesn't contain any significant markup like HTML tags or markdown syntax. Just regular sentences."
    success, result_detect_plain = await safe_tool_call(
        "Detect Type (Plain Text)", doc_tool.detect_content_type, text=sample_plain, tracker=tracker
    )
    if success:
        display_result("Detect Type (Plain Text)", result_detect_plain)


async def demo_section_5_analyze_structure(
    doc_tool: DocumentProcessingTool, sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate the dedicated PDF structure analysis tool."""
    console.print(Rule("[bold green]Demo 5: Analyze PDF Structure Tool[/]", style="green"))
    logger.info("Starting Demo Section 5: Analyze PDF Structure")

    pdf_digital = sample_files.get("pdf_digital")
    buffett_pdf = sample_files.get("buffett_pdf") 
    backprop_pdf = sample_files.get("backprop_pdf")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")
    
    pdf_files_to_process = [pdf for pdf in [pdf_digital, buffett_pdf, backprop_pdf] if pdf is not None]
    
    if not pdf_files_to_process:
        console.print("[yellow]Skipping Demo 5: No PDF files available.[/]")
        return

    # Function to generate output path
    def get_output_path(file_name: str, analysis_type: str) -> str:
        """Generate standardized output path for analysis outputs"""
        return str(conversion_outputs_dir / f"{file_name}_analysis_{analysis_type}.json")

    for pdf_file in pdf_files_to_process:
        console.print(
            Panel(Text.from_markup(f"Analyzing PDF Structure: [cyan]{pdf_file.name}[/]"), border_style="blue")
        )

        # 5a: Analyze Structure (Default options)
        output_path = get_output_path(pdf_file.stem, "default")
        success, result = await safe_tool_call(
            f"Analyze {pdf_file.name} Structure (Defaults)",
            doc_tool.analyze_pdf_structure,
            tracker=tracker,
            file_path=str(pdf_file),
            # Defaults: extract_metadata=True, extract_outline=True, estimate_ocr_needs=True
            # Fonts & Images off by default
        )
        # Save analysis result
        if success:
            try:
                # Create a clean copy without internal keys
                result_to_save = {k: v for k, v in result.items() if not k.startswith('_')}
                with open(output_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result_to_save, f, indent=2)
                console.print(Text.from_markup(f"[green]✓ Saved PDF analysis to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving PDF analysis: {e}[/]"))
                
            display_result(f"Analyze {pdf_file.name} Structure (Defaults)", result)

        # 5b: Analyze Structure (All options enabled)
        console.print(Panel(Text.from_markup(f"Analyzing {pdf_file.name} Structure (All Options Enabled)"), border_style="blue"))
        output_path = get_output_path(pdf_file.stem, "all_options")
        # This might be slightly slower if font/image extraction is intensive
        success, result_all = await safe_tool_call(
            f"Analyze {pdf_file.name} Structure (All Options)",
            doc_tool.analyze_pdf_structure,
            tracker=tracker,
            file_path=str(pdf_file),
            extract_metadata=True,
            extract_outline=True,
            extract_fonts=True,  # Request fonts
            extract_images=True,  # Request image info
            estimate_ocr_needs=True,
        )
        if success:
            # Save analysis result
            try:
                # Create a clean copy without internal keys
                result_to_save = {k: v for k, v in result_all.items() if not k.startswith('_')}
                with open(output_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result_to_save, f, indent=2)
                console.print(Text.from_markup(f"[green]✓ Saved detailed PDF analysis to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving PDF analysis: {e}[/]"))
                
            display_result(Text.from_markup(f"Analyze {pdf_file.name} Structure (All Options)"), result_all)

    # 5c: Analyze Structure from Bytes (just for one PDF)
    if pdf_digital:
        console.print(Panel(Text.from_markup("Analyzing PDF Structure from Bytes Data"), border_style="blue"))
        try:
            pdf_bytes = pdf_digital.read_bytes()
            output_path = get_output_path(pdf_digital.stem, "from_bytes")
            success, result_bytes = await safe_tool_call(
                "Analyze PDF Structure (Bytes)",
                doc_tool.analyze_pdf_structure,
                tracker=tracker,
                document_data=pdf_bytes,
                extract_metadata=True,
                estimate_ocr_needs=True,  # Fewer options for brevity
            )
            if success:
                # Save analysis result
                try:
                    # Create a clean copy without internal keys
                    result_to_save = {k: v for k, v in result_bytes.items() if not k.startswith('_')}
                    with open(output_path, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(result_to_save, f, indent=2)
                    console.print(Text.from_markup(f"[green]✓ Saved PDF bytes analysis to: [blue underline]{output_path}[/][/]"))
                except Exception as e:
                    console.print(Text.from_markup(f"[red]Error saving PDF analysis: {e}[/]"))
                    
                display_result(
                    "Analyze PDF Structure (Bytes)",
                    result_bytes,
                    display_options={"display_keys": ["page_count", "metadata", "ocr_assessment"]},
                )
        except Exception as e:
            console.print(Text.from_markup(f"[red]Error analyzing PDF from bytes: {e}[/]"))


async def demo_section_6_chunking_tables(
    doc_tool: DocumentProcessingTool, sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate Document Chunking and Table Extraction tools."""
    console.print(Rule(Text.from_markup("[bold green]Demo 6: Chunking & Table Extraction[/]", style="green")))
    logger.info("Starting Demo Section 6: Chunking & Table Extraction")

    pdf_digital = sample_files.get("pdf_digital")
    buffett_pdf = sample_files.get("buffett_pdf") 
    backprop_pdf = sample_files.get("backprop_pdf")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")
    
    pdf_files_to_process = [pdf for pdf in [pdf_digital, buffett_pdf, backprop_pdf] if pdf is not None]
    
    if not pdf_files_to_process:
        console.print(Text.from_markup("[yellow]Skipping Demo 6: No PDF files available.[/]"))
        return
    
    # Function to generate output path
    def get_output_path(base_name: str, process_type: str, format_name: str) -> str:
        """Generate standardized output path for outputs"""
        return str(conversion_outputs_dir / f"{base_name}_{process_type}.{format_name}")

    for pdf_file in pdf_files_to_process:
        try:
            # --- Get Markdown Content First ---
            console.print(
                Panel(
                    Text.from_markup(f"Preparing Content for Chunking/Tables from: [cyan]{pdf_file.name}[/]"),
                    border_style="dim",
                )
            )
            success, conv_result = await safe_tool_call(
                f"Get Markdown Content for {pdf_file.name}",
                doc_tool.convert_document,
                tracker=tracker,
                document_path=str(pdf_file),
                output_format="markdown",
                extraction_strategy="direct_text",
                enhance_with_llm=False,  # Raw text for speed
                save_to_file=True,
                output_path=str(conversion_outputs_dir / f"{pdf_file.stem}_for_chunking.md"),
            )
            if not success or not conv_result.get("content"):
                console.print(Text.from_markup(f"[red]Failed to get markdown content for {pdf_file.name}.[/]"))
                continue
            markdown_content = conv_result["content"]
            console.print(Text.from_markup("[green]✓ Content prepared.[/]"))

            # --- Chunking Demonstrations ---
            console.print(Rule(Text.from_markup(f"Document Chunking for {pdf_file.name}"), style="dim"))

            chunking_configs = [
                {"method": "paragraph", "size": 500, "overlap": 50},
                {"method": "character", "size": 800, "overlap": 100},
                {"method": "token", "size": 200, "overlap": 20},  # Requires tiktoken
                {"method": "section", "size": 1000, "overlap": 0},  # Relies on section identification
            ]

            for config in chunking_configs:
                method, size, overlap = config["method"], config["size"], config["overlap"]
                if method == "token" and not doc_tool._tiktoken_available:
                    console.print(
                        Text.from_markup(f"[yellow]Skipping chunking method '{method}': Tiktoken not available.[/]")
                    )
                    continue

                output_path = get_output_path(pdf_file.stem, f"chunks_{method}", "json")
                success, result = await safe_tool_call(
                    f"Chunking {pdf_file.name} ({method.capitalize()})",
                    doc_tool.chunk_document,
                    tracker=tracker,
                    document=markdown_content,
                    chunk_method=method,
                    chunk_size=size,
                    chunk_overlap=overlap,
                )
                if success:
                    # Save chunks to file
                    try:
                        # Create a clean copy without internal keys
                        result_to_save = {k: v for k, v in result.items() if not k.startswith('_')}
                        with open(output_path, 'w', encoding='utf-8') as f:
                            import json
                            json.dump(result_to_save, f, indent=2)
                        console.print(Text.from_markup(f"[green]✓ Saved chunks to: [blue underline]{output_path}[/][/]"))
                    except Exception as e:
                        console.print(Text.from_markup(f"[red]Error saving chunks: {e}[/]"))
                        
                    display_result(f"Chunking {pdf_file.name} ({method}, size={size}, overlap={overlap})", result)

            # --- Table Extraction (Requires Docling) ---
            console.print(Rule(f"Table Extraction for {pdf_file.name} (Requires Docling)", style="dim"))
            if doc_tool._docling_available:
                console.print(
                    Panel(Text.from_markup(f"Extracting Tables (Docling): [cyan]{pdf_file.name}[/]"), border_style="blue")
                )
                # Tables directory for this specific PDF
                tables_dir = conversion_outputs_dir / f"{pdf_file.stem}_tables"
                tables_dir.mkdir(exist_ok=True)
                
                # 6a: Extract as CSV
                success, result_csv = await safe_tool_call(
                    f"Extract {pdf_file.name} Tables (CSV)",
                    doc_tool.extract_tables,
                    tracker=tracker,
                    document_path=str(pdf_file),
                    table_mode="csv",
                    output_dir=str(tables_dir / "csv"),
                )
                if success and result_csv.get("tables"):
                    display_result(
                        f"Extract {pdf_file.name} Tables (CSV)",
                        result_csv,
                        display_keys=["tables", "saved_files"],
                        display_options={"detail_level": 0},
                    )  # Show less detail for list of tables
                    console.print(
                        Panel(
                            escape(result_csv["tables"][0][:500]) + "...", title=f"First Table Preview from {pdf_file.name} (CSV)"
                        )
                    )
                elif success:
                    console.print(Text.from_markup(f"[yellow]No tables found by Docling in {pdf_file.name}.[/]"))

                # 6b: Extract as JSON
                success, result_json = await safe_tool_call(
                    f"Extract {pdf_file.name} Tables (JSON)",
                    doc_tool.extract_tables,
                    tracker=tracker,
                    document_path=str(pdf_file),
                    table_mode="json",
                    output_dir=str(tables_dir / "json"),
                )
                if success and result_json.get("tables"):
                    display_result(
                        f"Extract {pdf_file.name} Tables (JSON)",
                        result_json,
                        display_keys=["tables"],
                        display_options={"detail_level": 1},
                    )  # Show more structure for JSON

                # 6c: Extract as Pandas DataFrame (if available)
                if doc_tool._pandas_available:
                    success, result_pd = await safe_tool_call(
                        f"Extract {pdf_file.name} Tables (Pandas)",
                        doc_tool.extract_tables,
                        tracker=tracker,
                        document_path=str(pdf_file),
                        table_mode="pandas",
                        output_dir=str(tables_dir / "pandas"),
                    )
                    if success and result_pd.get("tables"):
                        display_result(
                            f"Extract {pdf_file.name} Tables (Pandas)",
                            result_pd,
                            display_keys=["tables"],
                            display_options={"detail_level": 0},
                        )
                        # Cannot directly display DataFrame easily, show preview info
                        first_df = result_pd["tables"][0]
                        console.print(
                            Panel(
                                f"First DataFrame Info:\nShape: {first_df.shape}\nColumns: {list(first_df.columns)}",
                                title=f"First DataFrame Preview from {pdf_file.name}",
                            )
                        )
                else:
                    console.print(Text.from_markup("[yellow]Pandas unavailable, skipping Pandas table extraction.[/]"))
            else:
                console.print(Text.from_markup("[yellow]Docling unavailable, skipping table extraction demo.[/]"))
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}", exc_info=True)
            console.print(Text.from_markup(f"[bold red]Error processing {pdf_file.name}:[/] {e}"))


async def demo_section_7_analysis(
    doc_tool: DocumentProcessingTool, sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate the full suite of document analysis tools."""
    console.print(Rule(Text.from_markup("[bold green]Demo 7: Document Analysis Suite[/]", style="green")))
    logger.info("Starting Demo Section 7: Document Analysis Suite")

    pdf_digital = sample_files.get("pdf_digital")
    buffett_pdf = sample_files.get("buffett_pdf") 
    backprop_pdf = sample_files.get("backprop_pdf")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")
    
    pdf_files_to_process = [pdf for pdf in [pdf_digital, buffett_pdf, backprop_pdf] if pdf is not None]
    
    if not pdf_files_to_process:
        console.print(Text.from_markup("[yellow]Skipping Demo 7: No PDF files available.[/]"))
        return

    # Function to generate output path
    def get_output_path(base_name: str, analysis_type: str, format_name: str = "json") -> str:
        """Generate standardized output path for analysis outputs"""
        return str(conversion_outputs_dir / f"{base_name}_analysis_{analysis_type}.{format_name}")

    for pdf_file in pdf_files_to_process:
        # --- Get Text Content for Analysis ---
        console.print(
            Panel(Text.from_markup(f"Preparing Text for Analysis from: [cyan]{pdf_file.name}[/]"), border_style="dim")
        )
        success, conv_result = await safe_tool_call(
            f"Get Text for {pdf_file.name} Analysis",
            doc_tool.convert_document,
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="markdown",  # Use Markdown for better structure
            extraction_strategy="direct_text",
            enhance_with_llm=False,
            save_to_file=True,
            output_path=str(conversion_outputs_dir / f"{pdf_file.stem}_for_analysis.md"),
        )
        if not success or not conv_result.get("content"):
            console.print(Text.from_markup(f"[red]Failed to extract text for analysis of {pdf_file.name}.[/]"))
            continue
        analysis_text = conv_result["content"]
        console.print(Text.from_markup("[green]✓ Content prepared.[/]"))
        console.print(
            Panel(
                escape(truncate_text_by_lines(analysis_text[:600], 300)) + "...",
                title=Text.from_markup(f"Text Prepared for Analysis of {pdf_file.name} (Markdown Preview)"),
                border_style="dim",
            )
        )

        # --- Run Analysis Tools ---
        entities_result_for_canon = None

        # 7.1 Identify Sections (Regex-based)
        output_path = get_output_path(pdf_file.stem, "sections")
        success, result = await safe_tool_call(
            f"Identify Sections in {pdf_file.name}", 
            doc_tool.identify_sections, 
            document=analysis_text,
            tracker=tracker,
        )
        if success:
            # Save result to file
            try:
                # Create a clean copy without internal keys
                result_to_save = {k: v for k, v in result.items() if not k.startswith('_')}
                with open(output_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result_to_save, f, indent=2)
                console.print(Text.from_markup(f"[green]✓ Saved sections analysis to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving analysis: {e}[/]"))
                
            display_result(Text.from_markup(f"Identify Sections in {pdf_file.name}"), result)

        # 7.2 Extract Entities (LLM-based) - Broad focus
        output_path = get_output_path(pdf_file.stem, "entities")
        success, result = await safe_tool_call(
            f"Extract Entities from {pdf_file.name} (Broad)", 
            doc_tool.extract_entities, 
            document=analysis_text,
            tracker=tracker,
        )
        if success:
            # Save result to file
            try:
                # Create a clean copy without internal keys
                result_to_save = {k: v for k, v in result.items() if not k.startswith('_')}
                with open(output_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result_to_save, f, indent=2)
                console.print(Text.from_markup(f"[green]✓ Saved entities analysis to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving analysis: {e}[/]"))
                
            display_result(Text.from_markup(f"Extract Entities from {pdf_file.name} (Broad)"), result)
            entities_result_for_canon = result

        # 7.3 Canonicalise Entities (if extraction succeeded)
        if entities_result_for_canon and entities_result_for_canon.get("entities"):
            output_path = get_output_path(pdf_file.stem, "canon_entities")
            success, result = await safe_tool_call(
                f"Canonicalise Entities for {pdf_file.name}",
                doc_tool.canonicalise_entities,
                entities_input=entities_result_for_canon,
                tracker=tracker,
            )
            if success:
                # Save result to file
                try:
                    # Create a clean copy without internal keys
                    result_to_save = {k: v for k, v in result.items() if not k.startswith('_')}
                    with open(output_path, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(result_to_save, f, indent=2)
                    console.print(Text.from_markup(f"[green]✓ Saved canonicalized entities to: [blue underline]{output_path}[/][/]"))
                except Exception as e:
                    console.print(Text.from_markup(f"[red]Error saving analysis: {e}[/]"))
                    
                display_result(Text.from_markup(f"Canonicalise Entities for {pdf_file.name}"), result)
        else:
            console.print(
                Text.from_markup(f"[yellow]Skipping entity canonicalization for {pdf_file.name} as entity extraction failed or yielded no results.[/]")
            )

        # 7.4 Generate QA Pairs (LLM-based)
        output_path = get_output_path(pdf_file.stem, "qa_pairs")
        success, result = await safe_tool_call(
            f"Generate QA Pairs for {pdf_file.name}", 
            doc_tool.generate_qa_pairs, 
            document=analysis_text, 
            num_questions=4,
            tracker=tracker,
        )
        if success:
            # Save result to file
            try:
                # Create a clean copy without internal keys
                result_to_save = {k: v for k, v in result.items() if not k.startswith('_')}
                with open(output_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result_to_save, f, indent=2)
                console.print(Text.from_markup(f"[green]✓ Saved QA pairs to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving QA pairs: {e}[/]"))
                
            display_result(Text.from_markup(f"Generate QA Pairs for {pdf_file.name}"), result)

        # 7.5 Summarize Document (LLM-based)
        output_path = get_output_path(pdf_file.stem, "summary", "md")
        success, result = await safe_tool_call(
            f"Summarize {pdf_file.name}", 
            doc_tool.summarize_document, 
            document=analysis_text, 
            max_length=100,
            tracker=tracker,
        )
        if success:
            # Save summary to file
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.get("summary", ""))
                console.print(Text.from_markup(f"[green]✓ Saved summary to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving summary: {e}[/]"))
                
            display_result(Text.from_markup(f"Summarize {pdf_file.name}"), result)

        # 7.6 Classify Document (LLM-based) - Using custom labels
        output_path = get_output_path(pdf_file.stem, "classification")
        success, result = await safe_tool_call(
            f"Classify {pdf_file.name} (Custom)",
            doc_tool.classify_document,
            document=analysis_text,
            custom_labels=["AI Research", "Software Manual", "Financial Report", "News Article"],
            tracker=tracker,
        )
        if success:
            # Save result to file
            try:
                # Create a clean copy without internal keys
                result_to_save = {k: v for k, v in result.items() if not k.startswith('_')}
                with open(output_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result_to_save, f, indent=2)
                console.print(Text.from_markup(f"[green]✓ Saved classification to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving classification: {e}[/]"))
                
            display_result(Text.from_markup(f"Classify {pdf_file.name} (Custom Labels)"), result)

        # 7.7 Extract Metrics (Regex-based) - Might not find much in this paper
        output_path = get_output_path(pdf_file.stem, "metrics")
        success, result = await safe_tool_call(
            f"Extract Metrics from {pdf_file.name}", 
            doc_tool.extract_metrics, 
            document=analysis_text,
            tracker=tracker,
        )
        if success:
            # Save result to file
            try:
                # Create a clean copy without internal keys
                result_to_save = {k: v for k, v in result.items() if not k.startswith('_')}
                with open(output_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result_to_save, f, indent=2)
                console.print(Text.from_markup(f"[green]✓ Saved metrics to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving metrics: {e}[/]"))
                
            display_result(Text.from_markup(f"Extract Metrics from {pdf_file.name}"), result)
        if success and not result.get("metrics"):
            console.print(Text.from_markup(f"[yellow]Note: No pre-defined metrics found in {pdf_file.name}.[/]"))

        # 7.8 Flag Risks (Regex-based) - Might not find much in this paper
        output_path = get_output_path(pdf_file.stem, "risks")
        success, result = await safe_tool_call(
            f"Flag Risks in {pdf_file.name}", 
            doc_tool.flag_risks, 
            document=analysis_text,
            tracker=tracker,
        )
        if success:
            # Save result to file
            try:
                # Create a clean copy without internal keys
                result_to_save = {k: v for k, v in result.items() if not k.startswith('_')}
                with open(output_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result_to_save, f, indent=2)
                console.print(Text.from_markup(f"[green]✓ Saved risks analysis to: [blue underline]{output_path}[/][/]"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]Error saving risks analysis: {e}[/]"))
                
            display_result(Text.from_markup(f"Flag Risks in {pdf_file.name}"), result)
        if success and not result.get("risks"):
            console.print(Text.from_markup(f"[yellow]Note: No pre-defined risks found in {pdf_file.name}.[/]"))


async def demo_section_8_batch_processing(
    doc_tool: DocumentProcessingTool, sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate complex batch processing pipeline with detailed results."""
    console.print(Rule(Text.from_markup("[bold green]Demo 8: Advanced Batch Processing[/]", style="green")))
    logger.info("Starting Demo Section 8: Batch Processing")


async def main():
    """Main function to run the DocumentProcessingTool demo."""
    try:
        # Create a CostTracker instance
        tracker = CostTracker()  # Create tracker instance
        
        # Create a gateway instance
        gateway = Gateway("document-processing-demo", register_tools=True)
        
        # Initialize providers
        logger.info("Initializing gateway and providers...", emoji_key="provider")
        await gateway._initialize_providers()
        
        # Create the document processing tool
        logger.info("Creating DocumentProcessingTool instance...", emoji_key="info")
        doc_tool = DocumentProcessingTool(gateway)  # Pass gateway as mcp_server argument
        
        # Prepare sample files and directories
        logger.info("Setting up sample files and directories...", emoji_key="info")
        
        # Create sample directory if it doesn't exist
        DEFAULT_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        DOWNLOADED_FILES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Output directory for conversion results
        conversion_outputs_dir = DEFAULT_SAMPLE_DIR / "conversion_outputs"
        conversion_outputs_dir.mkdir(exist_ok=True)
        
        # Download or locate sample files
        sample_files = {
            "conversion_outputs_dir": conversion_outputs_dir
        }
        
        # Download digital PDF if needed (Attention Is All You Need paper)
        pdf_digital_path = DOWNLOADED_FILES_DIR / "attention_is_all_you_need.pdf"
        pdf_digital = await download_file_with_progress(
            DEFAULT_SAMPLE_PDF_URL, pdf_digital_path, "Transformer Paper (PDF)"
        )
        sample_files["pdf_digital"] = pdf_digital
        
        # Download sample OCR image if needed
        image_path = DOWNLOADED_FILES_DIR / "sample_ocr_image.png"
        image_file = await download_file_with_progress(
            DEFAULT_SAMPLE_IMAGE_URL, image_path, "Sample OCR Image"
        )
        sample_files["image"] = image_file
        
        # Download HTML sample if needed
        html_path = DOWNLOADED_FILES_DIR / "transformer_wiki.html"
        html_file = await download_file_with_progress(
            SAMPLE_HTML_URL, html_path, "Transformer Wiki Page (HTML)"
        )
        sample_files["html"] = html_file
        
        # Run the demo sections, passing the tracker to each
        await demo_section_1_conversion_ocr(doc_tool, sample_files, tracker)
        await demo_section_2_dedicated_ocr(doc_tool, sample_files, tracker)
        await demo_section_3_enhance_text(doc_tool, sample_files, tracker)
        await demo_section_4_html_markdown(doc_tool, sample_files, tracker)
        await demo_section_5_analyze_structure(doc_tool, sample_files, tracker)
        await demo_section_6_chunking_tables(doc_tool, sample_files, tracker)
        await demo_section_7_analysis(doc_tool, sample_files, tracker)
        await demo_section_8_batch_processing(doc_tool, sample_files, tracker)
        
        # Display cost summary at the end
        tracker.display_summary(console)
        
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
