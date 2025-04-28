#!/usr/bin/env python
"""
DETAILED Demonstration script for the SmartBrowserTool in Ultimate MCP Server,
showcasing browsing, interaction, search, download, macro, and autopilot features.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime  # Corrected import for timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root to path for imports when running as script
# Adjust this relative path if your script structure is different
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    print(f"INFO: Added {_PROJECT_ROOT} to sys.path")

# Rich imports for enhanced terminal UI
from rich import box, get_console  # noqa: E402
from rich.console import Group  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402
from rich.traceback import install as install_rich_traceback  # noqa: E402

# --- Attempt to import required MCP Server components ---
try:
    from ultimate_mcp_server.core.server import Gateway
    from ultimate_mcp_server.exceptions import ToolError, ToolInputError
    from ultimate_mcp_server.tools.smart_browser import SmartBrowserTool
    from ultimate_mcp_server.utils import get_logger
    from ultimate_mcp_server.utils.display import (
        CostTracker,
    )  # Though SB likely doesn't track costs directly

    MCP_COMPONENTS_LOADED = True
except ImportError as e:
    MCP_COMPONENTS_LOADED = False
    _IMPORT_ERROR_MSG = str(e)
    # Handle error in main

# Initialize Rich console and logger
console = get_console()
logger = get_logger("demo.smart_browser_tool")

# Install rich tracebacks
install_rich_traceback(show_locals=True, width=console.width, extra_lines=2)

# --- Configuration ---
# Base directory for Smart Browser outputs (matching the tool's default)
SMART_BROWSER_HOME = Path.home() / ".smart_browser"
DEMO_OUTPUTS_DIR = SMART_BROWSER_HOME / "sb_demo_outputs"

# Example URLs for demo
URL_EXAMPLE = "http://example.com"
URL_BOOKSTORE = "http://books.toscrape.com/"
URL_QUOTES = "http://quotes.toscrape.com/"  # Good for text extraction
URL_PDF_SAMPLE = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"  # Simple downloadable PDF
URL_GITHUB = "https://github.com/features/copilot"  # Example for doc collection (adjust if needed)


# Environment variables can influence behavior (see tool code)
# export SB_HEADLESS=0  # Run headful
# export SB_VNC_ENABLED=1 # Enable VNC (requires password set, usually via config)
# export SB_MAX_TABS=3    # Limit concurrent tabs

# --- Demo Helper Functions ---


def timestamp_str(short: bool = False) -> str:
    """Return a formatted timestamp string."""
    now = time.time()  # Use time.time for consistency
    dt_now = datetime.fromtimestamp(now)
    if short:
        return f"[dim]{dt_now.strftime('%H:%M:%S')}[/]"
    return f"[dim]{dt_now.strftime('%Y-%m-%d %H:%M:%S')}[/]"


def truncate_text_by_lines(text: str, max_lines: int = 50) -> str:
    """Truncates text to show first/last lines if too long."""
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    half_lines = max_lines // 2
    return "\n".join(lines[:half_lines] + ["[...TRUNCATED...]"] + lines[-half_lines:])


def format_value(key: str, value: Any, detail_level: int = 1) -> Any:
    """Format specific values for display."""
    if value is None:
        return "[dim]None[/]"
    if isinstance(value, bool):
        return "[green]Yes[/]" if value else "[red]No[/]"
    if isinstance(value, float):
        return f"{value:.3f}"
    if key.lower().endswith("time_seconds") or key.lower() == "duration_ms":
        val_s = float(value) / 1000.0 if key.lower() == "duration_ms" else float(value)
        return f"[green]{val_s:.3f}s[/]"
    if key.lower() == "size_bytes" and isinstance(value, int):
        if value > 1024 * 1024:
            return f"{value / (1024 * 1024):.2f} MB"
        if value > 1024:
            return f"{value / 1024:.2f} KB"
        return f"{value} Bytes"

    if isinstance(value, list):
        if not value:
            return "[dim]Empty List[/]"
        list_len = len(value)
        preview_count = 3 if detail_level < 2 else 5
        suffix = f" [dim]... ({list_len} items total)[/]" if list_len > preview_count else ""
        if detail_level >= 1:
            previews = [
                format_value(f"{key}[{i}]", item, detail_level=0)
                for i, item in enumerate(value[:preview_count])
            ]
            return f"[{', '.join(previews)}]{suffix}"
        else:
            return f"[List with {list_len} items]"

    if isinstance(value, dict):
        if not value:
            return "[dim]Empty Dict[/]"
        dict_len = len(value)
        preview_count = 4 if detail_level < 2 else 8
        preview_keys = list(value.keys())[:preview_count]
        suffix = f" [dim]... ({dict_len} keys total)[/]" if dict_len > preview_count else ""
        if detail_level >= 1:
            items_preview = [
                f"{repr(k)}: {format_value(k, value[k], detail_level=0)}" for k in preview_keys
            ]
            return f"{{{'; '.join(items_preview)}}}{suffix}"
        else:
            return f"[Dict with {dict_len} keys]"

    if isinstance(value, str):
        value = truncate_text_by_lines(
            value, 30
        )  # Truncate long strings more aggressively for inline display
        preview_len = 300 if detail_level < 2 else 600
        if len(value) > preview_len:
            return (
                escape(value[:preview_len]) + f"[dim]... (truncated, {len(value)} chars total)[/]"
            )
        return escape(value)

    return escape(str(value))


def display_page_state(state: Dict[str, Any], title: str = "Page State"):
    """Display the 'page_state' dictionary nicely."""
    panel_content = []
    url = state.get("url", "N/A")
    panel_content.append(f"[bold cyan]URL:[/bold cyan] [link={url}]{escape(url)}[/link]")
    panel_content.append(f"[bold cyan]Title:[/bold cyan] {escape(state.get('title', 'N/A'))}")

    # Display main text summary (truncated)
    main_text = state.get("main_text", "")
    if main_text:
        truncated_text = truncate_text_by_lines(main_text, 15)  # Show fewer lines for summary
        panel_content.append("\n[bold cyan]Main Text Summary:[/bold cyan]")
        panel_content.append(Panel(escape(truncated_text), border_style="dim", padding=(0, 1)))

    # Display elements in a table
    elements = state.get("elements", [])
    if elements:
        elements_table = Table(
            title=f"Interactive Elements ({len(elements)} found)",
            box=box.MINIMAL,
            show_header=True,
            padding=(0, 1),
            border_style="blue",
        )
        elements_table.add_column("ID", style="magenta", no_wrap=True)
        elements_table.add_column("Tag", style="cyan")
        elements_table.add_column("Role", style="yellow")
        elements_table.add_column(
            "Text Preview", style="white", max_width=60
        )  # Limit preview width
        elements_table.add_column("BBox", style="dim")

        preview_count = 15  # Show more elements
        for elem in elements[:preview_count]:
            elem_text = escape(
                truncate_text_by_lines(elem.get("text", ""), 5)
            )  # Limit text preview lines
            bbox = elem.get("bbox", ["?"] * 4)
            try:
                # Ensure bbox elements are convertible to string before joining
                bbox_str = f"({str(bbox[0])}x{str(bbox[1])}, {str(bbox[2])}w{str(bbox[3])}h)"
            except IndexError:
                bbox_str = "[Invalid Bbox]"

            elements_table.add_row(
                str(elem.get("id", "?")),  # Ensure ID is string
                str(elem.get("tag", "?")),  # Ensure tag is string
                str(elem.get("role", "")),  # Ensure role is string
                elem_text[:60] + ("..." if len(elem_text) > 60 else ""),  # Char limit
                bbox_str,
            )
        if len(elements) > preview_count:
            elements_table.add_row(
                "...", f"[dim]{len(elements) - preview_count} more...[/]", "", "", ""
            )

        panel_content.append("\n[bold cyan]Elements:[/bold cyan]")
        panel_content.append(elements_table)

    console.print(
        Panel(Group(*panel_content), title=title, border_style="blue", padding=(1, 2), expand=False)
    )


def display_result(
    title: str, result: Dict[str, Any], display_options: Optional[Dict] = None
) -> None:
    """Display operation result with enhanced formatting using Rich."""
    display_options = display_options or {}
    console.print(Rule(f"[bold cyan]{escape(title)}[/] {timestamp_str(short=True)}", style="cyan"))

    success = result.get("success", False)
    detail_level = display_options.get("detail_level", 1)
    hide_keys_set = set(
        display_options.get(
            "hide_keys",
            [
                "success",
                "page_state",
                "results",
                "steps",
                "download",
                "final_page_state",
                "documentation",
                "raw_response",  # Hide raw LLM responses by default
                "raw_llm_response",
            ],
        )
    )

    # --- Status Panel ---
    status_panel_content = Text(
        f"Status: {'[bold green]Success[/]' if success else '[bold red]Failed[/]'}\n", no_wrap=False
    )
    if not success:
        error_code = result.get("error_code", "N/A")
        error_msg = result.get("error", "Unknown error")
        status_panel_content.append(
            Text(f"Error Code: [yellow]{escape(str(error_code))}[/]\n")
        )  # Ensure string
        status_panel_content.append(
            Text(f"Message: [red]{escape(str(error_msg))}[/]\n")
        )  # Ensure string
        console.print(
            Panel(
                status_panel_content,
                title="Operation Status",
                border_style="red",
                padding=(1, 2),
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                status_panel_content,
                title="Operation Status",
                border_style="green",
                padding=(0, 1),
                expand=False,
            )
        )

    # --- Top Level Details ---
    details_table = Table(
        title="Result Summary", box=box.MINIMAL, show_header=False, padding=(0, 1)
    )
    details_table.add_column("Key", style="cyan", justify="right", no_wrap=True)
    details_table.add_column("Value", style="white")
    has_details = False
    for key, value in result.items():
        if key in hide_keys_set or key.startswith("_"):
            continue
        # Ensure key is a string before escaping
        details_table.add_row(escape(str(key)), format_value(key, value, detail_level=detail_level))
        has_details = True
    if has_details:
        console.print(details_table)

    # --- Special Section Displays ---

    # Page State
    if "page_state" in result and isinstance(result["page_state"], dict):
        display_page_state(result["page_state"], title="Page State After Action")
    elif "final_page_state" in result and isinstance(result["final_page_state"], dict):
        display_page_state(result["final_page_state"], title="Final Page State")

    # Search Results
    if "results" in result and isinstance(result["results"], list) and "query" in result:
        search_results = result["results"]
        search_table = Table(
            title=f"Search Results for '{escape(result['query'])}' ({len(search_results)} found)",
            box=box.ROUNDED,
            show_header=True,
            padding=(0, 1),
        )
        search_table.add_column("#", style="dim")
        search_table.add_column("Title", style="cyan")
        search_table.add_column("URL", style="blue", no_wrap=False)
        search_table.add_column("Snippet", style="white", no_wrap=False)
        for i, item in enumerate(search_results, 1):
            title = truncate_text_by_lines(item.get("title", ""), 3)
            snippet = truncate_text_by_lines(item.get("snippet", ""), 5)
            url = item.get("url", "")
            search_table.add_row(
                str(i), escape(title), f"[link={url}]{escape(url)}[/link]", escape(snippet)
            )
        console.print(search_table)

    # Download Result
    if "download" in result and isinstance(result["download"], dict):
        dl_info = result["download"]
        dl_table = Table(
            title="Download Details", box=box.MINIMAL, show_header=False, padding=(0, 1)
        )
        dl_table.add_column("Metric", style="cyan", justify="right")
        dl_table.add_column("Value", style="white")
        dl_table.add_row("File Path", escape(dl_info.get("file_path", "N/A")))
        dl_table.add_row("File Name", escape(dl_info.get("file_name", "N/A")))
        dl_table.add_row("SHA256", escape(dl_info.get("sha256", "N/A")))
        dl_table.add_row("Size", format_value("size_bytes", dl_info.get("size_bytes", 0)))
        dl_table.add_row("Source URL", escape(dl_info.get("url", "N/A")))
        dl_table.add_row(
            "Tables Extracted",
            format_value("tables_extracted", dl_info.get("tables_extracted", False)),
        )
        if dl_info.get("tables"):
            dl_table.add_row("Table Preview", format_value("tables", dl_info.get("tables")))
        console.print(
            Panel(dl_table, title="Download Result", border_style="green", padding=(1, 2))
        )

    # Macro/Autopilot Steps
    if "steps" in result and isinstance(result["steps"], list):
        steps = result["steps"]
        steps_table = Table(
            title=f"Macro/Autopilot Steps ({len(steps)} executed)",
            box=box.ROUNDED,
            show_header=True,
            padding=(0, 1),
        )
        steps_table.add_column("#", style="dim")
        steps_table.add_column("Action/Tool", style="cyan")
        steps_table.add_column("Arguments/Hint", style="white", no_wrap=False)
        steps_table.add_column("Status", style="yellow")
        steps_table.add_column("Result/Error", style="white", no_wrap=False)

        for i, step in enumerate(steps, 1):
            action = step.get("action", step.get("tool", "?"))
            args = step.get("args", step)  # Show full step if args missing
            args_preview = format_value("args", args, detail_level=0)
            success_step = step.get("success", False)  # Use different variable name
            status = "[green]OK[/]" if success_step else "[red]FAIL[/]"
            outcome = step.get("result", step.get("error", ""))
            outcome_preview = format_value("outcome", outcome, detail_level=0)
            steps_table.add_row(str(i), escape(action), args_preview, status, outcome_preview)
        console.print(steps_table)

    # Documentation
    if "documentation" in result and isinstance(result["documentation"], str):
        doc_content = result["documentation"]
        format_type = result.get("format", "markdown")
        content_to_display: Any = escape(doc_content)
        if format_type == "markdown":
            content_to_display = Markdown(doc_content)
        elif format_type == "json":
            try:
                parsed = json.loads(doc_content)
                content_to_display = Syntax(
                    json.dumps(parsed, indent=2),
                    "json",
                    theme="default",
                    line_numbers=False,
                    word_wrap=True,
                )
            except json.JSONDecodeError:
                content_to_display = Syntax(
                    doc_content, "text", theme="default", line_numbers=False, word_wrap=True
                )  # Fallback display
        console.print(
            Panel(
                content_to_display,
                title="Collected Documentation",
                border_style="magenta",
                padding=(1, 2),
            )
        )

    console.print()  # Add spacing


async def safe_tool_call(
    operation_name: str, tool_func: callable, *args, tracker: Optional[CostTracker] = None, **kwargs
) -> Tuple[bool, Dict[str, Any]]:
    """Safely call a tool function, handling exceptions and logging."""
    console.print(
        f"\n[cyan]Calling Tool:[/][bold] {escape(operation_name)}[/] {timestamp_str(short=True)}"
    )
    display_options = kwargs.pop("display_options", {})

    log_args_repr = {}
    MAX_ARG_LEN = 100
    for k, v in kwargs.items():
        if isinstance(v, (str, bytes)) and len(v) > MAX_ARG_LEN:
            log_args_repr[k] = f"{type(v).__name__}(len={len(v)})"
        elif isinstance(v, (list, dict)) and len(v) > 10:
            log_args_repr[k] = f"{type(v).__name__}(len={len(v)})"
        else:
            log_args_repr[k] = repr(v)
    logger.debug(f"Executing {operation_name} with kwargs: {log_args_repr}")

    try:
        result = await tool_func(*args, **kwargs)
        if not isinstance(result, dict):
            logger.error(f"Tool '{operation_name}' returned non-dict type: {type(result)}")
            return False, {
                "success": False,
                "error": f"Tool returned unexpected type: {type(result).__name__}",
                "error_code": "INTERNAL_ERROR",
                "_display_options": display_options,
            }

        result["_display_options"] = display_options
        logger.debug(f"Tool '{operation_name}' completed successfully.")
        return True, result
    except ToolInputError as e:
        logger.warning(f"Input error for {operation_name}: {e}")
        return False, {
            "success": False,
            "error": str(e),
            "error_code": e.error_code,
            "_display_options": display_options,
        }
    except ToolError as e:
        logger.error(f"Tool error during {operation_name}: {e}")
        return False, {
            "success": False,
            "error": str(e),
            "error_code": e.error_code,
            "_display_options": display_options,
        }
    except Exception as e:
        logger.error(f"Unexpected error during {operation_name}: {e}", exc_info=True)
        tb_str = traceback.format_exc(limit=1)
        return False, {
            "success": False,
            "error": f"{type(e).__name__}: {e}\n{tb_str}",
            "error_type": type(e).__name__,
            "error_code": "UNEXPECTED_ERROR",
            "_display_options": display_options,
        }


# --- Demo Sections (Reusing previous section functions) ---
# Demo section functions (demo_section_1_browse, etc.) are assumed to be defined
# as in the previous version.


async def demo_section_1_browse(tool: SmartBrowserTool, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 1: Basic Browsing[/]", style="green"))
    logger.info("Starting Demo Section 1: Basic Browsing")

    # 1a: Browse Example.com
    success, result = await safe_tool_call(
        "Browse Example.com", tool.browse_url, url=URL_EXAMPLE, tracker=tracker
    )
    display_result("Browse Example.com", result)

    # 1b: Browse Bookstore (wait for specific element)
    success, result = await safe_tool_call(
        "Browse Bookstore (wait for footer)",
        tool.browse_url,
        url=URL_BOOKSTORE,
        wait_for_selector="footer.footer",  # Example specific selector
        tracker=tracker,
    )
    display_result("Browse Bookstore (Wait)", result)


async def demo_section_2_interaction(tool: SmartBrowserTool, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 2: Page Interaction[/]", style="green"))
    logger.info("Starting Demo Section 2: Page Interaction")

    # 2a: Search on Bookstore
    console.print(f"--- Scenario: Search for 'Science' on {URL_BOOKSTORE} ---")
    success, initial_state_res = await safe_tool_call(
        "Load Bookstore Search Page", tool.browse_url, url=URL_BOOKSTORE, tracker=tracker
    )
    if not success:
        console.print("[red]Cannot proceed with interaction demo, failed to load page.[/]")
        return
    display_result("Bookstore Initial State", initial_state_res)

    # Fill the search form using task hints
    form_fields = [
        {"task_hint": "The search input field", "text": "Science"},
    ]
    success, fill_res = await safe_tool_call(
        "Fill Bookstore Search Form",
        tool.fill_form,
        url=URL_BOOKSTORE,
        form_fields=form_fields,
        submit_hint="The search button",
        wait_after_submit_ms=1500,
        tracker=tracker,
    )
    display_result("Fill Bookstore Search Form", fill_res)

    # 2b: Click the first search result (if successful)
    if success:
        console.print("--- Scenario: Click the first search result ---")
        current_url = fill_res.get("page_state", {}).get("url", URL_BOOKSTORE)

        success, click_res = await safe_tool_call(
            "Click First Book Result",
            tool.click_and_extract,
            url=current_url,
            task_hint="The link for the first book shown in the results list",
            wait_ms=1000,
            tracker=tracker,
        )
        display_result("Click First Book Result", click_res)


async def demo_section_3_search(tool: SmartBrowserTool, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 3: Web Search[/]", style="green"))
    logger.info("Starting Demo Section 3: Web Search")

    search_query = "latest advancements in large language models"

    # 3a: Search Bing
    success, result = await safe_tool_call(
        "Search Bing",
        tool.search,
        query=search_query,
        engine="bing",
        max_results=5,
        tracker=tracker,
    )
    display_result(f"Search Bing: '{search_query}'", result)

    # 3b: Search DuckDuckGo
    success, result = await safe_tool_call(
        "Search DuckDuckGo",
        tool.search,
        query=search_query,
        engine="duckduckgo",
        max_results=5,
        tracker=tracker,
    )
    display_result(f"Search DuckDuckGo: '{search_query}'", result)


async def demo_section_4_download(tool: SmartBrowserTool, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 4: File Download[/]", style="green"))
    logger.info("Starting Demo Section 4: File Download")

    # 4a: Download PDFs from a site (e.g., find PDFs on example.com - likely none)
    console.print("--- Scenario: Find and Download PDFs from Example.com ---")
    success, result = await safe_tool_call(
        "Download PDFs from Example.com",
        tool.download_site_pdfs,
        start_url=URL_EXAMPLE,
        max_depth=1,
        max_pdfs=5,
        dest_subfolder="example_com_pdfs",
        tracker=tracker,
    )
    display_result("Download PDFs from Example.com", result)
    if result.get("pdf_count", 0) == 0:
        console.print("[yellow]Note: No PDFs found on example.com as expected.[/]")

    # 4b: Click-based download (Requires a page with a clear download link)
    # Create a simple local HTML file for this purpose
    download_page_content = f"""
    <!DOCTYPE html>
    <html><head><title>Download Test</title></head>
    <body><h1>Download Page</h1>
    <p>Click the link to download a dummy PDF.</p>
    <a href="{URL_PDF_SAMPLE}" id="downloadLink">Download Dummy PDF</a>
    <p>Another paragraph.</p>
    </body></html>
    """
    download_page_path = DEMO_OUTPUTS_DIR / "download_test.html"
    try:
        download_page_path.write_text(download_page_content, encoding="utf-8")
        local_url = download_page_path.as_uri()  # Get file:// URL

        console.print("\n--- Scenario: Click a link to download a file ---")
        success, result = await safe_tool_call(
            "Click to Download PDF",
            tool.download_file,
            url=local_url,
            task_hint="The 'Download Dummy PDF' link",
            dest_dir=str(DEMO_OUTPUTS_DIR / "clicked_downloads"),
            tracker=tracker,
        )
        display_result("Click to Download PDF", result)
    except Exception as e:
        console.print(f"[red]Error setting up or running click-download demo: {e}[/]")
    finally:
        # Clean up the temporary HTML file
        if download_page_path.exists():
            try:
                download_page_path.unlink()
            except OSError:
                pass


async def demo_section_5_macro(tool: SmartBrowserTool, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 5: Execute Macro[/]", style="green"))
    logger.info("Starting Demo Section 5: Execute Macro")

    macro_task = f"Go to {URL_BOOKSTORE}, search for 'Travel', and click the first book result."
    console.print("--- Scenario: Execute Macro ---")
    console.print(f"[italic]Task:[/italic] {macro_task}")

    success, result = await safe_tool_call(
        "Execute Bookstore Search Macro",
        tool.execute_macro,
        url=URL_BOOKSTORE,
        task=macro_task,
        max_rounds=5,
        tracker=tracker,
    )
    display_result("Execute Bookstore Search Macro", result)


async def demo_section_6_autopilot(tool: SmartBrowserTool, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 6: Autopilot[/]", style="green"))
    logger.info("Starting Demo Section 6: Autopilot")

    autopilot_task = f"Find the price of the book 'A Light in the Attic' on {URL_BOOKSTORE}"
    console.print("--- Scenario: Autopilot ---")
    console.print(f"[italic]Task:[/italic] {autopilot_task}")

    success, result = await safe_tool_call(
        "Run Autopilot: Find Book Price",
        tool.autopilot,
        task=autopilot_task,
        max_steps=8,
        scratch_subdir="autopilot_demo_runs",
        tracker=tracker,
    )
    display_result("Run Autopilot: Find Book Price", result)
    if result.get("run_log"):
        console.print(f"[dim]Autopilot run log saved to: {result['run_log']}[/]")


async def demo_section_7_parallel(tool: SmartBrowserTool, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 7: Parallel Processing[/]", style="green"))
    logger.info("Starting Demo Section 7: Parallel Processing")

    urls_to_process = [
        URL_EXAMPLE,
        URL_BOOKSTORE,
        URL_QUOTES,
        "http://httpbin.org/delay/1",
        "https://webscraper.io/test-sites/e-commerce/static",
    ]
    console.print("--- Scenario: Get Page State for Multiple URLs in Parallel ---")
    console.print(f"[dim]URLs:[/dim] {urls_to_process}")

    success, result = await safe_tool_call(
        "Parallel Get Page State",
        tool.parallel_process,
        urls=urls_to_process,
        action="get_state",
        # max_tabs=3 # Override tool default if needed
        tracker=tracker,
    )

    # Custom display for parallel results
    console.print(Rule("[bold cyan]Parallel Processing Results[/]", style="cyan"))
    if success:
        console.print(f"Total URLs Processed: {result.get('processed_count', 0)}")
        console.print(f"Successful: {result.get('successful_count', 0)}")
        console.print("-" * 20)
        for i, item_result in enumerate(result.get("results", [])):
            url = item_result.get("url", f"URL {i + 1}")
            item_success = item_result.get("success", False)
            panel_title = f"Result for: {escape(url)}"
            border = "green" if item_success else "red"
            content = ""
            if item_success:
                state = item_result.get("page_state", {})
                content = f"Title: {escape(state.get('title', 'N/A'))}\nElements Found: {len(state.get('elements', []))}"
            else:
                content = f"[red]Error:[/red] {escape(item_result.get('error', 'Unknown'))}"
            console.print(
                Panel(content, title=panel_title, border_style=border, padding=(0, 1), expand=False)
            )
    else:
        console.print(
            Panel(
                f"[red]Parallel processing tool call failed:[/red]\n{escape(result.get('error', '?'))}",
                border_style="red",
            )
        )

    console.print()


async def demo_section_8_docs(tool: SmartBrowserTool, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 8: Documentation Collection[/]", style="green"))
    logger.info("Starting Demo Section 8: Documentation Collection")

    # 8a: Collect docs for a known package (e.g., requests)
    package_name = "requests"
    console.print(f"--- Scenario: Collect Documentation for '{package_name}' ---")

    success, result = await safe_tool_call(
        f"Collect Docs: {package_name}",
        tool.collect_documentation,
        package=package_name,
        max_pages=15,  # Limit pages for demo
        rate_limit_rps=2.0,
        tracker=tracker,
    )
    display_result(f"Collect Docs: {package_name}", result)
    if result.get("file_path"):
        console.print(f"[dim]Collected documentation saved to: {result['file_path']}[/]")
        try:
            with open(result["file_path"], "r", encoding="utf-8") as f:
                content_preview = f.read(1000)
            console.print(
                Panel(
                    escape(content_preview) + "\n[dim]... (file preview) ...[/]",
                    title="File Preview",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
        except Exception as e:
            console.print(f"[yellow]Could not read preview of doc file: {e}[/]")

    console.print()


# --- Main Function ---
async def main() -> int:
    """Run the SmartBrowserTool demo."""
    console.print(Rule("[bold magenta]Smart Browser Tool Demo[/bold magenta]"))

    if not MCP_COMPONENTS_LOADED:
        console.print(
            Panel(
                f"[bold red]Error: Failed to import required MCP Server components.[/]\n"
                f"Import Error: {_IMPORT_ERROR_MSG}\n\n"
                f"Please ensure the script is run from within the MCP Server environment or "
                f"that the necessary packages are installed and paths are configured correctly.",
                title="Initialization Error",
                border_style="red",
            )
        )
        return 1

    exit_code = 0
    gateway: Optional[Gateway] = None  # Use the actual Gateway type
    tool_instance: Optional[SmartBrowserTool] = None

    # Ensure output directory exists
    DEMO_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Demo outputs will be saved in: {DEMO_OUTPUTS_DIR}[/]")

    try:
        # --- CORRECTED: Use the actual Gateway ---
        console.print("[cyan]Initializing MCP Gateway...[/]")
        # Register tools=False since we instantiate manually
        gateway = Gateway("smart-browser-demo", register_tools=False)

        # Initialize providers (important for LLM calls within the tool)
        console.print("[cyan]Initializing Providers...[/]")
        # Use the internal method as done in the DocProc demo
        await gateway._initialize_providers()
        # -----------------------------------------

        # Create tool instance, passing the gateway
        console.print("[cyan]Initializing SmartBrowserTool...[/]")
        tool_instance = SmartBrowserTool(gateway)  # Pass the real gateway

        # Call the tool's setup (now handled by _ensure_initialized on first use, but explicit call is fine too)
        # await tool_instance.async_setup() # Explicit setup call

        # Initialize CostTracker (though likely unused by SmartBrowser)
        tracker = CostTracker()

        # Run Demo Sections
        await demo_section_1_browse(tool_instance, tracker)
        await demo_section_2_interaction(tool_instance, tracker)
        await demo_section_3_search(tool_instance, tracker)
        await demo_section_4_download(tool_instance, tracker)
        await demo_section_5_macro(tool_instance, tracker)
        # Autopilot can be slow and resource-intensive, maybe skip by default?
        # await demo_section_6_autopilot(tool_instance, tracker)
        console.print(
            "[yellow]Skipping Autopilot demo section (can be intensive). Uncomment to run.[/]"
        )
        await demo_section_7_parallel(tool_instance, tracker)
        await demo_section_8_docs(tool_instance, tracker)

        console.print(Rule("[bold magenta]Demo Complete[/bold magenta]"))

    except Exception as e:
        logger.critical(f"Demo failed with critical error: {e}", exc_info=True)
        console.print("[bold red]CRITICAL ERROR DURING DEMO:[/]")
        console.print_exception(show_locals=True)  # Use Rich's exception printing
        exit_code = 1
    finally:
        # Ensure graceful shutdown of the browser tool via its teardown method
        if tool_instance:
            console.print("\n[cyan]Shutting down Smart Browser Tool...[/]")
            # Call the tool's specific teardown method
            await tool_instance.async_teardown()
            console.print("[green]Smart Browser Tool shut down.[/]")
        # Gateway doesn't have an explicit shutdown in other demos,
        # but tool teardown should handle browser closing.

    return exit_code


if __name__ == "__main__":
    # Added basic logging setup for demo visibility
    import logging

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(
        os.environ.get("LOG_LEVEL", "INFO").upper()
    )  # Ensure demo logger respects level

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
