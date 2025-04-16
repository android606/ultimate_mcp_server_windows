#!/usr/bin/env python
"""Browser automation demonstration using LLM Gateway's Playwright tools."""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

# --- Import Instruction Packs ---
from examples.web_automation_instruction_packs import (
    ACADEMIC_PAPER_INSTRUCTIONS,
    COMPETITIVE_ANALYSIS_INSTRUCTIONS,
    CONTACT_FORM_WORKFLOW_INSTRUCTIONS,
    ECOMMERCE_PRODUCT_EXTRACTION_INSTRUCTIONS,
    GOVERNMENT_REPORT_INSTRUCTIONS,
    JOB_POSTING_EXTRACTION_INSTRUCTIONS,
    LEGAL_DOCUMENT_INSTRUCTIONS,
    MARKET_TREND_RESEARCH_INSTRUCTIONS,
    ORDER_STATUS_WORKFLOW_INSTRUCTIONS,
    PRODUCT_MANUAL_INSTRUCTIONS,
    PRODUCT_MONITORING_INSTRUCTIONS,
    SIMPLE_SEARCH_SUMMARY_INSTRUCTIONS,
    TECHNICAL_SEARCH_SUMMARY_INSTRUCTIONS,
    WEBSITE_SECTION_MONITORING_INSTRUCTIONS,
)
from llm_gateway.constants import TaskType
from llm_gateway.tools.browser_automation import (
    browser_checkbox,
    browser_click,
    browser_close,
    browser_execute_javascript,
    browser_get_console_logs,
    browser_get_text,
    browser_init,
    browser_navigate,
    browser_pdf,
    browser_screenshot,
    browser_select,
    browser_tab_close,
    browser_tab_list,
    browser_tab_new,
    browser_tab_select,
    browser_type,
    browser_upload_file,
    browser_wait,
    execute_web_workflow,
    extract_structured_data_from_pages,
    find_and_download_pdfs,
    monitor_web_data_points,
    multi_engine_search_summary,
    research_and_synthesize_report,
)
from llm_gateway.utils import get_logger
from llm_gateway.utils.logging.console import console

logger = get_logger("example.browser_automation")

# Config
DEMO_SITES = {
    "wikipedia": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "search_engine": "https://www.google.com",
    "form_demo": "https://www.selenium.dev/selenium/web/web-form.html",
    "dynamic_demo": "https://www.selenium.dev/selenium/web/dynamic.html"
}

SAVE_DIR = Path("./browser_demo_outputs")

# Import TaskType

# Add a class to track demo session information for reporting
class DemoSession:
    """Track information about demo session for reporting."""
    
    def __init__(self):
        self.actions = []
        self.start_time = time.time()
        self.end_time = None
        self.screenshots = {}
        self.results = {}
        self.demo_stats = {}
    
    def add_action(self, action_type: str, description: str, result: Dict[str, Any], 
                  screenshots: Optional[Dict[str, str]] = None, 
                  time_taken: Optional[float] = None):
        """Add an action to the session log."""
        action = {
            "type": action_type,
            "description": description,
            "result": result,
            "timestamp": time.time(),
            "time_taken": time_taken,
            "screenshots": screenshots or {}
        }
        self.actions.append(action)
        
    def add_screenshot(self, name: str, path: str):
        """Add a screenshot to the session."""
        self.screenshots[name] = path
    
    def add_demo_stats(self, demo_name: str, stats: Dict[str, Any]):
        """Add statistics for a specific demo."""
        self.demo_stats[demo_name] = stats
    
    def finish(self):
        """Mark the session as complete."""
        self.end_time = time.time()
        
    @property
    def total_duration(self) -> float:
        """Get total session duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

# Initialize global session tracking
demo_session = DemoSession()

def display_result(title: str, result: Dict[str, Any], include_snapshot: bool = False):
    """Display a browser tool result with consistent formatting."""
    console.print(Rule(f"[bold blue]{escape(title)}[/bold blue]"))
    
    # Basic metrics and stats
    metrics_table = Table(box=box.SIMPLE, show_header=False)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="white")
    
    # Success status
    success = result.get("success", False)
    status_text = "[green]Success[/green]" if success else "[red]Failed[/red]"
    metrics_table.add_row("Status", status_text)
    
    # Processing time
    if "processing_time" in result:
        metrics_table.add_row("Processing Time", f"{result['processing_time']:.3f}s")
    
    # Handle different types of results based on the tool
    if "url" in result:
        metrics_table.add_row("URL", result["url"])
    if "title" in result:
        metrics_table.add_row("Title", result["title"])
    if "status" in result and result["status"] is not None:
        metrics_table.add_row("HTTP Status", str(result["status"]))
    if "tab_id" in result:
        metrics_table.add_row("Tab ID", result["tab_id"])
    if "element_description" in result:
        metrics_table.add_row("Element", result["element_description"])
    if "text" in result and result["text"] is not None:
        if len(result["text"]) > 100:
            # For long text, show a preview and then the full text in a panel
            metrics_table.add_row("Text Preview", f"{result['text'][:100]}...")
        else:
            metrics_table.add_row("Text", result["text"])
    if "file_path" in result:
        metrics_table.add_row("File Path", result["file_path"])
    if "file_name" in result:
        metrics_table.add_row("File Name", result["file_name"])
    if "file_size" in result and result["file_size"]:
        metrics_table.add_row("File Size", f"{result['file_size'] / 1024:.2f} KB")
            
    console.print(metrics_table)
    
    # Show error if present
    if "error" in result and result["error"]:
        console.print(Panel(
            f"[red]{escape(str(result['error']))}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red"
        ))
    
    # Show text content if applicable and not already shown in metrics
    if "text" in result and result["text"] is not None and len(result["text"]) > 100:
        console.print(Panel(
            escape(result["text"]),
            title="[bold green]Text Content[/bold green]",
            border_style="green"
        ))
    
    # Show snapshot data if requested and available
    if include_snapshot and "snapshot" in result and result["snapshot"]:
        snapshot = result["snapshot"]
        console.print("[cyan]Page Snapshot:[/cyan]")
        # Display URL and title
        snapshot_table = Table(box=box.SIMPLE, show_header=False)
        snapshot_table.add_column("Property", style="cyan")
        snapshot_table.add_column("Value", style="white")
        snapshot_table.add_row("URL", snapshot.get("url", "N/A"))
        snapshot_table.add_row("Title", snapshot.get("title", "N/A"))
        console.print(snapshot_table)

        # Show accessibility tree structure preview (compact version)
        if "tree" in snapshot:
            tree_preview = _format_accessibility_tree(snapshot["tree"], max_depth=2)
            console.print(Panel(
                tree_preview,
                title="[bold]Accessibility Tree Preview[/bold]",
                border_style="dim blue",
                width=100
            ))
    
    # Show data preview for special result types
    if "data" in result and result["data"]:  # Screenshot data
        console.print(Panel(
            "[dim]Base64 image data (truncated):[/dim] " + result["data"][:50] + "...",
            title="[bold]Screenshot Data[/bold]",
            border_style="dim blue"
        ))
    
    if "result" in result:  # JavaScript execution result
        if isinstance(result["result"], dict):
            try:
                # Format as JSON for dict results
                import json
                js_result = json.dumps(result["result"], indent=2)
                console.print(Panel(
                    Syntax(js_result, "json", theme="monokai", line_numbers=True),
                    title="[bold green]JavaScript Result[/bold green]",
                    border_style="green"
                ))
            except Exception:
                # Fallback for non-serializable results
                console.print(Panel(
                    str(result["result"]),
                    title="[bold green]JavaScript Result[/bold green]",
                    border_style="green"
                ))
        else:
            # Plain display for non-dict results
            console.print(Panel(
                str(result["result"]),
                title="[bold green]JavaScript Result[/bold green]",
                border_style="green"
            ))

    if "logs" in result and result["logs"]:  # Console logs
        logs_table = Table(title="Console Logs", box=box.SIMPLE)
        logs_table.add_column("Type", style="cyan")
        logs_table.add_column("Message", style="white")
        
        for log in result["logs"]:
            log_type = log.get("type", "log")
            log_style = "red" if log_type == "error" else "yellow" if log_type == "warning" else "green"
            logs_table.add_row(
                f"[{log_style}]{log_type}[/{log_style}]",
                escape(log.get("text", ""))
            )
        
        console.print(logs_table)
    
    console.print()  # Add space after result


def _format_accessibility_tree(tree: Dict[str, Any], level: int = 0, max_depth: int = 3) -> str:
    """Format accessibility tree data for display, limiting depth."""
    if not tree or level > max_depth:
        return ""
    
    indent = "  " * level
    name = tree.get("name", "")
    role = tree.get("role", "")
    
    # Truncate long names
    if len(name) > 30:
        name = name[:27] + "..."
    
    # Format node with its role and name
    result = f"{indent}[cyan]{role}[/cyan]: {escape(name)}\n"
    
    # Add special properties if present
    properties = []
    if "value" in tree and tree["value"]:
        properties.append(f"value='{tree['value']}'")
    if "checked" in tree:
        properties.append(f"checked={str(tree['checked']).lower()}")
    if "disabled" in tree and tree["disabled"]:
        properties.append("disabled")
    if "expanded" in tree:
        properties.append(f"expanded={str(tree['expanded']).lower()}")
    
    if properties:
        result += f"{indent}  [dim]{', '.join(properties)}[/dim]\n"
    
    # Include children if we're not at max depth
    if level < max_depth and "children" in tree and tree["children"]:
        # For the last level before max_depth, show count instead of recursing further
        if level == max_depth - 1 and len(tree["children"]) > 2:
            result += f"{indent}  [dim]... {len(tree['children'])} child elements ...[/dim]\n"
        else:
            # Show up to 3 children at deep levels
            children_to_show = tree["children"][:3] if level >= max_depth - 1 else tree["children"]
            for child in children_to_show:
                result += _format_accessibility_tree(child, level + 1, max_depth)
            
            # Indicate if children were truncated
            if len(tree["children"]) > 3 and level >= max_depth - 1:
                result += f"{indent}  [dim]... {len(tree['children']) - 3} more elements ...[/dim]\n"
    
    return result


def setup_demo():
    """Create directories needed for the demo."""
    SAVE_DIR.mkdir(exist_ok=True)
    reports_dir = SAVE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    logger.info(f"Created output directories: {SAVE_DIR}", emoji_key="setup")


def create_demo_progress_tracker() -> Tuple[Progress, TaskID]:
    """Create a rich progress bar for tracking demo steps."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[cyan]{task.completed}/{task.total}[/cyan] steps"),
    )
    
    task_id = progress.add_task("[bold cyan]Running demo...", total=0)
    return progress, task_id


async def demo_browser_initialization():
    """Demonstrate browser initialization and basic properties."""
    console.print(Rule("[bold blue]Browser Initialization Demo[/bold blue]"))
    logger.info("Starting browser initialization", emoji_key="start")
    
    # Initialize browser in non-headless mode so users can see it
    result = await browser_init(
        browser_name="chromium",
        headless=False,
        default_timeout=30000
    )
    
    display_result("Browser Initialized", result)
    
    return result


async def demo_navigation_basics(progress=None, task_id=None):
    """Demonstrate basic navigation and page interaction."""
    console.print(Rule("[bold blue]Navigation Basics Demo[/bold blue]"))
    logger.info("Demonstrating basic navigation", emoji_key="navigation")
    
    # If no progress tracker is provided, create one
    own_progress = False
    if progress is None or task_id is None:
        progress, task_id = create_demo_progress_tracker()
        progress.update(task_id, total=4, description="[bold cyan]Navigation demo starting...[/bold cyan]")
        own_progress = True
    
    demo_start_time = time.time()  # noqa: F841
    demo_actions = 0
    
    if own_progress:
        with progress:
            result = await _run_navigation_steps(progress, task_id, demo_actions)
    else:
        result = await _run_navigation_steps(progress, task_id, demo_actions)
    
    return result

def _get_navigation_results(result, screenshot_result, text_result, click_result, demo_duration):
    return {
        "navigation": result,
        "screenshot": screenshot_result,
        "text": text_result,
        "click": click_result,
        "duration": demo_duration
    }

async def _run_navigation_steps(progress, task_id, demo_actions):
    # Navigate to Wikipedia page on AI
    progress.update(task_id, description="[cyan]Navigating to Wikipedia AI page...[/cyan]", advance=0)
    result = await browser_navigate(
        url=DEMO_SITES["wikipedia"],
        wait_until="load",
        timeout=30000,
        capture_snapshot=True
    )
    display_result("Navigated to Wikipedia AI Page", result, include_snapshot=True)
    demo_session.add_action("navigation", "Navigated to Wikipedia AI Page", result)
    demo_actions += 1
    progress.update(task_id, advance=1)
    # Take a screenshot of the page
    progress.update(task_id, description="[cyan]Taking screenshot...[/cyan]")
    screenshot_result = await browser_screenshot(
        full_page=False,
        quality=80
    )
    screenshot_path = None
    if screenshot_result.get("data"):
        import base64
        screenshot_path = SAVE_DIR / "wikipedia_ai_screenshot.jpg"
        try:
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(screenshot_result["data"]))
            screenshot_result["file_path"] = str(screenshot_path)
            screenshot_result["file_name"] = screenshot_path.name
            logger.success(f"Screenshot saved to {screenshot_path}", emoji_key="file")
            demo_session.add_screenshot("Wikipedia AI Page", str(screenshot_path))
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}", emoji_key="error")
    display_result("Page Screenshot", screenshot_result)
    demo_session.add_action("screenshot", "Wikipedia AI Page Screenshot", screenshot_result, 
                           screenshots={"Wikipedia AI": str(screenshot_path)} if screenshot_path else None)
    demo_actions += 1
    progress.update(task_id, advance=1)
    # Get text from a specific element (Wikipedia article lead paragraph)
    progress.update(task_id, description="[cyan]Extracting text content...[/cyan]")
    text_result = await browser_get_text(
        selector="div.mw-parser-output > p:nth-child(4)"
    )
    display_result("Article Lead Paragraph", text_result)
    demo_session.add_action("get_text", "Wikipedia AI Lead Paragraph", text_result)
    demo_actions += 1
    progress.update(task_id, advance=1)
    # Click on a link (e.g., the Machine Learning link)
    progress.update(task_id, description="[cyan]Clicking on Machine Learning link...[/cyan]")
    click_result = await browser_click(
        selector="a[title='Machine learning']",
        capture_snapshot=True
    )
    display_result("Clicked on Machine Learning Link", click_result, include_snapshot=True)
    demo_session.add_action("click", "Clicked on Machine Learning Link", click_result)
    demo_actions += 1
    progress.update(task_id, advance=1, description="[bold green]Navigation demo completed![/bold green]")
    # Record demo stats
    demo_duration = time.time() - progress.tasks[task_id].start_time if hasattr(progress.tasks[task_id], 'start_time') else time.time()
    demo_session.add_demo_stats("Navigation Basics", {
        "duration": demo_duration,
        "actions": demo_actions,
        "success": True
    })
    return _get_navigation_results(result, screenshot_result, text_result, click_result, demo_duration)


async def demo_form_interaction():
    """Demonstrate form interactions: typing, selecting, clicking checkboxes."""
    console.print(Rule("[bold blue]Form Interaction Demo[/bold blue]"))
    logger.info("Demonstrating form interactions", emoji_key="form")
    
    # Navigate to the Selenium test form
    result = await browser_navigate(
        url=DEMO_SITES["form_demo"],
        wait_until="load"
    )
    
    display_result("Navigated to Test Form", result)
    
    # Fill in a text field
    text_input_result = await browser_type(
        selector="input[name='my-text']",
        text="Hello from LLM Gateway Browser Automation!",
        delay=10  # Small delay for visibility
    )
    
    display_result("Entered Text in Input Field", text_input_result)
    
    # Select an option from a dropdown
    select_result = await browser_select(
        selector="select[name='my-select']",
        values="Three",
        by="label"
    )
    
    display_result("Selected Dropdown Option", select_result)
    
    # Check a checkbox
    checkbox_result = await browser_checkbox(
        selector="input[name='my-check']",
        check=True
    )
    
    display_result("Checked Checkbox", checkbox_result)
    
    # Fill a password field
    password_result = await browser_type(
        selector="input[name='my-password']",
        text="SecurePassword123",
        delay=10
    )
    
    display_result("Entered Password", password_result)
    
    # Submit the form by clicking the submit button
    submit_result = await browser_click(
        selector="button[type='submit']",
        capture_snapshot=True
    )
    
    display_result("Submitted Form", submit_result, include_snapshot=True)
    
    # Take a screenshot of the result page
    screenshot_result = await browser_screenshot(
        full_page=True,
        quality=90
    )
    
    # Save the screenshot
    if screenshot_result.get("data"):
        import base64
        screenshot_path = SAVE_DIR / "form_submission_result.jpg"
        try:
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(screenshot_result["data"]))
            screenshot_result["file_path"] = str(screenshot_path)
            screenshot_result["file_name"] = screenshot_path.name
            logger.success(f"Form result screenshot saved to {screenshot_path}", emoji_key="file")
        except Exception as e:
            logger.error(f"Failed to save form result screenshot: {e}", emoji_key="error")
    
    display_result("Form Submission Result Screenshot", screenshot_result)
    
    return {
        "navigation": result,
        "text_input": text_input_result,
        "select": select_result,
        "checkbox": checkbox_result,
        "password": password_result,
        "submit": submit_result,
        "screenshot": screenshot_result
    }


async def demo_javascript_execution():
    """Demonstrate JavaScript execution in the browser."""
    console.print(Rule("[bold blue]JavaScript Execution Demo[/bold blue]"))
    logger.info("Demonstrating JavaScript execution", emoji_key="javascript")
    
    # Navigate to a dynamic test page
    result = await browser_navigate(
        url=DEMO_SITES["dynamic_demo"],
        wait_until="load"
    )
    
    display_result("Navigated to Dynamic Test Page", result)
    
    # Execute JavaScript to extract metadata about the page
    js_result = await browser_execute_javascript(
        script="""() => {
            // Get basic page info
            const basicInfo = {
                title: document.title,
                url: location.href,
                domain: location.hostname,
                path: location.pathname,
                protocol: location.protocol,
                cookies: navigator.cookieEnabled,
                viewport: {
                    width: window.innerWidth,
                    height: window.innerHeight
                }
            };
            
            // Get meta tags
            const metaTags = Array.from(document.querySelectorAll('meta')).map(meta => {
                const attrs = {};
                Array.from(meta.attributes).forEach(attr => {
                    attrs[attr.name] = attr.value;
                });
                return attrs;
            });
            
            // Get all links
            const links = Array.from(document.querySelectorAll('a')).map(a => {
                return {
                    text: a.textContent.trim(),
                    href: a.href,
                    target: a.target || "_self"
                };
            });
            
            // Count elements by tag name
            const elementCounts = {};
            const tags = ['div', 'p', 'span', 'img', 'a', 'ul', 'ol', 'li', 'table', 'form', 'input', 'button', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
            tags.forEach(tag => {
                const count = document.getElementsByTagName(tag).length;
                if (count > 0) elementCounts[tag] = count;
            });
            
            return {
                basicInfo,
                metaTags,
                links,
                elementCounts
            };
        }"""
    )
    
    display_result("JavaScript Page Analysis", js_result)
    
    # Execute JavaScript to modify the page content
    modify_js_result = await browser_execute_javascript(
        script="""() => {
            // Change the page title
            document.title = "Modified by LLM Gateway Browser Automation";
            
            // Create a new styled element
            const banner = document.createElement('div');
            banner.style.backgroundColor = '#4CAF50';
            banner.style.color = 'white';
            banner.style.padding = '15px';
            banner.style.position = 'fixed';
            banner.style.top = '0';
            banner.style.left = '0';
            banner.style.width = '100%';
            banner.style.textAlign = 'center';
            banner.style.fontWeight = 'bold';
            banner.style.zIndex = '1000';
            banner.textContent = 'This page was modified by LLM Gateway Browser Automation!';
            
            // Add it to the page
            document.body.insertBefore(banner, document.body.firstChild);
            
            // Modify some existing content
            const paragraphs = document.querySelectorAll('p');
            if (paragraphs.length > 0) {
                paragraphs.forEach(p => {
                    p.style.color = '#2196F3';
                    p.style.fontWeight = 'bold';
                });
            }
            
            return {
                title: document.title,
                elementsModified: paragraphs.length,
                success: true
            };
        }"""
    )
    
    display_result("Modified Page with JavaScript", modify_js_result)
    
    # Take a screenshot to show the modifications
    screenshot_result = await browser_screenshot(
        full_page=True,
        quality=90
    )
    
    # Save the screenshot
    if screenshot_result.get("data"):
        import base64
        screenshot_path = SAVE_DIR / "js_modified_page.jpg"
        try:
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(screenshot_result["data"]))
            screenshot_result["file_path"] = str(screenshot_path)
            screenshot_result["file_name"] = screenshot_path.name
            logger.success(f"JavaScript modified page screenshot saved to {screenshot_path}", emoji_key="file")
        except Exception as e:
            logger.error(f"Failed to save JavaScript modified page screenshot: {e}", emoji_key="error")
    
    display_result("Modified Page Screenshot", screenshot_result)
    
    # Get console logs to see if our JavaScript produced any output
    logs_result = await browser_get_console_logs()
    
    display_result("Browser Console Logs", logs_result)
    
    return {
        "navigation": result,
        "js_analysis": js_result,
        "js_modification": modify_js_result,
        "screenshot": screenshot_result,
        "console_logs": logs_result
    }


async def demo_search_interaction():
    """Demonstrate a more complex interaction like performing a search."""
    console.print(Rule("[bold blue]Search Interaction Demo[/bold blue]"))
    logger.info("Demonstrating search interaction", emoji_key="search")
    
    # Navigate to DuckDuckGo
    result = await browser_navigate(
        url="https://duckduckgo.com", # <-- FIX: Changed URL
        wait_until="load",
        timeout=30000
    )
    
    display_result("Navigated to Search Engine", result)
    
    # Enter a search query
    search_query = "LLM Gateway Browser Automation"
    
    print("[cyan]Searching...[/cyan]")
    try:
        # Try different selectors for DuckDuckGo's search input
        search_selectors = [
            "#search_form_input_homepage", # DDG homepage input
            "input[name='q']"            # General query input
        ]
        
        search_successful = False
        for selector in search_selectors:
            try:
                # Directly try to type, relying on Playwright's ability to find the element
                type_result = await browser_type(  # noqa: F841
                    selector=selector,
                    text=search_query,
                    delay=20,  # Slow typing for visibility
                    press_enter=True  # Press Enter to submit search
                )
                search_successful = True
                logger.info(f"Successfully used selector '{selector}' for search input.", emoji_key="success")
                break # Exit loop once successful
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}", emoji_key="debug")
                continue
                
        if not search_successful:
            logger.warning("Could not find search input element using standard selectors. Trying JavaScript input approach.", emoji_key="warning")
            # Fallback: Use JavaScript to find and fill the search box
            js_search_result = await browser_execute_javascript(
                script=f"""() => {{
                    const searchInput = document.querySelector('input[name="q"]'); // <-- FIX: DDG selector
                    if (searchInput) {{
                        searchInput.value = "{search_query}";
                        const form = searchInput.closest('form');
                        if (form) form.submit();
                        return {{ success: true, method: "js-submit" }};
                    }}
                    return {{ success: false, error: "Could not find search input" }};
                }}"""
            )
            
            if not js_search_result.get("result", {}).get("success", False):
                logger.error("All methods to interact with search failed", emoji_key="error")
                raise Exception("Failed to interact with search input")
        
        # Wait for search results to load
        await browser_wait(
            wait_type="selector",
            value="#links",  # <-- FIX: DuckDuckGo's search results container
            timeout=20000 # <-- Increased timeout
        )
    finally:
        pass  # No progress.update needed
    
    # Take a screenshot of search results
    screenshot_result = await browser_screenshot(
        full_page=False,
        quality=80
    )
    
    # Save the screenshot
    if screenshot_result.get("data"):
        import base64
        screenshot_path = SAVE_DIR / "search_results.jpg"
        try:
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(screenshot_result["data"]))
            screenshot_result["file_path"] = str(screenshot_path)
            screenshot_result["file_name"] = screenshot_path.name
            logger.success(f"Search results screenshot saved to {screenshot_path}", emoji_key="file")
        except Exception as e:
            logger.error(f"Failed to save search results screenshot: {e}", emoji_key="error")
    
    display_result("Search Results Screenshot", screenshot_result)
    
    # Extract search results using JavaScript
    search_results_js = await browser_execute_javascript(
        script="""() => {
            // Extract search results
            const results = [];
            const resultElements = document.querySelectorAll('article.result'); // <-- FIX: DDG result selector
            
            resultElements.forEach((result, index) => {
                // Process only the first 5 results
                if (index >= 5) return;
                
                // Extract components of a result
                const linkElement = result.querySelector('a.result__a'); // <-- FIX: DDG link/title selector
                const snippetElement = result.querySelector('.result__snippet'); // <-- FIX: DDG snippet selector
                
                if (linkElement) { // Title might be within linkElement
                    results.push({
                        title: linkElement.textContent.trim(), // <-- FIX: Get text from link
                        url: linkElement.href,
                        snippet: snippetElement ? snippetElement.textContent.trim() : null
                    });
                }
            });
            
            return {
                count: results.length,
                results: results
            };
        }"""
    )
    
    display_result("Extracted Search Results", search_results_js)
    
    # Generate a PDF of the search results page
    pdf_result = await browser_pdf(
        full_page=True,
        save_path=str(SAVE_DIR),
        filename="search_results.pdf",
        landscape=False
    )
    
    display_result("Search Results PDF", pdf_result)
    
    return {
        "navigation": result,
        "screenshot": screenshot_result,
        "extracted_results": search_results_js,
        "pdf": pdf_result
    }


async def demo_tab_management():
    """Demonstrate tab management and parallel data extraction."""
    console.print(Rule("[bold blue]Tab Management Demo[/bold blue]"))
    logger.info("Demonstrating tab management and parallel browsing", emoji_key="tabs")
    
    # List of sites to open in different tabs
    sites = [
        {"name": "Wikipedia Python", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"},
        {"name": "Wikipedia JavaScript", "url": "https://en.wikipedia.org/wiki/JavaScript"},
        {"name": "Wikipedia Rust", "url": "https://en.wikipedia.org/wiki/Rust_(programming_language)"}
    ]
    
    # First tab is already open from previous demos
    tab_results = {}
    current_tab_id = None
    
    # Get info about current tab
    tabs_list_result = await browser_tab_list()
    
    if tabs_list_result.get("tabs"):
        for tab in tabs_list_result["tabs"]:
            if tab.get("is_current", False):
                current_tab_id = tab.get("id")
                break
    
    display_result("Current Tabs", tabs_list_result)
    
    # Open new tabs for each site
    tab_ids = []
    if current_tab_id:
        tab_ids.append(current_tab_id)
    
    for _i, site in enumerate(sites):
        console.print(f"[cyan]Opening new tab for:[/cyan] {site['name']}")
        
        new_tab_result = await browser_tab_new(
            url=site["url"],
            capture_snapshot=True
        )
        
        tab_id = new_tab_result.get("tab_id")
        if tab_id:
            tab_ids.append(tab_id)
            tab_results[tab_id] = {
                "name": site["name"],
                "result": new_tab_result
            }
        
        display_result(f"New Tab: {site['name']}", new_tab_result)
    
    # List all tabs
    updated_tabs_result = await browser_tab_list()
    
    display_result("All Open Tabs", updated_tabs_result)
    
    # Create a nice table showing the tabs
    tabs_table = Table(title="Open Browser Tabs", box=box.ROUNDED)
    tabs_table.add_column("Index", style="cyan")
    tabs_table.add_column("Tab ID", style="dim blue")
    tabs_table.add_column("Title", style="green")
    tabs_table.add_column("URL", style="yellow")
    tabs_table.add_column("Current", style="magenta")
    
    for tab in updated_tabs_result.get("tabs", []):
        tabs_table.add_row(
            str(tab.get("index")),
            tab.get("id", "unknown"),
            tab.get("title", "No title"),
            tab.get("url", "No URL"),
            "✓" if tab.get("is_current", False) else ""
        )
    
    console.print(tabs_table)
    
    # Demonstrate switching between tabs and performing actions
    console.print("\n[bold cyan]Switching Between Tabs and Extracting Data[/bold cyan]")
    
    # Extract data from each language tab
    language_data = {}
    
    for tab_id in tab_ids[1:]:  # Skip first tab (from previous demos)
        tab_info = tab_results.get(tab_id, {})
        tab_name = tab_info.get("name", "Unknown")
        
        console.print(f"[cyan]Switching to tab:[/cyan] {tab_name}")
        
        # Select the tab
        switch_result = await browser_tab_select(
            tab_index=tab_ids.index(tab_id) + 1  # 1-based index
        )
        
        display_result(f"Switched to Tab: {tab_name}", switch_result)
        
        # Execute JavaScript to extract information about the programming language
        js_result = await browser_execute_javascript(
            script="""() => {
                // Function to extract the first paragraph
                function getFirstParagraph() {
                    const paragraphs = document.querySelectorAll('.mw-parser-output > p');
                    for (const p of paragraphs) {
                        if (p.textContent.trim().length > 100) { // First substantial paragraph
                            return p.textContent.trim();
                        }
                    }
                    return "No description found";
                }
                
                // Extract infobox data if available
                function getInfoboxData() {
                    const infobox = document.querySelector('.infobox');
                    if (!infobox) return {};
                    
                    const data = {};
                    const rows = infobox.querySelectorAll('tr');
                    
                    rows.forEach(row => {
                        const header = row.querySelector('th');
                        const cell = row.querySelector('td');
                        if (header && cell) {
                            const key = header.textContent.trim();
                            const value = cell.textContent.trim();
                            if (key && value) {
                                data[key] = value;
                            }
                        }
                    });
                    
                    return data;
                }
                
                // Get section headings
                function getSectionHeadings() {
                    const headings = [];
                    document.querySelectorAll('h2 .mw-headline, h3 .mw-headline').forEach(el => {
                        headings.push(el.textContent.trim());
                    });
                    return headings.slice(0, 10); // First 10 headings
                }
                
                return {
                    title: document.title,
                    description: getFirstParagraph(),
                    infobox: getInfoboxData(),
                    headings: getSectionHeadings()
                };
            }"""
        )
        
        if js_result.get("success", False) and js_result.get("result"):
            language_data[tab_name] = js_result["result"]
            
        display_result(f"Extracted Data for {tab_name}", js_result)
        
        # Take a screenshot in this tab
        screenshot_result = await browser_screenshot(
            full_page=False,
            quality=80
        )
        
        # Save the screenshot
        if screenshot_result.get("data"):
            import base64
            screenshot_path = SAVE_DIR / f"{tab_name.lower().replace(' ', '_')}_screenshot.jpg"
            try:
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(screenshot_result["data"]))
                screenshot_result["file_path"] = str(screenshot_path)
                screenshot_result["file_name"] = screenshot_path.name
                logger.success(f"Screenshot saved to {screenshot_path}", emoji_key="file")
            except Exception as e:
                logger.error(f"Failed to save screenshot: {e}", emoji_key="error")
    
    # Display the extracted data comparison
    console.print("\n[bold cyan]Programming Languages Comparison[/bold cyan]")
    
    # Create a comparison table
    comparison_table = Table(title="Programming Language Comparison", box=box.ROUNDED)
    comparison_table.add_column("Feature", style="cyan")
    
    # Add a column for each language
    for tab_name in [info.get("name") for info in tab_results.values()]:
        if tab_name and tab_name in language_data:
            comparison_table.add_column(tab_name, style="green")
    
    # Add rows for comparison data
    # First paragraph description (truncated)
    comparison_table.add_row(
        "Description",
        *[language_data.get(lang, {}).get("description", "N/A")[:100] + "..." 
          for lang in language_data.keys()]
    )
    
    # Show paradigms if available
    comparison_table.add_row(
        "Paradigm",
        *[language_data.get(lang, {}).get("infobox", {}).get("Paradigm", "N/A") 
          for lang in language_data.keys()]
    )
    
    # Show designer if available
    comparison_table.add_row(
        "Designed by",
        *[language_data.get(lang, {}).get("infobox", {}).get("Designed by", "N/A") 
          for lang in language_data.keys()]
    )
    
    # First appeared
    comparison_table.add_row(
        "First appeared",
        *[language_data.get(lang, {}).get("infobox", {}).get("First appeared", "N/A") 
          for lang in language_data.keys()]
    )
    
    console.print(comparison_table)
    
    # Get the current number of tabs
    updated_tabs_result = await browser_tab_list()
    total_tabs = updated_tabs_result.get("total_tabs", len(tab_ids))
    # Close tabs from the last to the second (keep the first tab open)
    for i in range(total_tabs, 1, -1):
        close_result = await browser_tab_close(tab_index=i)
        display_result(f"Closed Tab {i}", close_result)
    
    # Select the first tab to return to previous state
    if tab_ids:
        await browser_tab_select(tab_index=1)
    
    return {
        "tabs_opened": len(tab_ids),
        "language_data": language_data
    }


async def demo_authentication_workflow():
    """Demonstrate a login workflow with credential handling."""
    console.print(Rule("[bold blue]Authentication Workflow Demo[/bold blue]"))
    logger.info("Demonstrating authentication workflow", emoji_key="login")
    
    # Navigate to the login page
    result = await browser_navigate(
        url="https://the-internet.herokuapp.com/login",
        wait_until="load"
    )
    
    display_result("Navigated to Login Page", result)
    
    # Take a screenshot before login
    screenshot_before = await browser_screenshot(
        full_page=True,
        quality=80
    )
    
    # Save the before screenshot
    if screenshot_before.get("data"):
        import base64
        screenshot_path = SAVE_DIR / "login_before.jpg"
        try:
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(screenshot_before["data"]))
            screenshot_before["file_path"] = str(screenshot_path)
            screenshot_before["file_name"] = screenshot_path.name
            logger.success(f"Pre-login screenshot saved to {screenshot_path}", emoji_key="file")
        except Exception as e:
            logger.error(f"Failed to save pre-login screenshot: {e}", emoji_key="error")
    
    display_result("Before Login", screenshot_before)
    
    # Show the login credentials for the demo site
    credentials_table = Table(title="Demo Login Credentials", box=box.SIMPLE)
    credentials_table.add_column("Field", style="cyan")
    credentials_table.add_column("Value", style="white")
    
    # The login credentials for this demo site
    username = "tomsmith"
    password = "SuperSecretPassword!"
    
    credentials_table.add_row("Username", username)
    credentials_table.add_row("Password", "********" + password[-2:])  # Masked password
    
    console.print(credentials_table)
    
    # Handle the login process
    print("[cyan]Logging in...[/cyan]")
    try:
        # Enter username
        username_result = await browser_type(  # noqa: F841
            selector="#username",
            text=username,
            delay=15  # Slow typing for visibility
        )
        
        # Enter password
        password_result = await browser_type(  # noqa: F841
            selector="#password",
            text=password,
            delay=15  # Slow typing for visibility
        )
        
        # Click the login button
        login_result = await browser_click(  # noqa: F841
            selector="button[type='submit']",
            delay=100  # Add a delay before click
        )
        
        # Wait for login to complete - look for success message
        await browser_wait(
            wait_type="selector",
            value=".flash.success",
            timeout=5000
        )
    finally:
        pass  # No progress.update needed
    
    # Take a screenshot after successful login
    screenshot_after = await browser_screenshot(
        full_page=True,
        quality=80
    )
    
    # Save the after screenshot
    if screenshot_after.get("data"):
        import base64
        screenshot_path = SAVE_DIR / "login_after.jpg"
        try:
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(screenshot_after["data"]))
            screenshot_after["file_path"] = str(screenshot_path)
            screenshot_after["file_name"] = screenshot_path.name
            logger.success(f"Post-login screenshot saved to {screenshot_path}", emoji_key="file")
        except Exception as e:
            logger.error(f"Failed to save post-login screenshot: {e}", emoji_key="error")
    
    display_result("After Login", screenshot_after)
    
    # Get the success message to confirm login
    success_message = await browser_get_text(
        selector=".flash.success",
        trim=True
    )
    
    display_result("Login Success Message", success_message)
    
    # Extract session info using JavaScript
    session_info = await browser_execute_javascript(
        script="""() => {
            // Get all cookies
            const cookies = document.cookie.split(';').map(cookie => {
                const [name, value] = cookie.trim().split('=');
                return { name, value };
            });
            
            // Get localStorage
            const localStorage = {};
            for (let i = 0; i < window.localStorage.length; i++) {
                const key = window.localStorage.key(i);
                localStorage[key] = window.localStorage.getItem(key);
            }
            
            // Get sessionStorage
            const sessionStorage = {};
            for (let i = 0; i < window.sessionStorage.length; i++) {
                const key = window.sessionStorage.key(i);
                sessionStorage[key] = window.sessionStorage.getItem(key);
            }
            
            return {
                url: window.location.href,
                title: document.title,
                cookies: cookies,
                localStorage: localStorage,
                sessionStorage: sessionStorage,
                authenticated: document.querySelector('.flash.success') !== null
            };
        }"""
    )
    
    display_result("Session Information", session_info)
    
    # Now logout to demonstrate session termination
    logger.info("Logging out to terminate session", emoji_key="logout")
    
    logout_result = await browser_click(
        selector="a[href='/logout']",
        delay=100
    )
    
    # Wait for logout to complete - redirected back to login page
    await browser_wait(
        wait_type="selector",
        value="#username",  # Looking for the login form again
        timeout=5000
    )
    
    display_result("Logout Result", logout_result)
    
    # Take a final screenshot after logout
    screenshot_logout = await browser_screenshot(
        full_page=True,
        quality=80
    )
    
    # Save the logout screenshot
    if screenshot_logout.get("data"):
        import base64
        screenshot_path = SAVE_DIR / "logout_result.jpg"
        try:
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(screenshot_logout["data"]))
            screenshot_logout["file_path"] = str(screenshot_path)
            screenshot_logout["file_name"] = screenshot_path.name
            logger.success(f"Post-logout screenshot saved to {screenshot_path}", emoji_key="file")
        except Exception as e:
            logger.error(f"Failed to save post-logout screenshot: {e}", emoji_key="error")
    
    display_result("After Logout", screenshot_logout)
    
    return {
        "login_success": success_message.get("success", False),
        "session_info": session_info.get("result", {}),
        "workflow_complete": True
    }


async def demo_network_monitoring():
    """Demonstrate network request monitoring and interception."""
    console.print(Rule("[bold blue]Network Monitoring Demo[/bold blue]"))
    logger.info("Demonstrating network monitoring capabilities", emoji_key="network")
    
    # Navigate to a site with multiple network requests
    result = await browser_navigate(
        url="https://httpbin.org/",
        wait_until="networkidle"  # Wait until network is idle
    )
    
    display_result("Navigated to HTTPBin", result)
    
    # First get information about all requests using JavaScript
    initial_request_data = await browser_execute_javascript(
        script="""() => {
            // Use the Performance API to get network data
            const performance = window.performance;
            const resources = performance.getEntriesByType('resource');
            
            // Process and extract key information from each request
            const requests = resources.map(res => {
                return {
                    name: res.name,
                    initiatorType: res.initiatorType,
                    duration: Math.round(res.duration),
                    size: Math.round(res.transferSize || 0),
                    startTime: Math.round(res.startTime)
                };
            });
            
            // Extract timing information
            const timing = {
                navigationStart: 0,
                domLoading: Math.round(performance.timing.domLoading - performance.timing.navigationStart),
                domInteractive: Math.round(performance.timing.domInteractive - performance.timing.navigationStart),
                domContentLoaded: Math.round(performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart),
                domComplete: Math.round(performance.timing.domComplete - performance.timing.navigationStart),
                loadEvent: Math.round(performance.timing.loadEventEnd - performance.timing.navigationStart)
            };
            
            return {
                url: document.location.href,
                numRequests: requests.length,
                requests: requests,
                timing: timing
            };
        }"""
    )
    
    display_result("Initial Network Activity", initial_request_data)
    
    # Display performance metrics in a nice table
    if initial_request_data.get("result") and initial_request_data["result"].get("timing"):
        timing = initial_request_data["result"]["timing"]
        performance_table = Table(title="Page Load Performance Metrics", box=box.ROUNDED)
        performance_table.add_column("Metric", style="cyan")
        performance_table.add_column("Time (ms)", style="green")
        
        for key, value in timing.items():
            performance_table.add_row(key, str(value))
        
        console.print(performance_table)
    
    # Now set up network monitoring to watch specific requests
    console.print("\n[bold cyan]Monitoring Specific Network Requests[/bold cyan]")
    
    # Set up JavaScript to monitor requests in real-time
    await browser_execute_javascript(
        script="""() => {
            // Create a global array to store request info
            window.monitoredRequests = [];
            
            // Create a PerformanceObserver to watch for resource loads
            const observer = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                
                entries.forEach(entry => {
                    if (entry.entryType === 'resource') {
                        window.monitoredRequests.push({
                            url: entry.name,
                            type: entry.initiatorType,
                            duration: Math.round(entry.duration),
                            size: Math.round(entry.transferSize || 0),
                            timestamp: new Date().toISOString()
                        });
                    }
                });
            });
            
            // Start observing
            observer.observe({entryTypes: ['resource']});
            
            // Also set up request logging on XMLHttpRequest
            const originalOpen = XMLHttpRequest.prototype.open;
            XMLHttpRequest.prototype.open = function(method, url) {
                this.addEventListener('load', function() {
                    const size = this.responseText ? this.responseText.length : 0;
                    window.monitoredRequests.push({
                        url: url,
                        method: method,
                        status: this.status,
                        size: size,
                        type: 'xhr',
                        timestamp: new Date().toISOString()
                    });
                });
                
                return originalOpen.apply(this, arguments);
            };
            
            return {started: true, message: "Network monitoring started"};
        }"""
    )
    
    # Navigate to a page that will generate some API requests
    console.print("\n[cyan]Navigating to a page with multiple API requests...[/cyan]")
    
    await browser_navigate(
        url="https://httpbin.org/forms/post",
        wait_until="networkidle" # Wait for initial requests to settle
    )
    
    # Interact with the form to trigger a request
    await browser_type(
        selector="input[name='custname']",
        text="LLM Gateway Test User"
    )
    
    # FIX: Use browser_click for radio button instead of browser_select
    logger.info("Selecting size 'Medium' using radio button")
    await browser_click(
        selector="input[type='radio'][name='size'][value='medium']",
        capture_snapshot=False # No need for snapshot here
    )
    
    # Select topping (checkbox interaction is likely correct)
    await browser_checkbox(
        selector="input[value='cheese']",
        check=True,
        capture_snapshot=False # No need for snapshot here
    )
    
    # Submit the form which will trigger an API request
    # FIX: Use a potentially more robust selector for the button
    logger.info("Clicking submit button")
    await browser_click(
        selector="form button", # More specific selector for the button within the form
        delay=100,
        capture_snapshot=False # No need for snapshot here
    )
    
    # Wait a bit for all requests to complete
    await asyncio.sleep(2)
    
    # Retrieve the monitored requests
    network_results = await browser_execute_javascript(
        script="""() => {
            // Return the collected requests, handle if undefined
            const requests = window.monitoredRequests || [];
            return {
                total: requests.length,
                requests: requests
            };
        }""" # <-- FIX: Handle undefined window.monitoredRequests
    )
    
    display_result("Monitored Network Requests", network_results)
    
    # Show network requests in a table
    if network_results.get("result") and network_results["result"].get("requests"):
        requests = network_results["result"]["requests"]
        
        requests_table = Table(title="Network Requests", box=box.ROUNDED)
        requests_table.add_column("URL", style="cyan")
        requests_table.add_column("Type", style="green")
        requests_table.add_column("Size", style="yellow")
        requests_table.add_column("Duration", style="magenta")
        
        for req in requests[:10]:  # Show first 10 requests
            url = req.get("url", "")
            # Truncate URL if too long
            if len(url) > 50:
                url = url[:47] + "..."
                
            requests_table.add_row(
                url,
                req.get("type", "unknown"),
                f"{req.get('size', 0)} bytes",
                f"{req.get('duration', 0)} ms" if "duration" in req else "N/A"
            )
        
        console.print(requests_table)
        
        if len(requests) > 10:
            console.print(f"[dim]...and {len(requests) - 10} more requests[/dim]")
    
    # Demonstrate waiting for a specific network request
    console.print("\n[bold cyan]Waiting for Specific Network Conditions[/bold cyan]")
    
    # Set up JavaScript to check for a specific request
    await browser_execute_javascript(
        script="""() => {
            // Reset the monitored requests array
            window.monitoredRequests = [];
            
            // Flag to track if our target request has been made
            window.targetRequestCompleted = false;
            
            // Original fetch function
            const originalFetch = window.fetch;
            
            // Override fetch to monitor for specific requests
            window.fetch = async function(...args) {
                const url = args[0].url || args[0];
                
                // Flag if this is our target URL
                if (url.includes('/json')) {
                    console.log('Target URL fetch started:', url);
                    
                    // Call original fetch
                    const response = await originalFetch.apply(this, args);
                    
                    // Clone the response to read the body
                    const clone = response.clone();
                    
                    // Process in the background
                    clone.json().then(data => {
                        console.log('Target request completed');
                        window.targetRequestCompleted = true;
                        window.lastJsonResponse = data;
                    }).catch(err => {
                        console.error('JSON parse error:', err);
                    });
                    
                    return response;
                }
                
                // Regular request
                return originalFetch.apply(this, args);
            };
            
            return {
                setup: true,
                message: "Network interception ready for target JSON endpoint"
            };
        }"""
    )
    
    # Navigate to a page with a JSON API endpoint
    console.print("\n[cyan]Navigating to JSON data endpoint...[/cyan]")
    
    print("[cyan]Waiting for JSON response...[/cyan]")
    try:
        # Navigate to JSON endpoint
        await browser_navigate(
            url="https://httpbin.org/json",
            wait_until="load"
        )
        
        # Wait for our specific network condition using JavaScript polling
        for _ in range(10):  # Try up to 10 times
            check_result = await browser_execute_javascript(
                script="""() => {
                    return {
                        completed: window.targetRequestCompleted === true,
                        data: window.lastJsonResponse || null
                    };
                }"""
            )
            
            if check_result.get("result", {}).get("completed", False):
                break
                
            await asyncio.sleep(0.5)
    finally:
        pass  # No progress.update needed
    
    # Check if we got data
    json_data_result = await browser_execute_javascript(
        script="""() => {
            return {
                success: window.targetRequestCompleted === true,
                data: window.lastJsonResponse || null
            };
        }"""
    )
    
    display_result("JSON Data from Intercepted Network Request", json_data_result)
    
    # Take a screenshot showing the JSON data
    screenshot_result = await browser_screenshot(
        full_page=True,
        quality=80
    )
    
    # Save the screenshot
    if screenshot_result.get("data"):
        import base64
        screenshot_path = SAVE_DIR / "network_json_response.jpg"
        try:
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(screenshot_result["data"]))
            screenshot_result["file_path"] = str(screenshot_path)
            screenshot_result["file_name"] = screenshot_path.name
            logger.success(f"Network response screenshot saved to {screenshot_path}", emoji_key="file")
        except Exception as e:
            logger.error(f"Failed to save network screenshot: {e}", emoji_key="error")
    
    display_result("Network Response Screenshot", screenshot_result)
    
    return {
        "initial_requests": initial_request_data.get("result", {}).get("numRequests", 0),
        "monitored_requests": network_results.get("result", {}).get("total", 0),
        "json_data_captured": json_data_result.get("result", {}).get("success", False)
    }


async def demo_file_upload():
    """Demonstrate file upload capabilities."""
    console.print(Rule("[bold blue]File Upload Demo[/bold blue]"))
    logger.info("Demonstrating file upload capabilities", emoji_key=TaskType.UPLOAD.value)
    
    # Navigate to the correct upload page first
    logger.info("Navigating to file upload demo page...", emoji_key="navigation")
    nav_result = await browser_navigate(
        url="https://the-internet.herokuapp.com/upload",
        wait_until="load"
    )
    display_result("Navigated to Upload Page", nav_result)

    # Create a dummy file for uploading
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as temp_file:
        temp_file.write("This is a test file created by LLM Gateway Browser Automation.\n")
        temp_file.write("This file demonstrates uploading a simple text file.\n")
        temp_file_path = temp_file.name
    
    # Create a small CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_csv:
        temp_csv.write(b"Name,Email,Role\n")
        temp_csv.write(b"Test User,user@example.com,Tester\n")
        temp_csv.write(b"Admin User,admin@example.com,Administrator\n")
        temp_csv.write(b"Guest User,guest@example.com,Guest\n")
        temp_csv_path = temp_csv.name
    
    # Create a simple HTML file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_html:
        temp_html.write(b"<!DOCTYPE html>\n<html>\n<head>\n<title>Test HTML File</title>\n</head>\n")
        temp_html.write(b"<body>\n<h1>Test HTML File</h1>\n<p>This file was created by the LLM Gateway Browser Automation demo.</p>\n</body>\n</html>\n")
        temp_html_path = temp_html.name
    
    # Show the temporary files we've created for upload
    files_table = Table(title="Files Prepared for Upload", box=box.ROUNDED)
    files_table.add_column("File Path", style="cyan")
    files_table.add_column("Type", style="green")
    files_table.add_column("Size", style="yellow")
    
    for file_path in [temp_file_path, temp_csv_path, temp_html_path]:
        file_size = os.path.getsize(file_path)
        file_type = file_path.split('.')[-1].upper()
        files_table.add_row(
            file_path,
            file_type,
            f"{file_size} bytes"
        )
    
    console.print(files_table)
    
    # Take a screenshot before upload
    screenshot_before = await browser_screenshot(
        full_page=True,
        quality=80
    )
    
    # Save the screenshot
    if screenshot_before.get("data"):
        import base64
        screenshot_path = SAVE_DIR / "upload_before.jpg"
        try:
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(screenshot_before["data"]))
            screenshot_before["file_path"] = str(screenshot_path)
            screenshot_before["file_name"] = screenshot_path.name
            logger.success(f"Pre-upload screenshot saved to {screenshot_path}", emoji_key="file")
        except Exception as e:
            logger.error(f"Failed to save pre-upload screenshot: {e}", emoji_key="error")
    
    # Upload each file one by one and show results
    for i, file_path in enumerate([temp_file_path, temp_csv_path, temp_html_path]):
        file_name = os.path.basename(file_path)
        file_type = file_path.split('.')[-1].upper()
        
        console.print(f"\n[bold cyan]Uploading {file_type} File: {file_name}[/bold cyan]")
        
        # Upload file
        upload_result = await browser_upload_file(
            selector="#file-upload",
            file_paths=file_path,
            capture_snapshot=True
        )
        
        display_result(f"File Upload {i+1}: {file_type}", upload_result)
        
        # Click submit
        await browser_click(
            selector="#file-submit",
            delay=100
        )
        
        # Wait for upload to complete
        await browser_wait(
            wait_type="selector",
            value="#uploaded-files",
            timeout=5000
        )
        
        # Take a screenshot of the result
        screenshot_result = await browser_screenshot(
            full_page=True,
            quality=80
        )
        
        # Save the screenshot
        if screenshot_result.get("data"):
            import base64
            screenshot_path = SAVE_DIR / f"upload_result_{file_type.lower()}.jpg"
            try:
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(screenshot_result["data"]))
                screenshot_result["file_path"] = str(screenshot_path)
                screenshot_result["file_name"] = screenshot_path.name
                logger.success(f"Upload result screenshot saved to {screenshot_path}", emoji_key="file")
            except Exception as e:
                logger.error(f"Failed to save upload result screenshot: {e}", emoji_key="error")
                
        # Get uploaded file info
        uploaded_file_info = await browser_get_text(
            selector="#uploaded-files"
        )
        
        display_result(f"Uploaded File {i+1} Info", uploaded_file_info)
        
        # Go back to upload page for the next file
        if i < 2:
            await browser_navigate(
                url="https://the-internet.herokuapp.com/upload",
                wait_until="load"
            )
    
    # Clean up temporary files
    for file_path in [temp_file_path, temp_csv_path, temp_html_path]:
        try:
            os.unlink(file_path)
            logger.info(f"Deleted temporary file: {file_path}", emoji_key="cleanup")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {e}", emoji_key="warning")
    
    console.print("\n[bold green]File Upload Demo Complete[/bold green]")
    
    return {
        "files_uploaded": 3,
        "temp_files_created": 3,
        "temp_files_cleaned": 3
    }


async def demo_structured_data_extraction_jobs(args):
    """Runs the structured data extraction demo for job postings."""
    console.print(Rule("[bold magenta]Demo: Extract Job Posting Details (Dynamic Crawl)[/bold magenta]"))
    # Note: The effectiveness depends heavily on Google's changing layout and the selectors
    # defined in JOB_POSTING_EXTRACTION_INSTRUCTIONS
    console.print("[yellow]Note: Job Posting demo uses selectors for Google search results which may break if Google changes its layout.[/yellow]")

    # Make a copy and set the LLM model from args
    instructions = JOB_POSTING_EXTRACTION_INSTRUCTIONS.copy()
    instructions["extraction_details"]["extraction_llm_model"] = args.model

    result = await extract_structured_data_from_pages(
        instructions=instructions,
        browser_options={"headless": args.headless},
        max_concurrent_pages=2 # Lower concurrency for demo stability
    )
    display_result("Job Posting Extraction Result", result) # Use existing display function
    demo_session.add_action("extract_structured_data", "Job Postings (Dynamic)", result) # Log action
    return result

async def demo_structured_data_extraction_products(args):
    """Runs the structured data extraction demo for product pages."""
    console.print(Rule("[bold magenta]Demo: Extract E-commerce Product Details (URL List)[/bold magenta]"))

    # IMPORTANT: Provide actual, valid product page URLs here for the demo!
    product_urls = [
        # "https://www.amazon.com/dp/B08H75RTZ8/", # Example Kindle Paperwhite - Replace/Add Real URLs
        # "https://www.bestbuy.com/site/sony-wh1000xm5-wireless-noise-cancelling-over-the-ear-headphones-black/6505725.p?skuId=6505725" # Example Sony XM5 - Replace/Add Real URLs
        # Add 1-3 valid product URLs from major e-commerce sites
    ]

    if not product_urls:
        console.print("[yellow]Skipping E-commerce Product Demo: No URLs provided in the script's `product_urls` list.[/yellow]")
        return {"success": True, "message": "Skipped: No product URLs provided in demo script."}

    # Create a copy and inject the URLs and LLM model for the demo run
    instructions = ECOMMERCE_PRODUCT_EXTRACTION_INSTRUCTIONS.copy()
    instructions["data_source"]["urls"] = product_urls
    instructions["extraction_details"]["extraction_llm_model"] = args.model

    result = await extract_structured_data_from_pages(
        instructions=instructions,
        browser_options={"headless": args.headless},
        max_concurrent_pages=2
    )
    display_result("E-commerce Product Extraction Result", result)
    demo_session.add_action("extract_structured_data", "E-commerce Products (List)", result)
    return result

async def demo_workflow_contact_form(args):
    """Runs the web workflow demo for submitting a contact form."""
    console.print(Rule("[bold blue]Demo: Execute Contact Form Submission Workflow[/bold blue]"))

    # Provide input data for the contact form
    contact_input = {
        # Keys here MUST match the keys expected by input_data_mapping in the instructions pack
        "user_name": "Test User via LLM Gateway",
        "user_email": "test@example.com",
        "user_message": "This is a test message sent by the execute_web_workflow tool. Time: " + datetime.now().isoformat()
    }

    # Make a copy and set the LLM model from args
    instructions = CONTACT_FORM_WORKFLOW_INSTRUCTIONS.copy()
    instructions["llm_model"] = args.model

    result = await execute_web_workflow(
        instructions=instructions,
        input_data=contact_input,
        browser_options={"headless": args.headless}
        # max_steps is defined within the instruction pack
    )
    display_result("Contact Form Workflow Result", result)
    demo_session.add_action("execute_web_workflow", "Contact Form Submission", result)
    return result

async def generate_session_report(format: str = "html") -> str:
    """Generate a comprehensive report of the demo session.
    
    Args:
        format: Output format, either "html" or "markdown"
        
    Returns:
        Path to the generated report file
    """
    global demo_session
    
    report_dir = SAVE_DIR / "reports"
    report_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if format.lower() == "html":
        report_path = report_dir / f"browser_automation_report_{timestamp}.html"
        content = _generate_html_report()
    else:
        report_path = report_dir / f"browser_automation_report_{timestamp}.md"
        content = _generate_markdown_report()
    
    # Write the report to file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.success(f"Generated {format} report at {report_path}", emoji_key="report")
    return str(report_path)


def _generate_html_report() -> str:
    """Generate HTML report of the demo session."""
    global demo_session
    
    # Use triple quote string literals directly without .format() for the CSS
    # This avoids the string placeholder issues
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Browser Automation Demo Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            border-left: 5px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
        .action {{
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .action-header {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}
        .action-title {{
            font-weight: bold;
            color: #2980b9;
        }}
        .action-time {{
            color: #7f8c8d;
        }}
        .success {{
            color: #27ae60;
        }}
        .error {{
            color: #e74c3c;
        }}
        .screenshots {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 10px;
        }}
        .screenshot {{
            max-width: 45%;
        }}
        .screenshot img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .screenshot-caption {{
            font-size: 0.9em;
            text-align: center;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .timing-chart {{
            height: 30px;
            background-color: #ecf0f1;
            position: relative;
            margin-top: 20px;
            border-radius: 5px;
            overflow: hidden;
        }}
        .timing-bar {{
            height: 100%;
            background-color: #3498db;
            position: absolute;
            top: 0;
            left: 0;
        }}
        .before-after {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .before-after-panel {{
            flex: 1;
            min-width: 45%;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
        }}
        .panel-header {{
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        }}
        .before {{
            background-color: #f8f9fa;
        }}
        .after {{
            background-color: #e8f4fc;
        }}
        .code {{
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .summary-metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-box {{
            flex: 1;
            min-width: 200px;
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2980b9;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Browser Automation Demo Report</h1>
    <div class="summary-box">
        <p><strong>Generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Total Duration:</strong> {demo_session.total_duration:.2f} seconds</p>
        <p><strong>Actions Performed:</strong> {len(demo_session.actions)}</p>
        <p><strong>Screenshots Captured:</strong> {len(demo_session.screenshots)}</p>
    </div>
    
    <div class="summary-metrics">
        <div class="metric-box">
            <div class="metric-value">{len(demo_session.actions)}</div>
            <div class="metric-label">Actions Performed</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{demo_session.total_duration:.1f}s</div>
            <div class="metric-label">Total Duration</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{len(demo_session.screenshots)}</div>
            <div class="metric-label">Screenshots</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{len(demo_session.demo_stats)}</div>
            <div class="metric-label">Demos Run</div>
        </div>
    </div>
    
    <h2>Action Timeline</h2>
    <div class="timing-chart">
        <!-- Timing bars would be generated here -->
    </div>
    
    <h2>Actions Performed</h2>
"""
    
    # Add each action
    for _i, action in enumerate(demo_session.actions):
        timestamp = time.strftime("%H:%M:%S", time.localtime(action["timestamp"]))
        result = action["result"]
        success = result.get("success", False)
        success_class = "success" if success else "error"
        success_text = "Success" if success else "Failed"
        
        # Calculate timing if available
        timing_info = ""
        if action["time_taken"]:
            timing_info = f" ({action['time_taken']:.2f}s)"
        
        html += f"""
    <div class="action">
        <div class="action-header">
            <span class="action-title">{escape_html(action["type"])}: {escape_html(action["description"])}</span>
            <span class="action-time">{timestamp}{timing_info}</span>
        </div>
        <div>Status: <span class="{success_class}">{success_text}</span></div>
"""
        
        # Add result details if available
        if "url" in result:
            html += f'        <div>URL: {escape_html(result["url"])}</div>\n'
        
        if "title" in result:
            html += f'        <div>Title: {escape_html(result["title"])}</div>\n'
        
        if "element_description" in result:
            html += f'        <div>Element: {escape_html(result["element_description"])}</div>\n'
        
        if "text" in result and result["text"]:
            text = result["text"]
            if len(text) > 200:
                text = text[:197] + "..."
            html += f'        <div>Text: {escape_html(text)}</div>\n'
        
        # Add screenshots if available
        if action["screenshots"]:
            html += '        <div class="screenshots">\n'
            for name, path in action["screenshots"].items():
                rel_path = os.path.relpath(path, SAVE_DIR / "reports")
                html += f"""            <div class="screenshot">
                <img src="{rel_path}" alt="{escape_html(name)}">
                <div class="screenshot-caption">{escape_html(name)}</div>
            </div>
"""
            html += '        </div>\n'
            
        # Close the action div
        html += '    </div>\n'
    
    # Add demo statistics
    html += """
    <h2>Demo Statistics</h2>
    <table>
        <tr>
            <th>Demo</th>
            <th>Duration</th>
            <th>Actions</th>
            <th>Status</th>
        </tr>
"""
    
    for demo_name, stats in demo_session.demo_stats.items():
        success = stats.get("success", True)
        success_class = "success" if success else "error"
        success_text = "✅ Success" if success else "❌ Failed"
        
        html += f"""        <tr>
            <td>{escape_html(demo_name)}</td>
            <td>{stats.get("duration", 0):.2f}s</td>
            <td>{stats.get("actions", 0)}</td>
            <td class="{success_class}">{success_text}</td>
        </tr>
"""
    
    html += "    </table>\n"
    
    # Add screenshots gallery
    if demo_session.screenshots:
        html += """
    <h2>Screenshots Gallery</h2>
    <div class="screenshots">
"""
        
        for name, path in demo_session.screenshots.items():
            rel_path = os.path.relpath(path, SAVE_DIR / "reports")
            html += f"""        <div class="screenshot">
            <img src="{rel_path}" alt="{escape_html(name)}">
            <div class="screenshot-caption">{escape_html(name)}</div>
        </div>
"""
            
        html += "    </div>\n"
    
    # Close the HTML document
    html += """
</body>
</html>
"""
    
    return html


def _generate_markdown_report() -> str:
    """Generate Markdown report of the demo session."""
    global demo_session
    
    # Start with Markdown template
    markdown = f"""# Browser Automation Demo Report

## Summary

- **Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Total Duration:** {demo_session.total_duration:.2f} seconds
- **Actions Performed:** {len(demo_session.actions)}
- **Screenshots Captured:** {len(demo_session.screenshots)}
- **Demos Run:** {len(demo_session.demo_stats)}

## Actions Performed

"""
    
    # Add each action
    for i, action in enumerate(demo_session.actions):
        timestamp = time.strftime("%H:%M:%S", time.localtime(action["timestamp"]))
        result = action["result"]
        success = result.get("success", False)
        success_text = "✅ Success" if success else "❌ Failed"
        
        # Calculate timing if available
        timing_info = ""
        if action["time_taken"]:
            timing_info = f" ({action['time_taken']:.2f}s)"
        
        markdown += f"### {i+1}. {action['type']}: {action['description']}\n\n"
        markdown += f"- **Time:** {timestamp}{timing_info}\n"
        markdown += f"- **Status:** {success_text}\n"
        
        # Add result details if available
        if "url" in result:
            markdown += f"- **URL:** {result['url']}\n"
        
        if "title" in result:
            markdown += f"- **Title:** {result['title']}\n"
        
        if "element_description" in result:
            markdown += f"- **Element:** {result['element_description']}\n"
        
        if "text" in result and result["text"]:
            text = result["text"]
            if len(text) > 200:
                text = text[:197] + "..."
            markdown += f"- **Text:** {text}\n"
        
        # Add screenshots if available
        if action["screenshots"]:
            markdown += "\n**Screenshots:**\n\n"
            for name, path in action["screenshots"].items():
                rel_path = os.path.relpath(path, SAVE_DIR / "reports")
                markdown += f"![{name}]({rel_path})\n"
            
        markdown += "\n"
    
    # Add demo statistics
    markdown += "## Demo Statistics\n\n"
    markdown += "| Demo | Duration | Actions | Status |\n"
    markdown += "|------|----------|---------|--------|\n"
    
    for demo_name, stats in demo_session.demo_stats.items():
        success = stats.get("success", True)
        success_text = "✅ Success" if success else "❌ Failed"
        
        markdown += f"| {demo_name} | {stats.get('duration', 0):.2f}s | {stats.get('actions', 0)} | {success_text} |\n"
    
    markdown += "\n"
    
    # Add screenshots gallery
    if demo_session.screenshots:
        markdown += "## Screenshots Gallery\n\n"
        
        for name, path in demo_session.screenshots.items():
            rel_path = os.path.relpath(path, SAVE_DIR / "reports")
            markdown += f"### {name}\n\n"
            markdown += f"![{name}]({rel_path})\n\n"
    
    return markdown


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))

async def cleanup():
    """Cleanup browser resources and generate final report."""
    console.print(Rule("[bold blue]Cleanup[/bold blue]"))
    logger.info("Cleaning up browser resources and generating report", emoji_key="cleanup")
    
    # Close the browser
    result = await browser_close()
    display_result("Browser Closed", result)
    
    # Generate session report
    demo_session.finish()
    report_path = await generate_session_report(format="html")
    markdown_path = await generate_session_report(format="markdown")
    
    console.print(Panel(
        f"[green]Session report generated:[/green]\n"
        f"HTML: [cyan]{report_path}[/cyan]\n"
        f"Markdown: [cyan]{markdown_path}[/cyan]",
        title="[bold]Demo Report[/bold]",
        border_style="green"
    ))
    
    return result

# Demo High Level Functions

async def demo_search_summary_with_instructions(
    query: str,
    instructions: Dict,
    llm_model: str,
    demo_title: str,
    headless: bool = True
):
    """Generic function to run the multi_engine_search_summary demo."""
    console.print(Rule(f"[bold yellow]Demo: {demo_title} ('{query}')[/bold yellow]"))

    result = await multi_engine_search_summary(
        query=query,
        instructions=instructions, # Pass the specific instructions
        llm_model=llm_model,
        browser_options={"headless": headless},
        max_concurrent_summaries=3 # Example concurrency limit
    )

    # Display result
    console.print(Panel(
        Syntax(json.dumps(result, indent=2, default=str), "json", theme="default", line_numbers=True),
        title=f"Result: {demo_title} ('{query}')",
        border_style="blue" if result.get("success") else "red"
    ))
    status_color = "green" if result.get("success") else "red"
    console.print(f"[bold {status_color}]Finished '{demo_title}' for '{query}'.[/bold {status_color}]")
    console.print("-" * 80)
    await asyncio.sleep(1)
    return result

async def demo_pdf_finder_with_instructions(
    topic: str,
    instructions: Dict, # Pass the specific instruction pack
    output_base_dir: str,
    llm_model: str,
    demo_title: str,
    steps: int = 15,
    headless: bool = True
):
    """Generic function to run the find_and_download_pdfs demo."""
    console.print(Rule(f"[bold cyan]Demo: {demo_title} ({topic})[/bold cyan]"))

    # Create the base output directory if it doesn't exist
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)

    result = await find_and_download_pdfs(
        topic=topic,
        instructions=instructions,
        output_directory=output_base_dir, # Tool will create topic subdir
        llm_model=llm_model,
        max_exploration_steps=steps,
        browser_options={"headless": headless}
    )

    # Display result
    console.print(Panel(
        Syntax(json.dumps(result, indent=2, default=str), "json", theme="default", line_numbers=True),
        title=f"Result: {demo_title} ({topic})",
        border_style="blue" if result.get("success") else "red"
    ))
    status_color = "green" if result.get("success") else "red"
    console.print(f"[bold {status_color}]Finished '{demo_title}' for '{topic}'.[/bold {status_color}]")
    console.print("-" * 80)
    await asyncio.sleep(1) # Pause between demos
    return result

async def demo_structured_data_extraction(
    instructions: Dict,
    demo_title: str,
    llm_model_override: Optional[str] = None, # Allow overriding model for demo if needed
    headless: bool = True
):
    """Generic function to run the extract_structured_data_from_pages demo."""
    console.print(Rule(f"[bold magenta]Demo: {demo_title}[/bold magenta]"))

    # If source URLs are needed, they must be populated in the instructions *before* calling
    # Example: instructions["data_source"]["urls"] = ["http://...", "http://..."]

    # Determine the LLM model to use
    # Priority: Override > Instructions > Default within tool
    model_to_use = llm_model_override or instructions.get("extraction_details", {}).get("extraction_llm_model")
    if not model_to_use:
         # Fallback if not specified anywhere (should be required in instructions ideally)
         model_to_use = "openai/gpt-4.1-mini"
         logger.warning(f"LLM model not specified for {demo_title}, defaulting to {model_to_use}")
         # Update instructions dict if needed, though tool expects it internally
         if "extraction_details" in instructions:
              instructions["extraction_details"]["extraction_llm_model"] = model_to_use

    result = await extract_structured_data_from_pages(
        instructions=instructions,
        # llm_model is now specified within instructions["extraction_details"]
        browser_options={"headless": headless},
        max_concurrent_pages=3 # Example concurrency limit for demo
    )

    # Display result
    console.print(Panel(
        Syntax(json.dumps(result, indent=2, default=str), "json", theme="default", line_numbers=True),
        title=f"Result: {demo_title}",
        border_style="blue" if result.get("success") else "red"
    ))
    status_color = "green" if result.get("success") else "red"
    console.print(f"[bold {status_color}]Finished '{demo_title}'.[/bold {status_color}]")
    console.print("-" * 80)
    await asyncio.sleep(1)
    return result

async def demo_web_workflow(
    instructions: Dict,
    input_data: Optional[Dict],
    demo_title: str,
    headless: bool = True
):
    """Generic function to run the execute_web_workflow demo."""
    console.print(Rule(f"[bold blue]Demo: {demo_title}[/bold blue]"))

    # Note: llm_model is specified *inside* the instructions dict for this tool
    result = await execute_web_workflow(
        instructions=instructions,
        input_data=input_data,
        browser_options={"headless": headless}
        # max_steps is also typically defined within instructions
    )

    # Display result
    console.print(Panel(
        Syntax(json.dumps(result, indent=2, default=str), "json", theme="default", line_numbers=True),
        title=f"Result: {demo_title}",
        border_style="blue" if result.get("success") else "red"
    ))
    status_color = "green" if result.get("success") else "red"
    console.print(f"[bold {status_color}]Finished '{demo_title}'.[/bold {status_color}]")
    console.print("-" * 80)
    await asyncio.sleep(1)
    return result

async def demo_data_point_monitoring(
    instructions: Dict,
    previous_values: Optional[Dict],
    demo_title: str,
    headless: bool = True
):
    """Generic function to run the monitor_web_data_points demo."""
    console.print(Rule(f"[bold yellow]Demo: {demo_title}[/bold yellow]"))

    # Pass previous values if available
    result = await monitor_web_data_points(
        instructions=instructions,
        previous_values=previous_values,
        # LLM model, browser options, concurrency are inside instructions
    )

    # Display result
    console.print(Panel(
        Syntax(json.dumps(result, indent=2, default=str), "json", theme="default", line_numbers=True),
        title=f"Result: {demo_title}",
        border_style="blue" if result.get("success") else "red"
    ))
    status_color = "green" if result.get("success") else "red"
    console.print(f"[bold {status_color}]Finished '{demo_title}'.[/bold {status_color}]")
    console.print("-" * 80)
    await asyncio.sleep(1)
    return result # Return the full result which includes current values

async def demo_research_synthesis(
    topic: str,
    instructions: Dict,
    demo_title: str,
    # Model is now inside instructions, but allow override for demo flexibility
    llm_model_override: Optional[str] = None,
    headless: bool = True
):
    """Generic function to run the research_and_synthesize_report demo."""
    console.print(Rule(f"[bold green]Demo: {demo_title} ('{topic}')[/bold green]"))

    # Override models in instructions if specified via CLI arg for demo purposes
    instructions_copy = instructions.copy() # Modify a copy
    if llm_model_override:
        if "extraction_phase" in instructions_copy and isinstance(instructions_copy["extraction_phase"], dict):
            instructions_copy["extraction_phase"]["extraction_llm_model"] = llm_model_override
        if "synthesis_phase" in instructions_copy and isinstance(instructions_copy["synthesis_phase"], dict):
            instructions_copy["synthesis_phase"]["synthesis_llm_model"] = llm_model_override
        # Also need to potentially override the selection model if we add that config later

    result = await research_and_synthesize_report(
        topic=topic,
        instructions=instructions_copy, # Pass the potentially modified instructions
        browser_options={"headless": headless},
        max_concurrent_extractions=2 # Lower concurrency for demo stability
    )

    # Display result
    console.print(Panel(
        Syntax(json.dumps(result, indent=2, default=str), "json", theme="default", line_numbers=True),
        title=f"Result: {demo_title} ('{topic}')",
        border_style="blue" if result.get("success") else "red"
    ))
    status_color = "green" if result.get("success") else "red"
    console.print(f"[bold {status_color}]Finished '{demo_title}' for '{topic}'.[/bold {status_color}]")
    console.print("-" * 80)
    await asyncio.sleep(1)
    return result


#  # --- Run Demos Using Instruction Packs ---

#         # 1. Academic Paper Demo
#         if "academic" in args.demos_to_run:
#             all_results["academic_papers"] = await demo_pdf_finder_with_instructions(
#                 topic="Quantum Computing Algorithms",
#                 instructions=ACADEMIC_PAPER_INSTRUCTIONS,
#                 output_base_dir=args.output_dir,
#                 llm_model=args.model,
#                 demo_title="Find Academic Papers (arXiv)",
#                 steps=args.max_steps,
#                 headless=args.headless
#             )

#         # 2. Government Report Demo
#         if "government" in args.demos_to_run:
#             all_results["gov_reports"] = await demo_pdf_finder_with_instructions(
#                 topic="UK AI Safety Summit Outcomes",
#                 instructions=GOVERNMENT_REPORT_INSTRUCTIONS,
#                 output_base_dir=args.output_dir,
#                 llm_model=args.model,
#                 demo_title="Find Government Reports",
#                 steps=args.max_steps,
#                 headless=args.headless
#             )

#         # 3. Product Manual Demo
#         if "manual" in args.demos_to_run:
#             all_results["product_manuals"] = await demo_pdf_finder_with_instructions(
#                 topic="Raspberry Pi 5",
#                 instructions=PRODUCT_MANUAL_INSTRUCTIONS,
#                 output_base_dir=args.output_dir,
#                 llm_model=args.model,
#                 demo_title="Find Product Manuals",
#                 steps=args.max_steps,
#                 headless=args.headless
#             )

#         # 4. Legal Document Demo (Example)
#         if "legal" in args.demos_to_run:
#              all_results["legal_docs"] = await demo_pdf_finder_with_instructions(
#                  topic="OpenAI lawsuit motion dismiss", # Example topic
#                  instructions=LEGAL_DOCUMENT_INSTRUCTIONS,
#                  output_base_dir=args.output_dir,
#                  llm_model=args.model, # May need a more capable model for legal
#                  demo_title="Find Legal Documents",
#                  steps=args.max_steps,
#                  headless=args.headless
#              )
#  # --- Run NEW Search Summary Demos ---
#         if "simple_search" in args.demos_to_run:
#             await demo_search_summary_with_instructions(
#                 query="Best family vacation destinations in Europe",
#                 instructions=SIMPLE_SEARCH_SUMMARY_INSTRUCTIONS,
#                 llm_model=args.model,
#                 demo_title="Simple Search Summary",
#                 headless=args.headless
#             )

#         if "tech_search" in args.demos_to_run:
#             await demo_search_summary_with_instructions(
#                 query="Playwright vs Selenium web scraping performance",
#                 instructions=TECHNICAL_SEARCH_SUMMARY_INSTRUCTIONS,
#                 llm_model=args.model,
#                 demo_title="Technical Search Summary",
#                 headless=args.headless
#             )

#  # --- Run NEW Workflow Demos ---
#         if "all" in args.demos_to_run or "workflow_login" in args.demos_to_run:
#              # Provide the necessary input data for the login workflow
#              login_input = {"username": "tomsmith", "password": "SuperSecretPassword!"}
#              await demo_web_workflow(
#                  instructions=ORDER_STATUS_WORKFLOW_INSTRUCTIONS,
#                  input_data=login_input,
#                  demo_title="Execute Login Workflow (the-internet.herokuapp.com)",
#                  headless=args.headless
#              )

#  # --- Run NEW Monitoring Demos ---
#         if "all" in args.demos_to_run or "monitor_product" in args.demos_to_run:
#              # IMPORTANT: You might need to update the URL and Selectors in
#              # PRODUCT_MONITORING_INSTRUCTIONS to match a live product page!
#              console.print("[yellow]Note: Product Monitoring Demo uses selectors that might break if the website changes.[/yellow]")
#              product_instructions = PRODUCT_MONITORING_INSTRUCTIONS.copy() # Use a copy
#              # Ensure the llm_model is set if overridden by args
#              product_instructions["llm_config"]["model"] = args.model

#              monitor_result = await demo_data_point_monitoring(
#                  instructions=product_instructions,
#                  # Pass the *current* state for change detection
#                  previous_values=monitor_state["previous_values"],
#                  demo_title="Monitor Product Price/Availability",
#                  headless=args.headless
#              )
#              # Update the state with the *new* current values for the next run
#              if monitor_result.get("success") and monitor_result.get("results"):
#                  for url, data_points in monitor_result["results"].items():
#                       if isinstance(data_points, dict) and "page_error" not in data_points:
#                            for dp_name, dp_data in data_points.items():
#                                 if isinstance(dp_data, dict) and "current_value" in dp_data and dp_data.get("error") is None:
#                                      state_key = f"{url}::{dp_name}"
#                                      monitor_state["previous_values"][state_key] = dp_data["current_value"]
#                  # In a real app: save monitor_state["previous_values"] to disk/db here

#         if "all" in args.demos_to_run or "monitor_news" in args.demos_to_run:
#              # IMPORTANT: Selectors for news sites are highly volatile!
#              console.print("[yellow]Note: News Monitoring Demo uses selectors that might break if the website changes.[/yellow]")
#              news_instructions = WEBSITE_SECTION_MONITORING_INSTRUCTIONS.copy()
#              news_instructions["llm_config"]["model"] = args.model # Set model

#              monitor_result_news = await demo_data_point_monitoring(
#                  instructions=news_instructions,
#                  previous_values=monitor_state["previous_values"],
#                  demo_title="Monitor Google News Headlines",
#                  headless=args.headless
#              )
#              # Update state
#              if monitor_result_news.get("success") and monitor_result_news.get("results"):
#                   # Similar logic as above to update previous_values in monitor_state
#                    for url, data_points in monitor_result_news["results"].items():
#                        if isinstance(data_points, dict) and "page_error" not in data_points:
#                            for dp_name, dp_data in data_points.items():
#                                 if isinstance(dp_data, dict) and "current_value" in dp_data and dp_data.get("error") is None:
#                                      state_key = f"{url}::{dp_name}"

#  # --- Run NEW Research & Synthesis Demos ---
#         if "all" in args.demos_to_run or "research_trends" in args.demos_to_run:
#             await demo_research_synthesis(
#                 topic="AI Agent Orchestration Frameworks",
#                 instructions=MARKET_TREND_RESEARCH_INSTRUCTIONS,
#                 demo_title="Research Market Trends",
#                 llm_model_override=args.model, # Pass CLI model if needed
#                 headless=args.headless
#             )

#         if "all" in args.demos_to_run or "research_competitors" in args.demos_to_run:
#             await demo_research_synthesis(
#                 topic="Notion productivity app", # Example product
#                 instructions=COMPETITIVE_ANALYSIS_INSTRUCTIONS,
#                 demo_title="Research Competitor Snippets",
#                 llm_model_override=args.model,
#                 headless=args.headless
#             )

async def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the demo script."""
    parser = argparse.ArgumentParser(
        description="LLM Gateway Browser Automation Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic demo type selector
    demo_type_group = parser.add_argument_group("Demo Type Selection")
    demo_type_group.add_argument(
        "--all", action="store_true",
        help="Run all low-level browser demos (default if no specific demo is selected)"
    )
    demo_type_group.add_argument(
        "--all-high-level", action="store_true",
        help="Run all high-level AI-powered demos"
    )
    
    # Low-level browser demos
    browser_demo_group = parser.add_argument_group("Low-Level Browser Demos")
    browser_demo_group.add_argument(
        "--basics", action="store_true",
        help="Run basic navigation demo"
    )
    browser_demo_group.add_argument(
        "--forms", action="store_true",
        help="Run form interaction demo"
    )
    browser_demo_group.add_argument(
        "--javascript", action="store_true",
        help="Run JavaScript execution demo"
    )
    browser_demo_group.add_argument(
        "--tabs", action="store_true",
        help="Run tab management demo"
    )
    browser_demo_group.add_argument(
        "--auth", action="store_true",
        help="Run authentication workflow demo"
    )
    browser_demo_group.add_argument(
        "--search", action="store_true",
        help="Run search interaction demo"
    )
    browser_demo_group.add_argument(
        "--file-upload", action="store_true",
        help="Run file upload demo"
    )
    browser_demo_group.add_argument(
        "--network", action="store_true",
        help="Run network monitoring demo"
    )

    # High-level AI-powered demos
    ai_demo_group = parser.add_argument_group("High-Level AI-Powered Demos")
    
    # Document finder demos
    ai_demo_group.add_argument(
        "--academic", action="store_true",
        help="Run academic paper finder demo"
    )
    ai_demo_group.add_argument(
        "--government", action="store_true",
        help="Run government report finder demo"
    )
    ai_demo_group.add_argument(
        "--manual", action="store_true",
        help="Run product manual finder demo"
    )
    ai_demo_group.add_argument(
        "--legal", action="store_true",
        help="Run legal document finder demo"
    )
    
    # Search summary demos
    ai_demo_group.add_argument(
        "--simple-search", action="store_true",
        help="Run simple search summary demo"
    )
    ai_demo_group.add_argument(
        "--tech-search", action="store_true",
        help="Run technical search summary demo"
    )
    
    # Workflow demos
    ai_demo_group.add_argument(
        "--workflow-login", action="store_true",
        help="Run login workflow demo"
    )
    
    # Monitoring demos
    ai_demo_group.add_argument(
        "--monitor-product", action="store_true",
        help="Run product price/availability monitoring demo"
    )
    ai_demo_group.add_argument(
        "--monitor-news", action="store_true",
        help="Run news headline monitoring demo"
    )
    
    # Research demos
    ai_demo_group.add_argument(
        "--research-trends", action="store_true",
        help="Run market trends research demo"
    )
    ai_demo_group.add_argument(
        "--research-competitors", action="store_true",
        help="Run competitor analysis research demo"
    )

    ai_demo_group.add_argument( "--extract-jobs", dest="extract_jobs", action="store_true", help="Run job posting extraction demo")
    ai_demo_group.add_argument( "--extract-products", dest="extract_products", action="store_true", help="Run e-commerce product extraction demo")
    ai_demo_group.add_argument( "--workflow-contact", dest="workflow_contact", action="store_true", help="Run contact form workflow demo")

    # LLM configuration
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument(
        "--model", type=str, default="openai/gpt-4o-mini",
        help="LLM model to use for AI-powered demos"
    )
    llm_group.add_argument(
        "--max-steps", type=int, default=15,
        help="Maximum exploration steps for document finder demos"
    )

    # Browser configuration
    browser_group = parser.add_argument_group("Browser Configuration")
    browser_group.add_argument(
        "--browser", choices=["chromium", "firefox", "webkit"], default="chromium",
        help="Browser to use for the demonstration"
    )
    browser_group.add_argument(
        "--headless", action="store_true",
        help="Run browser in headless mode (no visible UI)"
    )
    browser_group.add_argument(
        "--timeout", type=int, default=30000,
        help="Default timeout for browser operations in milliseconds"
    )
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir", type=str, default="./browser_demo_outputs",
        help="Directory to save screenshots, PDFs, and other outputs"
    )
    output_group.add_argument(
        "--no-screenshots", action="store_true",
        help="Disable saving screenshots to disk"
    )
    
    args = parser.parse_args()
    
    # If no specific demo is selected, default to running all low-level demos
    if not any([
        args.all, args.basics, args.forms, args.javascript, 
        args.tabs, args.auth, args.search, args.file_upload, args.network,
        args.all_high_level, args.academic, args.government, args.manual, args.legal,
        args.simple_search, args.tech_search, args.workflow_login,
        args.monitor_product, args.monitor_news, args.research_trends, args.research_competitors
    ]):
        args.all = True
    
    # Create a list of demos to run for easier handling in main()
    args.demos_to_run = []
    
    # Add low-level demos
    if args.all:
        args.demos_to_run.extend(["basics", "forms", "javascript", "tabs", "auth", "search", "file-upload", "network"])
    else:
        if args.basics:
            args.demos_to_run.append("basics")
        if args.forms:
            args.demos_to_run.append("forms")
        if args.javascript:
            args.demos_to_run.append("javascript")
        if args.tabs:
            args.demos_to_run.append("tabs")
        if args.auth:
            args.demos_to_run.append("auth")
        if args.search:
            args.demos_to_run.append("search")
        if args.file_upload:
            args.demos_to_run.append("file-upload")
        if args.network:
            args.demos_to_run.append("network")
    
    # Add high-level demos
    if args.all_high_level:
        args.demos_to_run.extend(["academic", "government", "manual", "legal", "simple-search", "tech-search", 
                                 "workflow-login", "monitor-product", "monitor-news", "research-trends", "research-competitors"])
    else:
        if args.academic:
            args.demos_to_run.append("academic")
        if args.government:
            args.demos_to_run.append("government")
        if args.manual:
            args.demos_to_run.append("manual")
        if args.legal:
            args.demos_to_run.append("legal")
        if args.simple_search:
            args.demos_to_run.append("simple-search")
        if args.tech_search:
            args.demos_to_run.append("tech-search")
        if args.workflow_login:
            args.demos_to_run.append("workflow-login")
        if args.monitor_product:
            args.demos_to_run.append("monitor-product")
        if args.monitor_news:
            args.demos_to_run.append("monitor-news")
        if args.research_trends:
            args.demos_to_run.append("research-trends")
        if args.research_competitors:
            args.demos_to_run.append("research-competitors")
    
    return args

async def main():
    """Run browser automation demonstrations based on command-line arguments."""
    # Parse command-line arguments
    args = await parse_arguments()
    
    # Update configuration based on arguments
    global SAVE_DIR
    SAVE_DIR = Path(args.output_dir)
    
    # Initialize monitor state for monitoring demos
    monitor_state = {"previous_values": {}}
    
    # Store results from all demos
    all_results = {}
    
    console.print(Rule("[bold magenta]Browser Automation Demonstration[/bold magenta]"))
    logger.info("Starting browser automation demo", emoji_key="start")
    logger.info(f"Using browser: {args.browser}", emoji_key="config")
    logger.info(f"Headless mode: {args.headless}", emoji_key="config")
    
    try:
        # Setup resources
        setup_demo()
        
        # Initialize browser with command-line arguments
        await browser_init(
            browser_name=args.browser,
            headless=args.headless,
            default_timeout=args.timeout
        )
        
        # Organize demos into categories for display and execution
        low_level_demos = []
        high_level_demos = []
        
        # Collect low-level demos
        if "basics" in args.demos_to_run:
            low_level_demos.append(("Navigation Basics", demo_navigation_basics))
        if "forms" in args.demos_to_run:
            low_level_demos.append(("Form Interaction", demo_form_interaction))
        if "javascript" in args.demos_to_run:
            low_level_demos.append(("JavaScript Execution", demo_javascript_execution))
        if "tabs" in args.demos_to_run:
            low_level_demos.append(("Tab Management", demo_tab_management))
        if "auth" in args.demos_to_run:
            low_level_demos.append(("Authentication Workflow", demo_authentication_workflow))
        if "search" in args.demos_to_run:
            low_level_demos.append(("Search Interaction", demo_search_interaction))
        if "file-upload" in args.demos_to_run:
            low_level_demos.append(("File Upload", demo_file_upload))
        if "network" in args.demos_to_run:
            low_level_demos.append(("Network Monitoring", demo_network_monitoring))
        
        # Collect high-level demos
        # Document finder demos
        if "academic" in args.demos_to_run:
            high_level_demos.append((
                "Academic Paper Finder", 
                lambda: demo_pdf_finder_with_instructions(
                    topic="Quantum Computing Algorithms",
                    instructions=ACADEMIC_PAPER_INSTRUCTIONS,
                    output_base_dir=args.output_dir,
                    llm_model=args.model,
                    demo_title="Find Academic Papers (arXiv)",
                    steps=args.max_steps,
                    headless=args.headless
                )
            ))
        if "government" in args.demos_to_run:
            high_level_demos.append((
                "Government Report Finder", 
                lambda: demo_pdf_finder_with_instructions(
                    topic="UK AI Safety Summit Outcomes",
                    instructions=GOVERNMENT_REPORT_INSTRUCTIONS,
                    output_base_dir=args.output_dir,
                    llm_model=args.model,
                    demo_title="Find Government Reports",
                    steps=args.max_steps,
                    headless=args.headless
                )
            ))
        if "manual" in args.demos_to_run:
            high_level_demos.append((
                "Product Manual Finder", 
                lambda: demo_pdf_finder_with_instructions(
                    topic="Raspberry Pi 5",
                    instructions=PRODUCT_MANUAL_INSTRUCTIONS,
                    output_base_dir=args.output_dir,
                    llm_model=args.model,
                    demo_title="Find Product Manuals",
                    steps=args.max_steps,
                    headless=args.headless
                )
            ))
        if "legal" in args.demos_to_run:
            high_level_demos.append((
                "Legal Document Finder", 
                lambda: demo_pdf_finder_with_instructions(
                    topic="OpenAI lawsuit motion dismiss",
                    instructions=LEGAL_DOCUMENT_INSTRUCTIONS,
                    output_base_dir=args.output_dir,
                    llm_model=args.model,
                    demo_title="Find Legal Documents",
                    steps=args.max_steps,
                    headless=args.headless
                )
            ))
        
        # Search summary demos
        if "simple-search" in args.demos_to_run:
            high_level_demos.append((
                "Simple Search Summary", 
                lambda: demo_search_summary_with_instructions(
                    query="Best family vacation destinations in Europe",
                    instructions=SIMPLE_SEARCH_SUMMARY_INSTRUCTIONS,
                    llm_model=args.model,
                    demo_title="Simple Search Summary",
                    headless=args.headless
                )
            ))
        if "tech-search" in args.demos_to_run:
            high_level_demos.append((
                "Technical Search Summary", 
                lambda: demo_search_summary_with_instructions(
                    query="Playwright vs Selenium web scraping performance",
                    instructions=TECHNICAL_SEARCH_SUMMARY_INSTRUCTIONS,
                    llm_model=args.model,
                    demo_title="Technical Search Summary",
                    headless=args.headless
                )
            ))
        
        # Workflow demos
        if "workflow-login" in args.demos_to_run:
            high_level_demos.append((
                "Login Workflow", 
                lambda: demo_web_workflow(
                    instructions=ORDER_STATUS_WORKFLOW_INSTRUCTIONS,
                    input_data={"username": "tomsmith", "password": "SuperSecretPassword!"},
                    demo_title="Execute Login Workflow (the-internet.herokuapp.com)",
                    headless=args.headless
                )
            ))
        
        # Monitoring demos
        if "monitor-product" in args.demos_to_run:
            high_level_demos.append((
                "Product Price/Availability Monitor", 
                lambda: demo_data_point_monitoring(
                    instructions={
                        **PRODUCT_MONITORING_INSTRUCTIONS,
                        "llm_config": {"model": args.model}
                    },
                    previous_values=monitor_state["previous_values"],
                    demo_title="Monitor Product Price/Availability",
                    headless=args.headless
                )
            ))
        if "monitor-news" in args.demos_to_run:
            high_level_demos.append((
                "News Headlines Monitor", 
                lambda: demo_data_point_monitoring(
                    instructions={
                        **WEBSITE_SECTION_MONITORING_INSTRUCTIONS,
                        "llm_config": {"model": args.model}
                    },
                    previous_values=monitor_state["previous_values"],
                    demo_title="Monitor Google News Headlines",
                    headless=args.headless
                )
            ))
        
        # Research & synthesis demos
        if "research-trends" in args.demos_to_run:
            high_level_demos.append((
                "Market Trends Research", 
                lambda: demo_research_synthesis(
                    topic="AI Agent Orchestration Frameworks",
                    instructions=MARKET_TREND_RESEARCH_INSTRUCTIONS,
                    demo_title="Research Market Trends",
                    llm_model_override=args.model,
                    headless=args.headless
                )
            ))
        if "research-competitors" in args.demos_to_run:
            high_level_demos.append((
                "Competitor Analysis", 
                lambda: demo_research_synthesis(
                    topic="Notion productivity app",
                    instructions=COMPETITIVE_ANALYSIS_INSTRUCTIONS,
                    demo_title="Research Competitor Snippets",
                    llm_model_override=args.model,
                    headless=args.headless
                )
            ))
        
        # Calculate total number of demos
        total_demo_count = len(low_level_demos) + len(high_level_demos)
        
        # Create overall progress bar for all demos
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan] demos"),
        ) as overall_progress:
            total_demo_task = overall_progress.add_task(
                "[bold blue]Running Browser Automation Demos[/bold blue]",
                total=total_demo_count
            )
            
            # First run low-level demos if any are selected
            if low_level_demos:
                console.print(Rule("[bold cyan]Running Low-Level Browser Demos[/bold cyan]"))
                
                for i, (demo_name, demo_func) in enumerate(low_level_demos):
                    overall_progress.update(
                        total_demo_task, 
                        description=f"[bold blue]Running Demo {i+1}/{len(low_level_demos)}: {demo_name}[/bold blue]",
                        advance=0
                    )
                    
                    console.print(Rule(f"[bold green]Running Demo: {demo_name}[/bold green]"))
                    
                    # Record start time for this demo
                    demo_start_time = time.time()
                    
                    # Run the demo
                    try:
                        if demo_name == "Navigation Basics":
                            # Use the shared progress context for navigation basics
                            task_id = overall_progress.add_task(f"[cyan]{demo_name} Steps", total=4)
                            result = await demo_func(progress=overall_progress, task_id=task_id)
                            overall_progress.remove_task(task_id)
                        else:
                            result = await demo_func()
                        all_results[demo_name.lower().replace(" ", "_")] = result
                        
                        # Record success
                        demo_duration = time.time() - demo_start_time
                        if demo_name not in demo_session.demo_stats:
                            # Only add if not already added by the demo function
                            demo_session.add_demo_stats(demo_name, {
                                "duration": demo_duration,
                                "success": True,
                                "actions": 0  # We don't know how many actions
                            })
                            
                    except Exception as e:
                        logger.error(f"Demo {demo_name} failed: {e}", emoji_key="error", exc_info=True)
                        console.print(f"[bold red]Demo Error:[/bold red] {escape(str(e))}")
                        
                        # Record failure
                        demo_duration = time.time() - demo_start_time
                        demo_session.add_demo_stats(demo_name, {
                            "duration": demo_duration,
                            "success": False,
                            "error": str(e)
                        })
                    
                    # Update progress
                    overall_progress.update(total_demo_task, advance=1)
            
            # Now run high-level demos if any are selected
            if high_level_demos:
                console.print(Rule("[bold yellow]Running High-Level AI-Powered Demos[/bold yellow]"))
                
                for i, (demo_name, demo_func) in enumerate(high_level_demos):
                    overall_progress.update(
                        total_demo_task, 
                        description=f"[bold yellow]Running Demo {i+1}/{len(high_level_demos)}: {demo_name}[/bold yellow]",
                        advance=0
                    )
                    
                    console.print(Rule(f"[bold green]Running Demo: {demo_name}[/bold green]"))
                    
                    # Record start time for this demo
                    demo_start_time = time.time()
                    
                    # Run the demo
                    try:
                        result = await demo_func()
                        all_results[demo_name.lower().replace(" ", "_")] = result
                        
                        # For monitoring demos, update monitor state
                        if "monitor" in demo_name.lower() and result.get("success") and result.get("results"):
                            for url, data_points in result["results"].items():
                                if isinstance(data_points, dict) and "page_error" not in data_points:
                                    for dp_name, dp_data in data_points.items():
                                        if isinstance(dp_data, dict) and "current_value" in dp_data and dp_data.get("error") is None:
                                            state_key = f"{url}::{dp_name}"
                                            monitor_state["previous_values"][state_key] = dp_data["current_value"]
                        
                        # Record success
                        demo_duration = time.time() - demo_start_time
                        if demo_name not in demo_session.demo_stats:
                            # Only add if not already added by the demo function
                            demo_session.add_demo_stats(demo_name, {
                                "duration": demo_duration,
                                "success": True,
                                "actions": 0  # We don't know how many actions
                            })
                            
                    except Exception as e:
                        logger.error(f"Demo {demo_name} failed: {e}", emoji_key="error", exc_info=True)
                        console.print(f"[bold red]Demo Error:[/bold red] {escape(str(e))}")
                        
                        # Record failure
                        demo_duration = time.time() - demo_start_time
                        demo_session.add_demo_stats(demo_name, {
                            "duration": demo_duration,
                            "success": False,
                            "error": str(e)
                        })
                    
                    # Update progress
                    overall_progress.update(total_demo_task, advance=1)
            
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"[bold red]Critical Demo Error:[/bold red] {escape(str(e))}")
        demo_session.add_action("error", "Critical Demo Error", {"success": False, "error": str(e)})
        return 1
    finally:
        # Always attempt to clean up browser resources and generate report
        try:
            await cleanup()
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}", emoji_key="error")
    
    # Final summary could go here if desired
    console.print(Rule("[bold magenta]Demo Run Complete[/bold magenta]", style="magenta"))
    
    # Show final statistics
    total_duration = demo_session.total_duration
    total_actions = len(demo_session.actions)
    total_screenshots = len(demo_session.screenshots)
    
    stats_table = Table(title="Demo Session Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Total Duration", f"{total_duration:.2f} seconds")
    stats_table.add_row("Total Actions", str(total_actions))
    stats_table.add_row("Screenshots Taken", str(total_screenshots))
    stats_table.add_row("Demos Run", str(len(demo_session.demo_stats)))
    stats_table.add_row("Report Path", str(SAVE_DIR / "reports"))
    
    console.print(stats_table)
    
    logger.success("Browser Automation Demo Completed Successfully", emoji_key="complete")
    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 