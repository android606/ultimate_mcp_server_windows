#!/usr/bin/env python
"""Browser automation demonstration using LLM Gateway's Playwright tools."""
import argparse
import asyncio
import os
import sys
import tempfile
import time
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
)
from llm_gateway.utils import get_logger

# --- Add Rich Imports ---
from llm_gateway.utils.logging.console import console

# ----------------------

# Initialize logger
logger = get_logger("example.browser_automation")

# Config
DEMO_SITES = {
    "wikipedia": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "search_engine": "https://www.google.com",
    "form_demo": "https://www.selenium.dev/selenium/web/web-form.html",
    "dynamic_demo": "https://www.selenium.dev/selenium/web/dynamic.html"
}

SAVE_DIR = Path("./browser_demo_outputs")

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


async def demo_navigation_basics():
    """Demonstrate basic navigation and page interaction."""
    console.print(Rule("[bold blue]Navigation Basics Demo[/bold blue]"))
    logger.info("Demonstrating basic navigation", emoji_key="navigation")
    
    # Create progress tracker for this demo
    progress, task_id = create_demo_progress_tracker()
    progress.update(task_id, total=4, description="[bold cyan]Navigation demo starting...[/bold cyan]")
    
    demo_start_time = time.time()
    demo_actions = 0
    
    with progress:
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
        
        # Save the screenshot to file
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
    demo_duration = time.time() - demo_start_time
    demo_session.add_demo_stats("Navigation Basics", {
        "duration": demo_duration,
        "actions": demo_actions,
        "success": True
    })
    
    return {
        "navigation": result,
        "screenshot": screenshot_result,
        "text": text_result,
        "click": click_result,
        "duration": demo_duration
    }


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
    
    # Navigate to Google
    result = await browser_navigate(
        url=DEMO_SITES["search_engine"],
        wait_until="load",
        timeout=30000
    )
    
    display_result("Navigated to Search Engine", result)
    
    # Enter a search query
    # Note: Google's search input selector may change; adjust if needed
    search_query = "LLM Gateway Browser Automation"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress_task = progress.add_task("[cyan]Searching...", total=None)
        
        try:
            # Try different selectors for Google's search input
            search_selectors = [
                "textarea[name='q']",  # Current Google search box
                "input[name='q']",     # Alternative/older Google search box
                "[name='q']"           # Fallback selector
            ]
            
            search_successful = False
            for selector in search_selectors:
                try:
                    # Check if element exists before trying to type
                    element_check = await browser_get_text(selector=selector)
                    if element_check.get("success", False):
                        type_result = await browser_type(  # noqa: F841
                            selector=selector,
                            text=search_query,
                            delay=20,  # Slow typing for visibility
                            press_enter=True  # Press Enter to submit search
                        )
                        search_successful = True
                        break
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}", emoji_key="debug")
                    continue
                    
            if not search_successful:
                logger.warning("Could not find search input element. Trying JavaScript input approach.", emoji_key="warning")
                # Fallback: Use JavaScript to find and fill the search box
                js_search_result = await browser_execute_javascript(
                    script=f"""() => {{
                        const searchInput = document.querySelector('[name="q"]');
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
                value="#search",  # Google's search results container
                timeout=10000
            )
        finally:
            progress.update(progress_task, completed=True)
    
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
            const resultElements = document.querySelectorAll('#search .g');
            
            resultElements.forEach((result, index) => {
                // Process only the first 5 results
                if (index >= 5) return;
                
                // Extract components of a result
                const titleElement = result.querySelector('h3');
                const linkElement = result.querySelector('a');
                const snippetElement = result.querySelector('.VwiC3b');
                
                if (titleElement && linkElement) {
                    results.push({
                        title: titleElement.textContent,
                        url: linkElement.href,
                        snippet: snippetElement ? snippetElement.textContent : null
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
    
    # Close tabs except the first one
    for i, _tab_id in enumerate(tab_ids[1:], start=2):  # Start at index 2 (1-based)
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
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress_task = progress.add_task("[cyan]Logging in...", total=None)
        
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
            progress.update(progress_task, completed=True)
    
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
        wait_until="networkidle"
    )
    
    # Interact with the form to trigger a request
    await browser_type(
        selector="input[name='custname']",
        text="LLM Gateway Test User"
    )
    
    await browser_select(
        selector="select[name='size']",
        values="medium",
        by="value"
    )
    
    await browser_checkbox(
        selector="input[value='cheese']",
        check=True
    )
    
    # Submit the form which will trigger an API request
    await browser_click(
        selector="button[type='submit']",
        delay=100
    )
    
    # Wait a bit for all requests to complete
    await asyncio.sleep(2)
    
    # Retrieve the monitored requests
    network_results = await browser_execute_javascript(
        script="""() => {
            // Return the collected requests
            return {
                total: window.monitoredRequests.length,
                requests: window.monitoredRequests
            };
        }"""
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
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress_task = progress.add_task("[cyan]Waiting for JSON response...", total=None)
        
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
        
        progress.update(progress_task, completed=True)
    
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
    """Demonstrate file upload capabilities with various file types."""
    console.print(Rule("[bold blue]File Upload Demo[/bold blue]"))
    logger.info("Demonstrating file upload capabilities", emoji_key="upload")
    
    # Navigate to a file upload demo page
    result = await browser_navigate(
        url="https://the-internet.herokuapp.com/upload",
        wait_until="load"
    )
    
    display_result("Navigated to File Upload Test Site", result)
    
    # Create some temporary files for upload
    temp_files = []
    
    # Create a text file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_txt:
        temp_txt.write(b"This is a test file created by LLM Gateway Browser Automation.\n")
        temp_txt.write(b"This file demonstrates uploading a simple text file.\n")
        temp_txt_path = temp_txt.name
        temp_files.append(temp_txt_path)
    
    # Create a small CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_csv:
        temp_csv.write(b"Name,Email,Role\n")
        temp_csv.write(b"Test User,user@example.com,Tester\n")
        temp_csv.write(b"Admin User,admin@example.com,Administrator\n")
        temp_csv.write(b"Guest User,guest@example.com,Guest\n")
        temp_csv_path = temp_csv.name
        temp_files.append(temp_csv_path)
    
    # Create a simple HTML file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_html:
        temp_html.write(b"<!DOCTYPE html>\n<html>\n<head>\n<title>Test HTML File</title>\n</head>\n")
        temp_html.write(b"<body>\n<h1>Test HTML File</h1>\n<p>This file was created by the LLM Gateway Browser Automation demo.</p>\n</body>\n</html>\n")
        temp_html_path = temp_html.name
        temp_files.append(temp_html_path)
    
    # Show the temporary files we've created for upload
    files_table = Table(title="Files Prepared for Upload", box=box.ROUNDED)
    files_table.add_column("File Path", style="cyan")
    files_table.add_column("Type", style="green")
    files_table.add_column("Size", style="yellow")
    
    for file_path in temp_files:
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
    for i, file_path in enumerate(temp_files):
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
        if i < len(temp_files) - 1:
            await browser_navigate(
                url="https://the-internet.herokuapp.com/upload",
                wait_until="load"
            )
    
    # Clean up temporary files
    for file_path in temp_files:
        try:
            os.unlink(file_path)
            logger.info(f"Deleted temporary file: {file_path}", emoji_key="cleanup")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {e}", emoji_key="warning")
    
    console.print("\n[bold green]File Upload Demo Complete[/bold green]")
    
    return {
        "files_uploaded": len(temp_files),
        "temp_files_created": len(temp_files),
        "temp_files_cleaned": len(temp_files)
    }


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


async def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the demo script."""
    parser = argparse.ArgumentParser(
        description="LLM Gateway Browser Automation Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Demo selection arguments
    demo_group = parser.add_argument_group("Demo Selection")
    demo_group.add_argument(
        "--all", action="store_true",
        help="Run all demos (default if no specific demo is selected)"
    )
    demo_group.add_argument(
        "--basics", action="store_true",
        help="Run basic navigation demo"
    )
    demo_group.add_argument(
        "--forms", action="store_true",
        help="Run form interaction demo"
    )
    demo_group.add_argument(
        "--javascript", action="store_true",
        help="Run JavaScript execution demo"
    )
    demo_group.add_argument(
        "--tabs", action="store_true",
        help="Run tab management demo"
    )
    demo_group.add_argument(
        "--auth", action="store_true",
        help="Run authentication workflow demo"
    )
    demo_group.add_argument(
        "--search", action="store_true",
        help="Run search interaction demo"
    )
    demo_group.add_argument(
        "--file-upload", action="store_true",
        help="Run file upload demo"
    )
    demo_group.add_argument(
        "--network", action="store_true",
        help="Run network monitoring demo"
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
    
    # If no specific demo is selected, default to running all
    if not any([
        args.all, args.basics, args.forms, args.javascript, 
        args.tabs, args.auth, args.search, args.file_upload, args.network
    ]):
        args.all = True
    
    return args


async def main():
    """Run browser automation demonstrations based on command-line arguments."""
    # Parse command-line arguments
    args = await parse_arguments()
    
    # Update configuration based on arguments
    global SAVE_DIR
    SAVE_DIR = Path(args.output_dir)
    
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
        
        # Run selected demos
        demos_to_run = []
        
        if args.all or args.basics:
            demos_to_run.append(("Navigation Basics", demo_navigation_basics))
            
        if args.all or args.forms:
            demos_to_run.append(("Form Interaction", demo_form_interaction))
            
        if args.all or args.javascript:
            demos_to_run.append(("JavaScript Execution", demo_javascript_execution))
            
        if args.all or args.tabs:
            demos_to_run.append(("Tab Management", demo_tab_management))
            
        if args.all or args.auth:
            demos_to_run.append(("Authentication Workflow", demo_authentication_workflow))
            
        if args.all or args.search:
            demos_to_run.append(("Search Interaction", demo_search_interaction))
            
        if args.all or args.file_upload:
            demos_to_run.append(("File Upload", demo_file_upload))
            
        if args.all or args.network:
            demos_to_run.append(("Network Monitoring", demo_network_monitoring))
        
        # Create overall progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan] demos"),
        ) as overall_progress:
            total_demo_task = overall_progress.add_task(
                "[bold blue]Running Browser Automation Demos[/bold blue]",
                total=len(demos_to_run)
            )
            
            # Run the selected demos
            for i, (demo_name, demo_func) in enumerate(demos_to_run):
                overall_progress.update(
                    total_demo_task, 
                    description=f"[bold blue]Running Demo {i+1}/{len(demos_to_run)}: {demo_name}[/bold blue]",
                    advance=0
                )
                
                console.print(Rule(f"[bold green]Running Demo: {demo_name}[/bold green]"))
                
                # Record start time for this demo
                demo_start_time = time.time()
                
                # Run the demo
                try:
                    result = await demo_func()  # noqa: F841
                    
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
    
    logger.success("Browser Automation Demo Completed Successfully", emoji_key="complete")
    console.print(Rule("[bold magenta]Browser Automation Demo Complete[/bold magenta]"))
    
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
    
    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 