#!/usr/bin/env python
"""Lightpanda browser tool demonstration for Ultimate MCP Server."""
import asyncio
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# Project imports
from ultimate_mcp_server.tools.lightpanda_browser_tool import (
    lightpanda_click,
    lightpanda_extract_links,
    lightpanda_extract_text,
    # Basic tools
    lightpanda_fetch,
    lightpanda_form_submit,
    lightpanda_get_element_info,
    lightpanda_javascript,
    lightpanda_screenshot,
    # CDP server
    lightpanda_start_cdp_server,
    lightpanda_stop_cdp_server,
)
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker
from ultimate_mcp_server.utils.logging.console import console

# Initialize logger
logger = get_logger("example.lightpanda_browser_demo")

# Keep track of CDP server port and PID
_current_port = None
_current_pid = None

async def restart_browser_connection():
    """Restart the browser connection by stopping and starting the CDP server with a unique port."""
    global _current_port, _current_pid
    
    try:
        # Try to stop any existing CDP server
        if _current_pid:
            try:
                # Try to kill the process directly
                logger.info(f"Attempting to kill CDP server process with PID {_current_pid}")
                try:
                    os.kill(_current_pid, 9)  # 9 is SIGKILL
                except ProcessLookupError:
                    pass  # Process already terminated
                except Exception as e:
                    logger.warning(f"Failed to kill process {_current_pid}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error terminating CDP process: {str(e)}")
        
        # Use force to kill any lingering Lightpanda processes
        try:
            # Find any Lightpanda processes and terminate them
            if sys.platform == "win32":
                subprocess.run("taskkill /f /im lightpanda* >nul 2>&1", shell=True)
            else:
                subprocess.run("pkill -9 -f lightpanda || true", shell=True)
        except Exception as e:
            logger.warning(f"Error force-killing Lightpanda processes: {str(e)}")
        
        # Wait a moment for processes to terminate
        await asyncio.sleep(1)
        
        # Generate a random port in the range 10000-65000
        new_port = random.randint(10000, 65000)
        _current_port = new_port
        
        # Start a new CDP server with the unique port
        logger.info(f"Starting CDP server on port {new_port}")
        start_result = await lightpanda_start_cdp_server(
            host="127.0.0.1",
            port=new_port
        )
        
        if start_result["success"]:
            _current_pid = start_result.get("server_pid")
            logger.info(f"CDP server started on port {new_port} with PID {_current_pid}")
            return True
        else:
            logger.warning(f"Failed to start CDP server on port {new_port}")
            return False
    except Exception as e:
        logger.warning(f"Error restarting browser connection: {str(e)}")
        return False

async def demonstrate_basic_fetch():
    """Demonstrate basic webpage fetching with Lightpanda."""
    console.print(Rule("[bold blue]Basic Webpage Fetching[/bold blue]"))
    logger.info("Starting basic webpage fetch example", emoji_key="start")
    
    url = "https://example.com"
    
    try:
        # Fetch the webpage
        logger.info(f"Fetching URL: {url}", emoji_key="processing")
        start_time = time.time()
        result = await lightpanda_fetch(url=url, dump_html=True)
        end_time = time.time()
        
        # Display results
        if result["success"]:
            logger.success(f"Successfully fetched {url}", emoji_key="success")
            
            # Show stats
            stats_table = Table(title="Fetch Statistics", show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            stats_table.add_row("URL", result["url"])
            stats_table.add_row("Status", str(result["status"]))
            if result.get("title"):
                stats_table.add_row("Page Title", result["title"])
            stats_table.add_row("Time Taken", f"{end_time - start_time:.3f}s")
            console.print(stats_table)
            
            # Show HTML content (truncated)
            html_content = result["html"]
            content_preview = html_content[:500] + "..." if len(html_content) > 500 else html_content
            console.print(Panel(
                escape(content_preview),
                title="HTML Content (Truncated)",
                border_style="green",
                expand=False
            ))
        else:
            logger.error(f"Failed to fetch {url}", emoji_key="error")
            
    except Exception as e:
        logger.error(f"Error in basic fetch example: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_javascript_execution():
    """Demonstrate JavaScript execution with Lightpanda."""
    console.print(Rule("[bold blue]JavaScript Execution[/bold blue]"))
    logger.info("Starting JavaScript execution example", emoji_key="start")
    
    # Restart browser connection
    await restart_browser_connection()
    
    url = "https://example.com"
    javascript = """
    // Get page information
    const data = {
        title: document.title,
        url: window.location.href,
        links: Array.from(document.querySelectorAll('a')).map(a => a.href),
        meta: {
            description: document.querySelector('meta[name="description"]')?.content,
            viewport: document.querySelector('meta[name="viewport"]')?.content
        }
    };
    
    // Return as JSON
    JSON.stringify(data, null, 2);
    """
    
    try:
        # Execute JavaScript
        logger.info(f"Executing JavaScript on {url}", emoji_key="processing")
        start_time = time.time()
        result = await lightpanda_javascript(url=url, javascript=javascript)
        end_time = time.time()
        
        # Display results
        if result["success"]:
            logger.success(f"Successfully executed JavaScript on {url}", emoji_key="success")
            
            # Show stats
            stats_table = Table(title="JavaScript Execution Statistics", show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            stats_table.add_row("URL", result["url"])
            stats_table.add_row("Time Taken", f"{end_time - start_time:.3f}s")
            console.print(stats_table)
            
            # Show JavaScript result
            js_result = result["result"]
            console.print(Panel(
                escape(js_result),
                title="JavaScript Execution Result",
                border_style="green",
                expand=False
            ))
        else:
            logger.error(f"Failed to execute JavaScript on {url}", emoji_key="error")
            
    except Exception as e:
        logger.error(f"Error in JavaScript execution example: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_link_extraction():
    """Demonstrate link extraction with Lightpanda."""
    console.print(Rule("[bold blue]Link Extraction[/bold blue]"))
    logger.info("Starting link extraction example", emoji_key="start")
    
    # Restart browser connection
    await restart_browser_connection()
    
    url = "https://nodejs.org"  # A more complex page with links
    
    try:
        # Extract links
        logger.info(f"Extracting links from {url}", emoji_key="processing")
        start_time = time.time()
        result = await lightpanda_extract_links(url=url, include_text=True)
        end_time = time.time()
        
        # Display results
        if result["success"]:
            logger.success(f"Successfully extracted {result['count']} links from {url}", emoji_key="success")
            
            # Show stats
            stats_table = Table(title="Link Extraction Statistics", show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            stats_table.add_row("URL", result["url"])
            stats_table.add_row("Links Found", str(result["count"]))
            stats_table.add_row("Time Taken", f"{end_time - start_time:.3f}s")
            console.print(stats_table)
            
            # Show links (limited)
            links_table = Table(title=f"Extracted Links (Top 10 of {result['count']})")
            links_table.add_column("URL", style="cyan")
            links_table.add_column("Text", style="white")
            
            for _i, link in enumerate(result["links"][:10]):  # Only show top 10
                href = link.get("href", "N/A")
                text = link.get("text", "").strip()
                # Truncate long text
                if len(text) > 40:
                    text = text[:40] + "..."
                links_table.add_row(href, escape(text))
            
            console.print(links_table)
        else:
            logger.error(f"Failed to extract links from {url}", emoji_key="error")
            
    except Exception as e:
        logger.error(f"Error in link extraction example: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_text_extraction():
    """Demonstrate text extraction with Lightpanda."""
    console.print(Rule("[bold blue]Text Extraction[/bold blue]"))
    logger.info("Starting text extraction example", emoji_key="start")
    
    # Restart browser connection
    await restart_browser_connection()
    
    url = "https://en.wikipedia.org/wiki/Browser_automation"  # A content-rich page
    
    try:
        # Extract text
        logger.info(f"Extracting text from {url}", emoji_key="processing")
        start_time = time.time()
        result = await lightpanda_extract_text(
            url=url,
            remove_selectors=["#mw-navigation", "#footer", ".mw-editsection"]  # Remove navigation and footer
        )
        end_time = time.time()
        
        # Display results
        if result["success"]:
            logger.success(f"Successfully extracted text from {url}", emoji_key="success")
            
            # Show stats
            stats_table = Table(title="Text Extraction Statistics", show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            stats_table.add_row("URL", result["url"])
            stats_table.add_row("Title", result["title"])
            stats_table.add_row("Word Count", str(result["word_count"]))
            stats_table.add_row("Time Taken", f"{end_time - start_time:.3f}s")
            console.print(stats_table)
            
            # Show extracted text (truncated)
            text_content = result["text"]
            content_preview = text_content[:1000] + "..." if len(text_content) > 1000 else text_content
            console.print(Panel(
                escape(content_preview),
                title="Extracted Text (Truncated)",
                border_style="green",
                expand=False
            ))
        else:
            logger.error(f"Failed to extract text from {url}", emoji_key="error")
            
    except Exception as e:
        logger.error(f"Error in text extraction example: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_screenshot():
    """Demonstrate taking screenshots with Lightpanda."""
    console.print(Rule("[bold blue]Screenshot Capture[/bold blue]"))
    logger.info("Starting screenshot example", emoji_key="start")
    
    # Restart browser connection
    await restart_browser_connection()
    
    url = "https://news.ycombinator.com/"  # Hacker News has a distinctive layout
    
    try:
        # Create temporary directory for screenshots
        temp_dir = tempfile.mkdtemp(prefix="lightpanda_screenshots_")
        logger.info(f"Created temporary directory for screenshots: {temp_dir}", emoji_key="info")
        
        # Take full page screenshot
        logger.info(f"Taking full page screenshot of {url}", emoji_key="processing")
        start_time = time.time()
        result_full = await lightpanda_screenshot(
            url=url,
            output_path=os.path.join(temp_dir, "full_page.png"),
            full_page=True
        )
        full_time = time.time() - start_time
        
        # Take element screenshot
        logger.info("Taking screenshot of a specific element", emoji_key="processing")
        start_time = time.time()
        result_element = await lightpanda_screenshot(
            url=url,
            output_path=os.path.join(temp_dir, "element.png"),
            element_selector=".title",  # Screenshot the title elements
            format="png"
        )
        element_time = time.time() - start_time
        
        # Display results
        if result_full["success"] and result_element["success"]:
            logger.success(f"Successfully captured screenshots from {url}", emoji_key="success")
            
            # Show stats
            stats_table = Table(title="Screenshot Statistics", show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            stats_table.add_row("URL", url)
            stats_table.add_row("Full Page Screenshot", result_full["file_path"])
            stats_table.add_row("Full Page Size", f"{result_full.get('width', 0)}x{result_full.get('height', 0)} px")
            stats_table.add_row("Element Screenshot", result_element["file_path"])
            stats_table.add_row("Element Size", f"{result_element.get('width', 0)}x{result_element.get('height', 0)} px")
            stats_table.add_row("Full Page Time", f"{full_time:.3f}s")
            stats_table.add_row("Element Time", f"{element_time:.3f}s")
            console.print(stats_table)
            
            console.print(f"[green]Screenshots saved to: {temp_dir}[/green]")
        else:
            if not result_full["success"]:
                logger.error(f"Failed to capture full page screenshot: {result_full.get('error')}", emoji_key="error")
            if not result_element["success"]:
                logger.error(f"Failed to capture element screenshot: {result_element.get('error')}", emoji_key="error")
            
    except Exception as e:
        logger.error(f"Error in screenshot example: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_element_interaction():
    """Demonstrate element interaction with Lightpanda."""
    console.print(Rule("[bold blue]Element Interaction[/bold blue]"))
    logger.info("Starting element interaction example", emoji_key="start")
    
    # Restart browser connection
    await restart_browser_connection()
    
    url = "https://httpbin.org/forms/post"  # A page with a form
    
    try:
        # Get element information
        logger.info(f"Getting information about form elements on {url}", emoji_key="processing")
        start_time = time.time()
        
        # Get info about the form element
        form_info_result = await lightpanda_get_element_info(
            url=url,
            selector="form",
            include_attributes=True
        )
        
        # Get info about the input elements
        input_info_result = await lightpanda_get_element_info(  # noqa: F841
            url=url,
            selector="input[name='custname']",
            include_attributes=True
        )
        
        element_info_time = time.time() - start_time
        
        # Click on a checkbox
        logger.info("Clicking on a checkbox", emoji_key="processing")
        start_time = time.time()
        click_result = await lightpanda_click(
            url=url,
            selector="input[value='medium']",  # Medium pizza size radio button
            wait_for_navigation=False
        )
        click_time = time.time() - start_time
        
        # Fill and submit the form
        logger.info("Filling and submitting the form", emoji_key="processing")
        start_time = time.time()
        form_result = await lightpanda_form_submit(
            url=url,
            form_selector="form",
            field_values={
                "input[name='custname']": "John Doe",
                "input[value='medium']": True,  # Check the medium pizza size
                "input[name='custemail']": "john@example.com",
                "input[name='size']": "medium",
                "input[name='topping']": "bacon",
                "textarea[name='comments']": "Please deliver ASAP!"
            },
            submit_button_selector="button[type='submit']",
            wait_for_navigation=True
        )
        form_time = time.time() - start_time
        
        # Display results
        console.print(Panel(
            "Element Information and Interaction Demonstration",
            border_style="green",
            expand=False
        ))
        
        # Show element info results
        if form_info_result["success"] and form_info_result["element_found"]:
            element_table = Table(title="Form Element Information")
            element_table.add_column("Property", style="cyan")
            element_table.add_column("Value", style="white")
            
            element_table.add_row("Tag Name", form_info_result["tag_name"])
            element_table.add_row("Text Content", 
                                 form_info_result["text_content"][:50] + "..." 
                                 if len(form_info_result["text_content"]) > 50 
                                 else form_info_result["text_content"])
            
            if "attributes" in form_info_result:
                for attr_name, attr_value in form_info_result["attributes"].items():
                    element_table.add_row(f"Attribute: {attr_name}", attr_value)
            
            console.print(element_table)
        
        # Show click results
        if click_result["success"]:
            console.print(f"[green]Successfully clicked on element: {click_result['element_description']}[/green]")
        
        # Show form submission results
        if form_result["success"]:
            console.print(Panel(
                f"Form submission successful!\n"
                f"Fields filled: {', '.join(form_result['fields_filled'])}\n"
                f"Navigation occurred: {form_result['navigation_occurred']}\n"
                f"Final URL: {form_result['url']}",
                title="Form Submission Results",
                border_style="green",
                expand=False
            ))
            
        # Show timing information
        timing_table = Table(title="Timing Information")
        timing_table.add_column("Operation", style="cyan")
        timing_table.add_column("Time", style="white")
        
        timing_table.add_row("Element Info Retrieval", f"{element_info_time:.3f}s")
        timing_table.add_row("Element Click", f"{click_time:.3f}s")
        timing_table.add_row("Form Fill & Submit", f"{form_time:.3f}s")
        
        console.print(timing_table)
            
    except Exception as e:
        logger.error(f"Error in element interaction example: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_cdp_server():
    """Demonstrate CDP Server with Lightpanda."""
    console.print(Rule("[bold blue]CDP Server Management[/bold blue]"))
    logger.info("Starting CDP server management example", emoji_key="start")
    
    # Clean up any previous CDP server
    await restart_browser_connection()
    
    try:
        # Start a CDP server with a specific port for demo purposes
        demo_port = 9333  # Using a different port than the default
        logger.info(f"Starting a Lightpanda CDP server on port {demo_port}", emoji_key="processing")
        start_result = await lightpanda_start_cdp_server(
            host="127.0.0.1",
            port=demo_port
        )
        
        if start_result["success"]:
            global _current_pid
            _current_pid = start_result.get("server_pid") 
            
            # For display, handle both old and new key names
            server_pid = start_result.get("server_pid") or start_result.get("pid", "Unknown")
            server_ws = start_result["ws_endpoint"]
            server_log = start_result["log_file"]
            
            logger.success(f"Started Lightpanda CDP server with PID {server_pid}", emoji_key="success")
            
            # Show server information
            server_table = Table(title="Lightpanda CDP Server")
            server_table.add_column("Property", style="cyan")
            server_table.add_column("Value", style="white")
            
            server_table.add_row("PID", str(server_pid))
            server_table.add_row("WebSocket Endpoint", server_ws)
            server_table.add_row("Log File", server_log)
            
            console.print(server_table)
            
            # Display help message about connecting to CDP
            console.print(Panel(
                "This CDP server can be used with any CDP client, including Playwright and Puppeteer.\n\n"
                f"[bold]WebSocket endpoint:[/bold] {server_ws}\n\n"
                "Example connection with Playwright:\n"
                "```python\n"
                f'browser = await playwright.chromium.connect_over_cdp("{server_ws}")\n'
                "```",
                title="CDP Connection Information",
                border_style="cyan",
                expand=False
            ))
            
            # Wait a moment before stopping
            await asyncio.sleep(1)
            
            # Stop the CDP server
            logger.info(f"Stopping Lightpanda CDP server with PID {server_pid}", emoji_key="processing")
            stop_result = await lightpanda_stop_cdp_server(pid=server_pid)
            
            if stop_result["success"]:
                logger.success(f"Successfully stopped CDP server: {stop_result['message']}", emoji_key="success")
            else:
                logger.warning(f"Failed to stop CDP server: {stop_result['message']}", emoji_key="warning")
        else:
            logger.error(f"Failed to start CDP server: {start_result.get('error', 'Unknown error')}", emoji_key="error")
            
    except Exception as e:
        logger.error(f"Error in CDP server example: {str(e)}", emoji_key="error", exc_info=True)

async def main():
    """Run the expanded Lightpanda browser tool demonstration."""
    # Create tracker for cost monitoring
    tracker = CostTracker()
    
    # Create header
    console.print(Panel(
        "[bold cyan]Lightpanda Browser Tool Expanded Demonstration[/bold cyan]\n\n"
        "This demo showcases the complete functionality of the [bold]Lightpanda[/bold] browser integration "
        "with Ultimate MCP Server. Lightpanda offers 9x less memory usage and 11x faster execution than "
        "Chrome-based browsers, combined with powerful LLM-guided automation capabilities.",
        border_style="cyan",
        expand=False
    ))
    
    try:
        # Basic functionality demonstrations
        await demonstrate_basic_fetch()
        console.print()
        
        await demonstrate_javascript_execution()
        console.print()
        
        await demonstrate_link_extraction()
        console.print()
        
        await demonstrate_text_extraction()
        console.print()
        
        # New tool demonstrations
        await demonstrate_screenshot()
        console.print()
        
        await demonstrate_element_interaction()
        console.print()
        
        # Skip functions that haven't been implemented
        # Other demos like file_operations, etc. would be here
        
        # CDP server demonstration
        await demonstrate_cdp_server()
        console.print()
        
        # Display cost tracker summary
        tracker.display_summary(console)
        
        # Final success message
        logger.success("Expanded Lightpanda Browser Tool Demonstration completed successfully!", emoji_key="complete")
        return 0
        
    except Exception as e:
        logger.critical(f"Demonstration failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)