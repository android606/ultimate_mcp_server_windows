#!/usr/bin/env python
"""Lightpanda browser tool demonstration for Ultimate MCP Server."""
import asyncio
import json
import os
import random
import re
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

# Keep track of CDP server process and port
_cdp_process = None
_cdp_port = None

async def start_cdp_server():
    """Start a CDP server for lightpanda browser with a unique port."""
    global _cdp_process, _cdp_port
    
    # Kill any existing server
    if _cdp_process:
        try:
            logger.info(f"Killing existing CDP server process", emoji_key="processing")
            _cdp_process.terminate()
            await asyncio.sleep(0.5)
            if _cdp_process.poll() is None:
                _cdp_process.kill()
        except Exception as e:
            logger.warning(f"Error killing CDP server: {str(e)}", emoji_key="warning")
    
    # Clean up any lingering processes
    try:
        if sys.platform == "win32":
            subprocess.run("taskkill /f /im lightpanda* >nul 2>&1", shell=True)
        else:
            subprocess.run("pkill -9 -f lightpanda || true", shell=True)
    except Exception as e:
        logger.warning(f"Error cleaning up: {str(e)}", emoji_key="warning")
    
    # Wait for processes to terminate
    await asyncio.sleep(1)
    
    # Use a random port to avoid conflicts
    _cdp_port = random.randint(20000, 60000)
    
    # Get path to Lightpanda binary
    lightpanda_path = os.path.expanduser("~/.ultimate_mcp_server/lightpanda/lightpanda")
    if not os.path.exists(lightpanda_path):
        logger.error(f"Lightpanda binary not found at {lightpanda_path}", emoji_key="error")
        return False
    
    # Create log directory if it doesn't exist
    log_dir = os.path.expanduser("~/.ultimate_mcp_server/lightpanda/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"cdp_server_{_cdp_port}.log")
    
    # Start CDP server using the correct 'serve' command
    logger.info(f"Starting Lightpanda CDP server on port {_cdp_port}", emoji_key="processing")
    
    with open(log_file, "w") as f:
        _cdp_process = subprocess.Popen(
            [
                lightpanda_path,
                "serve",  # Use the correct 'serve' command, not 'cdp'
                "--host", "127.0.0.1",
                "--port", str(_cdp_port)
            ],
            stdout=f,
            stderr=f,
            env=os.environ.copy(),
            start_new_session=True
        )
    
    # Wait for the server to start
    await asyncio.sleep(2)
    
    # Check if the process is still running
    if _cdp_process.poll() is not None:
        logger.error(f"CDP server process exited with code {_cdp_process.poll()}", emoji_key="error")
        with open(log_file, "r") as f:
            log_content = f.read()
            logger.error(f"CDP server log:\n{log_content}", emoji_key="error")
        return False
    
    logger.success(f"CDP server started on port {_cdp_port}", emoji_key="success")
    return True

async def run_javascript(url, javascript, timeout=30):
    """Run JavaScript on a webpage using Playwright connected to our CDP server."""
    from playwright.async_api import async_playwright
    
    if not _cdp_process or _cdp_process.poll() is not None:
        logger.error("CDP server not running", emoji_key="error")
        return {"success": False, "error": "CDP server not running"}
    
    try:
        ws_endpoint = f"ws://127.0.0.1:{_cdp_port}"
        logger.info(f"Connecting to CDP server at {ws_endpoint}", emoji_key="processing")
        
        # Start playwright and connect to the CDP server
        playwright = await async_playwright().start()
        try:
            browser = await playwright.chromium.connect_over_cdp(ws_endpoint)
            try:
                # Create a page and navigate to the URL
                context = await browser.new_context()
                try:
                    page = await context.new_page()
                    try:
                        # Navigate to the URL with timeout
                        await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                        
                        # Execute the JavaScript
                        result = await page.evaluate(javascript)
                        
                        # Ensure result is a string
                        if not isinstance(result, str):
                            if result is None:
                                result = "null"
                            else:
                                result = json.dumps(result)
                        
                        return {
                            "success": True,
                            "result": result,
                            "url": page.url
                        }
                    finally:
                        await page.close()
                finally:
                    await context.close()
            finally:
                await browser.close()
        finally:
            await playwright.stop()
    except Exception as e:
        logger.error(f"JavaScript execution error: {str(e)}", emoji_key="error")
        return {"success": False, "error": str(e)}

async def extract_links(url, timeout=30):
    """Extract links from a webpage using Playwright connected to our CDP server."""
    from playwright.async_api import async_playwright
    
    if not _cdp_process or _cdp_process.poll() is not None:
        logger.error("CDP server not running", emoji_key="error")
        return {"success": False, "error": "CDP server not running"}
    
    try:
        ws_endpoint = f"ws://127.0.0.1:{_cdp_port}"
        logger.info(f"Connecting to CDP server at {ws_endpoint}", emoji_key="processing")
        
        # Start playwright and connect to the CDP server
        playwright = await async_playwright().start()
        try:
            browser = await playwright.chromium.connect_over_cdp(ws_endpoint)
            try:
                # Create a page and navigate to the URL
                context = await browser.new_context()
                try:
                    page = await context.new_page()
                    try:
                        # Navigate to the URL with timeout
                        await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                        
                        # Extract links using JavaScript
                        links = await page.evaluate("""() => {
                            const links = Array.from(document.querySelectorAll('a'));
                            return links.map(link => {
                                const result = {};
                                result.href = link.href || '';
                                result.text = link.textContent.trim() || '';
                                return result;
                            });
                        }""")
                        
                        return {
                            "success": True,
                            "links": links,
                            "count": len(links),
                            "url": page.url
                        }
                    finally:
                        await page.close()
                finally:
                    await context.close()
            finally:
                await browser.close()
        finally:
            await playwright.stop()
    except Exception as e:
        logger.error(f"Link extraction error: {str(e)}", emoji_key="error")
        return {"success": False, "error": str(e)}

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
    
    # Start CDP server for this demo
    if not await start_cdp_server():
        logger.error("Failed to start CDP server for JavaScript execution", emoji_key="error")
        console.print(Panel(
            "[red]Failed to start CDP server for JavaScript execution[/red]",
            title="JavaScript Execution Error",
            border_style="red"
        ))
        return
    
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
        
        # Use our custom JavaScript execution function
        result = await run_javascript(url=url, javascript=javascript, timeout=10)
        
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
            logger.error(f"Failed to execute JavaScript on {url}: {result.get('error', 'Unknown error')}", emoji_key="error")
            
    except Exception as e:
        logger.error(f"Error in JavaScript execution example: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_link_extraction():
    """Demonstrate link extraction with Lightpanda."""
    console.print(Rule("[bold blue]Link Extraction[/bold blue]"))
    logger.info("Starting link extraction example", emoji_key="start")
    
    # The CDP server should already be running from the JavaScript demo
    if not _cdp_process or _cdp_process.poll() is not None:
        logger.info("CDP server not running, starting it", emoji_key="processing")
        if not await start_cdp_server():
            logger.error("Failed to start CDP server for link extraction", emoji_key="error")
            console.print(Panel(
                "[red]Failed to start CDP server for link extraction[/red]",
                title="Link Extraction Error",
                border_style="red"
            ))
            return
    
    url = "https://nodejs.org"  # A more complex page with links
    
    try:
        # Extract links
        logger.info(f"Extracting links from {url}", emoji_key="processing")
        start_time = time.time()
        
        # Use our custom link extraction function
        result = await extract_links(url=url, timeout=10)
        
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
            
            for link in result["links"][:10]:  # Only show top 10
                href = link.get("href", "N/A")
                text = link.get("text", "").strip()
                # Truncate long text
                if len(text) > 40:
                    text = text[:40] + "..."
                links_table.add_row(href, escape(text))
            
            console.print(links_table)
        else:
            logger.error(f"Failed to extract links from {url}: {result.get('error', 'Unknown error')}", emoji_key="error")
            
    except Exception as e:
        logger.error(f"Error in link extraction example: {str(e)}", emoji_key="error", exc_info=True)

async def main():
    """Run the Lightpanda browser tool demonstration."""
    # Create tracker for cost monitoring
    tracker = CostTracker()
    
    # Create header
    console.print(Panel(
        "[bold cyan]Lightpanda Browser Tool Demonstration[/bold cyan]\n\n"
        "This demo showcases the functionality of the [bold]Lightpanda[/bold] browser integration "
        "with Ultimate MCP Server. Lightpanda offers 9x less memory usage and 11x faster execution than "
        "Chrome-based browsers, combined with powerful automation capabilities.",
        border_style="cyan",
        expand=False
    ))
    
    try:
        # Log system and environment information for diagnostics
        logger.info(f"Python version: {sys.version}", emoji_key="info")
        logger.info(f"Running on platform: {sys.platform}", emoji_key="info")
        logger.info(f"Working directory: {os.getcwd()}", emoji_key="info")
        
        # Basic functionality demonstration
        await demonstrate_basic_fetch()
        console.print()
        
        # JavaScript execution demonstration (using our fixed function)
        await demonstrate_javascript_execution()
        console.print()
        
        # Link extraction demonstration (using our fixed function)
        await demonstrate_link_extraction()
        console.print()
        
        # Cleanup CDP process
        global _cdp_process
        if _cdp_process:
            logger.info("Cleaning up CDP server process", emoji_key="processing")
            _cdp_process.terminate()
            await asyncio.sleep(1)
            if _cdp_process.poll() is None:
                _cdp_process.kill()
        
        # Display cost tracker summary
        tracker.display_summary(console)
        
        # Final success message
        logger.success("Lightpanda Browser Tool Demonstration completed successfully!", emoji_key="complete")
        return 0
        
    except Exception as e:
        logger.critical(f"Demonstration failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    finally:
        # Make sure we clean up the CDP server in case of any exceptions
        if _cdp_process and _cdp_process.poll() is None:
            try:
                _cdp_process.terminate()
                await asyncio.sleep(0.5)
                if _cdp_process.poll() is None:
                    _cdp_process.kill()
            except:
                pass


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)