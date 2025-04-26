#!/usr/bin/env python
"""Lightpanda browser tool demonstration for Ultimate MCP Server."""
import asyncio
import os
import subprocess
import sys
import signal
import time
import json
import random
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.markup import escape

# Project imports
from ultimate_mcp_server.tools.lightpanda_browser_tool import (
    # Basic tools
    lightpanda_fetch,
)

# Initialize console
console = Console()

# Global variables
cdp_process = None
cdp_port = None

# Set up a signal handler to prevent the script from being killed
def setup_signal_handler():
    def signal_handler(signum, frame):
        console.print("[red]Signal received. Cleaning up and exiting...[/red]")
        cleanup_and_exit()
    
    # Set up handlers for common signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set up SIGALRM only on platforms that support it (not Windows)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, signal_handler)

def cleanup_and_exit():
    """Clean up resources and exit the script."""
    global cdp_process
    
    if cdp_process and cdp_process.poll() is None:
        console.print("Cleaning up CDP server process...")
        try:
            cdp_process.terminate()
            time.sleep(0.5)
            if cdp_process.poll() is None:
                cdp_process.kill()
        except:
            pass
    
    sys.exit(0)

async def start_cdp_server():
    """Start the Chrome DevTools Protocol server."""
    global cdp_process, cdp_port
    
    # Find a random port between 20000 and 60000
    cdp_port = random.randint(20000, 60000)
    
    # Ensure no existing processes are using this port
    try:
        output = subprocess.check_output(["lsof", "-i", f":{cdp_port}"], stderr=subprocess.STDOUT)
        console.print(f"[yellow]Warning: Port {cdp_port} is already in use.[/yellow]")
        console.print(output.decode())
        await cleanup_and_exit(1)
        return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        # This is actually good - it means the port is free
        pass
    
    try:
        # Start the local CDP server (lightpanda) 
        cdp_cmd = ["python", "-m", "lightpanda.server", "--port", str(cdp_port), "--headless", "--no-sandbox"]
        console.print(f"Starting CDP server on port {cdp_port}: {' '.join(cdp_cmd)}")
        
        cdp_process = subprocess.Popen(
            cdp_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Wait for the server to start
        await asyncio.sleep(10)
        
        if cdp_process.poll() is not None:
            stdout, stderr = cdp_process.communicate()
            console.print(f"[red]Failed to start CDP server[/red]")
            console.print(f"[red]stdout: {stdout.decode()}[/red]")
            console.print(f"[red]stderr: {stderr.decode()}[/red]")
            return False
        
        console.print(f"[green]CDP server started on port {cdp_port}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error starting CDP server: {str(e)}[/red]")
        return False

async def execute_javascript(url, javascript, timeout=60):
    """Run JavaScript on a webpage using Playwright connected to our CDP server."""
    if not cdp_process or cdp_process.poll() is not None:
        console.print("[red]CDP server not running. Cannot execute JavaScript.[/red]")
        return {"success": False, "error": "CDP server not running"}
    
    try:
        from playwright.async_api import async_playwright
        
        ws_endpoint = f"ws://127.0.0.1:{cdp_port}"
        console.print(f"Connecting to CDP server at {ws_endpoint}")
        
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
                        # Navigate to the URL with increased timeout (5x the original timeout value)
                        await page.goto(url, timeout=timeout * 5000, wait_until="networkidle")
                        
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
        console.print(f"[red]JavaScript execution error: {str(e)}[/red]")
        return {"success": False, "error": str(e)}

async def demonstrate_basic_fetch():
    """Demonstrate basic webpage fetching with Lightpanda."""
    console.print(Rule("[bold blue]Basic Webpage Fetching[/bold blue]"))
    
    url = "https://example.com"
    
    try:
        # Fetch the webpage
        console.print(f"Fetching URL: {url}")
        start_time = time.time()
        result = await lightpanda_fetch(url=url, dump_html=True)
        end_time = time.time()
        
        # Display results
        if result["success"]:
            console.print(f"[green]Successfully fetched {url}[/green]")
            
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
            console.print(f"[red]Failed to fetch {url}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error in basic fetch example: {str(e)}[/red]")

async def demonstrate_javascript_execution():
    """Demonstrate executing JavaScript on a webpage."""
    console.rule("[bold blue]JavaScript Execution Demo[/bold blue]")
    
    # Basic Google search and page info extraction
    url = "https://www.google.com"
    
    # JavaScript to extract information about the page
    javascript = """
    () => {
        return {
            title: document.title,
            url: window.location.href,
            links: Array.from(document.querySelectorAll('a')).length,
            images: Array.from(document.querySelectorAll('img')).length,
            scripts: Array.from(document.querySelectorAll('script')).length,
            timestamp: new Date().toString()
        };
    }
    """
    
    console.print(f"[blue]Executing JavaScript on {url}...[/blue]")
    
    try:
        # Execute JavaScript with longer timeout
        result = await execute_javascript(url=url, javascript=javascript, timeout=60)
        
        if result["success"]:
            # Parse the JSON result
            data = json.loads(result["result"])
            
            # Display the results in a table
            table = Table(title="JavaScript Execution Results")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in data.items():
                table.add_row(key, str(value))
            
            console.print(table)
        else:
            console.print(f"[red]Failed to execute JavaScript: {result.get('error', 'Unknown error')}[/red]")
    
    except Exception as e:
        console.print(f"[red]Error in JavaScript demo: {str(e)}[/red]")

async def main():
    """Run the Lightpanda browser demonstration."""
    # Set up signal handlers to prevent being killed
    setup_signal_handler()
    
    # Create header
    console.print(Panel(
        "[bold cyan]Lightpanda Browser Tool Demonstration[/bold cyan]\n\n"
        "This demo showcases the functionality of the Lightpanda browser integration "
        "with Ultimate MCP Server. Lightpanda offers 9x less memory usage and 11x faster execution than "
        "Chrome-based browsers, combined with powerful automation capabilities.",
        title="Lightpanda Demo",
        border_style="cyan",
        expand=False
    ))
    
    try:
        # Basic functionality demonstration
        await demonstrate_basic_fetch()
        console.print()
        
        # JavaScript execution demonstration
        await demonstrate_javascript_execution()
        console.print()
        
        console.print("[green]Demo completed successfully![/green]")
        return 0
    except Exception as e:
        console.print(f"[red]Demo failed: {str(e)}[/red]")
        return 1
    finally:
        # Clean up resources
        cleanup_and_exit()


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 