#!/usr/bin/env python
"""Demo of Smart Browser tool capabilities in Ultimate MCP Server."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# Project imports
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.tools.smart_browser import SmartBrowserTool
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker

# Initialize logger and console for rich output
logger = get_logger("example.smart_browser_demo")
console = Console()

# Cost tracker to monitor API usage
tracker = CostTracker()

async def run_basic_browsing(browser_tool):
    """Demonstrate basic browsing capabilities."""
    logger.info("Starting basic browsing demo", emoji_key="start")
    console.print(Rule("[bold blue]Basic Web Browsing[/bold blue]"))
    
    url = "https://example.com"
    try:
        result = await browser_tool.browse_url(url=url)
        
        if result.get("success"):
            page_state = result.get("page_state", {})
            
            # Create a summary table
            table = Table(title=f"Page Information: {page_state.get('title', 'Unknown')}")
            table.add_column("Property", style="green")
            table.add_column("Value", style="white")
            
            table.add_row("URL", page_state.get("url", ""))
            table.add_row("Title", page_state.get("title", ""))
            table.add_row("Element Count", str(len(page_state.get("elements", []))))
            
            # Show text summary snippet
            text_summary = page_state.get("text_summary", "")
            if text_summary:
                text_preview = text_summary[:100] + "..." if len(text_summary) > 100 else text_summary
                table.add_row("Text Preview", text_preview)
            
            console.print(table)
            logger.success("Successfully browsed to example.com", emoji_key="success")
        else:
            logger.error(f"Failed to browse: {result.get('error', 'Unknown error')}", emoji_key="error")
    
    except Exception as e:
        logger.error(f"Error in basic browsing: {str(e)}", emoji_key="error", exc_info=True)


async def run_element_interaction(browser_tool):
    """Demonstrate smart element interaction."""
    logger.info("Starting element interaction demo", emoji_key="start")
    console.print(Rule("[bold blue]Smart Element Interaction[/bold blue]"))
    
    url = "https://httpbin.org/forms/post"
    try:
        # First browse to the URL
        browse_result = await browser_tool.browse_url(url=url)
        if not browse_result.get("success"):
            raise RuntimeError(f"Failed to browse to {url}")
            
        # Click on a checkbox
        click_result = await browser_tool.click_and_extract(
            url=url,
            target={"name": "Default", "role": "checkbox"},
            wait_ms=1000
        )
        
        if click_result.get("success"):
            logger.success("Successfully clicked on checkbox", emoji_key="success")
            
            # Display updated page state
            page_state = click_result.get("page_state", {})
            console.print(Panel(
                f"Clicked on checkbox. Found {len(page_state.get('elements', []))} elements on page.",
                title="Element Interaction",
                border_style="green"
            ))
        else:
            logger.error(f"Failed to click: {click_result.get('error', 'Unknown error')}", emoji_key="error")
    
    except Exception as e:
        logger.error(f"Error in element interaction: {str(e)}", emoji_key="error", exc_info=True)


async def run_form_filling(browser_tool):
    """Demonstrate form filling capabilities."""
    logger.info("Starting form filling demo", emoji_key="start")
    console.print(Rule("[bold blue]Form Filling[/bold blue]"))
    
    url = "https://httpbin.org/forms/post"
    try:
        # Define form fields to fill
        form_fields = [
            {
                "target": {"role": "textbox", "name": "custname"},
                "text": "John Doe"
            },
            {
                "target": {"role": "textbox", "name": "custtel"},
                "text": "555-123-4567"
            },
            {
                "target": {"role": "textbox", "name": "custemail"},
                "text": "john.doe@example.com"
            },
            {
                "target": {"role": "radio", "name": "size", "css": "input[value='medium']"},
                "text": ""  # No text needed for radio buttons, just clicking
            }
        ]
        
        # Define submit button
        submit_button = {"role": "button", "name": "Submit"}
        
        # Fill and submit form
        result = await browser_tool.fill_form(
            url=url,
            form_fields=form_fields,
            submit_button=submit_button,
            return_result=True
        )
        
        if result.get("success"):
            logger.success("Successfully filled and submitted form", emoji_key="success")
            
            # Display confirmation
            page_state = result.get("page_state", {})
            if page_state:
                title = page_state.get("title", "Unknown")
                url = page_state.get("url", "")
                console.print(Panel(
                    f"Form submitted successfully\nRedirected to: {url}\nPage title: {title}",
                    title="Form Submission Result",
                    border_style="green"
                ))
        else:
            logger.error(f"Failed to fill form: {result.get('error', 'Unknown error')}", emoji_key="error")
    
    except Exception as e:
        logger.error(f"Error in form filling: {str(e)}", emoji_key="error", exc_info=True)


async def run_web_search(browser_tool):
    """Demonstrate web search capabilities."""
    logger.info("Starting web search demo", emoji_key="start")
    console.print(Rule("[bold blue]Web Search[/bold blue]"))
    
    query = "python programming best practices"
    engines = ["bing", "duckduckgo", "yandex"]
    
    for engine in engines:
        try:
            console.print(f"[bold cyan]Searching with {engine}...[/bold cyan]")
            
            result = await browser_tool.search(
                query=query,
                engine=engine,
                max_results=5
            )
            
            if result.get("success"):
                search_results = result.get("results", [])
                
                # Create a table to display results
                table = Table(title=f"Search Results: {engine}")
                table.add_column("#", style="cyan", width=3)
                table.add_column("Title", style="white")
                table.add_column("URL", style="blue")
                
                for i, item in enumerate(search_results, 1):
                    title = item.get("title", "No title")
                    url = item.get("url", "")
                    # Truncate long titles
                    if len(title) > 50:
                        title = title[:47] + "..."
                    # Truncate long URLs
                    if len(url) > 60:
                        url = url[:57] + "..."
                    
                    table.add_row(str(i), title, url)
                
                console.print(table)
                logger.success(f"Found {len(search_results)} results with {engine}", emoji_key="success")
            else:
                logger.error(f"Search with {engine} failed: {result.get('error', 'Unknown error')}", emoji_key="error")
                
            # Add a small delay before the next search to avoid overwhelming search engines
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in {engine} search: {str(e)}", emoji_key="error", exc_info=True)
            
        # Print separator between engines
        if engine != engines[-1]:
            console.print()


async def run_file_download(browser_tool):
    """Demonstrate file download capabilities."""
    logger.info("Starting file download demo", emoji_key="start")
    console.print(Rule("[bold blue]File Download[/bold blue]"))
    
    # Example PDF download from a reliable source
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    try:
        # For PDFs from w3.org, we can directly download without a browser click
        dest_dir = "demo_downloads"
        
        # Use the download_site_pdfs tool which handles direct HTTP downloads
        result = await browser_tool.download_site_pdfs(
            start_url=url,
            dest_subfolder=dest_dir,
            max_pdfs=1
        )
        
        if result.get("success"):
            files = result.get("files", [])
            if files:
                file_info = files[0]
                
                # Create a summary table
                table = Table(title="File Download Information")
                table.add_column("Property", style="green")
                table.add_column("Value", style="white")
                
                table.add_row("Filename", Path(file_info.get("file", "")).name)
                table.add_row("Path", file_info.get("file", ""))
                table.add_row("Size", f"{file_info.get('size', 0):,} bytes")
                table.add_row("SHA-256", file_info.get("sha256", ""))
                
                console.print(table)
                logger.success("Successfully downloaded file", emoji_key="success")
            else:
                logger.warning("No files were downloaded", emoji_key="warning")
        else:
            logger.error(f"Failed to download file: {result.get('error', 'Unknown error')}", emoji_key="error")
    
    except Exception as e:
        logger.error(f"Error in file download: {str(e)}", emoji_key="error", exc_info=True)


async def run_documentation_collection(browser_tool):
    """Demonstrate documentation collection capabilities."""
    logger.info("Starting documentation collection demo", emoji_key="start")
    console.print(Rule("[bold blue]Documentation Collection[/bold blue]"))
    
    # Collect documentation for a well-known Python package
    package = "requests"
    
    try:
        result = await browser_tool.collect_documentation(
            package=package,
            max_pages=5  # Limit to 5 pages for the demo
        )
        
        if result.get("success"):
            # Create a summary table
            table = Table(title=f"Documentation Collection: {package}")
            table.add_column("Property", style="green")
            table.add_column("Value", style="white")
            
            table.add_row("Package", result.get("package", ""))
            table.add_row("Pages Collected", str(result.get("pages", 0)))
            table.add_row("Output File", result.get("file", ""))
            
            console.print(table)
            
            # If we have a file, show a preview
            file_path = result.get("file", "")
            if file_path:
                try:
                    file_content = Path(file_path).read_text(errors="replace")
                    preview_length = min(500, len(file_content))
                    preview = file_content[:preview_length] + ("..." if preview_length < len(file_content) else "")
                    
                    console.print(Panel(
                        preview,
                        title="Documentation Preview",
                        border_style="cyan",
                        width=100
                    ))
                except Exception as file_error:
                    logger.warning(f"Could not read file for preview: {str(file_error)}", emoji_key="warning")
            
            logger.success(f"Successfully collected {result.get('pages', 0)} pages of documentation", emoji_key="success")
        else:
            logger.error(f"Failed to collect documentation: {result.get('error', 'Unknown error')}", emoji_key="error")
    
    except Exception as e:
        logger.error(f"Error in documentation collection: {str(e)}", emoji_key="error", exc_info=True)


async def run_natural_language_macro(browser_tool):
    """Demonstrate natural language macro execution."""
    logger.info("Starting natural language macro demo", emoji_key="start")
    console.print(Rule("[bold blue]Natural Language Macro[/bold blue]"))
    
    url = "https://httpbin.org/forms/post"
    task = "Fill out the customer name field with 'Jane Smith', select the medium size pizza, add extra cheese and mushrooms as toppings, and submit the form."
    
    try:
        # Execute the natural language task
        result = await browser_tool.execute_macro(
            url=url,
            task=task,
            max_rounds=3
        )
        
        if result.get("success"):
            steps = result.get("steps", [])
            
            # Create a summary table of steps executed
            table = Table(title="Macro Execution Steps")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Action", style="green")
            table.add_column("Target", style="white")
            table.add_column("Success", style="yellow")
            
            for i, step in enumerate(steps, 1):
                action = step.get("action", "unknown")
                target_info = ""
                
                # Format target information based on action type
                if "target" in step:
                    target = step.get("target", {})
                    if "name" in target:
                        target_info = f"name='{target['name']}'"
                    elif "role" in target:
                        target_info = f"role='{target['role']}'"
                    else:
                        target_info = str(target)
                elif action == "type":
                    text = step.get("text", "")
                    target_info = f"{text} → {step.get('target', {})}"
                elif action == "wait":
                    target_info = f"{step.get('ms', 1000)}ms"
                elif action == "finish":
                    target_info = "completed task"
                
                success = "✓" if step.get("success", False) else "✗"
                
                table.add_row(str(i), action, target_info, success)
            
            console.print(table)
            
            # Display the final page state
            final_state = result.get("final_state", {})
            if final_state:
                console.print(Panel(
                    f"Final page: {final_state.get('url', '')}\nTitle: {final_state.get('title', '')}\n"
                    f"Elements: {len(final_state.get('elements', []))}",
                    title="Final State After Macro Execution",
                    border_style="green"
                ))
            
            logger.success(f"Successfully executed macro with {len(steps)} steps", emoji_key="success")
        else:
            logger.error(f"Failed to execute macro: {result.get('error', 'Unknown error')}", emoji_key="error")
    
    except Exception as e:
        logger.error(f"Error in natural language macro: {str(e)}", emoji_key="error", exc_info=True)


async def run_parallel_processing(browser_tool):
    """Demonstrate parallel URL processing."""
    logger.info("Starting parallel processing demo", emoji_key="start")
    console.print(Rule("[bold blue]Parallel URL Processing[/bold blue]"))
    
    # List of URLs to process in parallel
    urls = [
        "https://example.com",
        "https://httpbin.org",
        "https://www.w3.org"
    ]
    
    try:
        console.print(f"Processing {len(urls)} URLs in parallel...\n")
        
        start_time = time.time()
        result = await browser_tool.parallel_process(
            urls=urls,
            max_tabs=3
        )
        total_time = time.time() - start_time
        
        if result.get("success"):
            url_results = result.get("results", [])
            
            # Create a summary table
            table = Table(title=f"Parallel Processing Results (completed in {total_time:.2f} seconds)")
            table.add_column("URL", style="blue")
            table.add_column("Title", style="white")
            table.add_column("Elements", style="cyan")
            table.add_column("Status", style="green")
            
            for item in url_results:
                url = item.get("url", "")
                success = item.get("success", False)
                status = "✓ Success" if success else f"✗ Failed: {item.get('error', 'Unknown error')}"
                
                if success and "state" in item:
                    state = item.get("state", {})
                    title = state.get("title", "Unknown")
                    elements = str(len(state.get("elements", [])))
                else:
                    title = "N/A"
                    elements = "N/A"
                
                table.add_row(url, title, elements, status)
            
            console.print(table)
            logger.success(f"Successfully processed {len(urls)} URLs in parallel", emoji_key="success")
        else:
            logger.error(f"Failed to process URLs in parallel: {result.get('error', 'Unknown error')}", emoji_key="error")
    
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}", emoji_key="error", exc_info=True)


async def run_autopilot(browser_tool):
    """Demonstrate the autopilot feature."""
    logger.info("Starting autopilot demo", emoji_key="start")
    console.print(Rule("[bold blue]Autopilot Demo[/bold blue]"))
    
    task = "Search for 'python programming tutorials', find a good resource, and collect some basic information about Python programming concepts"
    
    try:
        console.print(f"Running autopilot with task: '{task}'")
        
        result = await browser_tool.autopilot(
            task=task,
            max_steps=10
        )
        
        if result.get("success"):
            # Display execution summary
            steps_executed = result.get("steps_executed", 0)
            run_log = result.get("run_log", "")
            results = result.get("results", [])
            
            console.print(Panel(
                f"Steps executed: {steps_executed}\nLog file: {run_log}",
                title="Autopilot Execution Summary",
                border_style="green"
            ))
            
            # Show the most recent results
            if results:
                table = Table(title="Recent Step Results")
                table.add_column("Tool", style="cyan")
                table.add_column("Success", style="green")
                table.add_column("Details", style="white")
                
                for step in results:
                    tool = step.get("tool", "unknown")
                    success = "✓" if step.get("success", False) else "✗"
                    
                    # Format details based on what's available
                    details = ""
                    if "result" in step:
                        result_data = step.get("result", {})
                        if "page_state" in result_data:
                            page = result_data.get("page_state", {})
                            details = f"URL: {page.get('url', 'N/A')}, Title: {page.get('title', 'N/A')}"
                        elif "results" in result_data and isinstance(result_data["results"], list):
                            details = f"Found {len(result_data['results'])} results"
                        else:
                            details = str(result_data)[:50] + "..." if len(str(result_data)) > 50 else str(result_data)
                    elif "error" in step:
                        details = f"Error: {step.get('error')}"
                    
                    table.add_row(tool, success, details)
                
                console.print(table)
            
            logger.success(f"Successfully executed autopilot with {steps_executed} steps", emoji_key="success")
        else:
            logger.error(f"Failed to execute autopilot: {result.get('error', 'Unknown error')}", emoji_key="error")
    
    except Exception as e:
        logger.error(f"Error in autopilot: {str(e)}", emoji_key="error", exc_info=True)


async def main():
    """Run all demo features of the Smart Browser tool."""
    console.print("[bold green]Smart Browser Tool Demo[/bold green]")
    console.print("This demo showcases the capabilities of the Smart Browser tool in Ultimate MCP Server.\n")
    
    try:
        # Create a gateway instance
        gateway = Gateway("smart-browser-demo", register_tools=False)
        
        # Initialize the Smart Browser tool
        browser_tool = SmartBrowserTool(gateway)
        
        # Run the demos with a menu
        demos = [
            ("Basic Web Browsing", run_basic_browsing),
            ("Smart Element Interaction", run_element_interaction),
            ("Form Filling", run_form_filling),
            ("Web Search", run_web_search),
            ("File Download", run_file_download),
            ("Documentation Collection", run_documentation_collection),
            ("Natural Language Macro", run_natural_language_macro),
            ("Parallel URL Processing", run_parallel_processing),
            ("Autopilot", run_autopilot)
        ]
        
        # Present a menu to select demo features
        table = Table(title="Available Demos")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Demo", style="green")
        table.add_column("Description", style="white")
        
        descriptions = [
            "Navigate to a web page and extract information",
            "Interact with page elements using smart locator",
            "Fill out and submit a form with multiple fields",
            "Search the web using different search engines",
            "Download files and extract information",
            "Collect documentation from project websites",
            "Execute complex tasks described in natural language",
            "Process multiple URLs concurrently",
            "Run multi-step tasks with automatic planning"
        ]
        
        for i, (name, _) in enumerate(demos, 1):
            table.add_row(str(i), name, descriptions[i-1])
        
        console.print(table)
        console.print()
        
        # Ask which demos to run
        console.print("[bold yellow]Select demos to run (comma-separated numbers, 'all' for all demos, or 'q' to quit):[/bold yellow]")
        choice = input("> ").strip().lower()
        
        if choice == 'q':
            console.print("[bold red]Exiting demo[/bold red]")
            return 0
        
        selected_demos = []
        if choice == 'all':
            selected_demos = list(range(len(demos)))
        else:
            try:
                for num in choice.split(','):
                    idx = int(num.strip()) - 1
                    if 0 <= idx < len(demos):
                        selected_demos.append(idx)
                    else:
                        console.print(f"[bold red]Invalid demo number: {int(num.strip())}[/bold red]")
            except ValueError:
                console.print("[bold red]Invalid input. Please enter numbers separated by commas.[/bold red]")
                return 1
        
        if not selected_demos:
            console.print("[bold red]No valid demos selected. Exiting.[/bold red]")
            return 1
        
        # Run selected demos
        for idx in selected_demos:
            name, demo_func = demos[idx]
            console.print(f"\n[bold magenta]Running {name} Demo[/bold magenta]")
            await demo_func(browser_tool)
            
            # Add separator between demos
            if idx != selected_demos[-1]:
                console.print("\n" + "="*80 + "\n")
        
        console.print("\n[bold green]Demo completed![/bold green]")
        
        # Ensure proper shutdown
        await browser_tool.tab_pool.cancel_all()
        
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demos
    exit_code = asyncio.run(main())
    sys.exit(exit_code)