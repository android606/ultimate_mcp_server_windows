#!/usr/bin/env python
"""Document conversion examples using Ultimate MCP Server."""
import asyncio
import sys
import time
from pathlib import Path

# Add direct stderr output for debugging
sys.stderr.write("Script starting...\n")
sys.stderr.flush()

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Debug information
sys.stderr.write("Starting document conversion demo with debugging enabled...\n")
sys.stderr.write(f"Python version: {sys.version}\n")
sys.stderr.write(f"Current path: {Path.cwd()}\n")
sys.stderr.flush()

try:
    # Third-party imports
    import httpx
    from rich import box
    from rich.markup import escape
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    sys.stderr.write("Imported third-party libraries...\n")
    sys.stderr.flush()

    # Project imports
    from ultimate_mcp_server.core.server import Gateway
    from ultimate_mcp_server.tools.document_conversion_tool import convert_document
    from ultimate_mcp_server.utils import get_logger
    from ultimate_mcp_server.utils.display import CostTracker
    from ultimate_mcp_server.utils.logging.console import console
    sys.stderr.write("Successfully imported project modules\n")
    sys.stderr.flush()
except Exception as e:
    sys.stderr.write(f"Import error: {str(e)}\n")
    sys.stderr.flush()
    raise

# Initialize logger
logger = get_logger("example.document_conversion_demo")

# Initialize MCP server (Gateway doesn't need manual tool registration for direct calls)
mcp = Gateway("document-conversion-demo", register_tools=False)

# Sample document URLs to use in the examples
SAMPLE_DOCUMENTS = {
    "pdf": "https://arxiv.org/pdf/2408.09869.pdf",  # Docling technical report
    "docx": "https://filesamples.com/samples/document/docx/sample3.docx",  # Generic sample docx
    "html": "https://docling-project.github.io/docling/",  # Docling documentation page
}

# Sample local files - these will be created if they don't exist
LOCAL_TEST_DIR = Path("test_documents")
LOCAL_TEST_FILES = {
    "pdf": LOCAL_TEST_DIR / "sample.pdf",
    "docx": LOCAL_TEST_DIR / "sample.docx",
    "html": LOCAL_TEST_DIR / "sample.html",
}


async def ensure_test_files_exist():
    """Ensure test files exist by downloading them if necessary using httpx."""
    LOCAL_TEST_DIR.mkdir(exist_ok=True, parents=True)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for file_type, file_path in LOCAL_TEST_FILES.items():
            if not file_path.exists():
                # Get the URL for this file type
                url = SAMPLE_DOCUMENTS.get(file_type)
                if not url:
                    logger.warning(
                        f"No download URL defined for {file_type}. Skipping download.",
                        emoji_key="warning"
                    )
                    continue
                
                logger.info(f"Downloading {file_type} sample from {url}...", emoji_key="download")
                try:
                    # Perform the download
                    response = await client.get(url)
                    response.raise_for_status()  # Raise exception for HTTP errors
                    
                    # Save the file
                    file_path.write_bytes(response.content)
                    
                    logger.success(
                        f"Successfully downloaded {file_type} sample to {file_path}",
                        emoji_key="success"
                    )
                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"HTTP error downloading {url}: {e.response.status_code}",
                        emoji_key="error"
                    )
                except httpx.RequestError as e:
                    logger.error(
                        f"Network error downloading {url}: {str(e)}",
                        emoji_key="error"
                    )
                except Exception as e:
                    logger.error(
                        f"Error downloading {url}: {str(e)}",
                        emoji_key="error",
                        exc_info=True
                    )
            else:
                logger.info(f"Local test file {file_path} already exists", emoji_key="info")

def display_conversion_result(title: str, result: dict, content_preview: bool = True):
    """Display document conversion results in a formatted way."""
    # Create table for metadata and stats
    stats_table = Table(title=f"{title} Stats", box=box.ROUNDED, show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    
    # Add rows for each piece of metadata, safely handling missing values
    stats_table.add_row("Output Format", result.get("output_format", "unknown"))
    stats_table.add_row("Processing Time", f"{result.get('processing_time', 0):.2f}s")
    
    # Add document metadata if available, with safe access
    if "document_metadata" in result and result["document_metadata"]:
        metadata = result["document_metadata"]
        stats_table.add_row("Pages", str(metadata.get("num_pages", "N/A")))
        stats_table.add_row("Has Tables", "✓" if metadata.get("has_tables", False) else "✗")
        stats_table.add_row("Has Figures", "✓" if metadata.get("has_figures", False) else "✗")
        stats_table.add_row("Has Sections", "✓" if metadata.get("has_sections", False) else "✗")
    
    # Add file path if available
    if "file_path" in result and result["file_path"]:
        stats_table.add_row("Output File", str(result["file_path"]))
    
    # Display the table
    console.print(stats_table)
    
    # Display content preview if requested and content is available
    if content_preview and "content" in result and result["content"]:
        content = result["content"]
        preview_length = min(1000, len(content))  # Limit preview to 1000 chars
        preview = content[:preview_length]
        if len(content) > preview_length:
            preview += "..."
        
        console.print(Panel(
            escape(preview),
            title=f"[bold green]Content Preview ({result.get('output_format', 'unknown')})[/bold green]",
            border_style="green",
            expand=False
        ))


async def demonstrate_basic_conversion(tracker: CostTracker):
    """Demonstrate basic document conversion to markdown."""
    console.print(Rule("[bold blue]Basic Document Conversion[/bold blue]"))
    logger.info("Starting basic document conversion example", emoji_key="start")
    
    # Use a sample PDF URL for conversion
    document_path = SAMPLE_DOCUMENTS["pdf"]
    logger.info(f"Converting document: {document_path}", emoji_key="processing")
    
    try:
        # Call the convert_document tool
        result = await convert_document(
            document_path=document_path,
            output_format="markdown"
        )
        
        # Log success
        logger.success("Document converted successfully to markdown", emoji_key="success")
        
        # Display results
        display_conversion_result("Basic Conversion", result)
        
        # Note: Document conversion doesn't use tokens/LLMs directly, so no cost to track
        logger.info("Document conversion doesn't incur token costs (no LLM usage)", emoji_key="info")
        
    except Exception as e:
        logger.error(f"Error converting document: {str(e)}", emoji_key="error", exc_info=True)


async def demonstrate_format_options(tracker: CostTracker):
    """Demonstrate conversion to different output formats."""
    console.print(Rule("[bold blue]Multiple Output Formats[/bold blue]"))
    logger.info("Demonstrating conversion to multiple output formats", emoji_key="start")
    
    # Use a sample document
    document_path = SAMPLE_DOCUMENTS["pdf"]
    
    # Output formats to demonstrate
    formats = ["markdown", "text", "html", "json", "doctags"]
    
    # Create a table to compare formats
    format_table = Table(title="Format Comparison", box=box.ROUNDED, show_header=True)
    format_table.add_column("Format", style="magenta")
    format_table.add_column("Processing Time", style="cyan", justify="right")
    format_table.add_column("Content Size", style="blue", justify="right")
    format_table.add_column("Features", style="green")
    
    for output_format in formats:
        try:
            logger.info(f"Converting to {output_format}...", emoji_key="processing")
            
            # Call the convert_document tool
            start_time = time.time()
            result = await convert_document(
                document_path=document_path,
                output_format=output_format
            )
            processing_time = time.time() - start_time
            
            # Get content size
            content_size = len(result.get("content", ""))
            
            # Determine features based on format
            if output_format == "markdown":
                features = "Headings, Links, Tables, Images"
            elif output_format == "text":
                features = "Plain text only, no formatting"
            elif output_format == "html":
                features = "Rich formatting, CSS styling, images"
            elif output_format == "json":
                features = "Full document structure and metadata"
            elif output_format == "doctags":
                features = "Format for NLP tasks, token-based"
            else:
                features = "Unknown"
            
            # Add row to table
            format_table.add_row(
                output_format,
                f"{processing_time:.2f}s",
                f"{content_size} chars",
                features
            )
            
            logger.success(f"Successfully converted to {output_format}", emoji_key="success")
            
            # For just one format, show a content preview
            if output_format == "markdown":
                display_conversion_result(f"{output_format.capitalize()} Conversion", result)
            
        except Exception as e:
            logger.error(f"Error converting to {output_format}: {str(e)}", emoji_key="error")
            format_table.add_row(output_format, "Error", "N/A", f"Error: {str(e)}")
    
    # Display the format comparison table
    console.print(format_table)


async def demonstrate_url_conversion(tracker: CostTracker):
    """Demonstrate conversion of documents from URLs."""
    console.print(Rule("[bold blue]URL Document Conversion[/bold blue]"))
    logger.info("Demonstrating conversion from different URL sources", emoji_key="start")
    
    # Table to compare different document types
    url_table = Table(title="URL Document Type Comparison", box=box.ROUNDED, show_header=True)
    url_table.add_column("Document Type", style="magenta")
    url_table.add_column("URL", style="dim blue")
    url_table.add_column("Processing Time", style="cyan", justify="right")
    url_table.add_column("Pages", style="yellow", justify="right")
    url_table.add_column("Status", style="green")
    
    # Try converting each sample document
    for doc_type, url in SAMPLE_DOCUMENTS.items():
        try:
            logger.info(f"Converting {doc_type.upper()} from URL: {url}", emoji_key="processing")
            
            # Call the convert_document tool
            result = await convert_document(
                document_path=url,
                output_format="markdown"
            )
            
            # Get page count from metadata
            page_count = result.get("document_metadata", {}).get("num_pages", "N/A")
            
            # Add row to table
            url_table.add_row(
                doc_type.upper(),
                url[:40] + "..." if len(url) > 40 else url,
                f"{result.get('processing_time', 0):.2f}s",
                str(page_count),
                "✓ Success"
            )
            
            logger.success(f"Successfully converted {doc_type.upper()} from URL", emoji_key="success")
            
        except Exception as e:
            logger.error(f"Error converting {doc_type.upper()} from URL: {str(e)}", emoji_key="error")
            url_table.add_row(
                doc_type.upper(),
                url[:40] + "..." if len(url) > 40 else url,
                "N/A",
                "N/A",
                f"✗ Error: {str(e)[:30]}"
            )
    
    # Display the URL comparison table
    console.print(url_table)


async def demonstrate_file_saving(tracker: CostTracker):
    """Demonstrate saving document conversion results to files."""
    console.print(Rule("[bold blue]Save Conversion to Files[/bold blue]"))
    logger.info("Demonstrating saving conversion results to files", emoji_key="start")
    
    # Create output directory
    output_dir = Path("conversion_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a sample document
    document_path = SAMPLE_DOCUMENTS["pdf"]
    document_name = Path(document_path).stem or "sample_document"
    
    # Output formats to save
    formats_to_save = ["markdown", "html", "text"]
    
    # Table to track saved files
    files_table = Table(title="Saved Output Files", box=box.ROUNDED, show_header=True)
    files_table.add_column("Format", style="magenta")
    files_table.add_column("Output Path", style="cyan")
    files_table.add_column("File Size", style="blue", justify="right")
    files_table.add_column("Status", style="green")
    
    for output_format in formats_to_save:
        try:
            # Generate output path
            output_path = output_dir / f"{document_name}.{output_format}"
            if output_format == "text":
                output_path = output_dir / f"{document_name}.txt"
            elif output_format == "html":
                output_path = output_dir / f"{document_name}.html"
            
            logger.info(f"Converting and saving to {output_path}", emoji_key="processing")
            
            # Call the convert_document tool with save_to_file=True
            result = await convert_document(
                document_path=document_path,
                output_format=output_format,
                output_path=str(output_path),
                save_to_file=True
            )
            
            # Check if file was created
            if output_path.exists():
                file_size = output_path.stat().st_size
                file_size_str = f"{file_size / 1024:.1f} KB"
                status = "✓ File created"
            else:
                file_size_str = "N/A"
                status = "⚠ File not found"
            
            # Add row to table
            files_table.add_row(
                output_format,
                str(output_path),
                file_size_str,
                status
            )
            
            logger.success(f"Successfully saved {output_format} output", emoji_key="success")
            
        except Exception as e:
            logger.error(f"Error saving {output_format} output: {str(e)}", emoji_key="error")
            files_table.add_row(
                output_format,
                str(output_path) if 'output_path' in locals() else "N/A",
                "N/A",
                f"✗ Error: {str(e)[:30]}"
            )
    
    # Display the files table
    console.print(files_table)
    
    # Display a list of all files in the output directory
    console.print(f"\n[bold cyan]Files in output directory ({output_dir})[/bold cyan]:")
    if list(output_dir.glob("*")):
        for file_path in output_dir.glob("*"):
            console.print(f"  - {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")
    else:
        console.print("  [dim]No files found[/dim]")


async def demonstrate_advanced_options(tracker: CostTracker):
    """Demonstrate advanced conversion options like acceleration settings."""
    console.print(Rule("[bold blue]Advanced Conversion Options[/bold blue]"))
    logger.info("Demonstrating advanced conversion options", emoji_key="start")
    
    # Use a sample document
    document_path = SAMPLE_DOCUMENTS["pdf"]
    
    # Create a table to compare different acceleration settings
    accel_table = Table(title="Acceleration Options Comparison", box=box.ROUNDED, show_header=True)
    accel_table.add_column("Accelerator", style="magenta")
    accel_table.add_column("Threads", style="yellow", justify="right")
    accel_table.add_column("Processing Time", style="cyan", justify="right")
    accel_table.add_column("Speed Ratio", style="green", justify="right")
    
    # First run with default settings for baseline
    try:
        logger.info("Running baseline conversion with default settings", emoji_key="processing")
        baseline_result = await convert_document(
            document_path=document_path,
            output_format="markdown"
        )
        baseline_time = baseline_result.get("processing_time", 1.0)  # Fallback to 1.0 to avoid division by zero
        
        # Add baseline to table
        accel_table.add_row(
            "Default (auto)",
            "4",  # Default thread count is 4
            f"{baseline_time:.2f}s",
            "1.00x"  # Baseline speed ratio
        )
        
        logger.success("Baseline conversion completed", emoji_key="success")
        
    except Exception as e:
        logger.error(f"Error in baseline conversion: {str(e)}", emoji_key="error")
        baseline_time = 1.0  # Fallback to avoid division by zero
    
    # Test different configurations
    configs = [
        {"accelerator": "cpu", "threads": 1},
        {"accelerator": "cpu", "threads": 8},
        {"accelerator": "auto", "threads": 8}
    ]
    
    for config in configs:
        try:
            accel_device = config["accelerator"]
            num_threads = config["threads"]
            
            logger.info(f"Testing with accelerator={accel_device}, threads={num_threads}", emoji_key="processing")
            
            # Call convert_document with specific acceleration settings
            result = await convert_document(
                document_path=document_path,
                output_format="markdown",
                accelerator_device=accel_device,
                num_threads=num_threads
            )
            
            # Calculate processing time and speed ratio
            proc_time = result.get("processing_time", 0)
            speed_ratio = baseline_time / proc_time if proc_time > 0 else 0
            
            # Add row to table
            accel_table.add_row(
                accel_device,
                str(num_threads),
                f"{proc_time:.2f}s",
                f"{speed_ratio:.2f}x"
            )
            
            logger.success(f"Conversion with accelerator={accel_device}, threads={num_threads} completed", emoji_key="success")
            
        except Exception as e:
            logger.error(f"Error in conversion with accelerator={config['accelerator']}: {str(e)}", emoji_key="error")
            accel_table.add_row(
                config["accelerator"],
                str(config["threads"]),
                "Error",
                "N/A"
            )
    
    # Display the acceleration comparison table
    console.print(accel_table)
    
    # Additional information about acceleration options
    console.print(Panel(
        "The document conversion tool supports different acceleration options:\n\n"
        "- 'auto': Automatically selects the best available device\n"
        "- 'cpu': Uses CPU processing only\n"
        "- 'cuda': Uses NVIDIA GPU acceleration (if available)\n"
        "- 'mps': Uses Apple Silicon GPU acceleration (if available)\n\n"
        "The number of threads controls CPU parallelism and can significantly impact performance.",
        title="[bold cyan]Acceleration Options[/bold cyan]",
        border_style="cyan",
        expand=False
    ))


async def main():
    """Run document conversion examples."""
    sys.stderr.write("Starting main function...\n")
    sys.stderr.flush()
    
    # Initialize cost tracker
    tracker = CostTracker()
    
    try:
        # Ensure test files exist
        sys.stderr.write("About to ensure test files exist...\n")
        sys.stderr.flush()
        await ensure_test_files_exist()
        sys.stderr.write("Test files check completed\n")
        sys.stderr.flush()
        
        console.print(Panel(
            "This demo showcases the document conversion tool which uses the Docling library "
            "to convert various document formats to structured outputs.\n\n"
            "The tool supports PDF, DOCX, HTML, and other formats, and can output to markdown, "
            "text, HTML, JSON, and other formats while preserving document structure.\n\n"
            "[yellow]Note: Unlike LLM-based tools, document conversion doesn't incur token costs.[/yellow]",
            title="[bold yellow]Document Conversion Tool Demo[/bold yellow]",
            border_style="yellow",
            expand=False
        ))
        
        # Run the demonstration functions, passing the tracker to each
        sys.stderr.write("Starting basic conversion demo...\n")
        sys.stderr.flush()
        await demonstrate_basic_conversion(tracker)
        sys.stderr.write("Basic conversion completed\n")
        sys.stderr.flush()
        
        console.print()  # Add space between sections
        
        sys.stderr.write("Starting format options demo...\n")
        sys.stderr.flush()
        await demonstrate_format_options(tracker)
        sys.stderr.write("Format options demo completed\n")
        sys.stderr.flush()
        
        console.print()  # Add space between sections
        
        sys.stderr.write("Starting URL conversion demo...\n")
        sys.stderr.flush()
        await demonstrate_url_conversion(tracker)
        sys.stderr.write("URL conversion demo completed\n")
        sys.stderr.flush()
        
        console.print()  # Add space between sections
        
        sys.stderr.write("Starting file saving demo...\n")
        sys.stderr.flush()
        await demonstrate_file_saving(tracker)
        sys.stderr.write("File saving demo completed\n")
        sys.stderr.flush()
        
        console.print()  # Add space between sections
        
        sys.stderr.write("Starting advanced options demo...\n")
        sys.stderr.flush()
        await demonstrate_advanced_options(tracker)
        sys.stderr.write("Advanced options demo completed\n")
        sys.stderr.flush()
        
        # Display the cost summary (which should show zero costs in this case)
        console.print()
        tracker.display_summary(console)
        
        logger.success("Document conversion examples completed successfully", emoji_key="complete")
        return 0
        
    except Exception as e:
        sys.stderr.write(f"Main function failed with: {e}\n")
        sys.stderr.flush()
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1


if __name__ == "__main__":
    # Run the examples
    sys.stderr.write("About to run main via asyncio.run...\n")
    sys.stderr.flush()
    exit_code = asyncio.run(main())
    sys.stderr.write(f"Main completed with exit code: {exit_code}\n")
    sys.stderr.flush()
    sys.exit(exit_code)