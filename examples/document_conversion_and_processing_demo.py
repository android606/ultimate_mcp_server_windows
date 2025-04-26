#!/usr/bin/env python
"""Demonstration script for DocumentProcessingTool in Ultimate MCP Server."""

import asyncio
import datetime as dt
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Rich imports for nice UI
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.traceback import install as install_rich_traceback
from rich.tree import Tree

# Import the actual Gateway for server communication
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.exceptions import ToolError, ToolInputError

# Import the DocumentProcessingTool
from ultimate_mcp_server.tools.document_conversion_and_processing import DocumentProcessingTool
from ultimate_mcp_server.utils import get_logger

# Initialize Rich console and logger
console = Console()
logger = get_logger("demo.document_processing_tool")

# Install rich tracebacks for better error display
install_rich_traceback(show_locals=False, width=console.width)

# --- Configuration ---
DEFAULT_PAPERS_DIR = "quantum_computing_papers"  # Directory containing PDF papers
SAMPLE_HTML_URL = "https://en.wikipedia.org/wiki/Quantum_computing"  # Sample HTML content
DOWNLOADED_FILES_DIR = Path("downloaded_files")  # Directory for downloaded files

# Use environment variables if available
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "2"))
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
ACCELERATOR_DEVICE = "cuda" if USE_GPU else "cpu"

# Define result types using Union and Tuple
ResultData = Dict[str, Any]
OperationResult = Tuple[bool, ResultData]
FileResult = Union[Path, None]

# --- Demo Helper Functions ---

def create_demo_layout() -> Layout:
    """Create a layout for the demo UI."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="content", ratio=3),
        Layout(name="stats", ratio=1)
    )
    return layout

def timestamp_str() -> str:
    """Return a formatted timestamp string."""
    return f"[dim]{dt.datetime.now().strftime('%H:%M:%S')}[/dim]"

def display_markdown_content(content: str, title: str = "Markdown Content") -> None:
    """Display markdown content with proper rendering."""
    # Use the Markdown class to render markdown content
    md = Markdown(content)
    console.print(Panel(md, title=title, border_style="blue", padding=(1, 2)))

def display_result(title: str, result: Dict[str, Any], display_keys: Optional[List[str]] = None, 
                  hide_keys: Optional[List[str]] = None, format_key: Optional[Dict[str, str]] = None) -> None:
    """Display operation result with enhanced formatting."""
    start_time = dt.datetime.now()
    console.print(Rule(f"[bold cyan]{escape(title)}[/bold cyan] {timestamp_str()}"))
    
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        console.print(Panel(
            f"[bold red]:x: Operation Failed:[/]\n{escape(error_msg)}",
            title="Error",
            border_style="red",
            padding=(1, 2),
            expand=False
        ))
        return
    
    # Filter keys to display
    display_dict = {}
    hide_keys = hide_keys or ["success", "raw_llm_response"]
    
    for key, value in result.items():
        if key in hide_keys:
            continue
        if display_keys and key not in display_keys:
            continue
        display_dict[key] = value
    
    # Create panels for each key value pair
    for key, value in display_dict.items():
        key_title = key.replace("_", " ").title()
        
        # Handle different value types differently
        if key == "content" or key == "markdown_text":
            # Use Markdown renderer for markdown content
            if format_key and format_key.get(key) == "markdown":
                display_markdown_content(str(value), key_title)
            else:
                # Show content in a syntax panel with appropriate highlighting
                format_type = format_key.get(key, "markdown") if format_key else "markdown"
                console.print(Panel(
                    Syntax(str(value), format_type, theme="default", line_numbers=True, word_wrap=True),
                    title=key_title,
                    border_style="blue",
                    padding=(1, 2)
                ))
        elif isinstance(value, list) and key == "chunks":
            # Show chunks in a table
            chunks_table = Table(title=f"{key_title} (Total: {len(value)})", box=box.ROUNDED)
            chunks_table.add_column("#", style="cyan", justify="right")
            chunks_table.add_column("Content", style="white")
            chunks_table.add_column("Length", style="green", justify="right")
            
            for i, chunk in enumerate(value[:5], 1):  # Show first 5 chunks
                chunks_table.add_row(
                    str(i),
                    str(chunk)[:100] + "..." if len(str(chunk)) > 100 else str(chunk),
                    str(len(str(chunk)))
                )
            
            if len(value) > 5:
                chunks_table.add_row("...", f"[dim]{len(value) - 5} more chunks...[/dim]", "")
                
            console.print(chunks_table)
        elif isinstance(value, list) and key == "qa_pairs":
            # Show QA pairs in a dedicated format
            qa_panel = Panel(
                "\n\n".join([f"[bold cyan]Q: {qa.get('question', '')}[/]\n[green]A: {qa.get('answer', '')}[/]" 
                            for qa in value[:3]]),  # First 3 QA pairs
                title=f"{key_title} (Total: {len(value)})",
                border_style="blue",
                padding=(1, 2)
            )
            console.print(qa_panel)
            if len(value) > 3:
                console.print(f"[dim]... and {len(value) - 3} more Q&A pairs ...[/dim]")
        elif isinstance(value, dict) and key == "entities":
            # Format entities dictionary
            entities_table = Table(title=key_title, box=box.ROUNDED)
            entities_table.add_column("Entity Type", style="cyan")
            entities_table.add_column("Entities", style="white")
            
            for entity_type, entity_list in value.items():
                entities_table.add_row(
                    entity_type,
                    ", ".join(entity_list[:5]) + (f" [dim]... and {len(entity_list) - 5} more[/dim]" if len(entity_list) > 5 else "")
                )
                
            console.print(entities_table)
        elif isinstance(value, dict) and key == "risks":
            # Format risks dictionary
            risks_table = Table(title=key_title, box=box.ROUNDED)
            risks_table.add_column("Risk Type", style="red")
            risks_table.add_column("Count", style="yellow", justify="right")
            risks_table.add_column("Sample Context", style="white")
            
            for risk_type, risk_info in value.items():
                sample = risk_info.get("sample_contexts", [""])[0][:100] + "..." if risk_info.get("sample_contexts") else ""
                risks_table.add_row(
                    risk_type,
                    str(risk_info.get("count", 0)),
                    sample
                )
                
            console.print(risks_table)
        elif isinstance(value, (int, float)) and key.endswith("_time"):
            # Format time values
            console.print(f"[cyan]{key_title}:[/] [green]{value:.3f}s[/]")
        elif isinstance(value, dict) and key == "document_metadata":
            # Format document metadata
            meta_table = Table(title=key_title, box=box.SIMPLE, show_header=False)
            meta_table.add_column("Property", style="cyan", justify="right")
            meta_table.add_column("Value", style="white")
            
            for meta_key, meta_value in value.items():
                meta_table.add_row(meta_key.replace("_", " ").title(), str(meta_value))
                
            console.print(meta_table)
        elif isinstance(value, dict) and key == "sections":
            # Use a Tree to display document sections
            sections_tree = Tree(f"[bold]{key_title}[/]")
            
            # Helper function to create section tree
            def add_sections_to_tree(tree: Tree, sections_data: Dict[str, Any], level: int = 0) -> None:
                for section_title, section_info in sections_data.items():
                    section_node = tree.add(Text.from_markup(f"[cyan]{section_title}[/]"))
                    
                    # Add content preview if available
                    if "content" in section_info:
                        content_preview = section_info["content"][:100] + "..." if len(section_info["content"]) > 100 else section_info["content"]
                        section_node.add(Text.from_markup(f"[dim]{content_preview}[/]"))
                    
                    # Add subsections recursively
                    if "subsections" in section_info and section_info["subsections"]:
                        add_sections_to_tree(section_node, section_info["subsections"], level + 1)
            
            add_sections_to_tree(sections_tree, value)
            console.print(sections_tree)
        elif isinstance(value, dict):
            # Generic dict formatting
            dict_table = Table(title=key_title, box=box.SIMPLE, show_header=False)
            dict_table.add_column("Key", style="cyan", justify="right")
            dict_table.add_column("Value", style="white")
            
            for k, v in value.items():
                dict_table.add_row(str(k), str(v)[:100] + "..." if len(str(v)) > 100 else str(v))
                
            console.print(dict_table)
        else:
            # Default formatting for other types
            if isinstance(value, list):
                if len(value) > 5:
                    value_str = f"{value[:5]} ... and {len(value) - 5} more items"
                else:
                    value_str = str(value)
            else:
                value_str = str(value)
            
            # Use styled Text for display
            styled_text = Text()
            styled_text.append(f"{key_title}: ", style="cyan bold")
            styled_text.append(value_str)
            console.print(styled_text)
    
    # Display processing time
    end_time = dt.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    console.print(f"[dim]Result displayed in {elapsed:.3f}s[/]")
    console.print()  # Add spacing

async def download_sample_file(url: str, output_dir: Path) -> FileResult:
    """Download a sample file for demo purposes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = dt.datetime.now()
    
    # Extract filename from URL or use a default
    filename = url.split("/")[-1]
    if "?" in filename:
        filename = filename.split("?")[0]
    if not filename or filename.endswith("/"):
        filename = "sample.html"
        
    output_path = output_dir / filename
    
    # Skip if file already exists
    if output_path.exists():
        file_age = dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(output_path))
        logger.info(f"Using existing file: {output_path} (Age: {file_age.days} days)")
        return output_path
        
    try:
        with console.status(f"[cyan]Downloading {url}... {timestamp_str()}", spinner="dots"):
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Save content to file
                output_path.write_bytes(response.content)
                end_time = dt.datetime.now()
                download_time = (end_time - start_time).total_seconds()
                logger.info(f"Downloaded to {output_path} in {download_time:.2f}s")
                
                return output_path
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        console.print(f"[red]Failed to download file: {e}[/]")
        return None

async def safe_tool_call(operation_name: str, tool_func: callable, *args, **kwargs) -> OperationResult:
    """Safely call a tool function, handling specific tool exceptions."""
    try:
        result = await tool_func(*args, **kwargs)
        return True, result
    except ToolInputError as e:
        logger.error(f"Invalid input for {operation_name}: {e}")
        console.print(Panel(
            f"[bold red]Input Error:[/] {str(e)}",
            title=f"[red]{operation_name} Failed[/]",
            border_style="red"
        ))
        return False, {"success": False, "error": str(e), "error_type": "input_error"}
    except ToolError as e:
        logger.error(f"Tool error during {operation_name}: {e}")
        console.print(Panel(
            f"[bold red]Tool Error:[/] {str(e)}",
            title=f"[red]{operation_name} Failed[/]",
            border_style="red"
        ))
        return False, {"success": False, "error": str(e), "error_type": "tool_error"}
    except Exception as e:
        logger.error(f"Unexpected error during {operation_name}: {e}", exc_info=True)
        console.print(Panel(
            f"[bold red]Unexpected Error:[/] {str(e)}",
            title=f"[red]{operation_name} Failed[/]",
            border_style="red"
        ))
        return False, {"success": False, "error": str(e), "error_type": "unexpected_error"}

# --- Demo Functions ---

async def conversion_demo(doc_tool: DocumentProcessingTool) -> None:
    """Demonstrate document conversion capabilities."""
    start_time = dt.datetime.now()
    console.print(Rule(f"[bold green]1. Document Conversion Demo[/bold green] {timestamp_str()}", style="green"))
    logger.info("Starting document conversion demo")
    
    # Find PDF files in the papers directory
    papers_dir = Path(DEFAULT_PAPERS_DIR)
    if not papers_dir.exists():
        papers_dir.mkdir(parents=True, exist_ok=True)
        console.print(Panel(
            f"The directory [bold]{papers_dir}[/] doesn't exist or has no PDF files.\n"
            "Please add some PDF files to this directory for the demo.",
            title="[yellow]No PDF Files Found[/]",
            border_style="yellow",
            padding=(1, 2)
        ))
        return
    
    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(Panel(
            f"No PDF files found in [bold]{papers_dir}[/].\n"
            "Please add some PDF files to this directory for the demo.",
            title="[yellow]No PDF Files Found[/]",
            border_style="yellow",
            padding=(1, 2)
        ))
        return
    
    file_stats = []
    for pdf_file in pdf_files[:5]:  # Show stats for first 5 files
        size_bytes = os.path.getsize(pdf_file)
        mod_time = dt.datetime.fromtimestamp(os.path.getmtime(pdf_file))
        file_stats.append((pdf_file.name, size_bytes, mod_time))
    
    # Display files in a table with stats
    files_table = Table(title="Available PDF Files")
    files_table.add_column("Filename", style="cyan")
    files_table.add_column("Size", style="green", justify="right")
    files_table.add_column("Modified", style="yellow")
    
    for name, size, mod_time in file_stats:
        files_table.add_row(
            name,
            f"{size / 1024:.2f} KB",
            mod_time.strftime("%Y-%m-%d %H:%M")
        )
    
    console.print(files_table)
    
    # Select a PDF file for demonstration
    demo_pdf = pdf_files[0]
    console.print(f"Using [cyan]{demo_pdf.name}[/] for conversion demo")
    
    # Convert to different formats
    formats = ["markdown", "text", "html"]
    
    # Use Progress to track conversion process
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        conversion_task = progress.add_task("[cyan]Converting documents...", total=len(formats))
        
        for output_format in formats:
            progress.update(conversion_task, description=f"[cyan]Converting to {output_format.upper()}...")
            
            # Use the safe_tool_call helper which handles ToolError and ToolInputError
            success, result = await safe_tool_call(
                f"Convert to {output_format.upper()}", 
                doc_tool.convert_document,
                document_path=str(demo_pdf),
                output_format=output_format,
                save_to_file=True,
                accelerator_device=ACCELERATOR_DEVICE
            )
            
            if success and result.get("success"):
                # Display the result
                display_result(
                    f"Converted to {output_format.upper()}", 
                    result, 
                    display_keys=["content", "processing_time", "document_metadata", "file_path"],
                    format_key={"content": output_format}
                )
            
            progress.advance(conversion_task)
    
    end_time = dt.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    console.print(f"[dim]Conversion demo completed in {elapsed:.2f}s[/]")
    console.print()  # Add spacing

async def chunking_demo(doc_tool: DocumentProcessingTool) -> None:
    """Demonstrate document chunking capabilities."""
    console.print(Rule("[bold green]2. Document Chunking Demo[/bold green]", style="green"))
    logger.info("Starting document chunking demo")
    
    # Find the converted markdown file from the previous demo or use a sample
    temp_dir = Path(tempfile.gettempdir())
    markdown_files = list(temp_dir.glob("*.md"))
    
    if markdown_files:
        demo_md_path = markdown_files[0]
        demo_md = demo_md_path.read_text(encoding="utf-8")
        source = f"converted file: {demo_md_path.name}"
    else:
        # Use a sample text if no markdown file is found
        demo_md = """# Quantum Computing: An Introduction

Quantum computing is an emerging field that leverages quantum mechanics principles to process information. Unlike classical computing, which uses bits as the smallest unit of data, quantum computing uses quantum bits or qubits.

## Quantum Bits

Qubits can exist in multiple states simultaneously due to superposition. This property allows quantum computers to process vast amounts of data in parallel.

## Quantum Gates

Quantum gates manipulate qubits through operations like:
- Hadamard gates
- CNOT gates
- Pauli-X, Y, Z gates

## Quantum Algorithms

Several quantum algorithms demonstrate advantages over classical counterparts:

### Shor's Algorithm
Efficiently factors large numbers, threatening current cryptographic systems.

### Grover's Algorithm
Provides quadratic speedup for unstructured database searches.

## Challenges in Quantum Computing

Despite theoretical advantages, practical challenges remain:
1. Decoherence - Quantum states are fragile
2. Error correction - Quantum error correction is complex
3. Scalability - Building large-scale, stable quantum systems is difficult

## Future Prospects

Quantum computing promises revolutionary advances in:
- Drug discovery
- Materials science
- Optimization problems
- Machine learning
- Cryptography

Research continues at major tech companies and academic institutions worldwide."""
        source = "sample text"
    
    console.print(f"Using [cyan]{source}[/] for chunking demo")
    
    # Demonstrate different chunking methods
    chunking_methods = [
        ("paragraph", 150, 0),
        ("character", 200, 20),
        ("token", 100, 10),
        ("section", 300, 0)
    ]
    
    for method, size, overlap in chunking_methods:
        console.print(f"\n[bold cyan]Chunking with method: {method.upper()} (size={size}, overlap={overlap})[/]")
        
        with console.status(f"[cyan]Chunking document using {method} method...", spinner="dots"):
            try:
                result = await doc_tool.chunk_document(
                    document=demo_md,
                    chunk_method=method,
                    chunk_size=size,
                    chunk_overlap=overlap
                )
                
                if result.get("success"):
                    display_result(
                        f"Chunked document ({method})", 
                        result,
                        hide_keys=["success"]
                    )
                else:
                    console.print(Panel(
                        f"[bold red]Chunking failed:[/]\n{result.get('error', 'Unknown error')}",
                        title=f"[red]Failed to Chunk with {method}[/]",
                        border_style="red",
                        padding=(1, 2)
                    ))
            except Exception as e:
                logger.error(f"Error during {method} chunking: {e}", exc_info=True)
                console.print(f"[bold red]Error:[/] {e}")
    
    console.print()  # Add spacing

async def html_processing_demo(doc_tool: DocumentProcessingTool) -> None:
    """Demonstrate HTML processing and conversion to Markdown."""
    console.print(Rule("[bold green]3. HTML Processing Demo[/bold green]", style="green"))
    logger.info("Starting HTML processing demo")
    
    # Download a sample HTML file if needed
    html_path = await download_sample_file(SAMPLE_HTML_URL, DOWNLOADED_FILES_DIR)
    
    if not html_path:
        # Use a simple HTML example if download failed
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Quantum Computing Sample</title>
</head>
<body>
    <header>
        <h1>Introduction to Quantum Computing</h1>
        <nav>
            <ul>
                <li><a href="#basics">Basics</a></li>
                <li><a href="#applications">Applications</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <section id="basics">
            <h2>Quantum Computing Basics</h2>
            <p>Quantum computing leverages quantum mechanical phenomena to perform computations. The basic unit of quantum information is the <strong>qubit</strong>.</p>
            <p>Unlike classical bits that can be either 0 or 1, qubits can exist in a superposition of states.</p>
            <table border="1">
                <tr>
                    <th>Property</th>
                    <th>Classical Computing</th>
                    <th>Quantum Computing</th>
                </tr>
                <tr>
                    <td>Information Unit</td>
                    <td>Bit (0 or 1)</td>
                    <td>Qubit (Superposition)</td>
                </tr>
                <tr>
                    <td>Processing</td>
                    <td>Sequential</td>
                    <td>Parallel</td>
                </tr>
            </table>
        </section>
        <section id="applications">
            <h2>Applications</h2>
            <ul>
                <li>Cryptography</li>
                <li>Drug Discovery</li>
                <li>Optimization Problems</li>
                <li>Machine Learning</li>
            </ul>
        </section>
    </main>
    <footer>
        <p>&copy; 2023 Quantum Computing Demo</p>
    </footer>
</body>
</html>"""
        # Save content to a temporary file
        temp_html_path = DOWNLOADED_FILES_DIR / "sample_quantum.html"
        DOWNLOADED_FILES_DIR.mkdir(parents=True, exist_ok=True)
        temp_html_path.write_text(html_content, encoding="utf-8")
        html_path = temp_html_path
    
    html_content = html_path.read_text(encoding="utf-8")
    console.print(f"Using HTML from [cyan]{html_path.name}[/] for processing")
    
    # Demonstrate different HTML processing options
    with console.status("[cyan]Converting HTML to Markdown...", spinner="dots"):
        try:
            result = await doc_tool.clean_and_format_text_as_markdown(
                text=html_content,
                extraction_method="auto",
                preserve_tables=True,
                preserve_links=True,
                preserve_images=False
            )
            
            if result.get("success"):
                display_result(
                    "HTML to Markdown Conversion", 
                    result,
                    display_keys=["markdown_text", "was_html", "parser_used", "extraction_method_used", "processing_time"],
                    format_key={"markdown_text": "markdown"}
                )
            else:
                console.print(Panel(
                    f"[bold red]HTML processing failed:[/]\n{result.get('error', 'Unknown error')}",
                    title="[red]Failed to Convert HTML to Markdown[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
                
            # Try another extraction method
            console.print("\n[bold cyan]Using different extraction method: 'readability'[/]")
            result_readability = await doc_tool.clean_and_format_text_as_markdown(
                text=html_content,
                extraction_method="readability",
                preserve_tables=True,
                preserve_links=True,
                preserve_images=False
            )
            
            if result_readability.get("success"):
                display_result(
                    "HTML to Markdown using Readability", 
                    result_readability,
                    display_keys=["markdown_text", "extraction_method_used", "processing_time"],
                    format_key={"markdown_text": "markdown"}
                )
        except Exception as e:
            logger.error(f"Error during HTML processing: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/] {e}")
    
    # Demonstrate Markdown optimization
    if result.get("success") and result.get("markdown_text"):
        console.print("\n[bold cyan]Optimizing Markdown formatting...[/]")
        
        with console.status("[cyan]Optimizing Markdown...", spinner="dots"):
            try:
                optimize_result = await doc_tool.optimize_markdown_formatting(
                    markdown=result["markdown_text"],
                    normalize_headings=True,
                    fix_lists=True,
                    fix_links=True,
                    add_line_breaks=True
                )
                
                if optimize_result.get("success"):
                    display_result(
                        "Optimized Markdown", 
                        optimize_result,
                        display_keys=["optimized_markdown", "changes_made", "processing_time"],
                        format_key={"optimized_markdown": "markdown"}
                    )
            except Exception as e:
                logger.error(f"Error during Markdown optimization: {e}", exc_info=True)
                console.print(f"[bold red]Error:[/] {e}")
    
    # Demonstrate content type detection
    console.print("\n[bold cyan]Detecting content type...[/]")
    
    with console.status("[cyan]Detecting content type...", spinner="dots"):
        try:
            # Try with HTML
            detect_html_result = await doc_tool.detect_content_type(text=html_content[:5000])
            
            if detect_html_result.get("success"):
                display_result(
                    "Content Type Detection (HTML)", 
                    detect_html_result,
                    display_keys=["content_type", "confidence", "details"]
                )
                
            # Try with Markdown
            if result.get("markdown_text"):
                detect_md_result = await doc_tool.detect_content_type(text=result["markdown_text"][:5000])
                
                if detect_md_result.get("success"):
                    display_result(
                        "Content Type Detection (Markdown)", 
                        detect_md_result,
                        display_keys=["content_type", "confidence", "details"]
                    )
        except Exception as e:
            logger.error(f"Error during content type detection: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/] {e}")
    
    console.print()  # Add spacing

async def table_extraction_demo(doc_tool: DocumentProcessingTool) -> None:
    """Demonstrate table extraction from documents."""
    console.print(Rule("[bold green]4. Table Extraction Demo[/bold green]", style="green"))
    logger.info("Starting table extraction demo")
    
    # Find PDF files in the papers directory
    papers_dir = Path(DEFAULT_PAPERS_DIR)
    if not papers_dir.exists() or not list(papers_dir.glob("*.pdf")):
        console.print(Panel(
            "To demonstrate table extraction, we need PDF files with tables.\n"
            "No suitable files found in the papers directory.",
            title="[yellow]Table Extraction Demo Skipped[/]",
            border_style="yellow",
            padding=(1, 2)
        ))
        return
    
    # Try to find PDF files that might contain tables
    pdf_files = list(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        console.print("[yellow]No PDF files found for table extraction demo.[/]")
        return
        
    demo_pdf = pdf_files[0]
    console.print(f"Using [cyan]{demo_pdf.name}[/] for table extraction")
    
    with console.status(f"[cyan]Extracting tables from {demo_pdf.name}...", spinner="dots"):
        try:
            result = await doc_tool.extract_tables(
                document_path=str(demo_pdf),
                table_mode="csv",
                output_dir=str(DOWNLOADED_FILES_DIR)
            )
            
            if result.get("success"):
                if result.get("tables"):
                    display_result(
                        "Extracted Tables (CSV format)", 
                        result,
                        display_keys=["tables", "saved_files"]
                    )
                else:
                    console.print(Panel(
                        "No tables were found in the document.",
                        title="[yellow]No Tables Found[/]",
                        border_style="yellow",
                        padding=(1, 2)
                    ))
            else:
                console.print(Panel(
                    f"[bold red]Table extraction failed:[/]\n{result.get('error', 'Unknown error')}",
                    title="[red]Failed to Extract Tables[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
        except Exception as e:
            logger.error(f"Error during table extraction: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/] {e}")
    
    console.print()  # Add spacing

async def document_analysis_demo(doc_tool: DocumentProcessingTool) -> None:
    """Demonstrate document analysis capabilities (sections, entities, QA, summary, etc.)."""
    console.print(Rule("[bold green]5. Document Analysis Demo[/bold green]", style="green"))
    logger.info("Starting document analysis demo")
    
    # Find a suitable text file to analyze (converted markdown or sample)
    temp_dir = Path(tempfile.gettempdir())
    markdown_files = list(temp_dir.glob("*.md"))
    
    if markdown_files:
        demo_text_path = markdown_files[0]
        demo_text = demo_text_path.read_text(encoding="utf-8")
        source = f"converted file: {demo_text_path.name}"
    else:
        # Use a longer sample text for analysis
        demo_text = """# Quantum Computing: State of the Field

## Introduction

Quantum computing represents a paradigm shift in computational theory and practice. Unlike classical computers that operate on bits (0s and 1s), quantum computers leverage quantum mechanical phenomena such as superposition and entanglement to process information using quantum bits or "qubits." This fundamental difference enables quantum computers to solve certain problems exponentially faster than their classical counterparts.

## Current State of Technology

As of 2023, quantum computing remains in its early stages but has shown remarkable progress. IBM, Google, and several startups have built quantum processors with 50-100+ qubits. However, these systems are still prone to errors and decoherence, limiting their practical applications. 

The field has witnessed several milestone achievements:

- In 2019, Google claimed to achieve "quantum supremacy" by performing a calculation that would be practically impossible for classical supercomputers.
- IBM released its 127-qubit Eagle processor in 2021, followed by the 433-qubit Osprey in 2022.
- In 2023, multiple research teams demonstrated improved quantum error correction techniques, a critical step toward fault-tolerant quantum computing.

## Core Quantum Algorithms

Several quantum algorithms demonstrate theoretical advantages over classical methods:

### Shor's Algorithm
Developed by Peter Shor in 1994, this algorithm can factor large integers exponentially faster than the best known classical algorithms, potentially threatening current cryptographic systems based on RSA encryption.

### Grover's Algorithm
Developed by Lov Grover in 1996, this provides a quadratic speedup for unstructured database searches, reducing the complexity from O(N) to O(√N).

### Quantum Approximate Optimization Algorithm (QAOA)
A more recent development aimed at solving combinatorial optimization problems on near-term quantum devices.

## Industry Applications

While general-purpose quantum computers remain years away, specific applications are emerging:

1. **Cryptography**: Quantum-resistant encryption methods are being developed to counter the threat posed by quantum computers to current security systems.

2. **Drug Discovery**: Companies like Zapata Computing and Menten AI are using quantum computing to simulate molecular interactions for pharmaceutical research.

3. **Financial Modeling**: JPMorgan Chase, Goldman Sachs, and other financial institutions are exploring quantum computing for portfolio optimization and risk analysis.

4. **Materials Science**: Quantum simulations can help discover new materials with specific properties for batteries, solar cells, and superconductors.

## Challenges and Limitations

Despite progress, significant challenges remain:

- **Quantum Decoherence**: Qubits are extremely sensitive to environmental noise, causing them to lose their quantum properties rapidly.
- **Error Rates**: Current quantum gates have error rates around 0.1-1%, too high for complex algorithms without error correction.
- **Scalability**: Building large-scale quantum systems while maintaining coherence and connectivity is technically demanding.
- **Cost and Accessibility**: Quantum computers require specialized infrastructure including cryogenic cooling systems operating near absolute zero.

## Future Outlook

The quantum computing roadmap suggests several key developments in the coming decade:

- Improved error correction leading to logical qubits with significantly lower error rates
- Quantum processors with 1,000+ physical qubits by 2025
- The emergence of quantum advantage in specific commercial applications
- Greater integration between quantum and classical computing systems

Industry analysts project the quantum computing market to reach $1.7 billion by 2026, growing at an annual rate of 30.2%. Major investments continue from governments, tech giants, and venture capital firms.

## Conclusion

Quantum computing stands at a pivotal moment, transitioning from theoretical curiosity to practical technology. While universal quantum computers remain distant, specialized quantum processors may soon deliver value in targeted domains. The field's rapid evolution demands continued attention from researchers, policymakers, and business leaders prepared to harness its transformative potential.

"""
        source = "sample text"
    
    console.print(f"Using [cyan]{source}[/] for document analysis")
    
    # 5.1. Section Identification
    console.print("\n[bold cyan]Identifying document sections...[/]")
    
    with console.status("[cyan]Identifying sections...", spinner="dots"):
        try:
            sections_result = await doc_tool.identify_sections(document=demo_text)
            
            if sections_result.get("success"):
                display_result(
                    "Document Sections Identified", 
                    sections_result,
                    display_keys=["sections"]
                )
            else:
                console.print(Panel(
                    f"[bold red]Section identification failed:[/]\n{sections_result.get('error', 'Unknown error')}",
                    title="[red]Failed to Identify Sections[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
        except Exception as e:
            logger.error(f"Error during section identification: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/] {e}")
    
    # 5.2. Entity Extraction
    console.print("\n[bold cyan]Extracting named entities...[/]")
    
    with console.status("[cyan]Extracting entities...", spinner="dots"):
        try:
            entities_result = await doc_tool.extract_entities(
                document=demo_text,
                entity_types=["PERSON", "ORGANIZATION", "DATE", "PRODUCT", "ALGORITHM"]
            )
            
            if entities_result.get("success"):
                display_result(
                    "Named Entities Extracted", 
                    entities_result,
                    display_keys=["entities"]
                )
                
                # Try entity canonicalization if entities were found
                if entities_result.get("entities"):
                    console.print("\n[bold cyan]Canonicalizing entities...[/]")
                    
                    canon_result = await doc_tool.canonicalise_entities(
                        entities_input=entities_result
                    )
                    
                    if canon_result.get("success"):
                        display_result(
                            "Canonicalized Entities", 
                            canon_result,
                            display_keys=["canonicalized"]
                        )
            else:
                console.print(Panel(
                    f"[bold red]Entity extraction failed:[/]\n{entities_result.get('error', 'Unknown error')}",
                    title="[red]Failed to Extract Entities[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
        except Exception as e:
            logger.error(f"Error during entity extraction: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/] {e}")
    
    # 5.3. QA Generation
    console.print("\n[bold cyan]Generating QA pairs...[/]")
    
    with console.status("[cyan]Generating QA pairs...", spinner="dots"):
        try:
            qa_result = await doc_tool.generate_qa_pairs(
                document=demo_text,
                num_questions=3
            )
            
            if qa_result.get("success"):
                display_result(
                    "Generated QA Pairs", 
                    qa_result,
                    display_keys=["qa_pairs"]
                )
            else:
                console.print(Panel(
                    f"[bold red]QA generation failed:[/]\n{qa_result.get('error', 'Unknown error')}",
                    title="[red]Failed to Generate QA Pairs[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
        except Exception as e:
            logger.error(f"Error during QA generation: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/] {e}")
    
    # 5.4. Document Summarization
    console.print("\n[bold cyan]Summarizing document...[/]")
    
    with console.status("[cyan]Generating summary...", spinner="dots"):
        try:
            summary_result = await doc_tool.summarize_document(
                document=demo_text,
                max_length=100
            )
            
            if summary_result.get("success"):
                display_result(
                    "Document Summary", 
                    summary_result,
                    display_keys=["summary", "word_count"]
                )
                
                # Try focused summary
                focused_summary = await doc_tool.summarize_document(
                    document=demo_text,
                    max_length=50,
                    focus="industry applications and challenges"
                )
                
                if focused_summary.get("success"):
                    display_result(
                        "Focused Summary (Industry Applications and Challenges)", 
                        focused_summary,
                        display_keys=["summary", "word_count"]
                    )
            else:
                console.print(Panel(
                    f"[bold red]Summarization failed:[/]\n{summary_result.get('error', 'Unknown error')}",
                    title="[red]Failed to Summarize Document[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
        except Exception as e:
            logger.error(f"Error during summarization: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/] {e}")
    
    # 5.5. Document Classification
    console.print("\n[bold cyan]Classifying document...[/]")
    
    with console.status("[cyan]Classifying document...", spinner="dots"):
        try:
            classification_result = await doc_tool.classify_document(
                document=demo_text,
                custom_labels=["Research Paper", "Technical Report", "News Article", "Tutorial", "Review"]
            )
            
            if classification_result.get("success"):
                display_result(
                    "Document Classification", 
                    classification_result,
                    display_keys=["classification", "confidence", "raw_llm_output"]
                )
            else:
                console.print(Panel(
                    f"[bold red]Classification failed:[/]\n{classification_result.get('error', 'Unknown error')}",
                    title="[red]Failed to Classify Document[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
        except Exception as e:
            logger.error(f"Error during classification: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/] {e}")
    
    # 5.6. Extract Metrics
    console.print("\n[bold cyan]Extracting metrics...[/]")
    
    with console.status("[cyan]Extracting metrics...", spinner="dots"):
        try:
            metrics_result = await doc_tool.extract_metrics(document=demo_text)
            
            if metrics_result.get("success"):
                display_result(
                    "Extracted Metrics", 
                    metrics_result,
                    display_keys=["metrics"]
                )
            else:
                console.print(Panel(
                    f"[bold red]Metrics extraction failed:[/]\n{metrics_result.get('error', 'Unknown error')}",
                    title="[red]Failed to Extract Metrics[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
        except Exception as e:
            logger.error(f"Error during metrics extraction: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/] {e}")
    
    # 5.7. Flag Risks
    console.print("\n[bold cyan]Flagging potential risks...[/]")
    
    with console.status("[cyan]Flagging risks...", spinner="dots"):
        try:
            risks_result = await doc_tool.flag_risks(document=demo_text)
            
            if risks_result.get("success"):
                display_result(
                    "Flagged Risks", 
                    risks_result,
                    display_keys=["risks"]
                )
            else:
                console.print(Panel(
                    f"[bold red]Risk flagging failed:[/]\n{risks_result.get('error', 'Unknown error')}",
                    title="[red]Failed to Flag Risks[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
        except Exception as e:
            logger.error(f"Error during risk flagging: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/] {e}")
    
    console.print()  # Add spacing

async def batch_processing_demo(doc_tool: DocumentProcessingTool) -> None:
    """Demonstrate batch processing capabilities."""
    console.print(Rule("[bold green]6. Batch Processing Demo[/bold green]", style="green"))
    logger.info("Starting batch processing demo")
    
    # Find all PDFs for batch processing
    papers_dir = Path(DEFAULT_PAPERS_DIR)
    pdf_files = list(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        console.print(Panel(
            "No PDF files found for batch processing demo.",
            title="[yellow]Batch Processing Demo Skipped[/]",
            border_style="yellow",
            padding=(1, 2)
        ))
        return
    
    # Limit to 3 PDFs for the demo to avoid overwhelming
    pdf_files = pdf_files[:3]
    console.print(f"Using [green]{len(pdf_files)}[/] PDF files for batch processing demo")
    
    # Create batch inputs
    batch_inputs = [{"document_path": str(pdf), "index": i} for i, pdf in enumerate(pdf_files)]
    
    # Define batch operations
    batch_operations = [
        {
            "operation": "convert_document",
            "output_key": "conversion_result",
            "params": {
                "output_format": "markdown",
                "accelerator_device": "cpu"
            },
            "promote_output": "content"  # Promote content for next step
        },
        {
            "operation": "chunk_document",
            "output_key": "chunking_result",
            "params": {
                "chunk_method": "paragraph",
                "chunk_size": 200
            },
            "promote_output": "chunks"  # Promote chunks for next step
        },
        {
            "operation": "summarize_document",
            "output_key": "summary_result",
            "input_key": "content",  # Use the original full content, not the chunks
            "params": {
                "max_length": 100
            }
        }
    ]
    
    console.print("\n[bold cyan]Starting batch processing pipeline...[/]")
    
    with console.status("[cyan]Processing batch...", spinner="dots"):
        try:
            batch_result = await doc_tool.process_document_batch(
                inputs=batch_inputs,
                operations=batch_operations,
                max_concurrency=2
            )
            
            if batch_result:
                # Display summary of batch results
                console.print(Panel(
                    f"[green]✓ Successfully processed [bold]{len(batch_result)}[/] documents through [bold]{len(batch_operations)}[/] operations.[/]",
                    title="Batch Processing Results",
                    border_style="green",
                    padding=(1, 2)
                ))
                
                # Show more detailed results for first item
                if batch_result:
                    first_item = batch_result[0]
                    
                    # Create a table summarizing the results
                    results_table = Table(title="Results for First Document", box=box.ROUNDED)
                    results_table.add_column("Operation", style="cyan")
                    results_table.add_column("Success", style="green")
                    results_table.add_column("Details", style="white")
                    
                    # Check each operation's result
                    if "conversion_result" in first_item:
                        conv_result = first_item["conversion_result"]
                        results_table.add_row(
                            "Document Conversion",
                            "✓" if conv_result.get("success") else "✗",
                            f"Format: {conv_result.get('output_format', 'unknown')}, Time: {conv_result.get('processing_time', 0):.2f}s"
                        )
                    
                    if "chunking_result" in first_item:
                        chunk_result = first_item["chunking_result"]
                        results_table.add_row(
                            "Document Chunking",
                            "✓" if chunk_result.get("success") else "✗",
                            f"Chunks: {len(chunk_result.get('chunks', []))}"
                        )
                    
                    if "summary_result" in first_item:
                        summary_result = first_item["summary_result"]
                        results_table.add_row(
                            "Document Summarization",
                            "✓" if summary_result.get("success") else "✗",
                            f"Words: {summary_result.get('word_count', 0)}"
                        )
                    
                    console.print(results_table)
                    
                    # Show the summary if available
                    if "summary_result" in first_item and first_item["summary_result"].get("success"):
                        summary = first_item["summary_result"].get("summary", "")
                        if summary:
                            console.print(Panel(
                                f"[italic]{escape(summary)}[/]",
                                title="Generated Summary",
                                border_style="blue",
                                padding=(1, 2)
                            ))
                    
                    # Show error logs if any
                    if "_error_log" in first_item and first_item["_error_log"]:
                        error_panel = Panel(
                            "\n".join([f"[red]- {err}[/]" for err in first_item["_error_log"]]),
                            title="Error Log",
                            border_style="red",
                            padding=(1, 2)
                        )
                        console.print(error_panel)
            else:
                console.print(Panel(
                    "[bold red]Batch processing failed - no results returned.[/]",
                    title="[red]Batch Processing Failed[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
        except Exception as e:
            logger.error(f"Error during batch processing: {e}", exc_info=True)
            console.print(f"[bold red]Error during batch processing:[/] {e}")
    
    console.print()  # Add spacing

async def main() -> int:
    """Run the document processing tools demo."""
    start_time = dt.datetime.now()
    
    # Create and display the main layout
    layout = create_demo_layout()
    
    # Display system information in the header
    system_info = Text()
    system_info.append("Document Processing Demo ", style="bold magenta")
    system_info.append(f"| OS: {os.name} ", style="dim")
    system_info.append(f"| Python: {sys.version.split()[0]} ", style="dim")
    system_info.append(f"| Device: {ACCELERATOR_DEVICE} ", style="green" if USE_GPU else "yellow")
    system_info.append(f"| Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
    
    layout["header"].update(Panel(system_info, border_style="magenta"))
    console.print(layout["header"])
    
    exit_code = 0
    
    try:
        # Initialize Gateway and DocumentProcessingTool
        gateway = Gateway("document-processing-demo", register_tools=False)
        doc_tool = DocumentProcessingTool(gateway)
        
        # Ensure demo directories exist
        DOWNLOADED_FILES_DIR.mkdir(parents=True, exist_ok=True)
        Path(DEFAULT_PAPERS_DIR).mkdir(parents=True, exist_ok=True)
        
        # Check if we have PDF files
        pdf_files = list(Path(DEFAULT_PAPERS_DIR).glob("*.pdf"))
        if not pdf_files:
            # Show usage instructions as markdown
            demo_instructions = """
            # PDF Files Required

            No PDF files found in the `quantum_computing_papers` directory.

            ## Instructions:
            
            1. Add some PDF files to the directory
            2. Or run with `--download-samples` to get sample files
            
            The demo requires PDF files to demonstrate conversion capabilities.
            """
            display_markdown_content(demo_instructions, "Setup Requirements")
            return 1
        
        # Run the demonstrations
        await conversion_demo(doc_tool)
        await chunking_demo(doc_tool)
        await html_processing_demo(doc_tool)
        await table_extraction_demo(doc_tool)
        await document_analysis_demo(doc_tool)
        await batch_processing_demo(doc_tool)
        
        end_time = dt.datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        # Create a summary tree of operations
        demo_tree = Tree("[bold green]Demo Completed Successfully[/]")
        demo_tree.add(f"[cyan]Total runtime:[/] {elapsed:.2f} seconds")
        demo_tree.add(f"[cyan]System:[/] {os.name} / Python {sys.version.split()[0]}")
        demo_tree.add(f"[cyan]Accelerator:[/] {ACCELERATOR_DEVICE}")
        
        operations_node = demo_tree.add("[cyan]Operations:[/]")
        operations_node.add("[green]✓[/] Document Conversion")
        operations_node.add("[green]✓[/] Document Chunking")
        operations_node.add("[green]✓[/] HTML Processing")
        operations_node.add("[green]✓[/] Table Extraction")
        operations_node.add("[green]✓[/] Document Analysis")
        operations_node.add("[green]✓[/] Batch Processing")
        
        console.print(demo_tree)
        
    except Exception as e:
        logger.critical(f"Demo failed with unexpected error: {e}")
        console.print(f"[bold red]CRITICAL ERROR: {escape(str(e))}[/]")
        exit_code = 1
    
    return exit_code

if __name__ == "__main__":
    # Setup logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)