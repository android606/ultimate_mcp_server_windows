#!/usr/bin/env python
"""Demo of HTML to Markdown conversion capabilities using LLM Gateway MCP server."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Project imports
from llm_gateway.tools.html_to_markdown import (
    batch_format_texts,
    clean_and_format_text_as_markdown,
    detect_content_type,
    optimize_markdown_formatting,
)
from llm_gateway.utils import get_logger
from llm_gateway.utils.display import CostTracker
from llm_gateway.utils.logging.console import console

# Initialize logger
logger = get_logger("example.html_to_markdown_demo")

# Create a separate console for detailed debugging output
debug_console = Console(stderr=True, highlight=False)

# Sample HTML snippets for demos
SAMPLE_HTML = """
<div class="article">
    <h1>Welcome to Our Website</h1>
    <p>This is a <b>sample</b> HTML content with <a href="https://example.com">a link</a>.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
    <table>
        <tr>
            <th>Header 1</th>
            <th>Header 2</th>
        </tr>
        <tr>
            <td>Cell 1</td>
            <td>Cell 2</td>
        </tr>
        <tr>
            <td>Cell 3</td>
            <td>Cell 4</td>
        </tr>
    </table>
    <p>Some more text at the bottom.</p>
</div>
"""

COMPLEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Complex HTML Sample</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .container { max-width: 800px; margin: 0 auto; }
        .banner { background-color: #f0f0f0; padding: 20px; }
        .content { padding: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; }
        th { background-color: #f2f2f2; }
    </style>
    <script>
        function showMessage() {
            alert('Hello, World!');
        }
    </script>
</head>
<body>
    <div class="container">
        <div class="banner">
            <h1>Complex HTML Document</h1>
            <p>This document contains various HTML elements including script tags, style tags, and complex structure.</p>
        </div>
        <div class="content">
            <h2>Features</h2>
            <ul>
                <li>Nested elements</li>
                <li>CSS styling</li>
                <li>JavaScript functionality</li>
                <li>Tables and lists</li>
            </ul>
            
            <h2>Sample Table</h2>
            <table>
                <tr>
                    <th>Product</th>
                    <th>Price</th>
                    <th>Stock</th>
                </tr>
                <tr>
                    <td>Widget A</td>
                    <td>$25.99</td>
                    <td>In Stock</td>
                </tr>
                <tr>
                    <td>Widget B</td>
                    <td>$34.50</td>
                    <td>Out of Stock</td>
                </tr>
                <tr>
                    <td>Widget C</td>
                    <td>$15.75</td>
                    <td>In Stock</td>
                </tr>
            </table>
            
            <h2>Base64 Image (will be removed in processing)</h2>
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=" alt="Sample image">
            
            <p>Click <a href="javascript:showMessage()">here</a> to show a message!</p>
        </div>
    </div>
</body>
</html>
"""

MARKDOWN_SAMPLE = """
# Markdown Sample

This is a sample of **Markdown** text with some _formatting_.

## Subheading

* Bullet point 1
* Bullet point 2
* Bullet point 3

[A link](https://example.com)

```python
def hello_world():
    print("Hello, World!")
```

> A blockquote with some wisdom

1. Numbered item 1
2. Numbered item 2
3. Numbered item 3
"""

CODE_SAMPLE = """
function calculateTotal(items) {
    let total = 0;
    for (let i = 0; i < items.length; i++) {
        total += items[i].price * items[i].quantity;
    }
    return total;
}

// Example usage
const cart = [
    { name: 'Product 1', price: 10.99, quantity: 2 },
    { name: 'Product 2', price: 24.99, quantity: 1 },
    { name: 'Product 3', price: 5.49, quantity: 3 }
];

const totalPrice = calculateTotal(cart);
console.log(`Total: $${totalPrice.toFixed(2)}`);
"""

PLAIN_TEXT_SAMPLE = """
Sample Report
Date: April 15, 2025

This is a plain text document with no special formatting.
It has multiple lines but no real structure that would be
considered HTML, Markdown, or code.

Some points to consider:
- The weather is nice today
- Remember to submit your reports
- The meeting is scheduled for tomorrow

Contact: example@example.com if you have questions.
"""

HTML_FRAGMENT = """
<div class="post-content">
  <p>The new study published in <em>Nature Climate Change</em> reveals that Arctic temperatures have risen at <strong>nearly twice</strong> the global average over the past 40 years.</p>
  
  <blockquote>
    "Our findings confirm accelerated warming in the Arctic region, which has serious implications for global climate systems," said lead researcher Dr. Amelia Chen.
  </blockquote>
  
  <p>The research team analyzed data from 1980-2023, finding:</p>
  <ul>
    <li>Average Arctic warming of 3.2°C compared to 1.8°C globally</li>
    <li>Sea ice decline of 12.7% per decade</li>
    <li>Permafrost thaw extending to previously stable regions</li>
  </ul>
</div>
"""

async def demonstrate_clean_and_format(tracker: CostTracker):
    """Demonstrate the clean_and_format_text_as_markdown function."""
    console.print(Rule("[bold cyan]⚡ HTML to Markdown Conversion Demo [/bold cyan]", style="bold blue"))
    logger.info("Starting HTML to Markdown conversion demo", emoji_key="start")
    
    # Display sample HTML
    console.print(Panel(
        escape(SAMPLE_HTML),
        title="[bold yellow]Sample HTML Input[/bold yellow]",
        border_style="yellow",
        expand=False,
        padding=(1, 2)
    ))
    
    # Create progress display
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True
    ) as progress:
        task_id = progress.add_task("[cyan]Converting HTML to Markdown...", total=1)
        
        try:
            # Process HTML with different options
            start_time = time.time()
            
            # Call the tool directly
            result = await clean_and_format_text_as_markdown(
                text=SAMPLE_HTML,
                preserve_tables=True,
                preserve_links=True,
                preserve_images=False
            )
            
            processing_time = time.time() - start_time
            
            # Update progress
            progress.update(task_id, completed=1, description="[green]Conversion complete!")
            
            # Log success
            logger.success(
                f"HTML converted to Markdown successfully in {processing_time:.2f}s",
                emoji_key="success"
            )
            
            # Create markdown display panel
            markdown_panel = Panel(
                escape(result["markdown_text"]),
                title="[bold green]Converted Markdown[/bold green]",
                border_style="green",
                expand=True,
                padding=(1, 2)
            )
            
            # Create a table for metadata
            metadata_table = Table(box=box.MINIMAL, show_header=False, expand=True)
            metadata_table.add_column("Property", style="cyan")
            metadata_table.add_column("Value", style="white")
            metadata_table.add_row("Was HTML", f"[green]{result['was_html']}[/green]")
            metadata_table.add_row("Extraction Method", f"[blue]{result['extraction_method_used']}[/blue]")
            metadata_table.add_row("Processing Time", f"[yellow]{result['processing_time']:.4f}s[/yellow]")
            
            # Display both panels
            console.print(markdown_panel)
            console.print(Panel(metadata_table, title="[bold blue]Conversion Metadata[/bold blue]", border_style="blue"))
            
        except Exception as e:
            progress.update(task_id, completed=1, description="[red]Conversion failed!")
            logger.error(f"Error converting HTML to Markdown: {str(e)}", emoji_key="error", exc_info=True)
    
    # Try with a more complex HTML example
    console.print(Rule("[bold cyan]⚡ Complex HTML Conversion [/bold cyan]", style="bold blue"))
    
    console.print(Panel(
        "Converting a complex HTML document with JavaScript, CSS, and other elements that should be stripped...",
        title="[bold yellow]Complex HTML Example[/bold yellow]",
        border_style="yellow",
        expand=False,
        padding=(1, 2)
    ))
    
    try:
        # Process the complex HTML
        result = await clean_and_format_text_as_markdown(
            text=COMPLEX_HTML,
            extraction_method="auto",
            preserve_tables=True,
            preserve_links=True
        )
        
        # Create markdown display panel for complex example
        complex_markdown_panel = Panel(
            escape(result["markdown_text"]),
            title="[bold green]Converted Complex HTML[/bold green]",
            border_style="green",
            expand=True,
            padding=(1, 2)
        )
        
        # Create a compact stats display
        stats_text = (
            f"[cyan]Extraction Method:[/cyan] {result['extraction_method_used']} | "
            f"[cyan]Processing Time:[/cyan] {result['processing_time']:.4f}s"
        )
        
        console.print(complex_markdown_panel)
        console.print(Panel(stats_text, border_style="blue"))
        
        logger.success(
            f"Complex HTML converted successfully using {result['extraction_method_used']} method",
            emoji_key="success"
        )
        
    except Exception as e:
        logger.error(f"Error converting complex HTML: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_content_detection(tracker: CostTracker):
    """Demonstrate the detect_content_type function."""
    console.print(Rule("[bold cyan]⚡ Content Type Detection Demo [/bold cyan]", style="bold blue"))
    logger.info("Starting content type detection demo", emoji_key="start")
    
    # Create a table for displaying different content types
    content_samples = [
        ("HTML", HTML_FRAGMENT),
        ("Markdown", MARKDOWN_SAMPLE),
        ("Code", CODE_SAMPLE),
        ("Plain Text", PLAIN_TEXT_SAMPLE)
    ]
    
    # Create progress display
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True
    ) as progress:
        task_id = progress.add_task("[cyan]Detecting content types...", total=len(content_samples))
        
        # Create a table for results
        results_table = Table(title="[bold]Content Type Detection Results[/bold]", box=box.ROUNDED)
        results_table.add_column("Content Type", style="cyan")
        results_table.add_column("Detection Result", style="green")
        results_table.add_column("Confidence", style="yellow")
        results_table.add_column("Markers Found", style="magenta")
        
        for content_type, sample in content_samples:
            progress.update(task_id, description=f"[cyan]Detecting {content_type} content...")
            
            try:
                # Call the detect_content_type function
                result = await detect_content_type(text=sample)
                
                # Truncate sample for display
                truncated_sample = sample[:100] + "..." if len(sample) > 100 else sample  # noqa: F841
                
                # Prepare markers info
                markers_info = ""
                if content_type.lower() == "html":
                    markers_info = f"HTML: {result['details']['html_markers']}"
                elif content_type.lower() == "markdown":
                    markers_info = f"Markdown: {result['details']['markdown_markers']}"
                elif content_type.lower() == "code":
                    markers_info = f"Code: {result['details']['code_markers']}" + (
                        f" ({result['details']['detected_language']})" 
                        if result['details']['detected_language'] 
                        else ""
                    )
                
                # Add to results table
                results_table.add_row(
                    content_type,
                    result['content_type'],
                    f"{result['confidence']:.2f}",
                    markers_info
                )
                
                # Update progress
                progress.advance(task_id)
                
            except Exception as e:
                progress.update(task_id, description=f"[red]Error detecting {content_type} content")
                logger.error(f"Error detecting content type for {content_type}: {str(e)}", emoji_key="error")
                results_table.add_row(content_type, "[red]Error[/red]", "N/A", "N/A")
                progress.advance(task_id)
        
        # Complete progress
        progress.update(task_id, completed=len(content_samples), description="[green]Detection complete!")
    
    # Display results table
    console.print(results_table)
    
    # Try with an ambiguous example
    console.print(Panel(
        "Now let's try detection with an ambiguous example that mixes content types...",
        title="[bold yellow]Ambiguous Content Test[/bold yellow]",
        border_style="yellow",
        expand=False,
        padding=(1, 2)
    ))
    
    # Create ambiguous content
    ambiguous_content = """
    <div>
        # This looks like a Markdown heading
        
        function testFunction() {
            console.log("But this is JavaScript code");
        }
        
        <p>And this is HTML again</p>
        
        * This could be a Markdown list
        * With multiple items
    </div>
    """
    
    try:
        # Detect the ambiguous content
        ambiguous_result = await detect_content_type(text=ambiguous_content)
        
        # Create a detailed result panel
        details_table = Table(box=box.MINIMAL, show_header=False)
        details_table.add_column("Property", style="cyan")
        details_table.add_column("Value", style="white")
        details_table.add_row("Detected Type", f"[bold green]{ambiguous_result['content_type']}[/bold green]")
        details_table.add_row("Confidence", f"[yellow]{ambiguous_result['confidence']:.4f}[/yellow]")
        details_table.add_row("HTML Markers", f"{ambiguous_result['details']['html_markers']}")
        details_table.add_row("Markdown Markers", f"{ambiguous_result['details']['markdown_markers']}")
        details_table.add_row("Code Markers", f"{ambiguous_result['details']['code_markers']}")
        
        if ambiguous_result['details']['detected_language']:
            details_table.add_row("Language", f"{ambiguous_result['details']['detected_language']}")
        
        console.print(Panel(
            escape(ambiguous_content),
            title="[bold yellow]Ambiguous Content[/bold yellow]",
            border_style="yellow",
            expand=False,
            padding=(1, 2)
        ))
        
        console.print(Panel(
            details_table,
            title="[bold green]Detection Results for Ambiguous Content[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
        
        logger.success("Ambiguous content detection completed", emoji_key="success")
        
    except Exception as e:
        logger.error(f"Error detecting ambiguous content type: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_batch_processing(tracker: CostTracker):
    """Demonstrate the batch_format_texts function."""
    console.print(Rule("[bold cyan]⚡ Batch Processing Demo [/bold cyan]", style="bold blue"))
    logger.info("Starting batch processing demo", emoji_key="start")
    
    # Create a batch of different text types
    batch_texts = [
        HTML_FRAGMENT,
        "<p>Another simple <b>HTML</b> fragment with a <a href='#'>link</a>.</p>",
        "# Just a markdown heading\n\nWith some paragraph text.",
        "<div><h3>Mixed Content</h3><p>Some <em>emphasized</em> text</p><pre>var x = 10;</pre></div>"
    ]
    
    # Display batch info
    console.print(Panel(
        f"Processing a batch of {len(batch_texts)} texts with varying formats...",
        title="[bold yellow]Batch Processing[/bold yellow]",
        border_style="yellow",
        expand=False,
        padding=(1, 2)
    ))
    
    try:
        # Process the batch
        logger.info("Starting batch processing...", emoji_key="processing")
        start_time = time.time()
        
        # Store the max_concurrency value to use later
        batch_max_concurrency = 2
        
        result = await batch_format_texts(
            texts=batch_texts,
            force_markdown_conversion=False,
            extraction_method="auto",
            max_concurrency=batch_max_concurrency,
            preserve_tables=True
        )
        
        processing_time = time.time() - start_time  # noqa: F841
        
        # Create a table for results summary
        summary_table = Table(title="[bold]Batch Processing Summary[/bold]", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        summary_table.add_row("Total Items", f"{len(batch_texts)}")
        summary_table.add_row("Successful Conversions", f"[green]{result['success_count']}[/green]")
        summary_table.add_row("Failed Conversions", f"[red]{result['failure_count']}[/red]")
        summary_table.add_row("Total Processing Time", f"[yellow]{result['total_processing_time']:.4f}s[/yellow]")
        summary_table.add_row("Average Time Per Item", f"[blue]{result['total_processing_time']/len(batch_texts):.4f}s[/blue]")
        
        console.print(summary_table)
        
        # Display individual results
        for i, item_result in enumerate(result['results']):
            if item_result.get('success', False):
                console.print(Panel(
                    escape(item_result['markdown_text']),
                    title=f"[bold green]Item {i+1} Result[/bold green]",
                    subtitle=f"[dim]Was HTML: {item_result['was_html']} | Method: {item_result['extraction_method_used']}[/dim]",
                    border_style="green",
                    padding=(1, 1)
                ))
            else:
                console.print(Panel(
                    f"[red]Error: {item_result.get('error', 'Unknown error')}[/red]",
                    title=f"[bold red]Item {i+1} Failed[/bold red]",
                    border_style="red",
                    padding=(1, 1)
                ))
        
        # Show concurrency advantage
        console.print(Panel(
            f"Batch processing completed {len(batch_texts)} items in {result['total_processing_time']:.2f}s using concurrency.\n"
            f"Sequential processing would have taken approximately {result['total_processing_time'] * 0.6 * len(batch_texts) / batch_max_concurrency:.2f}s.",
            title="[bold blue]Concurrency Advantage[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        ))
        
        logger.success(f"Batch processing completed successfully: {result['success_count']} succeeded, {result['failure_count']} failed", emoji_key="success")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_markdown_optimization(tracker: CostTracker):
    """Demonstrate the optimize_markdown_formatting function."""
    console.print(Rule("[bold cyan]⚡ Markdown Optimization Demo [/bold cyan]", style="bold blue"))
    logger.info("Starting markdown optimization demo", emoji_key="start")
    
    # Create a messy markdown sample
    messy_markdown = """
    #Heading with no space
    
    * Inconsistent
    + list
    * markers
    
    
    Too many blank lines above
    
    [link with space] (https://example.com)
    
    ##Another heading
    No space after this paragraph.
    Next paragraph starts immediately.
    
    ```
    code block
    with poor formatting
    ```
    
    > Blockquote
    Regular text without proper spacing
    """
    
    # Display the messy markdown
    console.print(Panel(
        escape(messy_markdown),
        title="[bold yellow]Messy Markdown Input[/bold yellow]",
        border_style="yellow",
        expand=False,
        padding=(1, 2)
    ))
    
    try:
        # Optimize the markdown
        logger.info("Optimizing markdown formatting...", emoji_key="processing")
        start_time = time.time()
        
        result = await optimize_markdown_formatting(
            markdown=messy_markdown,
            normalize_headings=True,
            fix_lists=True,
            fix_links=True,
            add_line_breaks=True,
            compact_mode=False
        )
        
        processing_time = time.time() - start_time  # noqa: F841
        
        # Display optimized markdown
        optimized_panel = Panel(
            escape(result["optimized_markdown"]),
            title="[bold green]Optimized Markdown[/bold green]",
            border_style="green",
            expand=True,
            padding=(1, 2)
        )
        
        console.print(optimized_panel)
        
        # Create a table for changes made
        changes_table = Table(title="[bold]Optimization Changes[/bold]", box=box.MINIMAL)
        changes_table.add_column("Change Type", style="cyan")
        changes_table.add_column("Applied", style="white")
        
        for change_type, applied in result["changes_made"].items():
            status = "[green]Yes[/green]" if applied else "[yellow]No[/yellow]"
            changes_table.add_row(change_type.replace("_", " ").title(), status)
        
        console.print(Panel(
            changes_table,
            title=f"[bold blue]Changes Made in {result['processing_time']:.4f}s[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        ))
        
        # Try with compact mode
        logger.info("Optimizing with compact mode...", emoji_key="processing")
        
        compact_result = await optimize_markdown_formatting(
            markdown=messy_markdown,
            normalize_headings=True,
            fix_lists=True,
            fix_links=True,
            add_line_breaks=False,
            compact_mode=True
        )
        
        console.print(Panel(
            escape(compact_result["optimized_markdown"]),
            title="[bold magenta]Compact Mode Optimization[/bold magenta]",
            border_style="magenta",
            expand=True,
            padding=(1, 2)
        ))
        
        logger.success("Markdown optimization completed successfully", emoji_key="success")
        
    except Exception as e:
        logger.error(f"Error in markdown optimization: {str(e)}", emoji_key="error", exc_info=True)

async def demonstrate_html_fragment_handling(tracker: CostTracker):
    """Demonstrate handling of HTML fragments, which is a key strength of the tool."""
    console.print(Rule("[bold cyan]⚡ HTML Fragment Handling Demo [/bold cyan]", style="bold blue"))
    logger.info("Starting HTML fragment handling demo", emoji_key="start")
    
    # Create an array of increasingly challenging fragments
    fragments = [
        # Simple inline tags
        "<p>This is a <b>simple</b> paragraph with <i>some</i> formatting.</p>",
        
        # Fragment without root element
        "<h2>Heading</h2><p>Paragraph <a href='#'>with link</a></p>",
        
        # Fragment with partial table
        """<table><tr><th>Name</th><th>Value</th></tr>
        <tr><td>Item 1</td><td>Value 1</td></tr></table>""",
        
        # Real-world fragment from a blog post
        """<div class="entry-content">
            <p>Researchers at <a href="#">Example University</a> have discovered a new way to <strong>optimize machine learning models</strong> that reduces training time by up to 40%.</p>
            <blockquote class="wp-block-quote">
                <p>"This breakthrough could significantly accelerate AI development across multiple domains," said lead researcher Dr. Jane Smith.</p>
                <cite>From the official press release</cite>
            </blockquote>
            <h3>Key Findings</h3>
            <ul>
                <li>Training time reduced by 35-40%</li>
                <li>Memory consumption decreased by 22%</li>
                <li>Model accuracy maintained or slightly improved</li>
            </ul>
        </div>"""
    ]
    
    # Process each fragment
    for i, fragment in enumerate(fragments):
        fragment_number = i + 1
        console.print(Panel(
            escape(fragment),
            title=f"[bold yellow]Fragment {fragment_number}[/bold yellow]",
            border_style="yellow",
            expand=False,
            padding=(1, 2)
        ))
        
        try:
            # First detect content type
            detection_result = await detect_content_type(text=fragment)
            
            # Convert the fragment
            conversion_result = await clean_and_format_text_as_markdown(
                text=fragment,
                force_markdown_conversion=True,  # Force conversion even if detection is uncertain
                extraction_method="raw",  # Use raw extraction for fragments
                preserve_tables=True,
                preserve_links=True
            )
            
            # Display results
            result_panel = Panel(
                escape(conversion_result["markdown_text"]),
                title=f"[bold green]Fragment {fragment_number} Result[/bold green]",
                border_style="green",
                expand=True,
                padding=(1, 2)
            )
            
            # Show detection and conversion details
            details = [
                f"[cyan]Detection:[/cyan] {detection_result['content_type']} (Confidence: {detection_result['confidence']:.2f})",
                f"[cyan]HTML Markers:[/cyan] {detection_result['details']['html_markers']}",
                f"[cyan]Was HTML:[/cyan] {conversion_result['was_html']}",
                f"[cyan]Extraction Method:[/cyan] {conversion_result['extraction_method_used']}",
                f"[cyan]Processing Time:[/cyan] {conversion_result['processing_time']:.4f}s"
            ]
            
            details_panel = Panel(
                "\n".join(details),
                title="[bold blue]Processing Details[/bold blue]",
                border_style="blue",
                expand=False,
                padding=(1, 2)
            )
            
            console.print(result_panel)
            console.print(details_panel)
            
            logger.success(f"Fragment {fragment_number} processed successfully", emoji_key="success")
            
        except Exception as e:
            logger.error(f"Error processing fragment {fragment_number}: {str(e)}", emoji_key="error", exc_info=True)
        
        console.print()  # Add spacing between fragments

async def main():
    """Run HTML to Markdown conversion demos."""
    tracker = CostTracker()
    try:
        # Create title with padding
        title = Text("⚡ HTML to Markdown Conversion Showcase ⚡", style="bold white on blue")
        title.justify = "center"
        console.print(Panel(title, box=box.DOUBLE_EDGE, padding=(1, 0)))
        
        debug_console.print("[dim]Starting HTML to Markdown demo in debug mode[/dim]")
        
        # Run the demos
        await demonstrate_clean_and_format(tracker)
        console.print()
        
        await demonstrate_content_detection(tracker)
        console.print()
        
        await demonstrate_html_fragment_handling(tracker)
        console.print()
        
        await demonstrate_batch_processing(tracker)
        console.print()
        
        await demonstrate_markdown_optimization(tracker)
        
        # No cost tracking needed since we're not using API calls in this demo
        
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        debug_console.print_exception(show_locals=True)
        return 1
    
    logger.success("HTML to Markdown Demo Finished Successfully!", emoji_key="complete")
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)