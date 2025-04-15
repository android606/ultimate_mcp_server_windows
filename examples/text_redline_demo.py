#!/usr/bin/env python
"""Text redline comparison tool demonstration for LLM Gateway."""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
from rich import box
from rich.columns import Columns
from rich.console import Group
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
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Project imports
from llm_gateway.constants import Provider
from llm_gateway.core.server import Gateway

# Import our text redline tool
from llm_gateway.tools.text_redline_tools import compare_documents_redline, create_html_redline
from llm_gateway.utils import get_logger
from llm_gateway.utils.display import CostTracker
from llm_gateway.utils.logging.console import console

# Initialize logger
logger = get_logger("example.text_redline")

# Sample HTML documents for demonstration
ORIGINAL_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Company Policy Document</title>
</head>
<body>
    <h1>Employee Handbook</h1>
    <p>Welcome to our company. This handbook outlines our policies.</p>
    
    <h2>Work Hours</h2>
    <p>Standard work hours are 9:00 AM to 5:00 PM, Monday through Friday.</p>
    
    <h2>Vacation Policy</h2>
    <p>Full-time employees receive 10 days of paid vacation annually.</p>
    <p>Vacation requests must be submitted at least two weeks in advance.</p>
    
    <h2>Code of Conduct</h2>
    <p>Employees are expected to maintain professional behavior at all times.</p>
    <p>Respect for colleagues is essential to our workplace culture.</p>
    
    <table border="1">
        <tr>
            <th>Benefit</th>
            <th>Eligibility</th>
        </tr>
        <tr>
            <td>Health Insurance</td>
            <td>After 30 days</td>
        </tr>
        <tr>
            <td>401(k)</td>
            <td>After 90 days</td>
        </tr>
    </table>
</body>
</html>"""

MODIFIED_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Company Policy Document - 2025 Update</title>
</head>
<body>
    <h1>Employee Handbook</h1>
    <p>Welcome to our company. This handbook outlines our policies and procedures.</p>
    
    <h2>Flexible Work Schedule</h2>
    <p>With our new flexible work policy, employees can choose to work between 7:00 AM and 7:00 PM.</p>
    <p>A minimum of 6 hours must overlap with core hours (10:00 AM to 4:00 PM).</p>
    
    <h2>Code of Conduct</h2>
    <p>Employees are expected to maintain professional behavior at all times.</p>
    <p>Respect for colleagues is essential to our workplace culture.</p>
    
    <h2>Vacation Policy</h2>
    <p>Full-time employees receive 15 days of paid vacation annually.</p>
    <p>Vacation requests must be submitted at least one week in advance.</p>
    
    <table border="1">
        <tr>
            <th>Benefit</th>
            <th>Eligibility</th>
        </tr>
        <tr>
            <td>Health Insurance</td>
            <td>Immediate</td>
        </tr>
        <tr>
            <td>401(k)</td>
            <td>After 60 days</td>
        </tr>
        <tr>
            <td>Professional Development</td>
            <td>After 180 days</td>
        </tr>
    </table>
</body>
</html>"""

# Sample markdown documents
ORIGINAL_MD = """# Project Proposal

## Overview
This project aims to improve customer satisfaction by implementing a new feedback system.

## Goals
1. Increase response rate to customer surveys by 25%
2. Reduce resolution time for customer issues by 30%
3. Improve overall customer satisfaction score to 4.5/5

## Timeline
The project will be completed in 3 months.

## Budget
The estimated budget is $50,000.

## Team
- Project Manager: John Smith
- Developer: Jane Doe
- Designer: David Johnson
"""

MODIFIED_MD = """# Project Proposal: Customer Experience Enhancement

## Overview
This project aims to revolutionize customer experience by implementing an advanced feedback and resolution system.

## Goals
1. Increase response rate to customer surveys by 40%
2. Reduce resolution time for customer issues by 50%
3. Improve overall customer satisfaction score to 4.8/5
4. Implement AI-based feedback analysis

## Timeline
The project will be completed in 4 months, with monthly progress reviews.

## Budget
The estimated budget is $75,000, including software licensing costs.

## Team
- Project Manager: John Smith
- Lead Developer: Jane Doe
- UX Designer: David Johnson
- Data Analyst: Sarah Williams
"""

# Sample plain text
ORIGINAL_TEXT = """QUARTERLY BUSINESS REVIEW
Q1 2025

Revenue: $2.3M
Expenses: $1.8M
Profit: $0.5M

Key Achievements:
- Launched new product line
- Expanded to 2 new markets
- Hired 5 new team members

Challenges:
- Supply chain delays
- Increased competition
- Rising material costs

Next Steps:
- Evaluate pricing strategy
- Invest in marketing
- Explore partnership opportunities
"""

MODIFIED_TEXT = """QUARTERLY BUSINESS REVIEW
Q1 2025

Revenue: $2.5M
Expenses: $1.7M
Profit: $0.8M

Key Achievements:
- Launched new premium product line
- Expanded to 3 new markets
- Hired 8 new team members
- Secured major enterprise client

Challenges:
- Minor supply chain delays
- Increased competition in EU market
- Staff retention in technical roles

Next Steps:
- Implement dynamic pricing strategy
- Double marketing budget for Q2
- Finalize strategic partnership with TechCorp
- Develop employee retention program
"""


async def demonstrate_basic_redline():
    """Demonstrate basic HTML redlining capabilities."""
    console.print(Rule("[bold blue]Basic HTML Redline Demonstration[/bold blue]"))
    logger.info("Starting basic HTML redline demonstration", emoji_key="start")
    
    # Display input document information
    input_table = Table(title="[bold cyan]Input Documents[/bold cyan]", box=box.MINIMAL, show_header=False)
    input_table.add_column("Document", style="cyan")
    input_table.add_column("Details", style="white")
    
    original_lines = ORIGINAL_HTML.count('\n') + 1
    modified_lines = MODIFIED_HTML.count('\n') + 1
    
    input_table.add_row("Original Document", f"HTML, {original_lines} lines")
    input_table.add_row("Modified Document", f"HTML, {modified_lines} lines")
    console.print(input_table)
    
    # Create progress display
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True
    ) as progress:
        task = progress.add_task("[cyan]Generating redline...", total=1)
        
        try:
            # Generate the redline
            result = await create_html_redline(
                original_html=ORIGINAL_HTML,
                modified_html=MODIFIED_HTML,
                detect_moves=True,
                include_css=True,
                add_navigation=True
            )
            
            # Mark task as complete
            progress.update(task, completed=1)
            
            # Log success
            logger.success(
                "Redline generated successfully",
                emoji_key="success",
                stats=result["stats"]
            )
            
            # Display stats in a table
            stats_table = Table(title="[bold green]Redline Statistics[/bold green]", box=box.ROUNDED)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            for key, value in result["stats"].items():
                stats_table.add_row(key.replace('_', ' ').title(), str(value))
            
            stats_table.add_row("Processing Time", f"{result['processing_time']:.3f} seconds")
            console.print(stats_table)
            
            # Display a preview of the redline HTML
            # Extract just a portion for display to keep it manageable
            html_preview = result["redline_html"]
            
            # Find the body tag and extract a portion
            body_start = html_preview.find("<body")
            if body_start > 0:
                body_end = html_preview.find("</body>", body_start)
                if body_end > 0:
                    content = html_preview[body_start:body_end]
                    # Further trim if too long
                    if len(content) > 1000:
                        content = content[:1000] + "...(truncated)..."
                else:
                    content = html_preview[:1000] + "...(truncated)..."
            else:
                content = html_preview[:1000] + "...(truncated)..."
            
            # Create a syntax object with HTML highlighting
            syntax = Syntax(content, "html", theme="monokai", line_numbers=True)
            
            console.print(Panel(
                syntax,
                title="[bold]HTML Redline Preview[/bold]",
                subtitle="[dim](truncated for display)[/dim]",
                border_style="green",
                padding=(1, 2)
            ))
            
            # Calculate summary of changes
            total_changes = result["stats"]["total_changes"]
            original_size = len(ORIGINAL_HTML)
            modified_size = len(MODIFIED_HTML)
            
            console.print(Panel(
                f"The redline shows [bold cyan]{total_changes}[/bold cyan] changes between documents.\n"
                f"Original document size: [yellow]{original_size}[/yellow] characters\n"
                f"Modified document size: [yellow]{modified_size}[/yellow] characters\n"
                f"In a real application, this HTML would be displayed in a browser with full styling and navigation.",
                title="[bold]Summary[/bold]",
                border_style="blue",
                padding=(1, 2)
            ))
            
        except Exception as e:
            # Update progress bar to show error
            progress.update(task, description="[bold red]Error![/bold red]", completed=1)
            
            # Log error
            logger.error(f"Failed to generate redline: {str(e)}", emoji_key="error", exc_info=True)
            
            # Display error message
            console.print(Panel(
                f"[bold red]Error generating redline:[/bold red]\n{escape(str(e))}",
                title="[bold red]Error[/bold red]",
                border_style="red",
                padding=(1, 2)
            ))


async def demonstrate_advanced_redline_features():
    """Demonstrate advanced redline features like move detection."""
    console.print(Rule("[bold blue]Advanced Redline Features[/bold blue]"))
    logger.info("Demonstrating advanced redline features", emoji_key="start")
    
    # Create a comparison table for different redline options
    comparison_table = Table(title="[bold cyan]Redline Configuration Comparison[/bold cyan]")
    comparison_table.add_column("Configuration", style="cyan")
    comparison_table.add_column("Move Detection", style="green")
    comparison_table.add_column("Whitespace Handling", style="yellow")
    comparison_table.add_column("Output Format", style="magenta")
    
    # Add different configurations
    comparison_table.add_row(
        "Default", "Enabled", "Ignore whitespace", "Complete HTML document"
    )
    comparison_table.add_row(
        "No Move Detection", "Disabled", "Ignore whitespace", "Complete HTML document"
    )
    comparison_table.add_row(
        "Whitespace Sensitive", "Enabled", "Preserve whitespace", "Complete HTML document"
    )
    comparison_table.add_row(
        "Fragment Output", "Enabled", "Ignore whitespace", "HTML fragment (body only)"
    )
    
    console.print(comparison_table)
    
    # Setup for the advanced demos
    configs = [
        {
            "name": "With Move Detection",
            "detect_moves": True,
            "ignore_whitespace": True,
            "output_format": "html"
        },
        {
            "name": "Without Move Detection",
            "detect_moves": False,
            "ignore_whitespace": True,
            "output_format": "html"
        }
    ]
    
    # Create a progress display
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True
    ) as progress:
        # Create a task for each configuration
        tasks = {config["name"]: progress.add_task(f"[cyan]Processing {config['name']}...", total=1) for config in configs}
        
        results = {}
        
        # Process each configuration
        for config in configs:
            config_name = config["name"]
            task_id = tasks[config_name]
            
            try:
                # Generate redline with this configuration
                result = await create_html_redline(
                    original_html=ORIGINAL_HTML,
                    modified_html=MODIFIED_HTML,
                    detect_moves=config["detect_moves"],
                    ignore_whitespace=config["ignore_whitespace"],
                    output_format=config["output_format"]
                )
                
                # Store result
                results[config_name] = result
                
                # Update progress
                progress.update(task_id, completed=1)
                
                logger.info(
                    f"Generated redline with configuration: {config_name}",
                    emoji_key="success",
                    stats=result["stats"]
                )
                
            except Exception as e:
                progress.update(task_id, description=f"[bold red]Error processing {config_name}[/bold red]", completed=1)
                logger.error(f"Error with {config_name}: {str(e)}", emoji_key="error")
                results[config_name] = {"error": str(e)}
    
    # Compare the results
    if all(["error" not in results[name] for name in results]):
        # Create comparison panels
        comparison_panels = []
        
        for name, result in results.items():
            # Create a stats panel for this configuration
            stats = result["stats"]
            
            stats_group = []
            
            # Add title for this configuration
            stats_group.append(Text(f"Configuration: {name}", style="bold cyan"))
            
            # Create stats table
            stats_table = Table(box=box.MINIMAL, show_header=False, padding=(0, 1))
            stats_table.add_column("Metric", style="dim cyan")
            stats_table.add_column("Value", style="white")
            
            for key, value in stats.items():
                # Highlight the move count if this is the move detection config
                if key == "moves" and name == "With Move Detection":
                    stats_table.add_row(key.replace('_', ' ').title(), f"[bold green]{value}[/bold green]")
                elif key == "moves" and name == "Without Move Detection":
                    stats_table.add_row(key.replace('_', ' ').title(), f"[dim]{value}[/dim] (detection disabled)")
                else:
                    stats_table.add_row(key.replace('_', ' ').title(), str(value))
            
            stats_table.add_row("Processing Time", f"{result['processing_time']:.3f} seconds")
            stats_group.append(stats_table)
            
            # Create a panel for this configuration
            panel = Panel(
                Group(*stats_group),
                title=f"[bold]{name}[/bold]",
                border_style="green" if "With Move" in name else "yellow",
                padding=(1, 2)
            )
            comparison_panels.append(panel)
        
        # Display the panels in columns
        console.print(Columns(comparison_panels))
        
        # Create a visualized summary of the differences between approaches
        move_detected = results["With Move Detection"]["stats"]["moves"]
        move_disabled = results["Without Move Detection"]["stats"]["moves"]  # noqa: F841
        
        # Add explanation of difference
        if move_detected > 0:
            console.print(Panel(
                f"The redline with move detection identified [bold green]{move_detected}[/bold green] moved blocks of content.\n"
                f"Without move detection, these would appear as [bold red]{move_detected}[/bold red] deletions and [bold blue]{move_detected}[/bold blue] insertions instead.\n\n"
                f"Move detection helps reduce visual noise in the redline and makes it easier to understand the changes.",
                title="[bold]Impact of Move Detection[/bold]",
                border_style="blue",
                padding=(1, 2)
            ))
        else:
            console.print(Panel(
                "No moved blocks were detected in this example. With more complex documents that have\n"
                "rearranged sections, move detection would highlight moved content in green instead of\n"
                "showing it as deletions and insertions.",
                title="[bold]Impact of Move Detection[/bold]",
                border_style="blue",
                padding=(1, 2)
            ))
    else:
        # Handle errors in the results
        for name, result in results.items():
            if "error" in result:
                console.print(Panel(
                    f"[bold red]Error with {name}:[/bold red]\n{escape(result['error'])}",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                    padding=(1, 2)
                ))


async def demonstrate_multi_format_redline():
    """Demonstrate redlining different document formats."""
    console.print(Rule("[bold blue]Multi-Format Redline Comparison[/bold blue]"))
    logger.info("Demonstrating redline across different document formats", emoji_key="start")
    
    # Prepare format configs
    formats = [
        {
            "name": "HTML Format",
            "original": ORIGINAL_HTML,
            "modified": MODIFIED_HTML,
            "format": "html"
        },
        {
            "name": "Markdown Format",
            "original": ORIGINAL_MD,
            "modified": MODIFIED_MD,
            "format": "markdown"
        },
        {
            "name": "Plain Text Format",
            "original": ORIGINAL_TEXT,
            "modified": MODIFIED_TEXT,
            "format": "text"
        }
    ]
    
    # Create a table showing the formats to be compared
    format_table = Table(title="[bold cyan]Document Formats for Comparison[/bold cyan]", box=box.MINIMAL)
    format_table.add_column("Format", style="cyan")
    format_table.add_column("Original Size", style="green")
    format_table.add_column("Modified Size", style="blue")
    format_table.add_column("Description", style="white")
    
    for fmt in formats:
        format_table.add_row(
            fmt["name"],
            f"{len(fmt['original'])} chars",
            f"{len(fmt['modified'])} chars",
            f"Sample {fmt['format']} document"
        )
    
    console.print(format_table)
    
    # Process each format
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True
    ) as progress:
        tasks = {}
        results = {}
        
        for fmt in formats:
            task_id = progress.add_task(f"[cyan]Processing {fmt['name']}...", total=1)
            tasks[fmt["name"]] = task_id
            
            try:
                # Use create_html_redline for HTML, compare_documents_redline for others
                if fmt["format"] == "html":
                    result = await create_html_redline(
                        original_html=fmt["original"],
                        modified_html=fmt["modified"],
                        detect_moves=True,
                        include_css=True,
                        output_format="fragment"  # Use fragment for display
                    )
                else:
                    result = await compare_documents_redline(
                        original_text=fmt["original"],
                        modified_text=fmt["modified"],
                        file_format=fmt["format"],
                        detect_moves=True,
                        output_format="html",
                        diff_level="word"
                    )
                
                # Store result
                results[fmt["name"]] = result
                
                # Update progress
                progress.update(task_id, completed=1)
                
                logger.info(
                    f"Generated redline for {fmt['name']}",
                    emoji_key="success",
                    stats=result["stats"] if "stats" in result else {"format": fmt["format"]}
                )
                
            except Exception as e:
                progress.update(task_id, description=f"[bold red]Error processing {fmt['name']}[/bold red]", completed=1)
                logger.error(f"Error with {fmt['name']}: {str(e)}", emoji_key="error")
                results[fmt["name"]] = {"error": str(e)}
    
    # Display comparison of results
    comparison_panels = []
    
    for fmt in formats:
        name = fmt["name"]
        result = results.get(name, {})
        
        if "error" in result:
            # Create error panel
            panel = Panel(
                f"[bold red]Error processing {name}:[/bold red]\n{escape(result['error'])}",
                title=f"[bold red]{name} - Error[/bold red]",
                border_style="red",
                padding=(1, 2)
            )
        else:
            # Create success panel with stats
            stats_table = Table(box=box.MINIMAL, show_header=False, padding=(0, 1))
            stats_table.add_column("Metric", style="dim cyan")
            stats_table.add_column("Value", style="white")
            
            if "stats" in result:
                for key, value in result["stats"].items():
                    key_display = key.replace('_', ' ').title()
                    stats_table.add_row(key_display, str(value))
            
            processing_time = result.get("processing_time", 0)
            stats_table.add_row("Processing Time", f"{processing_time:.3f} seconds")
            
            # Add a small preview (first ~300 chars) of the redline
            redline_content = result.get("redline_html", result.get("redline", "No preview available"))
            preview = redline_content[:300] + "..." if len(redline_content) > 300 else redline_content
            
            # Determine the appropriate color based on format
            color = "green" if name == "HTML Format" else "blue" if name == "Markdown Format" else "yellow"
            
            panel = Panel(
                Group(
                    stats_table,
                    Text("\nRedline Preview (truncated):", style="bold"),
                    Panel(
                        escape(preview),
                        border_style="dim",
                        padding=(1, 2)
                    )
                ),
                title=f"[bold {color}]{name}[/bold {color}]",
                border_style=color,
                padding=(1, 2)
            )
        
        comparison_panels.append(panel)
    
    # Display the panels in a column layout
    console.print(Columns(comparison_panels))
    
    # Add comparison summary
    if all(["error" not in results.get(fmt["name"], {}) for fmt in formats]):
        stats_comparisons = []
        
        for fmt in formats:
            name = fmt["name"]
            result = results.get(name, {})
            stats = result.get("stats", {})
            
            total_changes = stats.get("total_changes", 0)
            insertions = stats.get("insertions", 0)
            deletions = stats.get("deletions", 0)
            
            stats_comparisons.append({
                "name": name,
                "format": fmt["format"],
                "total_changes": total_changes,
                "insertions": insertions,
                "deletions": deletions
            })
        
        # Display comparison table
        comp_table = Table(title="[bold]Format Comparison[/bold]")
        comp_table.add_column("Format", style="cyan")
        comp_table.add_column("Total Changes", style="bold white")
        comp_table.add_column("Insertions", style="blue")
        comp_table.add_column("Deletions", style="red")
        
        for comp in stats_comparisons:
            comp_table.add_row(
                comp["name"],
                str(comp["total_changes"]),
                str(comp["insertions"]),
                str(comp["deletions"])
            )
        
        console.print(comp_table)
        
        # Identify the format with the most changes
        try:
            most_changes = max(stats_comparisons, key=lambda x: x["total_changes"])
            console.print(Panel(
                f"The [bold cyan]{most_changes['name']}[/bold cyan] had the most detected changes with "
                f"[bold]{most_changes['total_changes']}[/bold] total changes.\n\n"
                f"Different document formats may result in different redline representations due to their structure "
                f"and parsing methods. HTML documents retain more structural information, while plain text relies "
                f"more heavily on textual comparison.",
                title="[bold]Format Comparison Summary[/bold]",
                border_style="blue",
                padding=(1, 2)
            ))
        except (ValueError, KeyError):
            # Handle empty or incomplete comparison
            pass


async def demonstrate_llm_redline_integration(tracker: CostTracker):
    """Demonstrate integration with LLMs to generate content for redlining."""
    console.print(Rule("[bold blue]LLM Integration for Redline Generation[/bold blue]"))
    logger.info("Demonstrating LLM integration with redline tool", emoji_key="start")
    
    # Initialize gateway with providers
    gateway = Gateway("redline-demo", register_tools=False)
    logger.info("Initializing providers...", emoji_key="provider")
    await gateway._initialize_providers()
    
    # Providers to try in order of preference
    providers_to_try = [
        Provider.OPENAI.value,
        Provider.ANTHROPIC.value,
        Provider.GEMINI.value,
        Provider.DEEPSEEK.value
    ]
    
    # Find an available provider
    provider = None
    provider_name = None
    
    for p_name in providers_to_try:
        if p_name in gateway.providers:
            provider = gateway.providers[p_name]
            provider_name = p_name
            logger.info(f"Using provider {p_name}", emoji_key="provider")
            break
    
    if not provider:
        logger.warning("No LLM providers available. Using predefined content for demo.", emoji_key="warning")
        console.print(Panel(
            "[yellow]No LLM providers available. Using predefined content instead of generating with LLM.[/yellow]",
            title="[bold yellow]Warning[/bold yellow]",
            border_style="yellow",
            padding=(1, 2)
        ))
        
        # Use our existing samples
        original_content = ORIGINAL_TEXT
        revised_content = MODIFIED_TEXT
    else:
        # Define the prompts
        base_prompt = """Create a project status update for a fictional software development project named "Phoenix". 
The update should include:
- Project name and brief description
- Current progress/milestone status
- Key achievements
- Challenges or blockers
- Next steps

Keep it brief and professional, around 10-15 lines total."""
        
        revision_prompt = """Create an updated version of the following project status report:

{original_content}

The updated version should:
- Reflect more positive progress (15% more completion)
- Add 1-2 new achievements
- Remove 1 challenge that was resolved
- Add 1-2 new next steps
- Maintain the same general structure and format

The updated report should still be brief (10-15 lines) but reflect these changes."""
        
        # Generate original content
        console.print(Panel(
            escape(base_prompt),
            title="[bold yellow]Base Prompt[/bold yellow]",
            border_style="yellow",
            padding=(1, 2)
        ))
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True
        ) as progress:
            original_task = progress.add_task("[cyan]Generating original content...", total=1)
            
            try:
                # Generate original content
                model = provider.get_default_model()
                logger.info(f"Generating original content with {provider_name}/{model}", emoji_key="processing")
                
                result1 = await provider.generate_completion(
                    prompt=base_prompt,
                    model=model,
                    temperature=0.7,
                    max_tokens=500
                )
                
                # Track cost
                tracker.add_call(result1)
                
                original_content = result1.text.strip()
                progress.update(original_task, completed=1)
                logger.success("Generated original content", emoji_key="success")
                
                # Display original content
                console.print(Panel(
                    escape(original_content),
                    title=f"[bold green]Original Content ({provider_name}/{model})[/bold green]",
                    subtitle=f"[dim]Tokens: {result1.input_tokens} in, {result1.output_tokens} out | Cost: ${result1.cost:.6f}[/dim]",
                    border_style="green",
                    padding=(1, 2)
                ))
                
                # Generate revised content
                revision_prompt_filled = revision_prompt.format(original_content=original_content)
                
                revision_task = progress.add_task("[cyan]Generating revised content...", total=1)
                
                result2 = await provider.generate_completion(
                    prompt=revision_prompt_filled,
                    model=model,
                    temperature=0.7,
                    max_tokens=500
                )
                
                # Track cost
                tracker.add_call(result2)
                
                revised_content = result2.text.strip()
                progress.update(revision_task, completed=1)
                logger.success("Generated revised content", emoji_key="success")
                
                # Display revised content
                console.print(Panel(
                    escape(revised_content),
                    title=f"[bold blue]Revised Content ({provider_name}/{model})[/bold blue]",
                    subtitle=f"[dim]Tokens: {result2.input_tokens} in, {result2.output_tokens} out | Cost: ${result2.cost:.6f}[/dim]",
                    border_style="blue",
                    padding=(1, 2)
                ))
                
            except Exception as e:
                logger.error(f"Error generating content with LLM: {str(e)}", emoji_key="error", exc_info=True)
                console.print(Panel(
                    f"[bold red]Error generating content:[/bold red]\n{escape(str(e))}\n\n"
                    f"Falling back to predefined content for the redline demo.",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                    padding=(1, 2)
                ))
                
                # Use predefined content as fallback
                original_content = ORIGINAL_TEXT
                revised_content = MODIFIED_TEXT
    
    # Generate redline from the content
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True
    ) as progress:
        redline_task = progress.add_task("[cyan]Generating redline between versions...", total=1)
        
        try:
            # Generate redline
            result = await compare_documents_redline(
                original_text=original_content,
                modified_text=revised_content,
                file_format="text",
                detect_moves=True,
                output_format="html",
                diff_level="word"
            )
            
            progress.update(redline_task, completed=1)
            logger.success("Generated redline between versions", emoji_key="success")
            
            # Display results
            redline_html = result["redline"]
            stats = result["stats"]
            
            # Create stats display
            stats_table = Table(box=box.ROUNDED, show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            for key, value in stats.items():
                stats_table.add_row(key.replace('_', ' ').title(), str(value))
                
            stats_table.add_row("Processing Time", f"{result['processing_time']:.3f} seconds")
            console.print(stats_table)
            
            # Create preview of redline
            if len(redline_html) > 1000:
                preview = redline_html[:1000] + "...(truncated)..."
            else:
                preview = redline_html
                
            console.print(Panel(
                escape(preview),
                title="[bold green]Redline Preview (HTML)[/bold green]",
                subtitle="[dim](Truncated for display)[/dim]",
                border_style="green",
                padding=(1, 2)
            ))
            
            console.print(Panel(
                f"The redline shows [bold cyan]{stats['total_changes']}[/bold cyan] changes between versions:\n"
                f"- [blue]{stats['insertions']}[/blue] insertions\n"
                f"- [red]{stats['deletions']}[/red] deletions\n"
                f"- [green]{stats['moves']}[/green] moved blocks\n\n"
                f"This demonstrates how the redline tool can be combined with LLMs to generate and compare document versions.",
                title="[bold]LLM Redline Integration Summary[/bold]",
                border_style="blue",
                padding=(1, 2)
            ))
            
            # Display cost summary for the LLM operations
            tracker.display_summary(console)
            
        except Exception as e:
            progress.update(redline_task, description="[bold red]Error![/bold red]", completed=1)
            logger.error(f"Failed to generate redline: {str(e)}", emoji_key="error", exc_info=True)
            
            console.print(Panel(
                f"[bold red]Error generating redline:[/bold red]\n{escape(str(e))}",
                title="[bold red]Error[/bold red]",
                border_style="red",
                padding=(1, 2)
            ))


async def main():
    """Run the text redline tool demonstration."""
    console.print(Panel(
        Text("📝 Text Redline Tool Demonstration 📝", style="bold white on blue").center(),
        box=box.DOUBLE_EDGE,
        padding=(1, 0)
    ))
    
    logger.info("Starting Text Redline Tool Demonstration", emoji_key="start")
    
    # Initialize cost tracker
    tracker = CostTracker()
    
    try:
        # Basic demonstration
        await demonstrate_basic_redline()
        console.print()  # Add space between sections
        
        # Advanced features
        await demonstrate_advanced_redline_features()
        console.print()  # Add space between sections
        
        # Multi-format comparison
        await demonstrate_multi_format_redline()
        console.print()  # Add space between sections
        
        # LLM integration
        await demonstrate_llm_redline_integration(tracker)
        
        logger.success("Text Redline Tool Demo Completed Successfully!", emoji_key="complete")
        return 0
        
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)