#!/usr/bin/env python
"""Text redline comparison tool demonstration for LLM Gateway."""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
# Adjust the path depth as necessary for your project structure
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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

# Project imports (Ensure these paths are correct)
try:
    from llm_gateway.constants import Provider
    from llm_gateway.core.server import Gateway
    # Import our text redline tool
    from llm_gateway.tools.text_redline_tools import compare_documents_redline, create_html_redline
    from llm_gateway.tools.filesystem import write_file
    from llm_gateway.exceptions import ToolError
    from llm_gateway.utils import get_logger
    from llm_gateway.utils.display import CostTracker
    from llm_gateway.utils.logging.console import console
except ImportError as e:
     print(f"Error importing LLM Gateway components: {e}")
     print(f"Ensure the script is run from the correct directory or the project path is set correctly.")
     print(f"Project root added to sys.path: {project_root}")
     sys.exit(1)


# Initialize logger
logger = get_logger("example.text_redline")

# Define output directory
OUTPUT_DIR = Path(__file__).parent / "redline_outputs"

# --- Sample Data (Keep the existing sample data) ---
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
    <title>Company Policy Document - 2025 Update</title> <!-- Title updated -->
</head>
<body>
    <h1>Employee Handbook</h1>
    <p>Welcome to our company. This handbook outlines our policies and procedures.</p> <!-- Text updated -->

    <h2>Flexible Work Schedule</h2> <!-- Header Changed -->
    <p>With our new flexible work policy, employees can choose to work between 7:00 AM and 7:00 PM.</p> <!-- New Paragraph -->
    <p>A minimum of 6 hours must overlap with core hours (10:00 AM to 4:00 PM).</p> <!-- New Paragraph -->

    <h2>Code of Conduct</h2> <!-- This section MOVED -->
    <p>Employees are expected to maintain professional behavior at all times.</p>
    <p>Respect for colleagues is essential to our workplace culture.</p>

    <h2>Vacation Policy</h2> <!-- This section MOVED -->
    <p>Full-time employees receive 15 days of paid vacation annually.</p> <!-- Text updated -->
    <p>Vacation requests must be submitted at least one week in advance.</p> <!-- Text updated -->

    <table border="1"> <!-- Table content updated, row added -->
        <tr>
            <th>Benefit</th>
            <th>Eligibility</th>
        </tr>
        <tr>
            <td>Health Insurance</td>
            <td>Immediate</td> <!-- Text updated -->
        </tr>
        <tr>
            <td>401(k)</td>
            <td>After 60 days</td> <!-- Text updated -->
        </tr>
        <tr> <!-- New Row -->
            <td>Professional Development</td>
            <td>After 180 days</td>
        </tr>
    </table>
</body>
</html>"""

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
# --- End Sample Data ---


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
        task = progress.add_task("[cyan]Generating HTML redline...", total=1)

        try:
            # Generate the redline using create_html_redline directly
            result = await create_html_redline(
                original_html=ORIGINAL_HTML,
                modified_html=MODIFIED_HTML,
                detect_moves=True, # Explicitly enable move detection
                include_css=True,
                add_navigation=True,
                output_format="html"
            )

            # Mark task as complete
            progress.update(task, completed=1)

            # Log success
            logger.success(
                "HTML Redline generated successfully",
                emoji_key="success",
                stats=result["stats"]
            )

            # Display stats in a table
            stats_table = Table(title="[bold green]Redline Statistics[/bold green]", box=box.ROUNDED)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")

            # Display all stats returned by the formatter
            for key, value in result["stats"].items():
                stats_table.add_row(key.replace('_', ' ').title(), str(value))

            stats_table.add_row("Processing Time", f"{result['processing_time']:.3f} seconds")
            console.print(stats_table)

            # Display a preview of the redline HTML
            html_preview_full = result["redline_html"]

            # Find the body tag and extract a portion
            body_start = html_preview_full.find("<body")
            content = "Preview extraction failed."
            if body_start >= 0:
                body_end = html_preview_full.find("</body>", body_start)
                if body_end > body_start:
                     # Extract slightly more content for better preview
                    content = html_preview_full[body_start:min(body_end, body_start + 1500)]
                    if len(content) >= 1500:
                          content += "...(truncated)...</body>"
                    else:
                         content += "</body>" # Close the tag if not truncated there
                else:
                     content = html_preview_full[:1500] + "...(truncated)..." # Fallback if no body end
            else:
                 content = html_preview_full[:1500] + "...(truncated)..." # Fallback if no body start

            # Create a syntax object with HTML highlighting
            syntax = Syntax(content, "html", theme="default", line_numbers=True, word_wrap=True)

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

            # --- Save the full redline HTML ---
            output_filename = "basic_html_redline.html"
            output_path = OUTPUT_DIR / output_filename
            save_message = "[yellow]Save skipped[/yellow]"
            try:
                # Make sure the output path is absolute for write_file validation
                abs_output_path = str(output_path.resolve())
                await write_file(path=abs_output_path, content=html_preview_full)
                logger.info(f"Saved full redline HTML to: {abs_output_path}", emoji_key="save")
                save_message = f"Full redline saved to [cyan]{output_path}[/cyan]"
            except (ToolError, Exception) as save_err:
                logger.warning(f"Failed to save redline to {output_path}: {save_err}", emoji_key="warning")
                save_message = f"[yellow]Warning: Failed to save redline to {output_path} ({save_err})[/yellow]"
            # --- End Save ---

            console.print(Panel(
                f"The redline shows [bold cyan]{total_changes}[/bold cyan] changes between documents.\n"
                f"Original document size: [yellow]{original_size}[/yellow] characters\n"
                f"Modified document size: [yellow]{modified_size}[/yellow] characters\n"
                f"{save_message}\n"
                f"In a real application, this HTML would be displayed in a browser with full styling and navigation.",
                title="[bold]Summary[/bold]",
                border_style="blue",
                padding=(1, 2)
            ))

        except ToolError as te:
            progress.update(task, description="[bold red]Tool Error![/bold red]", completed=1)
            logger.error(f"Tool Error generating redline: {te}", emoji_key="error", exc_info=True)
            console.print(Panel(f"[bold red]Tool Error:[/bold red]\n{escape(str(te))}", title="[bold red]Error[/bold red]", border_style="red"))
        except Exception as e:
            # Update progress bar to show error
            progress.update(task, description="[bold red]Error![/bold red]", completed=1)
            logger.error(f"Failed to generate redline: {str(e)}", emoji_key="error", exc_info=True)
            console.print(Panel(f"[bold red]Error generating redline:[/bold red]\n{escape(str(e))}", title="[bold red]Error[/bold red]", border_style="red"))


async def demonstrate_advanced_redline_features():
    """Demonstrate advanced redline features like move detection."""
    console.print(Rule("[bold blue]Advanced Redline Features (Move Detection)[/bold blue]"))
    logger.info("Demonstrating impact of move detection", emoji_key="start")

    # Setup for the advanced demos
    configs = [
        {"name": "With Move Detection", "detect_moves": True, "filename": "adv_html_moves_enabled.html"},
        {"name": "Without Move Detection", "detect_moves": False, "filename": "adv_html_moves_disabled.html"},
    ]

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"), TaskProgressColumn(), TimeElapsedColumn(),
        console=console, expand=True
    ) as progress:
        tasks = {config["name"]: progress.add_task(f"[cyan]Processing {config['name']}...", total=1) for config in configs}
        results = {}
        save_statuses = {}

        for config in configs:
            config_name = config["name"]
            task_id = tasks[config_name]
            try:
                result = await create_html_redline(
                    original_html=ORIGINAL_HTML,
                    modified_html=MODIFIED_HTML,
                    detect_moves=config["detect_moves"],
                    ignore_whitespace=True, # Keep whitespace ignore consistent
                    output_format="html",
                    include_css=True, add_navigation=True
                )
                results[config_name] = result
                progress.update(task_id, completed=1)
                logger.info(f"Generated redline for {config_name}", emoji_key="success", stats=result["stats"])

                # --- Save Output ---
                output_path = OUTPUT_DIR / config["filename"]
                try:
                    abs_output_path = str(output_path.resolve())
                    await write_file(path=abs_output_path, content=result["redline_html"])
                    save_statuses[config_name] = f"Saved to [cyan]{output_path}[/cyan]"
                except Exception as save_err:
                    save_statuses[config_name] = f"[yellow]Failed to save to {output_path}[/yellow]"
                # --- End Save ---

            except Exception as e:
                progress.update(task_id, description=f"[bold red]Error processing {config_name}[/bold red]", completed=1)
                logger.error(f"Error with {config_name}: {str(e)}", emoji_key="error")
                results[config_name] = {"error": str(e)}
                save_statuses[config_name] = "[red]Error during generation[/red]"

    # Compare the results
    comparison_panels = []
    for name, result in results.items():
        if "error" in result:
            panel = Panel(f"[bold red]Error:[/bold red]\n{escape(result['error'])}", title=f"[bold red]{name}[/bold red]", border_style="red")
        else:
            stats = result["stats"]
            stats_table = Table(box=box.MINIMAL, show_header=False, padding=(0, 1))
            stats_table.add_column("Metric", style="dim cyan")
            stats_table.add_column("Value", style="white")
            for key, value in stats.items():
                style = "bold green" if key == "moves" and name == "With Move Detection" else "dim" if key == "moves" and name == "Without Move Detection" else "white"
                key_display = key.replace('_', ' ').title()
                value_display = f"[{style}]{value}[/{style}]" + (" (detection disabled)" if key == "moves" and name == "Without Move Detection" else "")
                stats_table.add_row(key_display, value_display)
            stats_table.add_row("Processing Time", f"{result['processing_time']:.3f} seconds")

            panel = Panel(
                Group(Text(f"Configuration: {name}", style="bold cyan"), stats_table),
                title=f"[bold]{name}[/bold]",
                border_style="green" if "With Move" in name else "yellow",
                subtitle=save_statuses.get(name, "")
            )
        comparison_panels.append(panel)

    if len(comparison_panels) == 2:
         console.print(Columns(comparison_panels))
         # Add explanation
         moves_enabled_stats = results.get("With Move Detection", {}).get("stats", {})
         moves_disabled_stats = results.get("Without Move Detection", {}).get("stats", {})
         move_count = moves_enabled_stats.get("moves", 0)
         ins_diff = moves_disabled_stats.get("insertions", 0) - moves_enabled_stats.get("insertions", 0)
         del_diff = moves_disabled_stats.get("deletions", 0) - moves_enabled_stats.get("deletions", 0)

         explanation = f"With move detection, [bold green]{move_count}[/bold green] moved blocks were identified.\n"
         if move_count > 0 and ins_diff >= move_count and del_diff >= move_count:
              explanation += f"Without it, these moves appear as [bold red]~{move_count}[/bold red] extra deletions and [bold blue]~{move_count}[/bold blue] extra insertions."
         elif move_count > 0:
              explanation += "Without it, these moves would appear as separate deletions and insertions."
         else:
               explanation += "No significant moves detected in this specific comparison."
         explanation += "\nMove detection provides a clearer picture of structural rearrangements."
         console.print(Panel(explanation, title="[bold]Impact of Move Detection[/bold]", border_style="blue"))

    elif comparison_panels: # Handle case where only one succeeded
         console.print(comparison_panels[0])


async def demonstrate_multi_format_redline():
    """Demonstrate redlining different document formats using compare_documents_redline."""
    console.print(Rule("[bold blue]Multi-Format Redline Comparison[/bold blue]"))
    logger.info("Demonstrating redline across different document formats", emoji_key="start")

    formats = [
        {"name": "Markdown Format", "original": ORIGINAL_MD, "modified": MODIFIED_MD, "format": "markdown", "filename": "multi_markdown_redline.html"},
        {"name": "Plain Text Format", "original": ORIGINAL_TEXT, "modified": MODIFIED_TEXT, "format": "text", "filename": "multi_text_redline.html"},
        # We already tested HTML in basic demo, focus on compare_documents_redline here
    ]

    format_table = Table(title="[bold cyan]Document Formats for Comparison[/bold cyan]", box=box.MINIMAL)
    format_table.add_column("Format", style="cyan"); format_table.add_column("Original Size", style="green"); format_table.add_column("Modified Size", style="blue")
    for fmt in formats:
        format_table.add_row(fmt["name"], f"{len(fmt['original'])} chars", f"{len(fmt['modified'])} chars")
    console.print(format_table)

    with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console, expand=True) as progress:
        tasks = {}
        results = {}
        save_statuses = {}

        for fmt in formats:
            task_id = progress.add_task(f"[cyan]Processing {fmt['name']}...", total=1)
            tasks[fmt["name"]] = task_id
            try:
                # Use compare_documents_redline, requesting HTML output
                result = await compare_documents_redline(
                    original_text=fmt["original"],
                    modified_text=fmt["modified"],
                    file_format=fmt["format"], # Explicitly set format
                    detect_moves=True,
                    output_format="html" # Request HTML output
                )
                results[fmt["name"]] = result
                progress.update(task_id, completed=1)
                logger.info(f"Generated redline for {fmt['name']}", emoji_key="success", stats=result.get("stats", {}))

                # --- Save Output ---
                redline_content = result.get("redline")
                output_path = OUTPUT_DIR / fmt["filename"]
                if redline_content:
                     try:
                         abs_output_path = str(output_path.resolve())
                         await write_file(path=abs_output_path, content=redline_content)
                         save_statuses[fmt["name"]] = f"Saved to [cyan]{output_path}[/cyan]"
                     except Exception as save_err:
                         save_statuses[fmt["name"]] = f"[yellow]Failed to save to {output_path}[/yellow]"
                else:
                     save_statuses[fmt["name"]] = "[yellow]No redline content generated[/yellow]"
                # --- End Save ---

            except Exception as e:
                progress.update(task_id, description=f"[bold red]Error processing {fmt['name']}[/bold red]", completed=1)
                logger.error(f"Error processing {fmt['name']}: {str(e)}", emoji_key="error", exc_info=True)
                results[fmt["name"]] = {"error": str(e)}
                save_statuses[fmt["name"]] = "[red]Error during generation[/red]"

    # Display comparison of results
    comparison_panels = []
    for fmt in formats:
        name = fmt["name"]
        result = results.get(name, {})
        color = "blue" if name == "Markdown Format" else "yellow"
        if "error" in result:
            panel = Panel(f"[bold red]Error:[/bold red]\n{escape(result['error'])}", title=f"[bold red]{name} - Error[/bold red]", border_style="red")
        else:
            stats_table = Table(box=box.MINIMAL, show_header=False, padding=(0, 1))
            stats_table.add_column("Metric", style="dim cyan")
            stats_table.add_column("Value", style="white")
            if "stats" in result:
                for key, value in result["stats"].items():
                    stats_table.add_row(key.replace('_', ' ').title(), str(value))
            stats_table.add_row("Processing Time", f"{result.get('processing_time', 0):.3f} seconds")

            redline_content = result.get("redline", "No preview available")
            preview = "Preview extraction failed."
            body_start = redline_content.find("<body")
            if body_start >= 0:
                preview = redline_content[body_start:min(len(redline_content), body_start + 500)] + "..."
            else:
                 preview = redline_content[:500] + "..."


            panel = Panel(
                Group(stats_table, Text("\nRedline Preview (HTML, truncated):", style="bold"), Panel(escape(preview), border_style="dim")),
                title=f"[bold {color}]{name}[/bold {color}]",
                border_style=color,
                subtitle=save_statuses.get(name, "")
            )
        comparison_panels.append(panel)

    if comparison_panels:
        console.print(Columns(comparison_panels))

    # Add summary note
    console.print(Panel(
        "The `compare_documents_redline` tool converts Markdown and Plain Text into HTML "
        "(wrapping text in `<pre>` tags if necessary) and then uses the HTML diff engine (`create_html_redline`). "
        "This allows leveraging the structural diff capabilities even for non-HTML inputs when HTML output is desired.",
        title="[bold]Multi-Format Handling Note[/bold]",
        border_style="blue"
    ))


async def demonstrate_llm_redline_integration(tracker: CostTracker):
    """Demonstrate integration with LLMs to generate content for redlining."""
    console.print(Rule("[bold blue]LLM Integration for Redline Generation[/bold blue]"))
    logger.info("Demonstrating LLM integration with redline tool", emoji_key="start")

    # --- Initialize Gateway and Find Provider (Same as before) ---
    gateway = Gateway("redline-demo", register_tools=False) # Avoid registering redline tool itself
    logger.info("Initializing providers...", emoji_key="provider")
    # await gateway._initialize_providers() # Make sure providers are initialized

    # Handle provider initialization (using a simplified check for demo purposes)
    # In a real app, ensure providers load correctly via Gateway setup
    available_providers = list(gateway.providers.keys()) # Get statically configured providers
    if not available_providers:
         logger.warning("No LLM providers configured in Gateway. Using predefined content for demo.", emoji_key="warning")
         console.print(Panel("[yellow]No LLM providers configured. Using predefined content.[/yellow]", title="[bold yellow]Warning[/bold yellow]"))
         original_content = ORIGINAL_TEXT
         revised_content = MODIFIED_TEXT
         provider_name = "N/A"
         model = "N/A"
    else:
        # Try to use the first configured provider
        provider_name = available_providers[0]
        provider = gateway.providers[provider_name]
        model = provider.get_default_model()
        logger.info(f"Using provider {provider_name} with model {model}", emoji_key="provider")


        # --- Define Prompts (Same as before) ---
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


        original_content = None
        revised_content = None

        with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console, expand=True) as progress:
            gen_task = progress.add_task("[cyan]Generating content with LLM...", total=2)
            try:
                logger.info(f"Generating original content with {provider_name}/{model}")
                result1 = await provider.generate_completion(prompt=base_prompt, model=model, temperature=0.7, max_tokens=500)
                tracker.add_call(result1)
                original_content = result1.text.strip()
                progress.advance(gen_task)
                logger.success("Generated original content")
                console.print(Panel(escape(original_content), title=f"[bold green]Original Content ({provider_name}/{model})[/bold green]", subtitle=f"[dim]Cost: ${result1.cost:.6f}[/dim]"))

                revision_prompt_filled = revision_prompt.format(original_content=original_content)
                logger.info(f"Generating revised content with {provider_name}/{model}")
                result2 = await provider.generate_completion(prompt=revision_prompt_filled, model=model, temperature=0.7, max_tokens=500)
                tracker.add_call(result2)
                revised_content = result2.text.strip()
                progress.advance(gen_task)
                logger.success("Generated revised content")
                console.print(Panel(escape(revised_content), title=f"[bold blue]Revised Content ({provider_name}/{model})[/bold blue]", subtitle=f"[dim]Cost: ${result2.cost:.6f}[/dim]"))

            except Exception as e:
                progress.update(gen_task, description="[bold red]LLM Error![/bold red]", completed=progress.tasks[gen_task].total)
                logger.error(f"Error generating content with LLM: {str(e)}", emoji_key="error", exc_info=True)
                console.print(Panel(f"[bold red]Error generating content:[/bold red]\n{escape(str(e))}\nFalling back to predefined content.", title="[bold red]LLM Error[/bold red]"))
                original_content = ORIGINAL_TEXT # Fallback
                revised_content = MODIFIED_TEXT # Fallback


    # --- Generate Redline (Using compare_documents_redline) ---
    if original_content and revised_content:
        with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console, expand=True) as progress:
            redline_task = progress.add_task("[cyan]Generating redline between versions...", total=1)
            try:
                result = await compare_documents_redline(
                    original_text=original_content,
                    modified_text=revised_content,
                    file_format="text", # Treat LLM output as text
                    detect_moves=True,
                    output_format="html" # Request HTML output
                )
                progress.update(redline_task, completed=1)
                logger.success("Generated redline between versions", emoji_key="success")

                # Display results
                redline_html = result["redline"]
                stats = result["stats"]

                stats_table = Table(box=box.ROUNDED, show_header=False)
                stats_table.add_column("Metric", style="cyan"); stats_table.add_column("Value", style="white")
                for key, value in stats.items(): stats_table.add_row(key.replace('_', ' ').title(), str(value))
                stats_table.add_row("Processing Time", f"{result['processing_time']:.3f} seconds")
                console.print(stats_table)

                preview = "Preview extraction failed."
                body_start = redline_html.find("<body")
                if body_start >= 0:
                    preview = redline_html[body_start:min(len(redline_html), body_start + 500)] + "..."
                else:
                    preview = redline_html[:500] + "..."

                console.print(Panel(escape(preview), title="[bold green]Redline Preview (HTML)[/bold green]", subtitle="[dim](Truncated)[/dim]"))

                # --- Save Output ---
                output_filename = "llm_generated_redline.html"
                output_path = OUTPUT_DIR / output_filename
                save_message = ""
                try:
                    abs_output_path = str(output_path.resolve())
                    await write_file(path=abs_output_path, content=redline_html)
                    save_message = f"Full redline saved to [cyan]{output_path}[/cyan]"
                except Exception as save_err:
                    save_message = f"[yellow]Warning: Failed to save redline to {output_path} ({save_err})[/yellow]"
                # --- End Save ---

                console.print(Panel(
                    f"The redline shows [bold cyan]{stats.get('total_changes', 0)}[/bold cyan] changes between LLM-generated versions.\n"
                    f"{save_message}\n"
                    f"This demonstrates comparing text content generated dynamically.",
                    title="[bold]LLM Redline Integration Summary[/bold]"
                ))
                tracker.display_summary(console)

            except Exception as e:
                progress.update(redline_task, description="[bold red]Error![/bold red]", completed=1)
                logger.error(f"Failed to generate redline for LLM content: {str(e)}", emoji_key="error", exc_info=True)
                console.print(Panel(f"[bold red]Error generating redline:[/bold red]\n{escape(str(e))}", title="[bold red]Error[/bold red]"))
    else:
        console.print("[yellow]Skipping redline generation as LLM content generation failed.[/yellow]")


async def main():
    """Run the text redline tool demonstration."""
    console.print(Panel(
        Text("📝 Text Redline Tool Demonstration 📝", style="bold white on blue", justify="center"),
        box=box.DOUBLE_EDGE, padding=(1, 0)
    ))

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {OUTPUT_DIR}")
    except OSError as e:
        logger.error(f"Could not create output directory {OUTPUT_DIR}: {e}", emoji_key="error")
        console.print(f"[bold red]Error:[/bold red] Could not create output directory: {OUTPUT_DIR}. Please check permissions.")
        # Optionally exit if saving is critical
        # return 1

    logger.info("Starting Text Redline Tool Demonstration", emoji_key="start")
    tracker = CostTracker()

    try:
        await demonstrate_basic_redline()
        console.print()
        await demonstrate_advanced_redline_features()
        console.print()
        await demonstrate_multi_format_redline()
        console.print()
        # Only run LLM demo if Gateway likely initialized correctly
        if 'Gateway' in globals():
             await demonstrate_llm_redline_integration(tracker)
        else:
             console.print("[yellow]Skipping LLM integration demo due to import errors.[/yellow]")

        logger.success("Text Redline Tool Demo Completed Successfully!", emoji_key="complete")
        return 0

    except Exception as e:
        logger.critical(f"Demo failed unexpectedly: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(Panel(f"[bold red]Critical Demo Failure:[/bold red]\n{escape(str(e))}", title="[bold red]Demo Failed[/bold red]", border_style="red"))
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
