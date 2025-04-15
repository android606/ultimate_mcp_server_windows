#!/usr/bin/env python
"""Entity relationship graph extraction and visualization demo using LLM Gateway."""
import asyncio
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

# Project imports
from llm_gateway.constants import Provider
from llm_gateway.core.server import Gateway
from llm_gateway.tools.entity_relation_graph import (
    COMMON_SCHEMAS,
    GraphStrategy,
    OutputFormat,
    VisualizationFormat,
    extract_entity_graph,
)
from llm_gateway.utils import get_logger
from llm_gateway.utils.logging.console import console

# Initialize logger
logger = get_logger("example.entity_relation_graph")

# Setup
SAMPLE_DIR = Path(__file__).parent / "sample"
OUTPUT_DIR = Path(__file__).parent / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TextDomain(Enum):
    """Domain types for demonstration examples."""
    BUSINESS = "business"
    ACADEMIC = "academic"
    LEGAL = "legal"
    MEDICAL = "medical"

# Console instances
main_console = console
detail_console = Console(width=100, highlight=True)

def display_header(title: str) -> None:
    """Display a section header."""
    main_console.print()
    main_console.print(Rule(f"[bold blue]{title}[/bold blue]"))
    main_console.print()

def display_dataset_info(dataset_path: Path, title: str) -> None:
    """Display information about a dataset."""
    if not dataset_path.exists():
        main_console.print(f"[red]Dataset file {dataset_path} not found![/red]")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Count entities/characters for display
    char_count = len(content)
    word_count = len(content.split())
    sentence_count = len([s for s in content.split(".") if s.strip()])
    
    # Preview of the content (first 200 chars)
    preview = content[:200] + "..." if len(content) > 200 else content
    
    main_console.print(Panel(
        f"[bold cyan]Dataset:[/bold cyan] {dataset_path.name}\n"
        f"[bold cyan]Size:[/bold cyan] {char_count} characters, {word_count} words, ~{sentence_count} sentences\n\n"
        f"[bold cyan]Preview:[/bold cyan]\n{escape(preview)}",
        title=title,
        border_style="cyan",
        expand=False
    ))

def display_extraction_params(params: Dict[str, Any]) -> None:
    """Display extraction parameters."""
    param_table = Table(title="Extraction Parameters", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    param_table.add_column("Parameter", style="cyan")
    param_table.add_column("Value", style="green")
    
    # Add key parameters to table
    for key, value in params.items():
        # Format enums and lists nicely
        if isinstance(value, Enum):
            value_str = value.value
        elif isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        elif isinstance(value, bool):
            value_str = "[green]Yes[/green]" if value else "[red]No[/red]"
        elif value is None:
            value_str = "[dim italic]None[/dim italic]"
        else:
            value_str = str(value)
            
        param_table.add_row(key, value_str)
    
    main_console.print(param_table)

def display_entity_stats(result: Dict[str, Any]) -> None:
    """Display statistics about extracted entities."""
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])
    
    if not entities:
        main_console.print("[yellow]No entities found in extraction result.[/yellow]")
        return
    
    # Count entity types
    entity_types = {}
    for entity in entities:
        ent_type = entity.get("type", "Unknown")
        entity_types[ent_type] = entity_types.get(ent_type, 0) + 1
    
    # Count relationship types
    rel_types = {}
    for rel in relationships:
        rel_type = rel.get("type", "Unknown")
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    
    # Create entity stats table
    stats_table = Table(title="Extraction Statistics", box=box.ROUNDED, show_header=True)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Count", style="green", justify="right")
    
    stats_table.add_row("Total Entities", str(len(entities)))
    stats_table.add_row("Total Relationships", str(len(relationships)))
    
    # Add entity type counts
    for ent_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        stats_table.add_row(f"Entity Type: {ent_type}", str(count))
    
    # Add relationship type counts (top 5)
    for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:5]:
        stats_table.add_row(f"Relationship Type: {rel_type}", str(count))
    
    main_console.print(stats_table)

def display_graph_metrics(result: Dict[str, Any]) -> None:
    """Display graph metrics if available."""
    metrics = result.get("metrics", {})
    if not metrics:
        return
        
    metrics_table = Table(title="Graph Metrics", box=box.ROUNDED, show_header=True)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green", justify="right")
    
    # Add metrics to table
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            metrics_table.add_row(key.replace("_", " ").title(), formatted_value)
    
    main_console.print(metrics_table)

def display_entities_table(result: Dict[str, Any], limit: int = 10) -> None:
    """Display extracted entities in a table."""
    entities = result.get("entities", [])
    if not entities:
        main_console.print("[yellow]No entities found to display.[/yellow]")
        return
        
    # Sort entities by centrality if available, otherwise by mentions count
    if "centrality" in entities[0]:
        entities = sorted(entities, key=lambda x: x.get("centrality", 0), reverse=True)
    elif "mentions" in entities[0]:
        entities = sorted(entities, key=lambda x: len(x.get("mentions", [])), reverse=True)
    
    # Limit to top entities
    display_entities = entities[:limit]
    
    entity_table = Table(title=f"Top {limit} Entities", box=box.ROUNDED, show_header=True)
    entity_table.add_column("ID", style="dim")
    entity_table.add_column("Name", style="cyan")
    entity_table.add_column("Type", style="green")
    
    # Add columns for additional information if available
    has_centrality = any("centrality" in entity for entity in display_entities)
    has_mentions = any("mentions" in entity for entity in display_entities)
    has_attributes = any("attributes" in entity for entity in display_entities)
    
    if has_centrality:
        entity_table.add_column("Centrality", style="magenta", justify="right")
    if has_mentions:
        entity_table.add_column("Mentions", style="yellow", justify="right")
    if has_attributes:
        entity_table.add_column("Attributes", style="blue")
    
    # Add rows for each entity
    for entity in display_entities:
        row = [
            entity.get("id", ""),
            entity.get("name", ""),
            entity.get("type", "Unknown")
        ]
        
        if has_centrality:
            centrality = entity.get("centrality", 0)
            row.append(f"{centrality:.4f}" if isinstance(centrality, float) else "N/A")
            
        if has_mentions:
            mentions = entity.get("mentions", [])
            row.append(str(len(mentions)))
            
        if has_attributes:
            attributes = entity.get("attributes", {})
            attr_str = ", ".join(f"{k}: {v}" for k, v in attributes.items())
            row.append(attr_str[:50] + ("..." if len(attr_str) > 50 else ""))
        
        entity_table.add_row(*row)
    
    main_console.print(entity_table)
    
    if len(entities) > limit:
        main_console.print(f"[dim italic]...and {len(entities) - limit} more entities[/dim italic]")

def display_relationships_table(result: Dict[str, Any], limit: int = 10) -> None:
    """Display extracted relationships in a table."""
    relationships = result.get("relationships", [])
    entities = {entity["id"]: entity for entity in result.get("entities", [])}
    
    if not relationships:
        main_console.print("[yellow]No relationships found to display.[/yellow]")
        return
        
    # Sort relationships by confidence
    relationships = sorted(relationships, key=lambda x: x.get("confidence", 0), reverse=True)
    
    # Limit to top relationships
    display_relationships = relationships[:limit]
    
    rel_table = Table(title=f"Top {limit} Relationships", box=box.ROUNDED, show_header=True)
    rel_table.add_column("Type", style="cyan")
    rel_table.add_column("Source", style="green")
    rel_table.add_column("Target", style="green")
    rel_table.add_column("Confidence", style="magenta", justify="right")
    
    # Check if we have evidence
    has_evidence = any("evidence" in rel for rel in display_relationships)
    if has_evidence:
        rel_table.add_column("Evidence", style="yellow")
    
    # Add rows for each relationship
    for rel in display_relationships:
        source_id = rel.get("source", "")
        target_id = rel.get("target", "")
        
        # Get entity names if available
        source_name = entities.get(source_id, {}).get("name", source_id)
        target_name = entities.get(target_id, {}).get("name", target_id)
        
        row = [
            rel.get("type", "Unknown"),
            source_name,
            target_name,
            f"{rel.get('confidence', 0):.2f}" if "confidence" in rel else "N/A",
        ]
        
        if has_evidence:
            evidence = rel.get("evidence", "")
            row.append(evidence[:50] + ("..." if len(evidence) > 50 else ""))
        
        rel_table.add_row(*row)
    
    main_console.print(rel_table)
    
    if len(relationships) > limit:
        main_console.print(f"[dim italic]...and {len(relationships) - limit} more relationships[/dim italic]")

def display_entity_graph_tree(result: Dict[str, Any], max_depth: int = 2, max_children: int = 3) -> None:
    """Display a tree representation of the entity graph."""
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])
    
    if not entities or not relationships:
        main_console.print("[yellow]Cannot display graph tree: insufficient data.[/yellow]")
        return
        
    # Sort entities by centrality if available
    if "centrality" in entities[0]:
        entities = sorted(entities, key=lambda x: x.get("centrality", 0), reverse=True)
    
    # Get most central entity as root
    root_entity = entities[0]
    root_id = root_entity["id"]
    
    # Create rich Tree
    tree = Tree(f"[bold cyan]{root_entity.get('name', root_id)}[/bold cyan] ([italic]{root_entity.get('type', 'Unknown')}[/italic])")
    
    # Track visited entities to prevent cycles
    visited = {root_id}
    
    # Recursively build tree
    def add_children(parent_tree, parent_id, depth, path):
        if depth >= max_depth:
            return
            
        # Find relationships where parent is source
        outgoing = [r for r in relationships if r.get("source") == parent_id and r.get("target") not in path]
        
        # Sort by confidence if available
        if outgoing and "confidence" in outgoing[0]:
            outgoing = sorted(outgoing, key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Limit number of children
        for rel in outgoing[:max_children]:
            target_id = rel.get("target")
            target_entity = next((e for e in entities if e.get("id") == target_id), None)
            
            if not target_entity:
                continue
                
            visited.add(target_id)
            target_name = target_entity.get("name", target_id)
            rel_type = rel.get("type", "related to")
            
            # Add branch with relationship type and entity
            branch_text = f"[green]{rel_type}[/green] → [cyan]{target_name}[/cyan] ([italic]{target_entity.get('type', 'Unknown')}[/italic])"
            branch = parent_tree.add(branch_text)
            
            # Recursively add children
            new_path = path + [target_id]
            add_children(branch, target_id, depth + 1, new_path)
    
    # Start building the tree
    add_children(tree, root_id, 0, [root_id])
    
    main_console.print(Panel(tree, title="Entity Graph Tree View", border_style="blue"))

def display_extraction_summary(result: Dict[str, Any]) -> None:
    """Display a summary of the extraction results."""
    model = result.get("model", "Unknown")
    provider = result.get("provider", "Unknown")
    tokens = result.get("tokens", {})
    cost = result.get("cost", 0)
    time = result.get("processing_time", 0)
    
    summary_table = Table(box=box.ROUNDED, show_header=False, title="Extraction Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Provider", provider)
    summary_table.add_row("Model", model)
    summary_table.add_row("Input Tokens", str(tokens.get("input", "N/A")))
    summary_table.add_row("Output Tokens", str(tokens.get("output", "N/A")))
    summary_table.add_row("Total Tokens", str(tokens.get("total", "N/A")))
    summary_table.add_row("Cost", f"${cost:.6f}" if isinstance(cost, (int, float)) else "N/A")
    summary_table.add_row("Processing Time", f"{time:.2f} seconds" if isinstance(time, (int, float)) else "N/A")
    
    main_console.print(summary_table)

def save_visualization(result: Dict[str, Any], domain: str, strategy: str, output_dir: Path) -> Optional[str]:
    """Save visualization to file and return path."""
    visualization = result.get("visualization", {})
    if not visualization:
        return None
        
    # Get visualization content based on format
    html_content = visualization.get("html")
    svg_content = visualization.get("svg")
    dot_content = visualization.get("dot")
    
    # Save to appropriate file
    timestamp = int(time.time())
    if html_content:
        output_path = output_dir / f"graph_{domain}_{strategy}_{timestamp}.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return str(output_path)
    elif svg_content:
        output_path = output_dir / f"graph_{domain}_{strategy}_{timestamp}.svg"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return str(output_path)
    elif dot_content:
        output_path = output_dir / f"graph_{domain}_{strategy}_{timestamp}.dot"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(dot_content)
        return str(output_path)
        
    return None

async def run_entity_extraction(
    text: str,
    domain: TextDomain,
    strategy: GraphStrategy,
    model: str,
    output_format: OutputFormat = OutputFormat.JSON,
    visualization_format: VisualizationFormat = VisualizationFormat.HTML,
    provider: str = Provider.ANTHROPIC.value
) -> Dict[str, Any]:
    """Run entity graph extraction with progress indicator."""
    # Setup extraction parameters based on domain
    params = {
        "text": text,
        "provider": provider,
        "model": model,
        "strategy": strategy.value,
        "output_format": output_format.value,
        "visualization_format": visualization_format.value,
        "include_evidence": True,
        "include_attributes": True,
        "include_positions": True,
        "include_temporal_info": True,
        "normalize_entities": True,
        "max_entities": 50,
        "max_relations": 100,
        "min_confidence": 0.5,
        "enable_reasoning": True
    }
    
    # Add domain-specific schema if available
    if domain is not None and domain.value in COMMON_SCHEMAS:
        params["domain"] = domain.value
    
    # Display parameters
    display_extraction_params(params)
    
    # Run extraction with progress spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=main_console,
        transient=True
    ) as progress:
        task = progress.add_task(f"Extracting entity graph using {strategy.value} strategy...", total=None)
        
        try:
            result = await extract_entity_graph(**params)
            progress.update(task, completed=True, description="Entity graph extraction complete!")
            return result
        except Exception as e:
            progress.update(task, completed=True, description=f"Entity graph extraction failed: {str(e)}")
            raise

async def demonstrate_business_extraction():
    """Demonstrate entity graph extraction for business text."""
    display_header("Business Domain Entity Graph Extraction")
    
    # Load business article
    business_path = SAMPLE_DIR / "article.txt"
    display_dataset_info(business_path, "Business News Article")
    
    with open(business_path, "r", encoding="utf-8") as f:
        business_text = f.read()
    
    # Extract entity graph using standard strategy
    main_console.print("[bold]Running Standard Extraction Strategy[/bold]")
    
    try:
        result = await run_entity_extraction(
            text=business_text,
            domain=TextDomain.BUSINESS,
            strategy=GraphStrategy.STANDARD,
            model="claude-3-5-haiku-20241022",
            visualization_format=VisualizationFormat.HTML
        )
        
        # Display results
        display_entity_stats(result)
        display_graph_metrics(result)
        display_entities_table(result)
        display_relationships_table(result)
        display_entity_graph_tree(result)
        
        # Save visualization if available
        vis_path = save_visualization(result, "business", "standard", OUTPUT_DIR)
        if vis_path:
            main_console.print(f"\n[green]✓[/green] Visualization saved to: [blue]{vis_path}[/blue]")
        
        # Display summary
        display_extraction_summary(result)
        
    except Exception as e:
        main_console.print(f"[bold red]Error:[/bold red] {str(e)}")

async def demonstrate_academic_extraction():
    """Demonstrate entity graph extraction for academic text."""
    display_header("Academic Domain Entity Graph Extraction")
    
    # Load academic paper
    academic_path = SAMPLE_DIR / "research_paper.txt"
    display_dataset_info(academic_path, "Academic Research Paper")
    
    with open(academic_path, "r", encoding="utf-8") as f:
        academic_text = f.read()
    
    # Extract entity graph using multistage strategy
    main_console.print("[bold]Running Multistage Extraction Strategy[/bold]")
    
    try:
        result = await run_entity_extraction(
            text=academic_text,
            domain=TextDomain.ACADEMIC,
            strategy=GraphStrategy.MULTISTAGE,
            model="claude-3-5-haiku-20241022",
            visualization_format=VisualizationFormat.HTML
        )
        
        # Display results
        display_entity_stats(result)
        display_graph_metrics(result)
        display_entities_table(result)
        display_relationships_table(result)
        display_entity_graph_tree(result)
        
        # Save visualization if available
        vis_path = save_visualization(result, "academic", "multistage", OUTPUT_DIR)
        if vis_path:
            main_console.print(f"\n[green]✓[/green] Visualization saved to: [blue]{vis_path}[/blue]")
        
        # Display summary
        display_extraction_summary(result)
        
    except Exception as e:
        main_console.print(f"[bold red]Error:[/bold red] {str(e)}")

async def demonstrate_legal_extraction():
    """Demonstrate entity graph extraction for legal text."""
    display_header("Legal Domain Entity Graph Extraction")
    
    # Load legal contract
    legal_path = SAMPLE_DIR / "legal_contract.txt"
    display_dataset_info(legal_path, "Legal Contract")
    
    with open(legal_path, "r", encoding="utf-8") as f:
        legal_text = f.read()
    
    # Extract entity graph using structured strategy
    main_console.print("[bold]Running Structured Extraction Strategy[/bold]")
    
    try:
        result = await run_entity_extraction(
            text=legal_text,
            domain=TextDomain.LEGAL,
            strategy=GraphStrategy.STRUCTURED,
            model="claude-3-5-haiku-20241022",
            visualization_format=VisualizationFormat.HTML
        )
        
        # Display results
        display_entity_stats(result)
        display_graph_metrics(result)
        display_entities_table(result)
        display_relationships_table(result)
        display_entity_graph_tree(result)
        
        # Save visualization if available
        vis_path = save_visualization(result, "legal", "structured", OUTPUT_DIR)
        if vis_path:
            main_console.print(f"\n[green]✓[/green] Visualization saved to: [blue]{vis_path}[/blue]")
        
        # Display summary
        display_extraction_summary(result)
        
    except Exception as e:
        main_console.print(f"[bold red]Error:[/bold red] {str(e)}")

async def demonstrate_medical_extraction():
    """Demonstrate entity graph extraction for medical text."""
    display_header("Medical Domain Entity Graph Extraction")
    
    # Load medical case
    medical_path = SAMPLE_DIR / "medical_case.txt"
    display_dataset_info(medical_path, "Medical Case Report")
    
    with open(medical_path, "r", encoding="utf-8") as f:
        medical_text = f.read()
    
    # Extract entity graph using strict schema strategy
    main_console.print("[bold]Running Strict Schema Extraction Strategy[/bold]")
    
    try:
        result = await run_entity_extraction(
            text=medical_text,
            domain=TextDomain.MEDICAL,
            strategy=GraphStrategy.STRICT_SCHEMA,
            model="claude-3-5-haiku-20241022",
            visualization_format=VisualizationFormat.HTML
        )
        
        # Display results
        display_entity_stats(result)
        display_graph_metrics(result)
        display_entities_table(result)
        display_relationships_table(result)
        display_entity_graph_tree(result)
        
        # Save visualization if available
        vis_path = save_visualization(result, "medical", "strict_schema", OUTPUT_DIR)
        if vis_path:
            main_console.print(f"\n[green]✓[/green] Visualization saved to: [blue]{vis_path}[/blue]")
        
        # Display summary
        display_extraction_summary(result)
        
    except Exception as e:
        main_console.print(f"[bold red]Error:[/bold red] {str(e)}")

async def demonstrate_strategy_comparison():
    """Compare different extraction strategies on the same text."""
    display_header("Strategy Comparison")
    
    # Load business article for comparison
    business_path = SAMPLE_DIR / "article.txt"
    display_dataset_info(business_path, "Business News Article (For Strategy Comparison)")
    
    with open(business_path, "r", encoding="utf-8") as f:
        business_text = f.read()
    
    # Define strategies to compare
    strategies = [
        (GraphStrategy.STANDARD, "Standard (Default)"),
        (GraphStrategy.MULTISTAGE, "Multistage (Entities First)"),
        (GraphStrategy.CHUNKED, "Chunked (For Large Texts)"),
        (GraphStrategy.STRUCTURED, "Structured (Example-based)")
    ]
    
    # Setup comparison table
    comparison_table = Table(title="Strategy Comparison Results", box=box.ROUNDED, show_header=True)
    comparison_table.add_column("Strategy", style="cyan")
    comparison_table.add_column("Entities", style="green", justify="right")
    comparison_table.add_column("Relationships", style="green", justify="right")
    comparison_table.add_column("Time (s)", style="yellow", justify="right")
    comparison_table.add_column("Tokens", style="magenta", justify="right")
    comparison_table.add_column("Cost ($)", style="blue", justify="right")
    
    # Compare each strategy
    for strategy, desc in strategies:
        main_console.print(f"\n[bold]Running {desc} Strategy[/bold]")
        
        try:
            # Only run on a portion of the text to save time in the comparison
            text_excerpt = business_text[:2000]  # First 2000 chars
            
            result = await run_entity_extraction(
                text=text_excerpt,
                domain=TextDomain.BUSINESS,
                strategy=strategy,
                model="claude-3-5-haiku-20241022",
                visualization_format=VisualizationFormat.NONE  # Skip visualization for comparison
            )
            
            # Extract metrics for comparison
            entity_count = len(result.get("entities", []))
            rel_count = len(result.get("relationships", []))
            processing_time = result.get("processing_time", 0)
            token_count = result.get("tokens", {}).get("total", 0)
            cost = result.get("cost", 0)
            
            # Add to comparison table
            comparison_table.add_row(
                desc,
                str(entity_count),
                str(rel_count),
                f"{processing_time:.2f}",
                str(token_count),
                f"{cost:.6f}"
            )
            
            # Display stats for this strategy
            display_entity_stats(result)
            
        except Exception as e:
            main_console.print(f"[bold red]Error with {desc} strategy:[/bold red] {str(e)}")
            # Add error row to comparison table
            comparison_table.add_row(desc, "ERROR", "ERROR", "N/A", "N/A", "N/A")
    
    # Display final comparison table
    main_console.print(comparison_table)

async def demonstrate_output_formats():
    """Demonstrate different output formats."""
    display_header("Output Format Comparison")
    
    # Load academic paper for output format demo
    academic_path = SAMPLE_DIR / "research_paper.txt"
    display_dataset_info(academic_path, "Academic Paper (For Output Formats)")
    
    with open(academic_path, "r", encoding="utf-8") as f:
        academic_text = f.read()
    
    # Define output formats to demonstrate
    formats = [
        (OutputFormat.JSON, "Standard JSON"),
        (OutputFormat.NETWORKX, "NetworkX Graph"),
        (OutputFormat.CYTOSCAPE, "Cytoscape.js Format"),
        (OutputFormat.NEO4J, "Neo4j Cypher Queries")
    ]
    
    main_console.print("[bold yellow]Note:[/bold yellow] This demonstrates how the extracted data can be formatted for different applications.")
    
    # Extract with each output format
    for fmt, desc in formats:
        main_console.print(f"\n[bold]Demonstrating {desc} Output Format[/bold]")
        
        try:
            # Only run on a portion of the text to save time
            text_excerpt = academic_text[:1500]  # First 1500 chars
            
            result = await run_entity_extraction(
                text=text_excerpt,
                domain=TextDomain.ACADEMIC,
                strategy=GraphStrategy.STANDARD,  # Use standard strategy for all
                model="claude-3-5-haiku-20241022",
                output_format=fmt,
                visualization_format=VisualizationFormat.NONE  # Skip visualization for this demo
            )
            
            # Display format-specific output
            if fmt == OutputFormat.JSON:
                # Display a subset of the JSON
                json_subset = {
                    "entities": result.get("entities", [])[:3],
                    "relationships": result.get("relationships", [])[:3]
                }
                main_console.print(Panel(
                    Syntax(json.dumps(json_subset, indent=2), "json", theme="monokai", line_numbers=True),
                    title=f"Sample of {desc} Output",
                    border_style="green"
                ))
                
            elif fmt == OutputFormat.NETWORKX:
                # Just show the fact that we have a NetworkX graph object
                has_graph = "graph" in result
                main_console.print(Panel(
                    f"[green]✓[/green] NetworkX graph object created: {has_graph}\n"
                    f"With {len(result.get('entities', []))} nodes and {len(result.get('relationships', []))} edges.\n\n"
                    "This format enables graph algorithms like:\n"
                    "- Centrality analysis\n"
                    "- Path finding\n"
                    "- Community detection\n"
                    "- Network visualization",
                    title=f"{desc} Output",
                    border_style="green"
                ))
                
            elif fmt == OutputFormat.CYTOSCAPE:
                # Show sample of Cytoscape.js format
                if "cytoscape" in result:
                    cyto_data = result["cytoscape"]
                    sample = {
                        "nodes": cyto_data.get("nodes", [])[:2],
                        "edges": cyto_data.get("edges", [])[:2]
                    }
                    main_console.print(Panel(
                        Syntax(json.dumps(sample, indent=2), "json", theme="monokai", line_numbers=True),
                        title=f"Sample of {desc} Output",
                        border_style="green"
                    ))
                else:
                    main_console.print("[yellow]Cytoscape.js format not available in result.[/yellow]")
                    
            elif fmt == OutputFormat.NEO4J:
                # Show sample of Neo4j Cypher queries
                if "neo4j_queries" in result:
                    queries = result["neo4j_queries"]
                    sample_queries = queries[:3] if len(queries) > 3 else queries
                    main_console.print(Panel(
                        Syntax("\n\n".join(sample_queries), "cypher", theme="monokai", line_numbers=True),
                        title=f"Sample of {desc} Output (Cypher Queries)",
                        border_style="green"
                    ))
                else:
                    main_console.print("[yellow]Neo4j queries not available in result.[/yellow]")
            
        except Exception as e:
            main_console.print(f"[bold red]Error with {desc} format:[/bold red] {str(e)}")

async def main():
    """Run entity relation graph extraction demonstrations."""
    try:
        # Display welcome message
        main_console.print(Rule("[bold magenta]Entity Relationship Graph Extraction Demo[/bold magenta]"))
        main_console.print(
            "[bold]This demonstrates the entity_relation_graph tool for extracting and visualizing "
            "knowledge graphs from text across different domains.[/bold]\n"
        )
        
        # Initialize the Gateway (needed for setup, not directly used in this demo)
        Gateway("entity-graph-demo", register_tools=True)
        
        # Check if sample directory exists
        if not SAMPLE_DIR.exists():
            main_console.print(f"[bold red]Error:[/bold red] Sample directory {SAMPLE_DIR} not found!")
            return 1
            
        # Run demonstrations
        await demonstrate_business_extraction()
        await demonstrate_academic_extraction()
        await demonstrate_legal_extraction()
        await demonstrate_medical_extraction()
        await demonstrate_strategy_comparison()
        await demonstrate_output_formats()
        
        main_console.print(Rule("[bold green]Entity Relationship Graph Extraction Demo Complete[/bold green]"))
        main_console.print(
            f"\n[bold]Visualizations have been saved to: [blue]{OUTPUT_DIR}[/blue][/bold]\n"
            "Open the HTML files in a web browser to explore the interactive entity graphs."
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", emoji_key="error", exc_info=True)
        main_console.print(f"[bold red]Demo failed with error:[/bold red] {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 