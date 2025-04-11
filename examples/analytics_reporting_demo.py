#!/usr/bin/env python
"""Analytics and reporting demonstration for LLM Gateway."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from decouple import config as decouple_config
from rich import box
from rich.live import Live
from rich.markup import escape
from rich.rule import Rule
from rich.table import Table

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.services.analytics.metrics import get_metrics_tracker
from llm_gateway.services.analytics.reporting import AnalyticsReporting
from llm_gateway.utils import get_logger
from llm_gateway.utils.display import display_analytics_metrics

# --- Add Rich Imports ---
from llm_gateway.utils.logging.console import console

# ----------------------

# Initialize logger
logger = get_logger("example.analytics_reporting")


async def simulate_llm_usage():
    """Simulate various LLM API calls to generate analytics data."""
    console.print(Rule("[bold blue]Simulating LLM Usage[/bold blue]"))
    logger.info("Simulating LLM usage to generate analytics data", emoji_key="start")
    
    metrics = get_metrics_tracker()
    providers_info = []
    
    # Setup providers
    provider_configs = {
        Provider.OPENAI: decouple_config("OPENAI_API_KEY", default=None),
        Provider.ANTHROPIC: decouple_config("ANTHROPIC_API_KEY", default=None),
        Provider.GEMINI: decouple_config("GEMINI_API_KEY", default=None),
        Provider.OPENROUTER: decouple_config("OPENROUTER_API_KEY", default=None),
    }

    for provider_enum, api_key in provider_configs.items():
        if api_key:
            try:
                provider = get_provider(provider_enum.value, api_key=api_key)
                await provider.initialize()
                providers_info.append((provider_enum.value, provider))
                logger.info(f"Initialized provider: {provider_enum.value}", emoji_key="provider")
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_enum.value}: {e}", emoji_key="warning")
        else:
             logger.info(f"API key not found for {provider_enum.value}, skipping.", emoji_key="skip")

    if not providers_info:
        logger.error("No providers could be initialized. Cannot simulate usage.", emoji_key="error")
        console.print("[bold red]Error:[/bold red] No LLM providers could be initialized. Please check your API keys.")
        return metrics # Return empty metrics

    console.print(f"Simulating usage with [cyan]{len(providers_info)}[/cyan] providers.")

    prompts = [
        "What is machine learning?",
        "Explain the concept of neural networks in simple terms.",
        "Write a short story about a robot that learns to love.",
        "Summarize the key innovations in artificial intelligence over the past decade.",
        "What are the ethical considerations in developing advanced AI systems?"
    ]
    
    total_calls = len(providers_info) * len(prompts)
    call_count = 0
    
    for provider_name, provider in providers_info:
        # Use default model unless specific logic requires otherwise
        model_to_use = provider.get_default_model()
        if not model_to_use:
            logger.warning(f"No default model found for {provider_name}, skipping provider.", emoji_key="warning")
            continue # Skip this provider if no default model

        for prompt in prompts:
            call_count += 1
            logger.info(
                f"Simulating call ({call_count}/{total_calls}) for {provider_name}",
                emoji_key="processing"
            )
            
            try:
                start_time = time.time()
                result = await provider.generate_completion(
                    prompt=prompt,
                    model=model_to_use, # Use determined model
                    temperature=0.7,
                    max_tokens=150
                )
                completion_time = time.time() - start_time
                
                # Record metrics using the actual model returned in the result
                metrics.record_request(
                    provider=provider_name,
                    model=result.model, # Use model from result
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost=result.cost,
                    duration=completion_time,
                    success=True
                )
                
                # Log less verbosely during simulation
                # logger.success("Completion generated", emoji_key="success", provider=provider_name, model=result.model)
                
                await asyncio.sleep(0.2) # Shorter delay
            
            except Exception as e:
                logger.error(f"Error simulating completion for {provider_name}: {str(e)}", emoji_key="error")
                metrics.record_request(
                    provider=provider_name,
                    model=model_to_use, # Log error against intended model
                    input_tokens=0, # Assume 0 tokens on error for simplicity
                    output_tokens=0,
                    cost=0.0,
                    duration=time.time() - start_time, # Log duration even on error
                    success=False # Mark as failed
                )
    
    logger.info("Finished simulating LLM usage", emoji_key="complete")
    return metrics


async def demonstrate_metrics_tracking():
    """Demonstrate metrics tracking functionality using Rich."""
    console.print(Rule("[bold blue]Metrics Tracking Demonstration[/bold blue]"))
    logger.info("Starting metrics tracking demonstration", emoji_key="start")
    
    metrics = get_metrics_tracker(reset_on_start=True)
    await simulate_llm_usage()
    stats = metrics.get_stats()
    
    # Use the standardized display utility instead of custom code
    display_analytics_metrics(stats)
    
    return stats


async def demonstrate_analytics_reporting():
    """Demonstrate analytics reporting functionality."""
    console.print(Rule("[bold blue]Analytics Reporting Demonstration[/bold blue]"))
    logger.info("Starting analytics reporting demonstration", emoji_key="start")
    
    metrics = get_metrics_tracker()
    stats = metrics.get_stats()
    if stats["general"]["requests_total"] == 0:
        logger.warning("No metrics data found. Running simulation first.", emoji_key="warning")
        await simulate_llm_usage()
        stats = metrics.get_stats()
    
    # Initialize reporting service
    reporting = AnalyticsReporting()
    
    # Generate reports
    cost_by_provider = reporting.cost_by_provider(stats)
    cost_by_model = reporting.cost_by_model(stats)
    tokens_by_provider = reporting.tokens_by_provider(stats)
    tokens_by_model = reporting.tokens_by_model(stats)
    daily_cost_trend = reporting.daily_cost_trend(stats)
    
    # Display reports using tables
    # Provider cost report
    if cost_by_provider:
        provider_cost_table = Table(title="[bold green]Cost by Provider[/bold green]", box=box.ROUNDED)
        provider_cost_table.add_column("Provider", style="magenta")
        provider_cost_table.add_column("Cost", style="green", justify="right")
        provider_cost_table.add_column("Percentage", style="cyan", justify="right")
        
        for item in cost_by_provider:
            provider_cost_table.add_row(
                escape(item["name"]),
                f"${item['value']:.6f}",
                f"{item['percentage']:.1f}%"
            )
        console.print(provider_cost_table)
        console.print()
    
    # Model cost report
    if cost_by_model:
        model_cost_table = Table(title="[bold green]Cost by Model[/bold green]", box=box.ROUNDED)
        model_cost_table.add_column("Model", style="blue")
        model_cost_table.add_column("Cost", style="green", justify="right")
        model_cost_table.add_column("Percentage", style="cyan", justify="right")
        
        for item in cost_by_model:
            model_cost_table.add_row(
                escape(item["name"]),
                f"${item['value']:.6f}",
                f"{item['percentage']:.1f}%"
            )
        console.print(model_cost_table)
        console.print()
    
    # Tokens by provider report
    if tokens_by_provider:
        tokens_provider_table = Table(title="[bold green]Tokens by Provider[/bold green]", box=box.ROUNDED)
        tokens_provider_table.add_column("Provider", style="magenta")
        tokens_provider_table.add_column("Tokens", style="white", justify="right")
        tokens_provider_table.add_column("Percentage", style="cyan", justify="right")
        
        for item in tokens_by_provider:
            tokens_provider_table.add_row(
                escape(item["name"]),
                f"{item['value']:,}",
                f"{item['percentage']:.1f}%"
            )
        console.print(tokens_provider_table)
        console.print()
    
    # Tokens by model report
    if tokens_by_model:
        tokens_model_table = Table(title="[bold green]Tokens by Model[/bold green]", box=box.ROUNDED)
        tokens_model_table.add_column("Model", style="blue")
        tokens_model_table.add_column("Tokens", style="white", justify="right")
        tokens_model_table.add_column("Percentage", style="cyan", justify="right")
        
        for item in tokens_by_model:
            tokens_model_table.add_row(
                escape(item["name"]),
                f"{item['value']:,}",
                f"{item['percentage']:.1f}%"
            )
        console.print(tokens_model_table)
        console.print()
        
    # Daily cost trend report
    if daily_cost_trend:
        daily_trend_table = Table(title="[bold green]Daily Cost Trend[/bold green]", box=box.ROUNDED)
        daily_trend_table.add_column("Date", style="yellow")
        daily_trend_table.add_column("Cost", style="green", justify="right")
        daily_trend_table.add_column("Change", style="cyan", justify="right")
        
        for item in daily_cost_trend:
            change_str = f"{item.get('change', 0):.1f}%" if 'change' in item else "N/A"
            change_style = ""
            if 'change' in item:
                if item['change'] > 0:
                    change_style = "[red]+"
                elif item['change'] < 0:
                    change_style = "[green]"
                    
            daily_trend_table.add_row(
                item["date"],
                f"${item['cost']:.6f}",
                f"{change_style}{change_str}[/]" if change_style else change_str
            )
        console.print(daily_trend_table)
        console.print()
    
    return {
        "cost_by_provider": cost_by_provider,
        "cost_by_model": cost_by_model,
        "tokens_by_provider": tokens_by_provider,
        "tokens_by_model": tokens_by_model,
        "daily_cost_trend": daily_cost_trend
    }


async def demonstrate_real_time_monitoring():
    """Demonstrate real-time metrics monitoring using Rich Live."""
    console.print(Rule("[bold blue]Real-Time Monitoring Demonstration[/bold blue]"))
    logger.info("Starting real-time monitoring (updates every 2s for 10s)", emoji_key="start")
    
    metrics = get_metrics_tracker() # Use existing tracker
    
    def generate_stats_table() -> Table:
        """Generates a Rich Table with current stats."""
        stats = metrics.get_stats()["general"]
        table = Table(title="Live LLM Usage Stats", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white", justify="right")
        table.add_row("Total Requests", f"{stats['requests_total']:,}")
        table.add_row("Total Tokens", f"{stats['tokens_total']:,}")
        table.add_row("Total Cost", f"${stats['cost_total']:.6f}")
        table.add_row("Total Errors", f"{stats['errors_total']:,}")
        return table

    try:
        with Live(generate_stats_table(), refresh_per_second=0.5, console=console) as live:
            # Simulate some activity in the background while monitoring
            # We could run simulate_llm_usage again, but let's just wait for demo purposes
            end_time = time.time() + 10 # Monitor for 10 seconds
            while time.time() < end_time:
                # In a real app, other tasks would be modifying metrics here
                live.update(generate_stats_table())
                await asyncio.sleep(2) # Update display every 2 seconds
                
            # Final update
            live.update(generate_stats_table())
            
    except Exception as e:
         logger.error(f"Error during live monitoring: {e}", emoji_key="error", exc_info=True)

    logger.info("Finished real-time monitoring demonstration", emoji_key="complete")
    console.print()


async def main():
    """Run all analytics and reporting demonstrations."""
    try:
        # Demonstrate metrics tracking (includes simulation)
        await demonstrate_metrics_tracking()
        
        # Demonstrate report generation
        await demonstrate_analytics_reporting()
        
        # Demonstrate real-time monitoring
        await demonstrate_real_time_monitoring()
        
    except Exception as e:
        logger.critical(f"Analytics demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    
    logger.success("Analytics & Reporting Demo Finished Successfully!", emoji_key="complete")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 