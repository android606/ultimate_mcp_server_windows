#!/usr/bin/env python
"""Simple completion demo using LLM Gateway's direct provider functionality."""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from llm_gateway.constants import Provider
from llm_gateway.core.server import Gateway
from llm_gateway.utils import get_logger
from llm_gateway.utils.display import CostTracker
from llm_gateway.utils.logging.console import console

# Initialize logger
logger = get_logger("example.simple_completion")

async def run_model_demo(tracker: CostTracker):
    """Run a simple demo using direct provider access."""
    logger.info("Starting simple completion demo", emoji_key="start")
    # Use Rich Rule for title
    console.print(Rule("[bold blue]Simple Completion Demo[/bold blue]"))
    
    # Create Gateway instance
    gateway = Gateway("simple-demo", register_tools=False)
    
    # Initialize providers
    logger.info("Initializing providers", emoji_key="provider")
    await gateway._initialize_providers()
    
    # Get provider (OpenAI)
    provider_name = Provider.OPENAI.value
    provider = gateway.providers.get(provider_name)
    
    if not provider:
        logger.error(f"Provider {provider_name} not available", emoji_key="error")
        return 1
        
    logger.success(f"Provider {provider_name} initialized", emoji_key="success")
    
    # List available models
    models = await provider.list_models()
    logger.info(f"Available models: {len(models)}", emoji_key="model")
    
    # Pick a valid model from the provider
    model = "gpt-4.1-mini"  # A valid model from constants.py
    
    # Generate a completion
    prompt = "Explain quantum computing in simple terms."
    
    logger.info(f"Generating completion with {model}", emoji_key="processing")
    result = await provider.generate_completion(
        prompt=prompt,
        model=model,
        temperature=0.7,
        max_tokens=150
    )
    
    # Print the result using Rich Panel
    logger.success("Completion generated successfully!", emoji_key="success")
    console.print(Panel(
        result.text.strip(),
        title=f"Quantum Computing Explanation ({model})",
        subtitle=f"Prompt: {prompt}",
        border_style="green",
        expand=False
    ))
    
    # Print stats using Rich Table
    stats_table = Table(title="Completion Stats", show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    stats_table.add_row("Input Tokens", str(result.input_tokens))
    stats_table.add_row("Output Tokens", str(result.output_tokens))
    stats_table.add_row("Cost", f"${result.cost:.6f}")
    stats_table.add_row("Processing Time", f"{result.processing_time:.2f}s")
    console.print(stats_table)

    # Track the call
    tracker.add_call(result)

    # Display cost summary
    tracker.display_summary(console)

    return 0

async def main():
    """Run the demo."""
    tracker = CostTracker()
    try:
        return await run_model_demo(tracker)
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical")
        return 1

if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 