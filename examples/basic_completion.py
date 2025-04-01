#!/usr/bin/env python
"""Basic completion example using LLM Gateway."""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.utils import get_logger
from llm_gateway.utils.logging.console import console  # Keep themed console import

# Initialize logger
logger = get_logger("example.basic_completion")

async def run_basic_completion():
    """Run a basic completion example."""
    logger.info("Starting basic completion example", emoji_key="start")
    console.print(Rule("[bold blue]Basic Completion[/bold blue]"))

    # Prompt to complete
    prompt = "Explain the concept of federated learning in simple terms."
    
    # Get OpenAI provider - API key loaded automatically by config system
    provider_name = Provider.OPENAI.value
    try:
        provider = get_provider(provider_name)
        await provider.initialize()
        
        logger.info(
            f"Using provider: {provider_name}",
            emoji_key="provider",
        )
        
        # Get default model
        model = provider.get_default_model()
        logger.info(f"Using model: {model}", emoji_key="model")
        
        # Generate completion
        logger.info("Generating completion...", emoji_key="processing")
        result = await provider.generate_completion(
            prompt=prompt,
            model=model,
            temperature=0.7,
            max_tokens=200
        )
        
        # Log simple success message
        logger.success("Completion generated successfully!", emoji_key="success")

        # Display results using Rich Panel
        console.print(Panel(
            result.text.strip(),
            title="Federated Learning Explanation",
            border_style="green",
            expand=False
        ))
        
        # Display stats using Rich Table
        stats_table = Table(title="Completion Stats", show_header=False, box=None)
        stats_table.add_column("Metric", style="green")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Input Tokens", str(result.input_tokens))
        stats_table.add_row("Output Tokens", str(result.output_tokens))
        stats_table.add_row("Cost", f"${result.cost:.6f}")
        stats_table.add_row("Processing Time", f"{result.processing_time:.3f}s")
        console.print(stats_table)
        
    except Exception as e:
        # Use logger for errors, as DetailedLogFormatter handles error panels well
        logger.error(f"Error generating completion: {str(e)}", emoji_key="error", exc_info=True)
        raise


async def run_streaming_completion():
    """Run a streaming completion example."""
    logger.info("Starting streaming completion example", emoji_key="start")
    console.print(Rule("[bold blue]Streaming Completion[/bold blue]"))

    # Prompt to complete
    prompt = "Write a short poem about artificial intelligence."
    
    # Get OpenAI provider - API key loaded automatically by config system
    provider_name = Provider.OPENAI.value
    try:
        provider = get_provider(provider_name)
        await provider.initialize()
        
        logger.info(
            f"Using provider: {provider_name} in streaming mode",
            emoji_key="provider",
        )
        
        # Get default model
        model = provider.get_default_model()
        
        # Generate streaming completion
        logger.info("Generating streaming completion...", emoji_key="processing")
        
        # Use Panel for streaming output presentation
        output_panel = Panel("", title="AI Poem (Streaming)", border_style="cyan", expand=False)
        
        # Start timer
        import time
        start_time = time.time()
        
        full_text = ""
        token_count = 0
        
        # Use Live display for the streaming output panel
        from rich.live import Live
        with Live(output_panel, console=console, refresh_per_second=4) as live:  # noqa: F841
            stream = provider.generate_completion_stream(
                prompt=prompt,
                model=model,
                temperature=0.7,
                max_tokens=200
            )
            async for chunk, _metadata in stream:
                full_text += chunk
                token_count += 1
                # Update the panel content within the Live context
                output_panel.renderable = full_text 
                # live.update(output_panel) # This might be needed depending on Panel updates
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log simple success message
        logger.success("Streaming completion generated successfully!", emoji_key="success")

        # Display stats using Rich Table
        stats_table = Table(title="Streaming Stats", show_header=False, box=None)
        stats_table.add_column("Metric", style="green")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Chunks Received", str(token_count))
        stats_table.add_row("Processing Time", f"{processing_time:.3f}s")
        console.print(stats_table)
        
    except Exception as e:
        # Use logger for errors
        logger.error(f"Error generating streaming completion: {str(e)}", emoji_key="error", exc_info=True)
        raise


async def main():
    """Run completion examples."""
    try:
        # Run basic completion
        await run_basic_completion()
        
        console.print() # Add space
        
        # Run streaming completion
        await run_streaming_completion()
        
    except Exception as e:
        # Use logger for critical errors
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    # Add logging handler setup here, perhaps, if needed for non-direct prints?
    # For now, relying on root config or direct console prints
    
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)