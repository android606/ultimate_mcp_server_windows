#!/usr/bin/env python
"""Basic completion example using LLM Gateway."""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.rule import Rule

from llm_gateway.clients import CompletionClient
from llm_gateway.constants import Provider
from llm_gateway.utils import get_logger
from llm_gateway.utils.display import display_completion_result
from llm_gateway.utils.logging.console import console

# Initialize logger
logger = get_logger("example.basic_completion")

async def run_basic_completion():
    """Run a basic completion example."""
    logger.info("Starting basic completion example", emoji_key="start")
    console.print(Rule("[bold blue]Basic Completion[/bold blue]"))

    # Prompt to complete
    prompt = "Explain the concept of federated learning in simple terms."
    
    # Initialize completion client
    client = CompletionClient()
    
    try:
        # Generate completion using OpenAI
        logger.info("Generating completion...", emoji_key="processing")
        result = await client.generate_completion(
            prompt=prompt,
            provider=Provider.OPENAI.value,
            temperature=0.7,
            max_tokens=200
        )
        
        # Log simple success message
        logger.success("Completion generated successfully!", emoji_key="success")

        # Display results using the utility function
        display_completion_result(
            console=console,
            result=result,
            title="Federated Learning Explanation"
        )
        
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
    
    # Initialize completion client
    client = CompletionClient()
    
    try:
        logger.info("Generating streaming completion...", emoji_key="processing")
        
        # Use Panel for streaming output presentation
        from rich.panel import Panel
        output_panel = Panel("", title="AI Poem (Streaming)", border_style="cyan", expand=False)
        
        # Start timer
        import time
        start_time = time.time()
        
        full_text = ""
        token_count = 0
        
        # Use Live display for the streaming output panel
        from rich.live import Live
        with Live(output_panel, console=console, refresh_per_second=4) as live:  # noqa: F841
            # Get stream from the client
            stream = client.generate_completion_stream(
                prompt=prompt,
                provider=Provider.OPENAI.value,
                temperature=0.7,
                max_tokens=200
            )
            
            async for chunk, _metadata in stream:
                full_text += chunk
                token_count += 1
                # Update the panel content
                output_panel.renderable = full_text
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log simple success message
        logger.success("Streaming completion generated successfully!", emoji_key="success")

        # Display stats using Rich Table
        from rich.table import Table
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


async def run_cached_completion():
    """Run a completion with caching."""
    logger.info("Starting cached completion example", emoji_key="start")
    console.print(Rule("[bold blue]Cached Completion[/bold blue]"))

    # Prompt to complete
    prompt = "Explain the concept of federated learning in simple terms."
    
    # Initialize completion client
    client = CompletionClient()
    
    try:
        # First request (cache miss)
        logger.info("First request (expected cache miss)...", emoji_key="processing")
        result1 = await client.generate_completion(
            prompt=prompt,
            provider=Provider.OPENAI.value,
            temperature=0.7,
            max_tokens=200,
            use_cache=True
        )
        
        # Second request (cache hit)
        logger.info("Second request (expected cache hit)...", emoji_key="processing")
        result2 = await client.generate_completion(
            prompt=prompt,
            provider=Provider.OPENAI.value,
            temperature=0.7,
            max_tokens=200,
            use_cache=True
        )
        
        # Log speed comparison
        processing_ratio = result1.processing_time / result2.processing_time if result2.processing_time > 0 else float('inf')
        logger.success(f"Cache speed-up: {processing_ratio:.1f}x", emoji_key="success")
        
        # Display results
        display_completion_result(
            console=console,
            result=result1,
            title="Cached Result"
        )
        
    except Exception as e:
        logger.error(f"Error with cached completion: {str(e)}", emoji_key="error", exc_info=True)
        raise


async def run_multi_provider():
    """Run completion with multiple providers."""
    logger.info("Starting multi-provider example", emoji_key="start")
    console.print(Rule("[bold blue]Multi-Provider Completion[/bold blue]"))

    # Prompt to complete
    prompt = "List 3 benefits of quantum computing."
    
    # Initialize completion client
    client = CompletionClient()
    
    try:
        # Try providers in sequence
        logger.info("Trying multiple providers in sequence...", emoji_key="processing")
        result = await client.try_providers(
            prompt=prompt,
            providers=[Provider.OPENAI.value, Provider.ANTHROPIC.value, Provider.GEMINI.value],
            temperature=0.7,
            max_tokens=200
        )
        
        logger.success(f"Successfully used provider: {result.provider}", emoji_key="success")
        
        # Display results
        display_completion_result(
            console=console,
            result=result,
            title=f"Response from {result.provider}"
        )
        
    except Exception as e:
        logger.error(f"Error with multi-provider completion: {str(e)}", emoji_key="error", exc_info=True)
        raise


async def main():
    """Run completion examples."""
    try:
        # Run basic completion
        await run_basic_completion()
        
        console.print() # Add space
        
        # Run streaming completion
        await run_streaming_completion()
        
        console.print() # Add space
        
        # Run cached completion
        await run_cached_completion()
        
        console.print() # Add space
        
        # Run multi-provider completion
        await run_multi_provider()
        
    except Exception as e:
        # Use logger for critical errors
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)