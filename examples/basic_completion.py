#!/usr/bin/env python
"""Basic completion example using LLM Gateway."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
# These imports need to be below sys.path modification, which is why they have noqa comments
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

# Project imports
from llm_gateway.constants import Provider  # noqa: E402
from llm_gateway.core.server import Gateway  # noqa: E402
from llm_gateway.utils import get_logger  # noqa: E402
from llm_gateway.utils.display import CostTracker, display_completion_result  # Import CostTracker
from llm_gateway.utils.logging.console import console  # noqa: E402

# Initialize logger
logger = get_logger("example.basic_completion")

async def run_basic_completion(gateway, tracker: CostTracker):
    """Run a basic completion example."""
    logger.info("Starting basic completion example", emoji_key="start")
    console.print(Rule("[bold blue]Basic Completion[/bold blue]"))

    # Prompt to complete
    prompt = "Explain the concept of federated learning in simple terms."
    
    try:
        # Get OpenAI provider from gateway
        provider = gateway.providers.get(Provider.OPENAI.value)
        if not provider:
            logger.error(f"Provider {Provider.OPENAI.value} not available or initialized", emoji_key="error")
            return
        
        # Generate completion using OpenAI
        logger.info("Generating completion...", emoji_key="processing")
        result = await provider.generate_completion(
            prompt=prompt,
            temperature=0.7,
            max_tokens=200
        )
        
        # Log simple success message
        logger.success("Completion generated successfully!", emoji_key="success")

        # Display results
        # Create a result format similar to what CompletionClient would return
        # formatted_result = {
        #     "text": result.text,
        #     "model": result.model,
        #     "provider": result.provider,
        #     "tokens": {
        #         "input": result.input_tokens,
        #         "output": result.output_tokens,
        #         "total": result.total_tokens
        #     },
        #     "cost": result.cost,
        #     "processing_time": result.processing_time
        # }

        # Display results using the utility function
        display_completion_result(
            console=console,
            result=result, # Pass the original result object
            title="Federated Learning Explanation"
        )
        
        # Track cost
        tracker.add_call(result)

    except Exception as e:
        # Use logger for errors, as DetailedLogFormatter handles error panels well
        logger.error(f"Error generating completion: {str(e)}", emoji_key="error", exc_info=True)
        raise


async def run_streaming_completion(gateway):
    """Run a streaming completion example."""
    logger.info("Starting streaming completion example", emoji_key="start")
    console.print(Rule("[bold blue]Streaming Completion[/bold blue]"))

    # Prompt to complete
    prompt = "Write a short poem about artificial intelligence."
    
    try:
        # Get OpenAI provider from gateway
        provider = gateway.providers.get(Provider.OPENAI.value)
        if not provider:
            logger.error(f"Provider {Provider.OPENAI.value} not available or initialized", emoji_key="error")
            return
        
        logger.info("Generating streaming completion...", emoji_key="processing")
        
        # Use Panel for streaming output presentation
        output_panel = Panel("", title="AI Poem (Streaming)", border_style="cyan", expand=False)
        
        # Start timer
        start_time = time.time()
        
        full_text = ""
        token_count = 0
        
        # Use Live display for the streaming output panel
        with Live(output_panel, console=console, refresh_per_second=4) as live:  # noqa: F841
            # Get stream from the provider directly
            stream = provider.generate_completion_stream(
                prompt=prompt,
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


async def run_cached_completion(gateway, tracker: CostTracker):
    """Run a completion with caching.
    
    Note: Since we're not using CompletionClient which has built-in caching,
    this example will make two separate calls to the provider.
    """
    logger.info("Starting cached completion example", emoji_key="start")
    console.print(Rule("[bold blue]Cached Completion Demo[/bold blue]"))

    # Prompt to complete
    prompt = "Explain the concept of federated learning in simple terms."
    
    try:
        # Get OpenAI provider from gateway
        provider = gateway.providers.get(Provider.OPENAI.value)
        if not provider:
            logger.error(f"Provider {Provider.OPENAI.value} not available or initialized", emoji_key="error")
            return
        
        # First request
        logger.info("First request...", emoji_key="processing")
        start_time1 = time.time()
        result1 = await provider.generate_completion(
            prompt=prompt,
            temperature=0.7,
            max_tokens=200
        )
        processing_time1 = time.time() - start_time1
        
        # Track first call
        tracker.add_call(result1)
        
        # Note: We don't actually have caching here since we're not using CompletionClient
        # So instead we'll just make another call and compare times
        logger.info("Second request...", emoji_key="processing")
        start_time2 = time.time()
        result2 = await provider.generate_completion(  # noqa: F841
            prompt=prompt,
            temperature=0.7,
            max_tokens=200
        )
        processing_time2 = time.time() - start_time2
        
        # Track second call
        tracker.add_call(result2)

        # Log timing comparison
        processing_ratio = processing_time1 / processing_time2 if processing_time2 > 0 else 1.0
        logger.info(f"Time comparison - First call: {processing_time1:.3f}s, Second call: {processing_time2:.3f}s", emoji_key="processing")
        logger.info(f"Speed ratio: {processing_ratio:.1f}x", emoji_key="info")
        
        console.print("[yellow]Note: This example doesn't use actual caching since we're bypassing CompletionClient.[/yellow]")
        
        # Display results
        display_completion_result(
            console=console,
            result=result1, # Pass the original result object
            title="Federated Learning Explanation"
        )
        
    except Exception as e:
        logger.error(f"Error with cached completion demo: {str(e)}", emoji_key="error", exc_info=True)
        raise


async def run_multi_provider(gateway, tracker: CostTracker):
    """Run completion with multiple providers."""
    logger.info("Starting multi-provider example", emoji_key="start")
    console.print(Rule("[bold blue]Multi-Provider Completion[/bold blue]"))

    # Prompt to complete
    prompt = "List 3 benefits of quantum computing."
    
    providers_to_try = [
        Provider.OPENAI.value,
        Provider.ANTHROPIC.value, 
        Provider.GEMINI.value
    ]
    
    result_obj = None
    
    try:
        # Try providers in sequence
        logger.info("Trying multiple providers in sequence...", emoji_key="processing")
        
        for provider_name in providers_to_try:
            try:
                logger.info(f"Trying provider: {provider_name}", emoji_key="processing")
                
                # Get provider from gateway
                provider = gateway.providers.get(provider_name)
                if not provider:
                    logger.warning(f"Provider {provider_name} not available or initialized, skipping", emoji_key="warning")
                    continue
                
                # Generate completion
                result_obj = await provider.generate_completion(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=200
                )
                
                # Track cost
                tracker.add_call(result_obj)

                logger.success(f"Successfully used provider: {provider_name}", emoji_key="success")
                break  # Exit loop on success
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {str(e)}", emoji_key="warning")
                # Continue to next provider
        
        if result_obj:
            # Display results
            display_completion_result(
                console=console,
                result=result_obj, # Pass result_obj directly
                title=f"Response from {result_obj.provider}" # Use result_obj.provider
            )
        else:
            logger.error("All providers failed. No results available.", emoji_key="error")
        
    except Exception as e:
        logger.error(f"Error with multi-provider completion: {str(e)}", emoji_key="error", exc_info=True)
        raise


async def main():
    """Run completion examples."""
    tracker = CostTracker() # Instantiate tracker
    try:
        # Create a gateway instance for all examples to share
        gateway = Gateway("basic-completion-demo", register_tools=False)
        
        # Initialize providers
        logger.info("Initializing providers...", emoji_key="provider")
        await gateway._initialize_providers()
        
        # Run basic completion
        await run_basic_completion(gateway, tracker)
        
        console.print() # Add space
        
        # Run streaming completion
        await run_streaming_completion(gateway)
        
        console.print() # Add space
        
        # Run cached completion
        await run_cached_completion(gateway, tracker)
        
        console.print() # Add space
        
        # Run multi-provider completion
        await run_multi_provider(gateway, tracker)
        
        # Display cost summary at the end
        tracker.display_summary(console)

    except Exception as e:
        # Use logger for critical errors
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)