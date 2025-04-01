#!/usr/bin/env python
"""Cache demonstration for LLM Gateway."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from decouple import config as decouple_config

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.services.cache import get_cache_service, run_completion_with_cache
from llm_gateway.utils import get_logger
# --- Add Rich Imports ---
from llm_gateway.utils.logging.console import console
from llm_gateway.utils.display import display_cache_stats
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.markup import escape
from rich import box
# ----------------------

# Initialize logger
logger = get_logger("example.cache_demo")


async def run_completion_with_cache(
    prompt: str,
    provider_name: str = Provider.OPENAI.value,
    model: str = None,
    use_cache: bool = True
):
    """Run a completion with caching.
    
    Args:
        prompt: Text prompt
        provider_name: Provider to use
        model: Model name (optional)
        use_cache: Whether to use cache
        
    Returns:
        Completion result
    """
    # Get provider with API key directly from decouple
    api_key = None
    # Simplify key retrieval slightly
    key_map = {
        Provider.OPENAI.value: "OPENAI_API_KEY",
        Provider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
        Provider.GEMINI.value: "GEMINI_API_KEY",
        Provider.DEEPSEEK.value: "DEEPSEEK_API_KEY"
    }
    api_key_name = key_map.get(provider_name)
    if api_key_name:
        api_key = decouple_config(api_key_name, default=None)
    
    if not api_key:
        # Log warning but allow fallback if provider supports keyless (unlikely for these)
        logger.warning(f"API key for {provider_name} not found. Request may fail.", emoji_key="warning")

    try:
        provider = get_provider(provider_name, api_key=api_key)
        await provider.initialize()
    except Exception as e:
         logger.error(f"Failed to initialize provider {provider_name}: {e}", emoji_key="error")
         raise # Re-raise exception to stop execution if provider fails
    
    cache_service = get_cache_service()
    
    # Create a more robust cache key (consider all relevant params)
    model_id = model or provider.get_default_model() # Ensure we have a model id
    params_hash = hash((prompt, 0.1)) # Hash includes temp, etc. - simplified here
    cache_key = f"completion:{provider_name}:{model_id}:{params_hash}"
    
    if use_cache and cache_service.enabled:
        cached_result = await cache_service.get(cache_key)
        if cached_result is not None:
            logger.success("Cache hit! Using cached result", emoji_key="cache")
            # Simulate processing time for cache retrieval (negligible)
            cached_result.processing_time = 0.001 
            return cached_result
    
    # Generate completion if not cached or cache disabled
    if use_cache:
        logger.info("Cache miss. Generating new completion...", emoji_key="processing")
    else:
        logger.info("Cache disabled by request. Generating new completion...", emoji_key="processing")
        
    # Use the determined model_id
    result = await provider.generate_completion(
        prompt=prompt,
        model=model_id,
        temperature=0.1, 
    )
    
    # Save to cache if enabled
    if use_cache and cache_service.enabled:
        await cache_service.set(
            key=cache_key,
            value=result,
            ttl=3600 # 1 hour TTL
        )
        logger.info(f"Result saved to cache (key: ...{cache_key[-10:]})", emoji_key="cache")
        
    return result


async def demonstrate_cache():
    """Demonstrate cache functionality using Rich."""
    console.print(Rule("[bold blue]Cache Demonstration[/bold blue]"))
    logger.info("Starting cache demonstration", emoji_key="start")
    
    cache_service = get_cache_service()
    
    if not cache_service.enabled:
        logger.warning("Cache is disabled by default. Enabling for demonstration.", emoji_key="warning")
        cache_service.enabled = True
    
    cache_service.clear() # Start with a clean slate
    logger.info("Cache cleared for demonstration", emoji_key="cache")
    
    prompt = "Explain how caching works in distributed systems."
    console.print(f"[cyan]Using Prompt:[/cyan] {escape(prompt)}")
    console.print()

    results = {}
    times = {}
    stats_log = {}

    try:
        # 1. Cache Miss
        logger.info("1. Running first completion (expect cache MISS)...", emoji_key="processing")
        start_time = time.time()
        results[1] = await run_completion_with_cache(prompt, use_cache=True)
        times[1] = time.time() - start_time
        stats_log[1] = cache_service.get_stats()["stats"]
        console.print(f"   [yellow]MISS:[/yellow] Took [bold]{times[1]:.3f}s[/bold] (Cost: ${results[1].cost:.6f}, Tokens: {results[1].total_tokens})")

        # 2. Cache Hit
        logger.info("2. Running second completion (expect cache HIT)...", emoji_key="processing")
        start_time = time.time()
        results[2] = await run_completion_with_cache(prompt, use_cache=True)
        times[2] = time.time() - start_time
        stats_log[2] = cache_service.get_stats()["stats"]
        speedup = times[1] / times[2] if times[2] > 0 else float('inf')
        console.print(f"   [green]HIT:[/green]  Took [bold]{times[2]:.3f}s[/bold] (Speed-up: {speedup:.1f}x vs Miss)")

        # 3. Cache Bypass
        logger.info("3. Running third completion (BYPASS cache)...", emoji_key="processing")
        start_time = time.time()
        results[3] = await run_completion_with_cache(prompt, use_cache=False)
        times[3] = time.time() - start_time
        stats_log[3] = cache_service.get_stats()["stats"] # Stats shouldn't change much
        console.print(f"   [cyan]BYPASS:[/cyan] Took [bold]{times[3]:.3f}s[/bold] (Cost: ${results[3].cost:.6f}, Tokens: {results[3].total_tokens})")

        # 4. Another Cache Hit
        logger.info("4. Running fourth completion (expect cache HIT again)...", emoji_key="processing")
        start_time = time.time()
        results[4] = await run_completion_with_cache(prompt, use_cache=True)
        times[4] = time.time() - start_time
        stats_log[4] = cache_service.get_stats()["stats"]
        speedup_vs_bypass = times[3] / times[4] if times[4] > 0 else float('inf')
        console.print(f"   [green]HIT:[/green]  Took [bold]{times[4]:.3f}s[/bold] (Speed-up: {speedup_vs_bypass:.1f}x vs Bypass)")
        console.print()

    except Exception as e:
         logger.error(f"Error during cache demonstration run: {e}", emoji_key="error", exc_info=True)
         console.print(f"[bold red]Error during demo run:[/bold red] {escape(str(e))}")
         # Attempt to display stats even if error occurred mid-way
         stats = cache_service.get_stats()
    else:
         stats = cache_service.get_stats() # Use final stats if all runs succeeded

    # Display Final Cache Statistics using our display function
    display_cache_stats(stats, stats_log, console)
    
    console.print()
    if cache_service.enable_persistence:
        logger.info("Cache persistence is enabled.", emoji_key="cache")
        if hasattr(cache_service, 'cache_dir'):
            console.print(f"[dim]Cache Directory: {cache_service.cache_dir}[/dim]")
    else:
        logger.info("Cache persistence is disabled.", emoji_key="cache")
    console.print()


async def main():
    """Run cache demonstration."""
    try:
        await demonstrate_cache()
        
    except Exception as e:
        logger.critical(f"Cache demonstration failed: {str(e)}", emoji_key="critical")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code)