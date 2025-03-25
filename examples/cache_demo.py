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
from llm_gateway.services.cache import get_cache_service
from llm_gateway.utils import get_logger

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
    if provider_name == Provider.OPENAI.value:
        api_key = decouple_config("OPENAI_API_KEY", default=None)
    elif provider_name == Provider.ANTHROPIC.value:
        api_key = decouple_config("ANTHROPIC_API_KEY", default=None)
    elif provider_name == Provider.GEMINI.value:
        api_key = decouple_config("GEMINI_API_KEY", default=None)
    elif provider_name == Provider.DEEPSEEK.value:
        api_key = decouple_config("DEEPSEEK_API_KEY", default=None)
    
    provider = get_provider(provider_name, api_key=api_key)
    await provider.initialize()
    
    # Get cache service
    cache_service = get_cache_service()
    
    # Create cache key
    cache_key = f"completion:{provider_name}:{model or 'default'}:{hash(prompt)}"
    
    # Try to get from cache
    if use_cache:
        cached_result = await cache_service.get(cache_key)
        if cached_result is not None:
            logger.success("Cache hit! Using cached result", emoji_key="cache")
            return cached_result
    
    # Generate completion if not cached
    logger.info("Cache miss. Generating new completion...", emoji_key="processing")
    result = await provider.generate_completion(
        prompt=prompt,
        model=model,
        temperature=0.1,  # Low temperature for deterministic results
    )
    
    # Save to cache
    if use_cache:
        await cache_service.set(
            key=cache_key,
            value=result,
            ttl=3600  # 1 hour TTL
        )
        
    return result


async def demonstrate_cache():
    """Demonstrate cache functionality."""
    logger.info("Starting cache demonstration", emoji_key="start")
    
    # Get cache service
    cache_service = get_cache_service()
    
    # Make sure cache is enabled
    if not cache_service.enabled:
        logger.warning("Cache is disabled. Enabling for demonstration.", emoji_key="warning")
        cache_service.enabled = True
    
    # Clear existing cache for demonstration
    cache_service.clear()
    logger.info("Cache cleared for demonstration", emoji_key="cache")
    
    # Define a prompt to use
    prompt = "Explain how caching works in distributed systems."
    
    # First completion - should be a cache miss
    logger.info("Running first completion (should be a cache miss)...", emoji_key="processing")
    start_time = time.time()
    result1 = await run_completion_with_cache(prompt)
    time1 = time.time() - start_time
    logger.info(
        f"First completion took {time1:.2f}s",
        emoji_key="time",
        tokens=result1.total_tokens,
        cost=result1.cost
    )
    
    # Get stats after first completion
    stats_after_miss = cache_service.get_stats()
    
    # Second completion - should be a cache hit
    logger.info("Running second completion (should be a cache hit)...", emoji_key="processing")
    start_time = time.time()
    result2 = await run_completion_with_cache(prompt)  # noqa: F841
    time2 = time.time() - start_time
    logger.info(
        f"Second completion took {time2:.2f}s (Cache speed-up: {time1/time2:.1f}x)",
        emoji_key="time"
    )
    
    # Get stats after second completion (cache hit)
    stats_after_hit = cache_service.get_stats()
    
    # Third completion - bypass cache
    logger.info("Running third completion (bypassing cache)...", emoji_key="processing")
    start_time = time.time()
    result3 = await run_completion_with_cache(prompt, use_cache=False)
    time3 = time.time() - start_time
    logger.info(
        f"Third completion (bypassing cache) took {time3:.2f}s",
        emoji_key="time",
        tokens=result3.total_tokens,
        cost=result3.cost
    )
    
    # Another cache hit (fourth completion)
    logger.info("Running fourth completion (should be another cache hit)...", emoji_key="processing")
    start_time = time.time()
    result4 = await run_completion_with_cache(prompt)  # noqa: F841
    time4 = time.time() - start_time
    logger.info(
        f"Fourth completion took {time4:.2f}s (Cache speed-up: {time3/time4:.1f}x)",
        emoji_key="time"
    )
    
    # Get final cache statistics
    stats = cache_service.get_stats()
    logger.info("Cache statistics:", emoji_key="info")
    print("\n" + "-" * 80)
    
    # Print the stats
    print(f"Stats dictionary contains keys: {list(stats.keys())}")
    
    # Display cache stats changes
    print("\nCache Stats Progression:")
    hit_counter1 = stats_after_miss['stats'].get('hits', 0)
    hit_counter2 = stats_after_hit['stats'].get('hits', 0)
    hit_counter_final = stats['stats'].get('hits', 0)
    
    print(f"  After first miss: {hit_counter1} hits")
    print(f"  After first hit: {hit_counter2} hits")
    print(f"  Final: {hit_counter_final} hits")
    
    # Access stats safely
    if 'stats' in stats:
        print("\nFinal Cache Statistics:")
        print(f"  Cache hits: {stats['stats'].get('hits', 0)}")
        print(f"  Cache misses: {stats['stats'].get('misses', 0)}")
        print(f"  Hit ratio: {stats['stats'].get('hit_ratio', 0):.2%}")
        print(f"  Total saved tokens: {stats['stats'].get('total_saved_tokens', 0)}")
        print(f"  Estimated cost savings: ${stats['stats'].get('estimated_cost_savings', 0):.6f}")
        
        # If stats aren't being tracked, show what should have been saved
        if stats['stats'].get('total_saved_tokens', 0) == 0:
            print("\nExpected savings (if cache tracking worked correctly):")
            expected_tokens_saved = result1.total_tokens * 2  # 2 hits
            expected_cost_saved = result1.cost * 2
            print(f"  Expected tokens saved: {expected_tokens_saved}")
            print(f"  Expected cost savings: ${expected_cost_saved:.6f}")
    
    print("-" * 80 + "\n")
    
    # Demonstrating cache persistence
    if cache_service.enable_persistence:
        logger.info(
            "Cache is persistent and will be available across restarts",
            emoji_key="cache"
        )
        # Check if cache_dir property exists directly on the cache_service
        if hasattr(cache_service, 'cache_dir'):
            logger.info(f"Cache directory: {cache_service.cache_dir}", emoji_key="info")
        else:
            logger.info("Cache directory not specified", emoji_key="info")


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