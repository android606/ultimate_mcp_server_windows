#!/usr/bin/env python
"""Cache demonstration for LLM Gateway."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    # Get provider
    provider = get_provider(provider_name)
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
    
    # Second completion - should be a cache hit
    logger.info("Running second completion (should be a cache hit)...", emoji_key="processing")
    start_time = time.time()
    result2 = await run_completion_with_cache(prompt)
    time2 = time.time() - start_time
    logger.info(
        f"Second completion took {time2:.2f}s (Cache speed-up: {time1/time2:.1f}x)",
        emoji_key="time"
    )
    
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
    
    # Show cache statistics
    stats = cache_service.get_stats()
    logger.info("Cache statistics:", emoji_key="info")
    print("\n" + "-" * 80)
    print(f"  Cache hits: {stats['stats']['hits']}")
    print(f"  Cache misses: {stats['stats']['misses']}")
    print(f"  Hit ratio: {stats['stats']['hit_ratio']:.2%}")
    print(f"  Total saved tokens: {stats['stats']['total_saved_tokens']}")
    print(f"  Estimated cost savings: ${stats['stats']['estimated_cost_savings']:.6f}")
    print("-" * 80 + "\n")
    
    # Demonstrating cache persistence
    if cache_service.enable_persistence:
        logger.info(
            "Cache is persistent and will be available across restarts",
            emoji_key="cache"
        )
        logger.info(f"Cache directory: {stats['persistence']['cache_dir']}", emoji_key="info")
    

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