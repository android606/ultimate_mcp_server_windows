"""High-level client for LLM completion operations."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import BaseProvider, get_provider
from llm_gateway.services.cache import get_cache_service
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.clients.completion")

class CompletionClient:
    """High-level client for completion operations.
    
    This client provides a simplified interface for text completions
    with support for caching, streaming, and multi-provider operations.
    """
    
    def __init__(self, default_provider: str = Provider.OPENAI.value, use_cache_by_default: bool = True):
        """Initialize the completion client.
        
        Args:
            default_provider: Default provider to use for completions
            use_cache_by_default: Whether to use cache by default
        """
        self.default_provider = default_provider
        self.cache_service = get_cache_service()
        self.use_cache_by_default = use_cache_by_default
        
    async def initialize_provider(self, provider_name: str, api_key: Optional[str] = None) -> BaseProvider:
        """Initialize and return a provider instance."""
        try:
            provider = await get_provider(provider_name, api_key=api_key)
            # Ensure the provider is initialized (some might need async init)
            if hasattr(provider, 'initialize') and asyncio.iscoroutinefunction(provider.initialize):
                await provider.initialize()
            return provider
        except Exception as e:
            logger.error(f"Failed to initialize provider {provider_name}: {e}", emoji_key="error")
            raise
    
    async def generate_completion(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        **kwargs
    ):
        """Generate a completion for the given prompt.
        
        Args:
            prompt: Text prompt
            provider: Provider to use (defaults to client default)
            model: Model to use (if None, uses provider default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use cache
            cache_ttl: Time-to-live for cache entries in seconds
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Completion result
        """
        provider_name = provider or self.default_provider
        
        # Check cache if enabled
        if use_cache and self.cache_service.enabled:
            # Create a robust cache key
            provider_instance = await self.initialize_provider(provider_name)
            model_id = model or provider_instance.get_default_model()
            # Include relevant parameters in the cache key
            params_hash = hash((prompt, temperature, max_tokens, str(kwargs)))
            cache_key = f"completion:{provider_name}:{model_id}:{params_hash}"
            
            cached_result = await self.cache_service.get(cache_key)
            if cached_result is not None:
                logger.success("Cache hit! Using cached result", emoji_key="cache")
                # Set a nominal processing time for cached results
                cached_result.processing_time = 0.001
                return cached_result
                
        # Cache miss or cache disabled
        if use_cache and self.cache_service.enabled:
            logger.info("Cache miss. Generating new completion...", emoji_key="processing")
        else:
            logger.info("Generating completion...", emoji_key="processing")
            
        # Initialize provider and generate completion
        try:
            provider_instance = await self.initialize_provider(provider_name)
            model_id = model or provider_instance.get_default_model()
            
            result = await provider_instance.generate_completion(
                prompt=prompt,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Save to cache if enabled
            if use_cache and self.cache_service.enabled:
                await self.cache_service.set(
                    key=cache_key,
                    value=result,
                    ttl=cache_ttl
                )
                logger.info(f"Result saved to cache (key: ...{cache_key[-10:]})", emoji_key="cache")
                
            return result
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}", emoji_key="error")
            raise
    
    async def generate_completion_stream(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion.
        
        Args:
            prompt: Text prompt
            provider: Provider to use (defaults to client default)
            model: Model to use (if None, uses provider default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the provider
            
        Yields:
            Tuples of (chunk_text, metadata)
        """
        provider_name = provider or self.default_provider
        
        logger.info("Generating streaming completion...", emoji_key="processing")
        
        # Initialize provider and generate streaming completion
        try:
            provider_instance = await self.initialize_provider(provider_name)
            model_id = model or provider_instance.get_default_model()
            
            stream = provider_instance.generate_completion_stream(
                prompt=prompt,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            async for chunk, metadata in stream:
                yield chunk, metadata
                
        except Exception as e:
            logger.error(f"Error generating streaming completion: {str(e)}", emoji_key="error")
            raise
            
    async def try_providers(
        self,
        prompt: str,
        providers: List[str],
        models: Optional[List[str]] = None,
        **kwargs
    ):
        """Try multiple providers in sequence until one succeeds.
        
        Args:
            prompt: Text prompt
            providers: List of providers to try
            models: Optional list of models to use with each provider
            **kwargs: Additional parameters to pass to generate_completion
            
        Returns:
            Completion result from the first successful provider
        """
        if not providers:
            raise ValueError("No providers specified")
            
        models = models or [None] * len(providers)
        if len(models) != len(providers):
            raise ValueError("If models are specified, there must be one for each provider")
            
        last_error = None
        
        for i, provider_name in enumerate(providers):
            try:
                logger.info(f"Trying provider: {provider_name}", emoji_key="provider")
                result = await self.generate_completion(
                    prompt=prompt,
                    provider=provider_name,
                    model=models[i],
                    **kwargs
                )
                return result
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {str(e)}", emoji_key="warning")
                last_error = e
                
        # If we get here, all providers failed
        raise Exception(f"All providers failed. Last error: {str(last_error)}") 