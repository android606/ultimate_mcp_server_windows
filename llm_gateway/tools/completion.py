"""Text completion tools for LLM Gateway."""
import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from llm_gateway.constants import Provider, TaskType
from llm_gateway.core.providers.base import get_provider
from llm_gateway.services.cache import with_cache
from llm_gateway.tools.base import BaseTool, with_retry, with_tool_metrics
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.completion")


class CompletionTools(BaseTool):
    """Text completion tools for LLM Gateway."""
    
    tool_name = "completion"
    description = "Tools for text generation and completion."
    
    def __init__(self, mcp_server):
        """Initialize the completion tools.
        
        Args:
            mcp_server: MCP server instance
        """
        super().__init__(mcp_server)
        
    def _register_tools(self):
        """Register completion tools with MCP server."""
        
        @self.mcp.tool()
        @with_tool_metrics
        async def generate_completion(
            prompt: str,
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: float = 0.7,
            stream: bool = False,
            additional_params: Optional[Dict[str, Any]] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Generate a text completion based on the provided prompt.
            
            Args:
                prompt: Input text prompt
                provider: LLM provider to use (openai, anthropic, etc.)
                model: Model name to use (defaults to provider's default)
                max_tokens: Maximum tokens to generate
                temperature: Temperature parameter (0.0-1.0)
                stream: Whether to stream the response (not supported for this endpoint)
                additional_params: Additional provider-specific parameters
                
            Returns:
                Dictionary containing completion text and metadata
            """
            # Streaming not supported for this endpoint
            if stream:
                return {
                    "error": "Streaming not supported for this endpoint. Use the stream_completion endpoint.",
                    "text": None,
                }
                
            start_time = time.time()
            
            # Get provider instance
            try:
                provider_instance = get_provider(provider)
            except Exception as e:
                logger.error(
                    f"Failed to initialize provider: {str(e)}",
                    emoji_key="error",
                    provider=provider
                )
                return {
                    "error": f"Failed to initialize provider: {str(e)}",
                    "text": None,
                }
            
            # Set default additional params
            additional_params = additional_params or {}
            
            try:
                # Generate completion
                result = await provider_instance.generate_completion(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **additional_params
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log success
                logger.success(
                    f"Completion generated successfully with {provider}/{result.model}",
                    emoji_key=TaskType.COMPLETION.value,
                    tokens={
                        "input": result.input_tokens,
                        "output": result.output_tokens
                    },
                    cost=result.cost,
                    time=processing_time
                )
                
                # Return standardized result
                return {
                    "text": result.text,
                    "model": result.model,
                    "provider": provider,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "cost": result.cost,
                    "processing_time": processing_time,
                }
                
            except Exception as e:
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log error
                logger.error(
                    f"Completion generation failed: {str(e)}",
                    emoji_key="error",
                    provider=provider,
                    model=model,
                    time=processing_time
                )
                
                # Return error
                return {
                    "error": f"Completion generation failed: {str(e)}",
                    "text": None,
                    "provider": provider,
                    "model": model,
                    "processing_time": processing_time,
                }
        
        @self.mcp.tool()
        @with_tool_metrics
        async def stream_completion(
            prompt: str,
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: float = 0.7,
            additional_params: Optional[Dict[str, Any]] = None,
            ctx=None
        ) -> AsyncGenerator[Dict[str, Any], None]:
            """
            Stream a text completion based on the provided prompt.
            
            Args:
                prompt: Input text prompt
                provider: LLM provider to use (openai, anthropic, etc.)
                model: Model name to use (defaults to provider's default)
                max_tokens: Maximum tokens to generate
                temperature: Temperature parameter (0.0-1.0)
                additional_params: Additional provider-specific parameters
                
            Yields:
                Dictionaries containing text chunks and metadata
            """
            start_time = time.time()
            
            # Get provider instance
            try:
                provider_instance = get_provider(provider)
            except Exception as e:
                logger.error(
                    f"Failed to initialize provider: {str(e)}",
                    emoji_key="error",
                    provider=provider
                )
                yield {
                    "error": f"Failed to initialize provider: {str(e)}",
                    "text": None,
                    "finished": True,
                }
                return
            
            # Set default additional params
            additional_params = additional_params or {}
            
            # Log start of streaming
            logger.info(
                f"Starting streaming completion with {provider}",
                emoji_key=TaskType.COMPLETION.value,
                prompt_length=len(prompt)
            )
            
            try:
                # Generate streaming completion
                chunk_count = 0
                full_text = ""
                
                # Get stream
                stream = provider_instance.generate_completion_stream(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **additional_params
                )
                
                async for chunk, metadata in stream:
                    chunk_count += 1
                    full_text += chunk
                    
                    # Yield chunk with metadata
                    yield {
                        "text": chunk,
                        "chunk_index": chunk_count,
                        "provider": provider,
                        "model": metadata.get("model"),
                        "finish_reason": metadata.get("finish_reason"),
                        "finished": False,
                    }
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log success
                logger.success(
                    f"Streaming completion finished ({chunk_count} chunks)",
                    emoji_key="success",
                    provider=provider,
                    chunks=chunk_count,
                    time=processing_time
                )
                
                # Yield final chunk with complete metadata
                yield {
                    "text": "",  # Empty final chunk
                    "chunk_index": chunk_count + 1,
                    "provider": provider,
                    "full_text": full_text,
                    "processing_time": processing_time,
                    "finished": True,
                }
                
            except Exception as e:
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log error
                logger.error(
                    f"Streaming completion failed: {str(e)}",
                    emoji_key="error",
                    provider=provider,
                    model=model,
                    time=processing_time
                )
                
                # Yield error
                yield {
                    "error": f"Streaming completion failed: {str(e)}",
                    "text": None,
                    "provider": provider,
                    "model": model,
                    "processing_time": processing_time,
                    "finished": True,
                }
        
        @self.mcp.tool()
        @with_cache(ttl=24 * 60 * 60)  # Cache for 24 hours
        @with_tool_metrics
        @with_retry(max_retries=2, retry_delay=1.0)
        async def chat_completion(
            messages: List[Dict[str, Any]],
            provider: str = Provider.OPENAI.value,
            model: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: float = 0.7,
            system_prompt: Optional[str] = None,
            additional_params: Optional[Dict[str, Any]] = None,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Generate a chat completion based on the provided message history.
            
            Args:
                messages: List of message dictionaries (role, content)
                provider: LLM provider to use (openai, anthropic, etc.)
                model: Model name to use (defaults to provider's default)
                max_tokens: Maximum tokens to generate
                temperature: Temperature parameter (0.0-1.0)
                system_prompt: Optional system prompt to prepend
                additional_params: Additional provider-specific parameters
                
            Returns:
                Dictionary containing completion text and metadata
            """
            start_time = time.time()
            
            # Validate messages format
            if not messages or not isinstance(messages, list):
                return {
                    "error": "Invalid messages format. Must be a non-empty list of message objects.",
                    "text": None,
                }
                
            for msg in messages:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    return {
                        "error": "Invalid message format. Each message must have 'role' and 'content'.",
                        "text": None,
                    }
            
            # Get provider instance
            try:
                provider_instance = get_provider(provider)
            except Exception as e:
                logger.error(
                    f"Failed to initialize provider: {str(e)}",
                    emoji_key="error",
                    provider=provider
                )
                return {
                    "error": f"Failed to initialize provider: {str(e)}",
                    "text": None,
                }
            
            # Set default additional params
            additional_params = additional_params or {}
            
            # Add system prompt if provided
            if system_prompt:
                # Different handling depending on provider
                if provider == Provider.ANTHROPIC.value:
                    additional_params["system"] = system_prompt
                else:
                    # For other providers, prepend system message to messages
                    messages = [{"role": "system", "content": system_prompt}] + messages
            
            try:
                # Generate completion
                result = await provider_instance.generate_completion(
                    prompt="",  # Not used with messages
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                    **additional_params
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log success
                logger.success(
                    f"Chat completion generated successfully with {provider}/{result.model}",
                    emoji_key=TaskType.COMPLETION.value,
                    tokens={
                        "input": result.input_tokens,
                        "output": result.output_tokens
                    },
                    cost=result.cost,
                    time=processing_time
                )
                
                # Return standardized result
                return {
                    "text": result.text,
                    "model": result.model,
                    "provider": provider,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "cost": result.cost,
                    "processing_time": processing_time,
                }
                
            except Exception as e:
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log error
                logger.error(
                    f"Chat completion failed: {str(e)}",
                    emoji_key="error",
                    provider=provider,
                    model=model,
                    time=processing_time
                )
                
                # Return error
                return {
                    "error": f"Chat completion failed: {str(e)}",
                    "text": None,
                    "provider": provider,
                    "model": model,
                    "processing_time": processing_time,
                }
                
        @self.mcp.tool()
        @with_cache(ttl=7 * 24 * 60 * 60)  # Cache for 7 days
        @with_tool_metrics
        async def multi_completion(
            prompt: str,
            providers: List[Dict[str, Any]],
            max_concurrency: int = 3,
            timeout: Optional[float] = 30.0,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Generate completions from multiple providers and compare results.
            
            Args:
                prompt: Input text prompt
                providers: List of provider configurations
                max_concurrency: Maximum number of concurrent requests
                timeout: Maximum time to wait for completions in seconds
                
            Returns:
                Dictionary containing all completion results
            """
            start_time = time.time()
            
            # Validate providers format
            if not providers or not isinstance(providers, list):
                return {
                    "error": "Invalid providers format. Must be a non-empty list.",
                    "results": {},
                }
                
            # Create tasks for each provider
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def process_provider(provider_config):
                async with semaphore:
                    provider_name = provider_config.get("provider", Provider.OPENAI.value)
                    model = provider_config.get("model")
                    max_tokens = provider_config.get("max_tokens")
                    temperature = provider_config.get("temperature", 0.7)
                    additional_params = provider_config.get("additional_params", {})
                    
                    provider_key = f"{provider_name}/{model or 'default'}"
                    
                    try:
                        # Get provider instance
                        provider_instance = get_provider(provider_name)
                        
                        # Generate completion
                        result = await provider_instance.generate_completion(
                            prompt=prompt,
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **additional_params
                        )
                        
                        return {
                            "provider_key": provider_key,
                            "success": True,
                            "text": result.text,
                            "model": result.model,
                            "provider": provider_name,
                            "tokens": {
                                "input": result.input_tokens,
                                "output": result.output_tokens,
                                "total": result.total_tokens,
                            },
                            "cost": result.cost,
                        }
                        
                    except Exception as e:
                        logger.error(
                            f"Completion failed for {provider_key}: {str(e)}",
                            emoji_key="error",
                            provider=provider_name,
                            model=model
                        )
                        
                        return {
                            "provider_key": provider_key,
                            "success": False,
                            "error": str(e),
                            "provider": provider_name,
                            "model": model,
                        }
            
            # Create tasks
            for provider_config in providers:
                task = process_provider(provider_config)
                tasks.append(task)
                
            # Wait for all tasks (with timeout if specified)
            if timeout:
                # Run tasks with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Get results from completed tasks
                    results = []
                    for task in tasks:
                        if task.done():
                            try:
                                results.append(task.result())
                            except Exception as e:
                                # Handle exceptions from tasks
                                results.append({
                                    "provider_key": "unknown",
                                    "success": False,
                                    "error": str(e),
                                })
                    
                    # Add timeout entries for incomplete tasks
                    incomplete_count = len(providers) - len(results)
                    if incomplete_count > 0:
                        logger.warning(
                            f"{incomplete_count} providers timed out after {timeout}s",
                            emoji_key="warning",
                            time=timeout
                        )
            else:
                # Run tasks without timeout
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
            # Process results
            processed_results = {}
            successful_count = 0
            
            for result in results:
                # Handle exceptions
                if isinstance(result, Exception):
                    continue
                    
                if result["success"]:
                    successful_count += 1
                    
                processed_results[result["provider_key"]] = result
                
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log completion
            logger.success(
                f"Multi-completion finished: {successful_count}/{len(providers)} successful",
                emoji_key="success",
                time=processing_time
            )
            
            # Return all results
            return {
                "results": processed_results,
                "successful_count": successful_count,
                "total_providers": len(providers),
                "processing_time": processing_time,
            }