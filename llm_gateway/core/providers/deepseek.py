"""DeepSeek provider implementation."""
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from openai import AsyncOpenAI

from llm_gateway.constants import DEFAULT_MODELS, Provider, COST_PER_MILLION_TOKENS
from llm_gateway.core.providers.base import BaseProvider, ModelResponse
from llm_gateway.utils import get_logger
from llm_gateway.config import get_config

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("llm_gateway.providers.deepseek")


class DeepSeekProvider(BaseProvider):
    """Provider implementation for DeepSeek API (using OpenAI-compatible interface)."""
    
    provider_name = Provider.DEEPSEEK.value
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key
            **kwargs: Additional options
        """
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = kwargs.get("base_url", "https://api.deepseek.com")
        self.models_cache = None
        
    async def initialize(self) -> bool:
        """Initialize the DeepSeek client.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # DeepSeek uses OpenAI-compatible API
            self.client = AsyncOpenAI(
                api_key=self.api_key, 
                base_url=self.base_url,
            )
            
            self.logger.success(
                "DeepSeek provider initialized successfully", 
                emoji_key="provider"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize DeepSeek provider: {str(e)}", 
                emoji_key="error"
            )
            return False
        
    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate a completion using DeepSeek.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "deepseek-chat")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse with completion result
            
        Raises:
            Exception: If API call fails
        """
        if not self.client:
            await self.initialize()
            
        # Use default model if not specified
        model = model or self.get_default_model()
        
        # Strip provider prefix if present (e.g., "deepseek:deepseek-chat" -> "deepseek-chat")
        if ":" in model:
            original_model = model
            model = model.split(":", 1)[1]
            self.logger.debug(f"Stripped provider prefix from model name: {original_model} -> {model}")
        
        # Create messages
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]
        
        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
            
        # Add any additional parameters
        params.update(kwargs)
        
        # Log request
        self.logger.info(
            f"Generating completion with DeepSeek model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt)
        )
        
        try:
            # Make API call with timing
            response, processing_time = await self.process_with_timer(
                self.client.chat.completions.create, **params
            )
            
            # Extract response text
            completion_text = response.choices[0].message.content
            
            # Create standardized response
            result = ModelResponse(
                text=completion_text,
                model=model,
                provider=self.provider_name,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                processing_time=processing_time,
                raw_response=response,
            )
            
            # Log success
            self.logger.success(
                "DeepSeek completion successful",
                emoji_key="success",
                model=model,
                tokens={
                    "input": result.input_tokens,
                    "output": result.output_tokens
                },
                cost=result.cost,
                time=result.processing_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"DeepSeek completion failed: {str(e)}",
                emoji_key="error",
                model=model
            )
            raise
            
    async def generate_completion_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion using DeepSeek.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "deepseek-chat")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters
            
        Yields:
            Tuple of (text_chunk, metadata)
            
        Raises:
            Exception: If API call fails
        """
        if not self.client:
            await self.initialize()
            
        # Use default model if not specified
        model = model or self.get_default_model()
        
        # Strip provider prefix if present (e.g., "deepseek:deepseek-chat" -> "deepseek-chat")
        if ":" in model:
            original_model = model
            model = model.split(":", 1)[1]
            self.logger.debug(f"Stripped provider prefix from model name (stream): {original_model} -> {model}")
        
        # Create messages
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]
        
        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
            
        # Add any additional parameters
        params.update(kwargs)
        
        # Log request
        self.logger.info(
            f"Generating streaming completion with DeepSeek model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt)
        )
        
        start_time = time.time()
        total_chunks = 0
        
        try:
            # Make streaming API call
            stream = await self.client.chat.completions.create(**params)
            
            # Process the stream
            async for chunk in stream:
                total_chunks += 1
                
                # Extract content from the chunk
                delta = chunk.choices[0].delta
                content = delta.content or ""
                
                # Metadata for this chunk
                metadata = {
                    "model": model,
                    "provider": self.provider_name,
                    "chunk_index": total_chunks,
                    "finish_reason": chunk.choices[0].finish_reason,
                }
                
                yield content, metadata
                
            # Log success
            processing_time = time.time() - start_time
            self.logger.success(
                "DeepSeek streaming completion successful",
                emoji_key="success",
                model=model,
                chunks=total_chunks,
                time=processing_time
            )
            
        except Exception as e:
            self.logger.error(
                f"DeepSeek streaming completion failed: {str(e)}",
                emoji_key="error",
                model=model
            )
            raise
            
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available DeepSeek models.
        
        Returns:
            List of model information dictionaries
        """
        # DeepSeek doesn't have a comprehensive models endpoint, so we return a static list
        if self.models_cache:
            return self.models_cache
            
        models = [
            {
                "id": "deepseek-chat",
                "provider": self.provider_name,
                "description": "General-purpose chat model",
            },
            {
                "id": "deepseek-reasoner",
                "provider": self.provider_name,
                "description": "Enhanced reasoning capabilities",
            },
        ]
        
        # Cache results
        self.models_cache = models
        
        return models
            
    def get_default_model(self) -> str:
        """Get the default DeepSeek model.
        
        Returns:
            Default model name
        """
        from llm_gateway.config import get_config
        
        # Safely get from config if available
        try:
            config = get_config()
            provider_config = getattr(config, 'providers', {}).get(self.provider_name, None)
            if provider_config and provider_config.default_model:
                return provider_config.default_model
        except (AttributeError, TypeError):
            # Handle case when providers attribute doesn't exist or isn't a dict
            pass
            
        # Otherwise return hard-coded default
        return "deepseek-chat"
        
    async def check_api_key(self) -> bool:
        """Check if the DeepSeek API key is valid.
        
        Returns:
            bool: True if API key is valid
        """
        try:
            # Try a simple completion to validate the API key
            await self.client.chat.completions.create(
                model=self.get_default_model(),
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False