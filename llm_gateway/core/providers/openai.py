"""OpenAI provider implementation."""
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from openai import AsyncOpenAI

from llm_gateway.constants import DEFAULT_MODELS, Provider, COST_PER_MILLION_TOKENS
from llm_gateway.core.providers.base import BaseProvider, ModelResponse
from llm_gateway.utils import get_logger
from llm_gateway.config import get_config

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("llm_gateway.providers.openai")


class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI API."""
    
    provider_name = Provider.OPENAI.value
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            **kwargs: Additional options
        """
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = kwargs.get("base_url")
        self.organization = kwargs.get("organization")
        self.models_cache = None
        
    async def initialize(self) -> bool:
        """Initialize the OpenAI client.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key, 
                base_url=self.base_url,
                organization=self.organization,
            )
            
            # Skip API call if using a mock key (for tests)
            if self.api_key and "mock-" in self.api_key:
                self.logger.info(
                    "Using mock OpenAI key - skipping API validation",
                    emoji_key="mock"
                )
                return True
            
            # Test connection by listing models
            await self.list_models()
            
            self.logger.success(
                "OpenAI provider initialized successfully", 
                emoji_key="provider"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize OpenAI provider: {str(e)}", 
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
        """Generate a completion using OpenAI.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "gpt-4o")
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
        
        # Strip provider prefix if present (e.g., "openai/gpt-4o" -> "gpt-4o")
        if model.startswith(f"{self.provider_name}/"):
            original_model = model
            model = model.split("/", 1)[1]
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

        # --- Special handling for specific model parameter constraints ---
        if model == 'o3-mini':
            if 'temperature' in params:
                self.logger.debug(f"Removing unsupported 'temperature' parameter for model {model}")
                del params['temperature']
        elif model == 'o1-preview':
            current_temp = params.get('temperature')
            # Only allow temperature if it's explicitly set to 1.0, otherwise remove it to use API default.
            if current_temp is not None and current_temp != 1.0:
                self.logger.debug(f"Removing non-default 'temperature' ({current_temp}) for model {model}")
                del params['temperature']
        # --- End special handling ---
        
        # Log request
        self.logger.info(
            f"Generating completion with OpenAI model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt)
        )
        
        try:
            # API call with timing
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
                "OpenAI completion successful",
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
                f"OpenAI completion failed: {str(e)}",
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
        """Generate a streaming completion using OpenAI.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "gpt-4o")
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
        
        # Strip provider prefix if present (e.g., "openai/gpt-4o" -> "gpt-4o")
        if model.startswith(f"{self.provider_name}/"):
            original_model = model
            model = model.split("/", 1)[1]
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
            f"Generating streaming completion with OpenAI model {model}",
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
                "OpenAI streaming completion successful",
                emoji_key="success",
                model=model,
                chunks=total_chunks,
                time=processing_time
            )
            
        except Exception as e:
            self.logger.error(
                f"OpenAI streaming completion failed: {str(e)}",
                emoji_key="error",
                model=model
            )
            raise
            
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available OpenAI models.
        
        Returns:
            List of model information dictionaries
        """
        if self.models_cache:
            return self.models_cache
            
        try:
            if not self.client:
                await self.initialize()
                
            # Fetch models from API
            response = await self.client.models.list()
            
            # Process response
            models = []
            for model in response.data:
                # Filter to relevant models (chat-capable GPT models)
                if model.id.startswith("gpt-"):
                    models.append({
                        "id": model.id,
                        "provider": self.provider_name,
                        "created": model.created,
                        "owned_by": model.owned_by,
                    })
            
            # Cache results
            self.models_cache = models
            
            return models
            
        except Exception as e:
            self.logger.error(
                f"Failed to list OpenAI models: {str(e)}",
                emoji_key="error"
            )
            
            # Return basic models on error
            return [
                {
                    "id": "gpt-4o",
                    "provider": self.provider_name,
                    "description": "Most capable GPT-4 model",
                },
                {
                    "id": "gpt-4.1-mini",
                    "provider": self.provider_name,
                    "description": "Smaller, efficient GPT-4 model",
                },
                {
                    "id": "gpt-4.1-mini",
                    "provider": self.provider_name,
                    "description": "Fast and cost-effective GPT model",
                },
            ]
            
    def get_default_model(self) -> str:
        """Get the default OpenAI model.
        
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
        return "gpt-4.1-mini"
        
    async def check_api_key(self) -> bool:
        """Check if the OpenAI API key is valid.
        
        Returns:
            bool: True if API key is valid
        """
        try:
            # Just list models as a simple validation
            await self.list_models()
            return True
        except Exception:
            return False