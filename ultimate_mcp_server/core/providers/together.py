"""Together AI provider implementation.

Together AI provides an OpenAI-compatible API for various open-source models.
This provider allows access to models like Llama, Mistral, and others through
Together AI's cloud infrastructure.
"""
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import BaseProvider, ModelResponse
from ultimate_mcp_server.utils import get_logger

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("ultimate_mcp_server.providers.together")


class TogetherProvider(BaseProvider):
    """Provider implementation for Together AI API."""
    
    provider_name = Provider.TOGETHER.value
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Together AI provider.
        
        Args:
            api_key: Together AI API key
            **kwargs: Additional options including base_url, max_tokens
        """
        super().__init__(api_key=api_key, **kwargs)
        
        # Together AI uses OpenAI-compatible API
        self.base_url = kwargs.get("base_url", "https://api.together.xyz/v1")
        self.max_tokens = kwargs.get("max_tokens")
        self.models_cache = None
        
        # Load configuration from environment variables if not provided
        if not self.api_key:
            self.api_key = os.getenv("TOGETHERAI_API_KEY")
        
        if not kwargs.get("base_url"):
            endpoint = os.getenv("TOGETHERAI_API_ENDPOINT")
            if endpoint:
                self.base_url = endpoint
                
        if not self.max_tokens:
            max_tokens_env = os.getenv("TOGETHERAI_MAX_TOKENS")
            if max_tokens_env:
                try:
                    self.max_tokens = int(max_tokens_env)
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid TOGETHERAI_MAX_TOKENS value: {max_tokens_env}")
        
    async def initialize(self) -> bool:
        """Initialize the Together AI client.
        
        Returns:
            bool: True if initialization was successful
        """
        if not self.api_key:
            self.logger.error(
                "Together AI API key not provided. Set TOGETHERAI_API_KEY environment variable.",
                emoji_key="error"
            )
            return False
            
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key, 
                base_url=self.base_url,
            )
            
            # Skip API call if using a mock key (for tests)
            if self.api_key and "mock-" in self.api_key:
                self.logger.info(
                    "Using mock Together AI key - skipping API validation",
                    emoji_key="mock"
                )
                return True
            
            # Test connection by making a simple request
            # We'll use a very small request to minimize cost
            test_response = await self.client.chat.completions.create(
                model=self.get_default_model(),
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            
            if test_response:
                self.logger.success(
                    "Together AI provider initialized successfully", 
                    emoji_key="provider"
                )
                return True
            else:
                self.logger.error(
                    "Together AI API test failed - no response received",
                    emoji_key="error"
                )
                return False
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize Together AI provider: {str(e)}", 
                emoji_key="error"
            )
            return False
        
    async def generate_completion(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate a completion using Together AI.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse with completion result
            
        Raises:
            Exception: If API call fails
        """
        if not self.client:
            initialized = await self.initialize()
            if not initialized:
                raise RuntimeError("Failed to initialize Together AI provider")
            
        # Use default model if not specified
        model = model or self.get_default_model()
        
        # Strip provider prefix if present (e.g., "together/meta-llama..." -> "meta-llama...")
        if model.startswith(f"{self.provider_name}/"):
            original_model = model
            model = model.split("/", 1)[1]
            self.logger.debug(f"Stripped provider prefix from model name: {original_model} -> {model}")
        
        # Handle case when messages are provided instead of prompt (for chat_completion)
        messages = kwargs.pop("messages", None)
        
        # If neither prompt nor messages are provided, raise an error
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
            
        # Create messages if not already provided
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        
        # Use instance max_tokens if no max_tokens specified
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
            
        # Check for json_mode flag and remove it from kwargs
        json_mode = kwargs.pop("json_mode", False)
        if json_mode:
            # Use the correct response_format for JSON mode
            params["response_format"] = {"type": "json_object"}
            self.logger.debug("Setting response_format to JSON mode for Together AI")

        # Handle any legacy response_format passed directly, but prefer json_mode
        if "response_format" in kwargs and not json_mode:
             # Support both direct format object and type-only specification
             response_format = kwargs.pop("response_format")
             if isinstance(response_format, dict):
                 params["response_format"] = response_format
             elif isinstance(response_format, str) and response_format in ["json_object", "text"]:
                 params["response_format"] = {"type": response_format}
             self.logger.debug(f"Setting response_format from direct param: {params.get('response_format')}")

        # Add any remaining additional parameters
        params.update(kwargs)
        
        # Log request
        prompt_length = len(prompt) if prompt else sum(len(m.get("content", "")) for m in messages)
        self.logger.info(
            f"Generating completion with Together AI model {model}",
            emoji_key=self.provider_name,
            prompt_length=prompt_length,
            json_mode=json_mode
        )
        
        try:
            # API call with timing
            response, processing_time = await self.process_with_timer(
                self.client.chat.completions.create, **params
            )
            
            # Extract response text
            completion_text = response.choices[0].message.content
            
            # Create message object for chat_completion
            message = {
                "role": "assistant",
                "content": completion_text
            }
            
            # Create standardized response
            result = ModelResponse(
                text=completion_text,
                model=f"{self.provider_name}/{model}",
                provider=self.provider_name,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                processing_time=processing_time,
                raw_response=response,
            )
            
            # Add message to result for chat_completion
            result.message = message
            
            # Log success
            self.logger.success(
                "Together AI completion successful",
                emoji_key="completion_success",
                tokens={"input": result.input_tokens, "output": result.output_tokens},
                time=processing_time,
                model=model
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Together AI API error: {str(e)}",
                emoji_key="error",
                model=model
            )
            raise
        
    async def generate_completion_stream(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion using Together AI.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters
            
        Yields:
            Tuples of (text_chunk, metadata_dict)
        """
        if not self.client:
            initialized = await self.initialize()
            if not initialized:
                yield "", {
                    "error": "Failed to initialize Together AI provider",
                    "finished": True,
                    "provider": self.provider_name,
                    "model": model or self.get_default_model()
                }
                return
            
        # Use default model if not specified
        model = model or self.get_default_model()
        
        # Strip provider prefix if present
        if model.startswith(f"{self.provider_name}/"):
            model = model.split("/", 1)[1]
        
        # Handle case when messages are provided instead of prompt
        messages = kwargs.pop("messages", None)
        
        # If neither prompt nor messages are provided, yield error
        if prompt is None and not messages:
            yield "", {
                "error": "Either 'prompt' or 'messages' must be provided",
                "finished": True,
                "provider": self.provider_name,
                "model": model
            }
            return
            
        # Create messages if not already provided
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        
        # Use instance max_tokens if no max_tokens specified
        if max_tokens is None:
            max_tokens = self.max_tokens
        
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
        
        # Check for json_mode flag and remove it from kwargs
        json_mode = kwargs.pop("json_mode", False)
        if json_mode:
            params["response_format"] = {"type": "json_object"}
            
        # Add remaining parameters
        params.update(kwargs)
        
        # Log request
        prompt_length = len(prompt) if prompt else sum(len(m.get("content", "")) for m in messages)
        self.logger.info(
            f"Generating streaming completion with Together AI model {model}",
            emoji_key=self.provider_name,
            prompt_length=prompt_length,
            json_mode=json_mode
        )
        
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0
        chunk_count = 0
        
        try:
            # Create streaming response
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                chunk_count += 1
                
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    content = delta.content if delta else None
                    
                    if content:
                        # Yield content chunk
                        yield content, {
                            "provider": self.provider_name,
                            "model": f"{self.provider_name}/{model}",
                            "chunk_index": chunk_count,
                            "finished": False,
                        }
                
                # Check for usage information in the final chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
            
            # Final metadata chunk
            processing_time = time.time() - start_time
            final_metadata = {
                "model": f"{self.provider_name}/{model}",
                "provider": self.provider_name,
                "finished": True,
                "finish_reason": "stop",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "processing_time": processing_time,
            }
            
            yield "", final_metadata
            
            # Log success
            self.logger.success(
                "Together AI streaming completion successful",
                emoji_key="completion_success",
                tokens={"input": input_tokens, "output": output_tokens},
                time=processing_time,
                model=model
            )
            
        except Exception as e:
            # Yield error
            yield "", {
                "error": f"Together AI streaming error: {str(e)}",
                "finished": True,
                "provider": self.provider_name,
                "model": f"{self.provider_name}/{model}",
            }
            
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from Together AI.
        
        Returns:
            List of model information dictionaries
        """
        if not self.client:
            initialized = await self.initialize()
            if not initialized:
                return []
                
        try:
            # Together AI returns a raw list, not an OpenAI-compatible response
            # So we need to make the API call directly using httpx or requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                models_data = response.json()
            
            models = []
            
            # Handle the actual response format from Together AI
            if isinstance(models_data, list):
                # Direct list of models
                for model in models_data:
                    if isinstance(model, dict):
                        model_id = model.get('id', model.get('name', 'unknown'))
                        models.append({
                            "id": model_id,
                            "name": model_id,
                            "description": f"Together AI model: {model_id}",
                            "provider": self.provider_name,
                            "created": model.get('created', None),
                            "context_length": model.get('context_length', None),
                            "pricing": model.get('pricing', {}),
                        })
            elif isinstance(models_data, dict) and 'data' in models_data:
                # OpenAI-compatible format (fallback)
                for model in models_data['data']:
                    if isinstance(model, dict):
                        model_id = model.get('id', model.get('name', 'unknown'))
                        models.append({
                            "id": model_id,
                            "name": model_id,
                            "description": f"Together AI model: {model_id}",
                            "provider": self.provider_name,
                            "created": model.get('created', None),
                            "context_length": model.get('context_length', None),
                            "pricing": model.get('pricing', {}),
                        })
            
            self.models_cache = models
            
            self.logger.info(
                f"Listed {len(models)} Together AI models",
                emoji_key="model"
            )
            
            return models
            
        except Exception as e:
            self.logger.error(
                f"Failed to list Together AI models: {str(e)}",
                emoji_key="error"
            )
            return []
    
    def get_default_model(self) -> str:
        """Get the default model for Together AI."""
        return "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        
    async def check_api_key(self) -> bool:
        """Check if the Together AI API key is valid.
        
        Returns:
            bool: True if API key is valid and accessible
        """
        if not self.api_key:
            return False
            
        try:
            # Try to list models as a simple API test
            models = await self.list_models()
            return len(models) > 0
            
        except Exception as e:
            self.logger.warning(
                f"Together AI API key validation failed: {str(e)}",
                emoji_key="warning"
            )
            return False 