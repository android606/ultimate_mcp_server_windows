"""Google Gemini provider implementation."""
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from google import genai

from llm_gateway.constants import DEFAULT_MODELS, Provider, COST_PER_MILLION_TOKENS
from llm_gateway.core.providers.base import BaseProvider, ModelResponse
from llm_gateway.utils import get_logger

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("llm_gateway.providers.gemini")


class GeminiProvider(BaseProvider):
    """Provider implementation for Google Gemini API."""
    
    provider_name = Provider.GEMINI.value
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Gemini provider.
        
        Args:
            api_key: Google API key
            **kwargs: Additional options
        """
        super().__init__(api_key=api_key, **kwargs)
        self.models_cache = None
        
    async def initialize(self) -> bool:
        """Initialize the Gemini client.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Skip real API calls if using mock key for tests
            if self.api_key and "mock-" in self.api_key:
                self.logger.info(
                    "Using mock Gemini key - skipping API initialization",
                    emoji_key="mock"
                )
                self.client = {"mock_client": True}
                return True
                
            # Create a client instance instead of configuring globally
            self.client = genai.Client(
                api_key=self.api_key,
                http_options={"api_version": "v1alpha"}
            )
            
            self.logger.success(
                "Gemini provider initialized successfully", 
                emoji_key="provider"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize Gemini provider: {str(e)}", 
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
        """Generate a completion using Google Gemini.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "gemini-2.0-flash-lite")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse: Standardized response
            
        Raises:
            Exception: If API call fails
        """
        if not self.client:
            await self.initialize()
            
        # Use default model if not specified
        model = model or self.get_default_model()
        
        # Strip provider prefix if present (e.g., "gemini:gemini-2.0-pro" -> "gemini-2.0-pro")
        if ":" in model:
            original_model = model
            model = model.split(":", 1)[1]
            self.logger.debug(f"Stripped provider prefix from model name: {original_model} -> {model}")
        
        # Log request
        self.logger.info(
            f"Generating completion with Gemini model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt)
        )
        
        start_time = time.time()
        
        try:
            # Check if we're using a mock client for testing
            if isinstance(self.client, dict) and self.client.get("mock_client"):
                # Return mock response for tests
                completion_text = "Mock Gemini response for testing"
                processing_time = 0.1
            else:
                # Use direct models.generate_content approach with correct parameters
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    **kwargs
                )
                
                processing_time = time.time() - start_time
                
                # Extract response text
                completion_text = response.text
            
            # Estimate token usage (Gemini doesn't provide token counts)
            # Roughly 4 characters per token as a crude approximation
            char_to_token_ratio = 4.0
            estimated_input_tokens = len(prompt) / char_to_token_ratio
            estimated_output_tokens = len(completion_text) / char_to_token_ratio
            
            # Create standardized response
            result = ModelResponse(
                text=completion_text,
                model=model,
                provider=self.provider_name,
                input_tokens=int(estimated_input_tokens),
                output_tokens=int(estimated_output_tokens),
                processing_time=processing_time,
                raw_response=None,  # Don't need raw response for tests
                metadata={"token_count_estimated": True}
            )
            
            # Log success
            self.logger.success(
                "Gemini completion successful",
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
                f"Gemini completion failed: {str(e)}",
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
        """Generate a streaming completion using Google Gemini.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "gemini-2.0-flash-lite")
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
        
        # Strip provider prefix if present (e.g., "gemini:gemini-2.0-pro" -> "gemini-2.0-pro")
        if ":" in model:
            original_model = model
            model = model.split(":", 1)[1]
            self.logger.debug(f"Stripped provider prefix from model name (stream): {original_model} -> {model}")
        
        # Prepare generation config
        generation_config = {
            "temperature": temperature,
        }
        
        # Add max_tokens if specified (Gemini uses max_output_tokens)
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        # Log request
        self.logger.info(
            f"Generating streaming completion with Gemini model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt)
        )
        
        start_time = time.time()
        total_chunks = 0
        
        try:
            # Use the streaming approach with the client's models API
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                stream=True,
                **kwargs
            )
            
            # Process the stream
            for chunk in response:
                total_chunks += 1
                
                # Extract content from the chunk
                chunk_text = chunk.text if hasattr(chunk, 'text') else ""
                
                # Metadata for this chunk
                metadata = {
                    "model": model,
                    "provider": self.provider_name,
                    "chunk_index": total_chunks,
                }
                
                yield chunk_text, metadata
                
            # Log success
            processing_time = time.time() - start_time
            self.logger.success(
                "Gemini streaming completion successful",
                emoji_key="success",
                model=model,
                chunks=total_chunks,
                time=processing_time
            )
            
        except Exception as e:
            self.logger.error(
                f"Gemini streaming completion failed: {str(e)}",
                emoji_key="error",
                model=model
            )
            raise
            
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Gemini models.
        
        Returns:
            List of model information dictionaries
        """
        # Gemini doesn't have a comprehensive models endpoint, so we return a static list
        if self.models_cache:
            return self.models_cache
            
        models = [
            {
                "id": "gemini-2.0-flash-lite",
                "provider": self.provider_name,
                "description": "Fastest and most efficient Gemini model",
            },
            {
                "id": "gemini-2.0-flash",
                "provider": self.provider_name,
                "description": "Fast Gemini model with good quality",
            },
            {
                "id": "gemini-2.0-pro",
                "provider": self.provider_name,
                "description": "More capable Gemini model",
            },
            {
                "id": "gemini-2.5-pro-preview-03-25",
                "provider": self.provider_name,
                "description": "Most capable Gemini model",
            },
        ]
        
        # Cache results
        self.models_cache = models
        
        return models
            
    def get_default_model(self) -> str:
        """Get the default Gemini model.
        
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
        return "gemini-2.0-flash-lite"
        
    async def check_api_key(self) -> bool:
        """Check if the Gemini API key is valid.
        
        Returns:
            bool: True if API key is valid
        """
        try:
            # Try listing models to validate the API key
            # Use the client's models API to check if API key is valid
            self.client.models.list()
            return True
        except Exception:
            return False