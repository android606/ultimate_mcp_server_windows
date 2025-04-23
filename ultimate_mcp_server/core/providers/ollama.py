"""Ollama provider implementation for the Ultimate MCP Server.

This module implements the Ollama provider, enabling interaction with locally running
Ollama models through a standard interface. Ollama is an open-source framework for
running LLMs locally with minimal setup.

The implementation supports:
- Text completion (generate) and chat completations
- Streaming responses
- Model listing and information retrieval
- Embeddings generation
- Cost tracking (estimated since Ollama is free to use locally)

Ollama must be installed and running locally (by default on localhost:11434)
for this provider to work properly.
"""

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
from pydantic import BaseModel

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.constants import COST_PER_MILLION_TOKENS, Provider
from ultimate_mcp_server.core.providers.base import (
    BaseProvider,
    ModelResponse,
)
from ultimate_mcp_server.exceptions import ProviderError
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.providers.ollama")


# Define the Model class locally since it's not available in base.py
class Model(dict):
    """Model information returned by providers."""
    
    def __init__(self, id: str, name: str, description: str, provider: str, **kwargs):
        """Initialize a model info dictionary.
        
        Args:
            id: Model identifier (e.g., "llama3.2")
            name: Human-readable model name
            description: Longer description of the model
            provider: Provider name
            **kwargs: Additional model metadata
        """
        super().__init__(
            id=id,
            name=name,
            description=description,
            provider=provider,
            **kwargs
        )


# Define ProviderFeatures locally since it's not available in base.py
class ProviderFeatures:
    """Features supported by a provider."""
    
    def __init__(
        self,
        supports_chat_completions: bool = False,
        supports_streaming: bool = False,
        supports_function_calling: bool = False,
        supports_multiple_functions: bool = False,
        supports_embeddings: bool = False,
        supports_json_mode: bool = False,
        max_retries: int = 3,
    ):
        """Initialize provider features.
        
        Args:
            supports_chat_completions: Whether the provider supports chat completions
            supports_streaming: Whether the provider supports streaming responses
            supports_function_calling: Whether the provider supports function calling
            supports_multiple_functions: Whether the provider supports multiple functions
            supports_embeddings: Whether the provider supports embeddings
            supports_json_mode: Whether the provider supports JSON mode
            max_retries: Maximum number of retries for failed requests
        """
        self.supports_chat_completions = supports_chat_completions
        self.supports_streaming = supports_streaming
        self.supports_function_calling = supports_function_calling
        self.supports_multiple_functions = supports_multiple_functions
        self.supports_embeddings = supports_embeddings
        self.supports_json_mode = supports_json_mode
        self.max_retries = max_retries


# Define ProviderStatus locally since it's not available in base.py
class ProviderStatus:
    """Status information for a provider."""
    
    def __init__(
        self,
        name: str,
        enabled: bool = False,
        available: bool = False,
        api_key_configured: bool = False,
        features: Optional[ProviderFeatures] = None,
        default_model: Optional[str] = None,
    ):
        """Initialize provider status.
        
        Args:
            name: Provider name
            enabled: Whether the provider is enabled
            available: Whether the provider is available
            api_key_configured: Whether an API key is configured
            features: Provider features
            default_model: Default model for the provider
        """
        self.name = name
        self.enabled = enabled
        self.available = available
        self.api_key_configured = api_key_configured
        self.features = features
        self.default_model = default_model


class OllamaConfig(BaseModel):
    """Configuration for the Ollama provider."""

    # API endpoint (default is localhost:11434)
    api_url: str = "http://127.0.0.1:11434"
    
    # Default model to use if none specified
    default_model: str = "llama3.2"
    
    # Timeout settings
    request_timeout: int = 300
    
    # Whether this provider is enabled
    enabled: bool = True


class OllamaProvider(BaseProvider):
    """
    Provider implementation for Ollama.
    
    Ollama allows running open-source language models locally with minimal setup.
    This provider implementation connects to a locally running Ollama instance and
    provides a standard interface for generating completions and embeddings.
    
    Unlike cloud providers, Ollama runs models locally, so:
    - No API key is required
    - Costs are estimated (since running locally is free)
    - Model availability depends on what models have been downloaded locally
    
    The Ollama provider supports both chat completions and text completions,
    as well as streaming responses. It requires that the Ollama service is 
    running and accessible at the configured endpoint.
    """

    provider_name = Provider.OLLAMA
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Ollama provider.
        
        Args:
            api_key: Not used by Ollama, included for API compatibility with other providers
            **kwargs: Additional provider-specific options
        """
        # Skip API key, it's not used by Ollama but we accept it for compatibility
        super().__init__()
        self.logger = get_logger(f"provider.{Provider.OLLAMA}")
        self.logger.info("Initializing Ollama provider...")
        self.config = self._load_config()
        self.logger.info(f"Loaded config: API URL={self.config.api_url}, default_model={self.config.default_model}, enabled={self.config.enabled}")
        
        # Initialize session to None, we'll create it when needed
        self._session = None
        
        self.client_session_params = {
            "timeout": aiohttp.ClientTimeout(total=self.config.request_timeout)
        }
        
        # Unlike other providers, Ollama doesn't require an API key
        # But we'll still set this flag to True for consistency
        self._api_key_configured = True
        self._initialized = False
        
        # Set feature flags
        self.features = ProviderFeatures(
            supports_chat_completions=True,
            supports_streaming=True,
            supports_function_calling=False,  # Ollama doesn't support function calling natively
            supports_multiple_functions=False,
            supports_embeddings=True,
            supports_json_mode=False,  # Ollama doesn't have a dedicated JSON mode
            max_retries=3,
        )
        
        # Set default costs for Ollama models (very low estimated costs)
        # Since Ollama runs locally, the actual cost is hardware usage/electricity
        # We'll use very low values for tracking purposes
        self._default_token_cost = {
            "input": 0.0001,  # $0.0001 per 1M tokens (effectively free)
            "output": 0.0001,  # $0.0001 per 1M tokens (effectively free)
        }
        self.logger.info("Ollama provider initialization completed")

    @property
    async def session(self) -> aiohttp.ClientSession:
        """Get the current session or create a new one if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(**self.client_session_params)
        return self._session
    
    async def __aenter__(self):
        """Enter async context, initializing the provider."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context, ensuring proper shutdown."""
        await self.shutdown()

    async def initialize(self) -> bool:
        """Initialize the provider, creating a new HTTP session.
        
        This method handles the initialization of the connection to Ollama.
        If Ollama isn't available (not installed or not running),
        it will gracefully report the issue without spamming errors.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Create a temporary session with a short timeout for the initial check
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as check_session:
                # Try to connect to Ollama and check if it's running
                self.logger.info(f"Attempting to connect to Ollama at {self.config.api_url}/api/tags", emoji_key="provider")
                
                # First try the configured URL
                try:
                    async with check_session.get(f"{self.config.api_url}/api/tags", timeout=5.0) as response:
                        if response.status == 200:
                            # Ollama is running, we'll create the main session when needed later
                            self.logger.info("Ollama service is available and running", emoji_key="provider")
                            self._initialized = True
                            return True
                        else:
                            self.logger.warning(
                                f"Ollama service responded with status {response.status}. "
                                "The service might be misconfigured.",
                                emoji_key="warning"
                            )
                except aiohttp.ClientConnectionError:
                    # Try alternate localhost format (127.0.0.1 instead of localhost or vice versa)
                    alternate_url = self.config.api_url.replace("localhost", "127.0.0.1") if "localhost" in self.config.api_url else self.config.api_url.replace("127.0.0.1", "localhost")
                    self.logger.info(f"Connection failed, trying alternate URL: {alternate_url}", emoji_key="provider")
                    
                    try:
                        async with check_session.get(f"{alternate_url}/api/tags", timeout=5.0) as response:
                            if response.status == 200:
                                # Update the config to use the working URL
                                self.logger.info(f"Connected successfully using alternate URL: {alternate_url}", emoji_key="provider")
                                self.config.api_url = alternate_url
                                self._initialized = True
                                return True
                            else:
                                self.logger.warning(
                                    f"Ollama service at alternate URL responded with status {response.status}. "
                                    "The service might be misconfigured.",
                                    emoji_key="warning"
                                )
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        self.logger.warning(
                            f"Could not connect to alternate URL: {str(e)}. "
                            "Make sure Ollama is installed and running: https://ollama.com/download",
                            emoji_key="warning"
                        )
                except aiohttp.ClientError as e:
                    # Other client errors
                    self.logger.warning(
                        f"Could not connect to Ollama service: {str(e)}. "
                        "Make sure Ollama is installed and running: https://ollama.com/download",
                        emoji_key="warning"
                    )
                except asyncio.TimeoutError:
                    # Timeout indicates Ollama is likely not responding
                    self.logger.warning(
                        "Connection to Ollama service timed out. "
                        "Make sure Ollama is installed and running: https://ollama.com/download",
                        emoji_key="warning"
                    )
            
            # If we got here, Ollama is not available
            self._initialized = False
            return False
            
        except Exception as e:
            # Catch any other exceptions to avoid spamming errors
            self.logger.error(f"Unexpected error initializing Ollama provider: {str(e)}", emoji_key="error")
            self._initialized = False
            return False

    async def shutdown(self) -> None:
        """Shutdown the provider, closing the HTTP session."""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
        except Exception as e:
            self.logger.warning(f"Error closing Ollama session during shutdown: {str(e)}", emoji_key="warning")
        finally:
            self._initialized = False

    def _load_config(self) -> OllamaConfig:
        """Load Ollama configuration from app configuration."""
        try:
            self.logger.info("Loading Ollama config from app configuration")
            config = get_config()
            # Print entire config for debugging
            self.logger.debug(f"Full config: {config}")
            
            if not hasattr(config, 'providers'):
                self.logger.warning("Config doesn't have 'providers' attribute")
                return OllamaConfig()
                
            if not hasattr(config.providers, Provider.OLLAMA):
                self.logger.warning(f"Config doesn't have '{Provider.OLLAMA}' provider configured")
                return OllamaConfig()
            
            provider_config = getattr(config.providers, Provider.OLLAMA, {})
            self.logger.info(f"Found provider config: {provider_config}")
            
            if hasattr(provider_config, 'dict'):
                self.logger.info("Provider config has 'dict' method, using it")
                return OllamaConfig(**provider_config.dict())
            else:
                self.logger.warning("Provider config doesn't have 'dict' method, attempting direct conversion")
                # Try to convert to dict directly
                config_dict = {}
                
                # Define mapping from ProviderConfig field names to OllamaConfig field names
                field_mapping = {
                    'base_url': 'api_url',       # ProviderConfig -> OllamaConfig
                    'default_model': 'default_model',
                    'timeout': 'request_timeout',
                    'enabled': 'enabled'
                }
                
                # Map fields from provider_config to OllamaConfig's expected field names
                for provider_key, ollama_key in field_mapping.items():
                    if hasattr(provider_config, provider_key):
                        config_dict[ollama_key] = getattr(provider_config, provider_key)
                        self.logger.info(f"Mapped {provider_key} to {ollama_key}: {getattr(provider_config, provider_key)}")
                
                self.logger.info(f"Created config dict: {config_dict}")
                return OllamaConfig(**config_dict)
        except Exception as e:
            self.logger.error(f"Error loading Ollama config: {e}", exc_info=True)
            return OllamaConfig()

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return self.config.default_model

    def get_status(self) -> ProviderStatus:
        """Get the current status of this provider."""
        return ProviderStatus(
            name=self.provider_name,
            enabled=self.config.enabled,
            available=self._initialized,
            api_key_configured=self._api_key_configured,
            features=self.features,
            default_model=self.get_default_model(),
        )

    async def check_api_key(self) -> bool:
        """
        Check if the Ollama service is accessible.
        
        Since Ollama doesn't use API keys, this just checks if the service is running.
        This check is designed to fail gracefully if Ollama is not installed or running,
        without causing cascading errors in the system.
        
        Returns:
            bool: True if Ollama service is running and accessible, False otherwise
        """
        if not self._initialized:
            try:
                # Attempt to initialize with a short timeout
                return await self.initialize()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Ollama during service check: {str(e)}", emoji_key="warning")
                return False
                
        try:
            # Use a dedicated session with short timeout for health check
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3.0)) as session:
                try:
                    async with session.get(f"{self.config.api_url}/api/tags") as response:
                        return response.status == 200
                except (aiohttp.ClientConnectionError, asyncio.TimeoutError, Exception) as e:
                    self.logger.warning(f"Ollama service check failed: {str(e)}", emoji_key="warning")
                    return False
        except Exception as e:
            self.logger.warning(f"Failed to create session for Ollama check: {str(e)}", emoji_key="warning")
            return False

    def _build_api_url(self, endpoint: str) -> str:
        """Build the full API URL for a given endpoint."""
        return f"{self.config.api_url}/api/{endpoint}"

    def _estimate_token_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the cost of a completion based on token counts.
        
        Since Ollama runs locally, the costs are just estimates and very low.
        """
        # Try to get model-specific costs if available
        model_costs = COST_PER_MILLION_TOKENS.get(model, self._default_token_cost)
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * model_costs.get("input", self._default_token_cost["input"])
        output_cost = (output_tokens / 1_000_000) * model_costs.get("output", self._default_token_cost["output"])
        
        return input_cost + output_cost

    async def list_models(self) -> List[Model]:
        """
        List all available models from Ollama.
        
        This method attempts to list all locally available Ollama models.
        If Ollama is not available or cannot be reached, it will return
        an empty list instead of raising an exception.
        
        Returns:
            List of available Ollama models, or empty list if Ollama is not available
        """
        if not self._initialized:
            try:
                initialized = await self.initialize()
                if not initialized:
                    self.logger.warning("Cannot list Ollama models because the service is not available", emoji_key="warning")
                    return []
            except Exception:
                self.logger.warning("Cannot list Ollama models because initialization failed", emoji_key="warning")
                return []
            
        try:
            # Create a dedicated session for this operation to avoid shared session issues
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0)) as session:
                return await self._fetch_models(session)
        except Exception as e:
            self.logger.warning(f"Error listing Ollama models: {str(e)}. The service may not be running.", emoji_key="warning")
            return []

    async def _fetch_models(self, session: aiohttp.ClientSession) -> List[Model]:
        """Fetch models using the provided session."""
        try:
            async with session.get(self._build_api_url("tags")) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to list Ollama models: {response.status}")
                    return []
                
                data = await response.json()
                models = []
                
                # Process the response
                for model_info in data.get("models", []):
                    model_id = model_info.get("name", "")
                    
                    # Extract additional info if available
                    description = f"Ollama model: {model_id}"
                    model_size = model_info.get("size", 0)
                    size_gb = None
                    
                    if model_size:
                        # Convert to GB for readability if size is provided in bytes
                        size_gb = model_size / (1024 * 1024 * 1024)
                        description += f" ({size_gb:.2f} GB)"
                    
                    models.append(Model(
                        id=model_id,
                        name=model_id,
                        description=description,
                        provider=self.provider_name,
                        size=f"{size_gb:.2f} GB" if size_gb else "Unknown"
                    ))
                
                return models
        except aiohttp.ClientConnectionError:
            self.logger.warning("Connection refused while listing Ollama models", emoji_key="warning")
            return []
        except asyncio.TimeoutError:
            self.logger.warning("Timeout while listing Ollama models", emoji_key="warning")
            return []
        except Exception as e:
            self.logger.warning(f"Error fetching Ollama models: {str(e)}", emoji_key="warning")
            return []

    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate a completion using the Ollama API.
        
        Args:
            prompt: The user prompt to generate a completion for.
            model: The model ID to use (defaults to provider's default).
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum number of tokens to generate.
            stop: List of strings that will stop generation when encountered.
            system: System message for chat-based models.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            A ModelResponse object with the generated text and metadata.
            
        Raises:
            ProviderError: If Ollama is not available or fails to generate a completion.
        """
        # Check if provider is initialized before attempting to generate
        if not self._initialized:
            try:
                initialized = await self.initialize()
                if not initialized:
                    # Return a clear error without raising an exception
                    return ModelResponse(
                        text="",
                        model=f"{self.provider_name}/{model or self.get_default_model()}",
                        provider=self.provider_name,
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        processing_time=0.0,
                        metadata={
                            "error": "Ollama service is not available. Make sure Ollama is installed and running: https://ollama.com/download"
                        }
                    )
            except Exception as e:
                # Return a clear error without raising an exception
                return ModelResponse(
                    text="",
                    model=f"{self.provider_name}/{model or self.get_default_model()}",
                    provider=self.provider_name,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    processing_time=0.0,
                    metadata={
                        "error": f"Failed to initialize Ollama provider: {str(e)}. Make sure Ollama is installed and running: https://ollama.com/download"
                    }
                )
            
        # Use default model if none specified
        model_id = model or self.get_default_model()
        
        # Remove any provider prefix if present (e.g., "ollama/llama3.2" -> "llama3.2")
        if "/" in model_id:
            parts = model_id.split("/", 1)
            if parts[0] == self.provider_name:
                model_id = parts[1]
        
        # Prepare the payload based on whether this is a chat or generate request
        # If system message is provided or if the model supports chat, use chat endpoint
        if system is not None or model_id.startswith(("llama", "gpt", "claude", "phi", "mistral")):
            # Use chat endpoint
            messages = []
            
            # Add system message if provided
            if system:
                messages.append({"role": "system", "content": system})
                
            # Add user message (the prompt)
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            
            # Add optional parameters
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if stop:
                payload["stop"] = stop
                
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in payload and value is not None:
                    payload[key] = value
            
            try:
                start_time = time.time()
                # Use a dedicated session for this request
                async with aiohttp.ClientSession(**self.client_session_params) as session:
                    async with session.post(self._build_api_url("chat"), json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            return ModelResponse(
                                text="",
                                model=f"{self.provider_name}/{model_id}",
                                provider=self.provider_name,
                                input_tokens=0,
                                output_tokens=0,
                                total_tokens=0,
                                processing_time=time.time() - start_time,
                                metadata={
                                    "error": f"Ollama API error: {response.status} - {error_text}",
                                    "cost": 0.0
                                }
                            )
                        
                        data = await response.json()
                        
                        # Extract response message
                        response_message = data.get("message", {})
                        content = response_message.get("content", "")
                        
                        # Extract token counts if available
                        input_tokens = data.get("prompt_eval_count", 0)  # Ollama's token count for input
                        output_tokens = data.get("eval_count", 0)  # Ollama's token count for output
                        
                        # If token counts are 0, make a rough estimate based on text length
                        if output_tokens == 0 and content:
                            # Rough estimate: ~1.3 tokens per word
                            output_tokens = len(content.split()) + len(content) // 10
                        
                        total_tokens = input_tokens + output_tokens
                        
                        # Calculate processing time
                        processing_time = time.time() - start_time
                        
                        # Calculate cost (estimated)
                        cost = self._estimate_token_cost(model_id, input_tokens, output_tokens)
                        
                        return ModelResponse(
                            text=content,
                            model=f"{self.provider_name}/{model_id}",
                            provider=self.provider_name,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                            processing_time=processing_time,
                            metadata={"cost": cost}
                        )
            except aiohttp.ClientConnectionError as e:
                processing_time = time.time() - start_time
                # Return connection error message
                return ModelResponse(
                    text="",
                    model=f"{self.provider_name}/{model_id}",
                    provider=self.provider_name,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    processing_time=processing_time,
                    metadata={
                        "error": f"Connection to Ollama failed: {str(e)}. Make sure Ollama is running and accessible.",
                        "cost": 0.0
                    }
                )
            except Exception as e:
                processing_time = time.time() - start_time
                # Return a proper error response instead of raising an exception
                if isinstance(e, ProviderError):
                    raise
                return ModelResponse(
                    text="",
                    model=f"{self.provider_name}/{model_id}",
                    provider=self.provider_name,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    processing_time=processing_time,
                    metadata={
                        "error": f"Error generating chat completion: {str(e)}",
                        "cost": 0.0
                    }
                )
        else:
            # Use generate endpoint for non-chat models
            payload = {
                "model": model_id,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
            }
            
            # Add optional parameters
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if stop:
                payload["stop"] = stop
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in payload and value is not None:
                    payload[key] = value
            
            try:
                start_time = time.time()
                # Use a dedicated session for this request
                async with aiohttp.ClientSession(**self.client_session_params) as session:
                    async with session.post(self._build_api_url("generate"), json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            return ModelResponse(
                                text="",
                                model=f"{self.provider_name}/{model_id}",
                                provider=self.provider_name,
                                input_tokens=0,
                                output_tokens=0,
                                total_tokens=0,
                                processing_time=time.time() - start_time,
                                metadata={
                                    "error": f"Ollama API error: {response.status} - {error_text}",
                                    "cost": 0.0
                                }
                            )
                        
                        data = await response.json()
                        
                        # Extract response text
                        content = data.get("response", "")
                        
                        # Extract token counts if available
                        input_tokens = data.get("prompt_eval_count", 0)
                        output_tokens = data.get("eval_count", 0)
                        
                        # If token counts are 0, make a rough estimate based on text length
                        if output_tokens == 0 and content:
                            # Rough estimate: ~1.3 tokens per word
                            output_tokens = len(content.split()) + len(content) // 10
                            
                        total_tokens = input_tokens + output_tokens
                        
                        # Calculate processing time
                        processing_time = time.time() - start_time
                        
                        # Calculate cost (estimated)
                        cost = self._estimate_token_cost(model_id, input_tokens, output_tokens)
                        
                        return ModelResponse(
                            text=content,
                            model=f"{self.provider_name}/{model_id}",
                            provider=self.provider_name,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                            processing_time=processing_time,
                            metadata={"cost": cost}
                        )
            except aiohttp.ClientConnectionError as e:
                processing_time = time.time() - start_time
                # Return connection error message
                return ModelResponse(
                    text="",
                    model=f"{self.provider_name}/{model_id}",
                    provider=self.provider_name,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    processing_time=processing_time,
                    metadata={
                        "error": f"Connection to Ollama failed: {str(e)}. Make sure Ollama is running and accessible.",
                        "cost": 0.0
                    }
                )
            except Exception as e:
                processing_time = time.time() - start_time
                # Return a proper error response instead of raising an exception
                if isinstance(e, ProviderError):
                    raise
                return ModelResponse(
                    text="",
                    model=f"{self.provider_name}/{model_id}",
                    provider=self.provider_name,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    processing_time=processing_time,
                    metadata={
                        "error": f"Error generating completion: {str(e)}",
                        "cost": 0.0
                    }
                )

    async def generate_completion_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """
        Generate a streaming completion using the Ollama API.
        
        Args:
            prompt: The user prompt to generate a completion for.
            model: The model ID to use (defaults to provider's default).
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum number of tokens to generate.
            stop: List of strings that will stop generation when encountered.
            system: System message for chat-based models.
            **kwargs: Additional parameters to pass to the API.
            
        Yields:
            Tuples containing (text_chunk, metadata) as they are received.
            If Ollama is not available, yields a single error message with metadata.
        """
        try:
            # Check if provider is initialized before attempting to generate
            if not self._initialized:
                try:
                    initialized = await self.initialize()
                    if not initialized:
                        # Yield an error message and immediately terminate
                        error_metadata = {
                            "model": f"{self.provider_name}/{model or self.get_default_model()}",
                            "provider": self.provider_name,
                            "error": "Ollama service is not available. Make sure Ollama is installed and running: https://ollama.com/download",
                            "finish_reason": "error",
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "processing_time": 0.0,
                        }
                        yield "", error_metadata
                        return
                except Exception as e:
                    # Yield an error message and immediately terminate
                    error_metadata = {
                        "model": f"{self.provider_name}/{model or self.get_default_model()}",
                        "provider": self.provider_name,
                        "error": f"Failed to initialize Ollama provider: {str(e)}. Make sure Ollama is installed and running: https://ollama.com/download",
                        "finish_reason": "error",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "processing_time": 0.0,
                    }
                    yield "", error_metadata
                    return
                    
            # Use default model if none specified
            model_id = model or self.get_default_model()
            
            # Remove any provider prefix if present
            if "/" in model_id:
                parts = model_id.split("/", 1)
                if parts[0] == self.provider_name:
                    model_id = parts[1]
            
            # Track token counts and timing for the final metadata
            start_time = time.time()
            input_tokens = 0
            output_tokens = 0
            last_token_count = 0
            
            # Create a dedicated session for streaming
            async with aiohttp.ClientSession(**self.client_session_params) as streaming_session:
                # Prepare the payload based on whether this is a chat or generate request
                if system is not None or model_id.startswith(("llama", "gpt", "claude", "phi", "mistral")):
                    # Use chat endpoint for chat-compatible models
                    messages = []
                    
                    # Add system message if provided
                    if system:
                        messages.append({"role": "system", "content": system})
                        
                    # Add user message (the prompt)
                    messages.append({"role": "user", "content": prompt})
                    
                    payload = {
                        "model": model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": True,  # Enable streaming
                    }
                    
                    # Add optional parameters
                    if max_tokens:
                        payload["max_tokens"] = max_tokens
                    if stop:
                        payload["stop"] = stop
                        
                    # Add any additional parameters
                    for key, value in kwargs.items():
                        if key not in payload and value is not None:
                            payload[key] = value
                    
                    try:
                        # Start streaming request
                        async with streaming_session.post(self._build_api_url("chat"), json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                error_metadata = {
                                    "model": f"{self.provider_name}/{model_id}",
                                    "provider": self.provider_name,
                                    "error": f"Ollama streaming API error: {response.status} - {error_text}",
                                    "finish_reason": "error",
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0,
                                    "cost": 0.0,
                                    "processing_time": time.time() - start_time,
                                }
                                yield "", error_metadata
                                return
                            
                            # Process streaming response
                            buffer = ""
                            try:
                                async for line in response.content:
                                    if not line.strip():
                                        continue
                                        
                                    # Parse the line as JSON
                                    try:
                                        data = json.loads(line)
                                        
                                        # Check if this is the final message with statistics
                                        if data.get("done", False):
                                            # Update token counts from the final message
                                            input_tokens = data.get("prompt_eval_count", input_tokens)
                                            total_output_tokens = data.get("eval_count", output_tokens)
                                            
                                            # Calculate the tokens in this last chunk
                                            new_tokens = total_output_tokens - last_token_count
                                            output_tokens = total_output_tokens
                                            
                                            # No more content to yield in final message
                                            break
                                        
                                        # Extract the content from the message
                                        if "message" in data:
                                            chunk_text = data["message"].get("content", "")
                                        else:
                                            chunk_text = data.get("content", "")
                                        
                                        # Update token count if available
                                        if "eval_count" in data:
                                            current_count = data["eval_count"]
                                            new_tokens = current_count - last_token_count
                                            last_token_count = current_count
                                            output_tokens = current_count
                                        else:
                                            # Estimate token count (very rough approximation)
                                            new_tokens = len(chunk_text.split()) // 4 or 1
                                        
                                        # Yield the chunk
                                        metadata = {
                                            "model": f"{self.provider_name}/{model_id}",
                                            "provider": self.provider_name,
                                            "finish_reason": None,
                                            "input_tokens": input_tokens,
                                            "output_tokens": output_tokens,
                                            "total_tokens": input_tokens + output_tokens,
                                            "cost": self._estimate_token_cost(model_id, input_tokens, new_tokens),
                                            "processing_time": time.time() - start_time,
                                        }
                                        
                                        yield chunk_text, metadata
                                        
                                    except json.JSONDecodeError:
                                        # Handle incomplete JSON or other parsing errors
                                        buffer += line.decode("utf-8")
                                        
                                        try:
                                            # Try to parse the accumulated buffer
                                            data = json.loads(buffer)
                                            buffer = ""
                                            
                                            # Process the data as above
                                            # (This is a simplified version; you might want to deduplicate this logic)
                                            if "message" in data:
                                                chunk_text = data["message"].get("content", "")
                                            else:
                                                chunk_text = data.get("content", "")
                                            
                                            new_tokens = 1  # Default estimation
                                            
                                            metadata = {
                                                "model": f"{self.provider_name}/{model_id}",
                                                "provider": self.provider_name,
                                                "finish_reason": None,
                                                "input_tokens": input_tokens,
                                                "output_tokens": output_tokens,
                                                "total_tokens": input_tokens + output_tokens,
                                                "cost": self._estimate_token_cost(model_id, input_tokens, new_tokens),
                                                "processing_time": time.time() - start_time,
                                            }
                                            
                                            yield chunk_text, metadata
                                            
                                        except json.JSONDecodeError:
                                            # Still incomplete, continue accumulating
                                            continue
                            except Exception as e:
                                # Handle errors during streaming content processing
                                self.logger.warning(f"Error processing streaming content: {str(e)}", emoji_key="warning")
                                error_metadata = {
                                    "model": f"{self.provider_name}/{model_id}",
                                    "provider": self.provider_name,
                                    "error": f"Error processing streaming content: {str(e)}",
                                    "finish_reason": "error",
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "total_tokens": input_tokens + output_tokens,
                                    "cost": self._estimate_token_cost(model_id, input_tokens, output_tokens),
                                    "processing_time": time.time() - start_time,
                                }
                                yield "", error_metadata
                                return
                        
                        # Final metadata with complete information
                        total_tokens = input_tokens + output_tokens
                        processing_time = time.time() - start_time
                        cost = self._estimate_token_cost(model_id, input_tokens, output_tokens)
                        
                        # Yield final empty chunk with complete metadata
                        final_metadata = {
                            "model": f"{self.provider_name}/{model_id}",
                            "provider": self.provider_name,
                            "finish_reason": "stop",
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "cost": cost,
                            "processing_time": processing_time,
                        }
                        
                        yield "", final_metadata
                        
                    except aiohttp.ClientConnectionError as e:
                        # Handle connection errors gracefully
                        processing_time = time.time() - start_time
                        error_metadata = {
                            "model": f"{self.provider_name}/{model_id}",
                            "provider": self.provider_name,
                            "error": f"Connection to Ollama failed: {str(e)}. Make sure Ollama is running and accessible.",
                            "finish_reason": "error",
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "cost": 0.0,
                            "processing_time": processing_time,
                        }
                        yield "", error_metadata
                    except asyncio.TimeoutError:
                        # Handle timeout errors gracefully
                        processing_time = time.time() - start_time
                        error_metadata = {
                            "model": f"{self.provider_name}/{model_id}",
                            "provider": self.provider_name,
                            "error": "Connection to Ollama timed out. Check if the service is overloaded or unresponsive.",
                            "finish_reason": "error",
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "cost": 0.0,
                            "processing_time": processing_time,
                        }
                        yield "", error_metadata
                    except Exception as e:
                        # Handle any other exceptions
                        processing_time = time.time() - start_time
                        if isinstance(e, ProviderError):
                            raise
                        
                        error_metadata = {
                            "model": f"{self.provider_name}/{model_id}",
                            "provider": self.provider_name,
                            "error": f"Error generating streaming chat completion: {str(e)}",
                            "finish_reason": "error",
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "cost": 0.0,
                            "processing_time": processing_time,
                        }
                        yield "", error_metadata
                else:
                    # Use generate endpoint for non-chat models
                    payload = {
                        "model": model_id,
                        "prompt": prompt,
                        "temperature": temperature,
                        "stream": True,  # Enable streaming
                    }
                    
                    # Add optional parameters
                    if max_tokens:
                        payload["max_tokens"] = max_tokens
                    if stop:
                        payload["stop"] = stop
                    
                    # Add any additional parameters
                    for key, value in kwargs.items():
                        if key not in payload and value is not None:
                            payload[key] = value
                    
                    try:
                        # Start streaming request
                        async with streaming_session.post(self._build_api_url("generate"), json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                error_metadata = {
                                    "model": f"{self.provider_name}/{model_id}",
                                    "provider": self.provider_name,
                                    "error": f"Ollama streaming API error: {response.status} - {error_text}",
                                    "finish_reason": "error",
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0,
                                    "cost": 0.0,
                                    "processing_time": time.time() - start_time,
                                }
                                yield "", error_metadata
                                return
                            
                            # Process streaming response
                            buffer = ""
                            try:
                                async for line in response.content:
                                    if not line.strip():
                                        continue
                                        
                                    # Parse the line as JSON
                                    try:
                                        data = json.loads(line)
                                        
                                        # Check if this is the final message
                                        if data.get("done", False):
                                            # Update token counts from the final message
                                            input_tokens = data.get("prompt_eval_count", input_tokens)
                                            total_output_tokens = data.get("eval_count", output_tokens)
                                            
                                            # Calculate the tokens in this last chunk
                                            new_tokens = total_output_tokens - last_token_count
                                            output_tokens = total_output_tokens
                                            
                                            # No more content to yield in final message
                                            break
                                        
                                        # Extract the content from the response
                                        chunk_text = data.get("response", "")
                                        
                                        # Update token count if available
                                        if "eval_count" in data:
                                            current_count = data["eval_count"]
                                            new_tokens = current_count - last_token_count
                                            last_token_count = current_count
                                            output_tokens = current_count
                                        else:
                                            # Estimate token count (very rough approximation)
                                            new_tokens = len(chunk_text.split()) // 4 or 1
                                        
                                        # Yield the chunk
                                        metadata = {
                                            "model": f"{self.provider_name}/{model_id}",
                                            "provider": self.provider_name,
                                            "finish_reason": None,
                                            "input_tokens": input_tokens,
                                            "output_tokens": output_tokens,
                                            "total_tokens": input_tokens + output_tokens,
                                            "cost": self._estimate_token_cost(model_id, input_tokens, new_tokens),
                                            "processing_time": time.time() - start_time,
                                        }
                                        
                                        yield chunk_text, metadata
                                        
                                    except json.JSONDecodeError:
                                        # Handle incomplete JSON or other parsing errors
                                        buffer += line.decode("utf-8")
                                        
                                        try:
                                            # Try to parse the accumulated buffer
                                            data = json.loads(buffer)
                                            buffer = ""
                                            
                                            # Process the data as above
                                            chunk_text = data.get("response", "")
                                            new_tokens = 1  # Default estimation
                                            
                                            metadata = {
                                                "model": f"{self.provider_name}/{model_id}",
                                                "provider": self.provider_name,
                                                "finish_reason": None,
                                                "input_tokens": input_tokens,
                                                "output_tokens": output_tokens,
                                                "total_tokens": input_tokens + output_tokens,
                                                "cost": self._estimate_token_cost(model_id, input_tokens, new_tokens),
                                                "processing_time": time.time() - start_time,
                                            }
                                            
                                            yield chunk_text, metadata
                                            
                                        except json.JSONDecodeError:
                                            # Still incomplete, continue accumulating
                                            continue
                            except Exception as e:
                                # Handle errors during streaming content processing
                                self.logger.warning(f"Error processing streaming content: {str(e)}", emoji_key="warning")
                                error_metadata = {
                                    "model": f"{self.provider_name}/{model_id}",
                                    "provider": self.provider_name,
                                    "error": f"Error processing streaming content: {str(e)}",
                                    "finish_reason": "error",
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "total_tokens": input_tokens + output_tokens,
                                    "cost": self._estimate_token_cost(model_id, input_tokens, output_tokens),
                                    "processing_time": time.time() - start_time,
                                }
                                yield "", error_metadata
                                return
                        
                        # Final metadata with complete information
                        total_tokens = input_tokens + output_tokens
                        processing_time = time.time() - start_time
                        cost = self._estimate_token_cost(model_id, input_tokens, output_tokens)
                        
                        # Yield final empty chunk with complete metadata
                        final_metadata = {
                            "model": f"{self.provider_name}/{model_id}",
                            "provider": self.provider_name,
                            "finish_reason": "stop",
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "cost": cost,
                            "processing_time": processing_time,
                        }
                        
                        yield "", final_metadata
                        
                    except aiohttp.ClientConnectionError as e:
                        # Handle connection errors gracefully
                        processing_time = time.time() - start_time
                        error_metadata = {
                            "model": f"{self.provider_name}/{model_id}",
                            "provider": self.provider_name,
                            "error": f"Connection to Ollama failed: {str(e)}. Make sure Ollama is running and accessible.",
                            "finish_reason": "error",
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "cost": 0.0,
                            "processing_time": processing_time,
                        }
                        yield "", error_metadata
                    except asyncio.TimeoutError:
                        # Handle timeout errors gracefully
                        processing_time = time.time() - start_time
                        error_metadata = {
                            "model": f"{self.provider_name}/{model_id}",
                            "provider": self.provider_name,
                            "error": "Connection to Ollama timed out. Check if the service is overloaded or unresponsive.",
                            "finish_reason": "error",
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "cost": 0.0,
                            "processing_time": processing_time,
                        }
                        yield "", error_metadata
                    except Exception as e:
                        # Handle any other exceptions
                        processing_time = time.time() - start_time
                        if isinstance(e, ProviderError):
                            raise
                        
                        error_metadata = {
                            "model": f"{self.provider_name}/{model_id}",
                            "provider": self.provider_name,
                            "error": f"Error generating streaming completion: {str(e)}",
                            "finish_reason": "error",
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "cost": 0.0,
                            "processing_time": processing_time,
                        }
                        yield "", error_metadata
        except Exception as e:
            # Handle any exceptions that occur outside the main processing logic
            error_metadata = {
                "model": f"{self.provider_name}/{model or self.get_default_model()}",
                "provider": self.provider_name,
                "error": f"Unexpected error in streaming: {str(e)}",
                "finish_reason": "error",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "processing_time": 0.0,
            }
            yield "", error_metadata

    async def create_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate embeddings for a list of texts using the Ollama API.
        
        Args:
            texts: List of texts to generate embeddings for.
            model: The model ID to use (defaults to provider's default).
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            An ModelResponse object with the embeddings and metadata.
            If Ollama is not available, returns an error in the metadata.
        """
        # Check if provider is initialized before attempting to generate
        if not self._initialized:
            try:
                initialized = await self.initialize()
                if not initialized:
                    # Return a clear error without raising an exception
                    return ModelResponse(
                        text="",
                        model=f"{self.provider_name}/{model or self.get_default_model()}",
                        provider=self.provider_name,
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        processing_time=0.0,
                        metadata={
                            "error": "Ollama service is not available. Make sure Ollama is installed and running: https://ollama.com/download",
                            "embeddings": []
                        }
                    )
            except Exception as e:
                # Return a clear error without raising an exception
                return ModelResponse(
                    text="",
                    model=f"{self.provider_name}/{model or self.get_default_model()}",
                    provider=self.provider_name,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    processing_time=0.0,
                    metadata={
                        "error": f"Failed to initialize Ollama provider: {str(e)}. Make sure Ollama is installed and running: https://ollama.com/download",
                        "embeddings": []
                    }
                )
            
        # Use default model if none specified
        model_id = model or self.get_default_model()
        
        # Remove any provider prefix if present
        if "/" in model_id:
            parts = model_id.split("/", 1)
            if parts[0] == self.provider_name:
                model_id = parts[1]
        
        # Get total number of tokens in all texts
        # This is an estimation since Ollama doesn't provide token counts for embeddings
        total_tokens = sum(len(text.split()) for text in texts)
        
        # Prepare the result
        result_embeddings = []
        errors = []
        all_dimensions = None
        
        try:
            start_time = time.time()
            
            # Create a dedicated session for this embeddings request
            async with aiohttp.ClientSession(**self.client_session_params) as session:
                # Process each text individually (Ollama supports batching but we'll use same pattern as other providers)
                for text in texts:
                    payload = {
                        "model": model_id,
                        "prompt": text,
                    }
                    
                    # Add any additional parameters
                    for key, value in kwargs.items():
                        if key not in payload and value is not None:
                            payload[key] = value
                    
                    try:
                        async with session.post(self._build_api_url("embeddings"), json=payload, timeout=30.0) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                errors.append(f"Ollama API error: {response.status} - {error_text}")
                                # Continue with the next text
                                continue
                            
                            data = await response.json()
                            
                            # Extract embeddings
                            embedding = data.get("embedding", [])
                            
                            if not embedding:
                                errors.append(f"No embedding returned for text: {text[:50]}...")
                                continue
                            
                            # Store the embedding
                            result_embeddings.append(embedding)
                            
                            # Check dimensions for consistency
                            dimensions = len(embedding)
                            if all_dimensions is None:
                                all_dimensions = dimensions
                            elif dimensions != all_dimensions:
                                errors.append(f"Inconsistent embedding dimensions: got {dimensions}, expected {all_dimensions}")
                    except aiohttp.ClientConnectionError as e:
                        errors.append(f"Connection to Ollama failed: {str(e)}. Make sure Ollama is running and accessible.")
                        break
                    except asyncio.TimeoutError:
                        errors.append("Connection to Ollama timed out. Check if the service is overloaded.")
                        break
                    except Exception as e:
                        errors.append(f"Error generating embedding: {str(e)}")
                        continue
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Calculate cost (estimated)
            estimated_cost = (total_tokens / 1_000_000) * 0.0001  # Very low cost estimation
            
            # Create response model with embeddings in metadata
            return ModelResponse(
                text="",  # Embeddings don't have text content
                model=f"{self.provider_name}/{model_id}",
                provider=self.provider_name,
                input_tokens=total_tokens,  # Use total tokens as input tokens for embeddings
                output_tokens=0,            # No output tokens for embeddings
                total_tokens=total_tokens,
                processing_time=processing_time,
                metadata={
                    "embeddings": result_embeddings,
                    "dimensions": all_dimensions or 0,
                    "errors": errors if errors else None,
                    "cost": estimated_cost
                }
            )
            
        except aiohttp.ClientConnectionError as e:
            # Return a clear error without raising an exception
            return ModelResponse(
                text="",
                model=f"{self.provider_name}/{model_id}",
                provider=self.provider_name,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                processing_time=0.0,
                metadata={
                    "error": f"Connection to Ollama failed: {str(e)}. Make sure Ollama is running and accessible.",
                    "embeddings": [],
                    "cost": 0.0
                }
            )
        except Exception as e:
            # Return a clear error without raising an exception
            if isinstance(e, ProviderError):
                raise
            return ModelResponse(
                text="",
                model=f"{self.provider_name}/{model_id}",
                provider=self.provider_name,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                processing_time=0.0,
                metadata={
                    "error": f"Error generating embeddings: {str(e)}",
                    "embeddings": result_embeddings,
                    "cost": 0.0
                }
            )