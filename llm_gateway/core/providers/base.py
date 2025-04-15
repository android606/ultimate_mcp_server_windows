"""Base LLM provider interface."""
import abc
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from llm_gateway.config import get_config
from llm_gateway.constants import COST_PER_MILLION_TOKENS, Provider
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class ModelResponse:
    """Standard response format for all LLM providers."""
    
    def __init__(
        self,
        text: str,
        model: str,
        provider: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        processing_time: float = 0.0,
        raw_response: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a model response.
        
        Args:
            text: Generated text content
            model: Model name used
            provider: Provider name
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            total_tokens: Total tokens used (if not provided, calculated from input + output)
            processing_time: Time taken to process the request in seconds
            raw_response: Raw provider response for debugging
            metadata: Additional response metadata
        """
        self.text = text
        self.model = model
        self.provider = provider
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens or (input_tokens + output_tokens)
        self.processing_time = processing_time
        self.raw_response = raw_response
        self.metadata = metadata or {}
        
        # Calculate cost based on token usage
        self.cost = self._calculate_cost()
        
    def _calculate_cost(self) -> float:
        """Calculate the cost of the request based on token usage."""
        if not self.model or not self.input_tokens or not self.output_tokens:
            return 0.0
            
        # Get cost per token for this model
        model_costs = COST_PER_MILLION_TOKENS.get(self.model, None)
        if not model_costs:
            # If model not found, use a default estimation
            model_costs = {"input": 0.5, "output": 1.5}
            logger.warning(
                f"Cost data not found for model {self.model}. Using estimates.", 
                emoji_key="cost"
            )
            
        # Calculate cost
        input_cost = (self.input_tokens / 1_000_000) * model_costs["input"]
        output_cost = (self.output_tokens / 1_000_000) * model_costs["output"]
        
        return input_cost + output_cost
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
            "processing_time": self.processing_time,
            "cost": self.cost,
            "metadata": self.metadata,
        }


class BaseProvider(abc.ABC):
    """Abstract base class for LLM providers."""
    
    provider_name: str = "base"
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the provider.
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific options
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = self.get_api_key_from_env()
            
        self.api_key = api_key
        self.options = kwargs
        self.client = None
        self.logger = get_logger(f"provider.{self.provider_name}")
        
    # Removed get_api_key_from_env - API key comes from config
    # def get_api_key_from_env(self) -> Optional[str]:
    #     """Get API key from environment variables.
    #     
    #     Returns:
    #         API key string or None if not found
    #     """
    #     import os
    #     
    #     # Map provider names to environment variable names
    #     env_vars = {
    #         Provider.OPENAI.value: "OPENAI_API_KEY",
    #         Provider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
    #         Provider.DEEPSEEK.value: "DEEPSEEK_API_KEY",
    #         Provider.GEMINI.value: "GEMINI_API_KEY",
    #     }
    #     
    #     # Get the appropriate environment variable name
    #     env_var = env_vars.get(self.provider_name)
    #     if not env_var:
    #         return None
    #         
    #     # Try to get from environment
    #     return os.environ.get(env_var)
        
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """Initialize the client.
        
        Returns:
            bool: True if initialization was successful
        """
        pass
        
    @abc.abstractmethod
    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate a completion from the provider.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse: Standardized response
        """
        pass
        
    @abc.abstractmethod
    async def generate_completion_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion from the provider.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters
            
        Yields:
            Tuple of (text_chunk, metadata)
        """
        pass
        
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from this provider.
        
        Returns:
            List of model information dictionaries
        """
        # Default implementation - override in provider-specific classes
        return [
            {
                "id": "default-model",
                "provider": self.provider_name,
                "description": "Default model",
            }
        ]
        
    def get_default_model(self) -> str:
        """Get the default model for this provider.
        
        Returns:
            Default model name
        """
        raise NotImplementedError("Provider must implement get_default_model")
        
    async def check_api_key(self) -> bool:
        """Check if the API key is valid.
        
        Returns:
            bool: True if API key is valid
        """
        # Default implementation just checks if key exists
        return bool(self.api_key)
        
    async def process_with_timer(
        self, 
        func: callable, 
        *args, 
        **kwargs
    ) -> Tuple[Any, float]:
        """Process a request with timing.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, processing_time)
        """
        start_time = time.time()
        result = await func(*args, **kwargs)
        processing_time = time.time() - start_time
        
        return result, processing_time


async def get_provider(provider_name: str, **kwargs) -> BaseProvider:
    """Factory function to get an initialized provider instance by name.
    
    Args:
        provider_name: Provider name
        **kwargs: Provider-specific options
        
    Returns:
        BaseProvider: Initialized provider instance
        
    Raises:
        ValueError: If provider name is invalid or initialization fails
    """
    cfg = get_config()
    provider_name = provider_name.lower().strip()
    
    from llm_gateway.core.providers.anthropic import AnthropicProvider
    from llm_gateway.core.providers.deepseek import DeepSeekProvider
    from llm_gateway.core.providers.gemini import GeminiProvider
    from llm_gateway.core.providers.openai import OpenAIProvider
    
    providers = {
        Provider.OPENAI.value: OpenAIProvider,
        Provider.ANTHROPIC.value: AnthropicProvider,
        Provider.DEEPSEEK.value: DeepSeekProvider,
        Provider.GEMINI.value: GeminiProvider,
    }
    
    if provider_name not in providers:
        raise ValueError(f"Invalid provider name: {provider_name}")
        
    # Get the top-level 'providers' config object, default to None if it doesn't exist
    providers_config = getattr(cfg, 'providers', None)
    
    # Get the specific provider config (e.g., providers_config.openai) from the providers_config object
    # Default to None if providers_config is None or the specific provider attr doesn't exist
    provider_cfg = getattr(providers_config, provider_name, None) if providers_config else None
    
    # Now use provider_cfg to get the api_key if needed
    if 'api_key' not in kwargs and provider_cfg and hasattr(provider_cfg, 'api_key') and provider_cfg.api_key:
        kwargs['api_key'] = provider_cfg.api_key
    
    provider_class = providers[provider_name]
    instance = provider_class(**kwargs)
    
    # Initialize the provider immediately
    initialized = await instance.initialize()
    if not initialized:
        # Raise an error if initialization fails to prevent returning an unusable instance
        raise ValueError(f"Failed to initialize provider: {provider_name}")

    return instance