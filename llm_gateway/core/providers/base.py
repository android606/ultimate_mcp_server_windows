"""Base LLM provider interface."""
import abc
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from llm_gateway.config import config
from llm_gateway.constants import Provider, COST_PER_MILLION_TOKENS
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
        self.api_key = api_key
        self.options = kwargs
        self.client = None
        self.logger = get_logger(f"provider.{self.provider_name}")
        
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


def get_provider(provider_name: str, **kwargs) -> BaseProvider:
    """Factory function to get a provider instance by name.
    
    Args:
        provider_name: Provider name
        **kwargs: Provider-specific options
        
    Returns:
        BaseProvider: Provider instance
        
    Raises:
        ValueError: If provider name is invalid
    """
    # Normalize provider name
    provider_name = provider_name.lower().strip()
    
    # Import here to avoid circular imports
    from llm_gateway.core.providers.anthropic import AnthropicProvider
    from llm_gateway.core.providers.deepseek import DeepSeekProvider
    from llm_gateway.core.providers.gemini import GeminiProvider
    from llm_gateway.core.providers.openai import OpenAIProvider
    
    # Map provider names to classes
    providers = {
        Provider.OPENAI.value: OpenAIProvider,
        Provider.ANTHROPIC.value: AnthropicProvider,
        Provider.DEEPSEEK.value: DeepSeekProvider,
        Provider.GEMINI.value: GeminiProvider,
    }
    
    if provider_name not in providers:
        raise ValueError(f"Invalid provider name: {provider_name}")
        
    # Get provider configuration
    provider_config = getattr(config.providers, provider_name, None)
    
    # Create provider instance
    provider_cls = providers[provider_name]
    api_key = kwargs.pop("api_key", None) or (
        provider_config.api_key if provider_config else None
    )
    
    return provider_cls(api_key=api_key, **kwargs)