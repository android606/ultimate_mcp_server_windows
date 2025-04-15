# llm_gateway/core/providers/openrouter.py
"""OpenRouter provider implementation."""
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

from openai import AsyncOpenAI

from llm_gateway.constants import DEFAULT_MODELS, Provider, COST_PER_MILLION_TOKENS
from llm_gateway.core.providers.base import BaseProvider, ModelResponse
from llm_gateway.utils import get_logger
from llm_gateway.config import get_config

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("llm_gateway.providers.openrouter")

# Default OpenRouter Base URL (can be overridden by config)
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

class OpenRouterProvider(BaseProvider):
    """Provider implementation for OpenRouter API (using OpenAI-compatible interface)."""

    provider_name = Provider.OPENROUTER.value

    def __init__(self, **kwargs):
        """Initialize the OpenRouter provider.

        Args:
            **kwargs: Additional options:
                - base_url (str): Override the default OpenRouter API base URL.
                - http_referer (str): Optional HTTP-Referer header.
                - x_title (str): Optional X-Title header.
        """
        config = get_config().providers.openrouter
        super().__init__(api_key=config.api_key, **kwargs)
        self.name = "openrouter"
        self.default_model = config.default_model

        # Get base_url from config, fallback to kwargs, then constant
        self.base_url = config.base_url or kwargs.get("base_url", DEFAULT_OPENROUTER_BASE_URL)

        # Get additional headers from config's additional_params
        self.http_referer = config.additional_params.get("http_referer") or kwargs.get("http_referer")
        self.x_title = config.additional_params.get("x_title") or kwargs.get("x_title")

        # Additional initialization for headers, client etc.
        self._initialize_client(**kwargs)
        self.available_models = self.fetch_available_models()

        logger.info(f"OpenRouter provider initialized. Base URL: {self.base_url}, Default Model: {self.default_model}")

    def _initialize_client(self, **kwargs):
        """Initialize the OpenAI async client with OpenRouter specifics."""
        if not self.api_key:
            logger.warning(f"{self.name} API key not found in configuration. Some operations might fail.")
            # Proceed without client if no key, some methods like list_models might still work partially
            self.client = None
            return

        headers = {}
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title
        
        config = get_config().providers.openrouter # Get timeout from config
        timeout = config.timeout or kwargs.get("timeout", 30.0) # Default timeout 30s

        try:
            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers=headers,
                timeout=timeout
            )
            logger.debug("AsyncOpenAI client initialized for OpenRouter.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client for OpenRouter: {e}", exc_info=True)
            self.client = None # Ensure client is None if init fails

    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate a completion using OpenRouter.

        Args:
            prompt: Text prompt to send to the model
            model: Model name (e.g., "openai/gpt-4.1-mini", "google/gemini-flash-1.5")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters, including:
                - extra_headers (Dict): Additional headers for this specific call.
                - extra_body (Dict): OpenRouter-specific arguments.

        Returns:
            ModelResponse with completion result

        Raises:
            Exception: If API call fails
        """
        if not self.client:
            initialized = await self._initialize_client()
            if not initialized:
                raise RuntimeError(f"{self.provider_name} provider not initialized.")

        # Use default model if not specified
        model = model or self.default_model

        # Strip provider prefix only if it matches OUR provider name
        if model.startswith(f"{self.provider_name}:"):
            model = model.split(":", 1)[1]
            logger.debug(f"Stripped provider prefix from model name: {model}")
        # Note: Keep prefixes like 'openai/' or 'google/' as OpenRouter uses them.

        # Create messages
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]

        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Extract OpenRouter specific args from kwargs
        extra_headers = kwargs.pop("extra_headers", {})
        extra_body = kwargs.pop("extra_body", {})

        # Add any remaining kwargs to the main params (standard OpenAI args)
        params.update(kwargs)

        logger.info(
            f"Generating completion with {self.provider_name} model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt)
        )

        try:
            # Make API call with timing
            response, processing_time = await self.process_with_timer(
                self.client.chat.completions.create, **params, extra_headers=extra_headers, extra_body=extra_body
            )

            # Extract response text
            completion_text = response.choices[0].message.content

            # Create standardized response
            result = ModelResponse(
                text=completion_text,
                model=response.model, # Use model returned by API
                provider=self.provider_name,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                processing_time=processing_time,
                raw_response=response,
            )

            logger.success(
                f"{self.provider_name} completion successful",
                emoji_key="success",
                model=result.model,
                tokens={
                    "input": result.input_tokens,
                    "output": result.output_tokens
                },
                cost=result.cost, # Will be calculated by ModelResponse
                time=result.processing_time
            )

            return result

        except Exception as e:
            logger.error(
                f"{self.provider_name} completion failed for model {model}: {str(e)}",
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
        """Generate a streaming completion using OpenRouter.

        Args:
            prompt: Text prompt to send to the model
            model: Model name (e.g., "openai/gpt-4.1-mini")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters, including:
                - extra_headers (Dict): Additional headers for this specific call.
                - extra_body (Dict): OpenRouter-specific arguments.

        Yields:
            Tuple of (text_chunk, metadata)

        Raises:
            Exception: If API call fails
        """
        if not self.client:
            initialized = await self._initialize_client()
            if not initialized:
                raise RuntimeError(f"{self.provider_name} provider not initialized.")

        model = model or self.default_model
        if model.startswith(f"{self.provider_name}:"):
            model = model.split(":", 1)[1]

        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]

        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        extra_headers = kwargs.pop("extra_headers", {})
        extra_body = kwargs.pop("extra_body", {})
        params.update(kwargs)

        logger.info(
            f"Generating streaming completion with {self.provider_name} model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt)
        )

        start_time = time.time()
        total_chunks = 0
        final_model_name = model # Store initially requested model

        try:
            stream = await self.client.chat.completions.create(**params, extra_headers=extra_headers, extra_body=extra_body)

            async for chunk in stream:
                total_chunks += 1
                delta = chunk.choices[0].delta
                content = delta.content or ""

                # Try to get model name from the chunk if available (some providers include it)
                if chunk.model:
                    final_model_name = chunk.model

                metadata = {
                    "model": final_model_name,
                    "provider": self.provider_name,
                    "chunk_index": total_chunks,
                    "finish_reason": chunk.choices[0].finish_reason,
                }

                yield content, metadata

            processing_time = time.time() - start_time
            logger.success(
                f"{self.provider_name} streaming completion successful",
                emoji_key="success",
                model=final_model_name,
                chunks=total_chunks,
                time=processing_time
            )

        except Exception as e:
            logger.error(
                f"{self.provider_name} streaming completion failed for model {model}: {str(e)}",
                emoji_key="error",
                model=model
            )
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available OpenRouter models (provides examples, not exhaustive).

        OpenRouter offers a vast number of models. This list provides common examples.
        Refer to OpenRouter documentation for the full list.

        Returns:
            List of example model information dictionaries
        """
        # OpenRouter doesn't have a standard API endpoint listable via openai client
        # Return a static list of common examples. Users should refer to OpenRouter docs.
        if self.available_models:
            return self.available_models

        models = [
            {
                "id": "openai/gpt-4.1-mini",
                "provider": self.provider_name,
                "description": "OpenAI: Fast, balances cost and performance.",
            },
            {
                "id": "openai/gpt-4o",
                "provider": self.provider_name,
                "description": "OpenAI: Most capable model.",
            },
            {
                "id": "anthropic/claude-3.5-sonnet", # Check exact ID on OpenRouter
                "provider": self.provider_name,
                "description": "Anthropic: Strong general reasoning (check exact ID).",
            },
             {
                "id": "anthropic/claude-3-haiku", # Check exact ID on OpenRouter
                "provider": self.provider_name,
                "description": "Anthropic: Fast and affordable (check exact ID).",
            },
            {
                "id": "google/gemini-pro-1.5", # Check exact ID on OpenRouter
                "provider": self.provider_name,
                "description": "Google: Large context window (check exact ID).",
            },
            {
                "id": "google/gemini-flash-1.5", # Check exact ID on OpenRouter
                "provider": self.provider_name,
                "description": "Google: Fast and cost-effective (check exact ID).",
            },
            {
                "id": "mistralai/mistral-large", # Check exact ID on OpenRouter
                "provider": self.provider_name,
                "description": "Mistral: Strong open-weight model (check exact ID).",
            },
            {
                 "id": "meta-llama/llama-3-70b-instruct", # Check exact ID on OpenRouter
                 "provider": self.provider_name,
                 "description": "Meta: Powerful open-source instruction-tuned model (check exact ID).",
            }
        ]

        self.available_models = models
        logger.warning(f"{self.provider_name} model list is illustrative. Check OpenRouter for full details.", emoji_key="warning")
        return models

    def get_default_model(self) -> str:
        """Get the default OpenRouter model.

        Returns:
            Default model name (e.g., "openai/gpt-4.1-mini")
        """
        # Allow override via environment variable
        default_model_env = os.environ.get("OPENROUTER_DEFAULT_MODEL")
        if default_model_env:
            return default_model_env

        # Fallback to constants
        return DEFAULT_MODELS.get(self.provider_name, "openai/gpt-4.1-mini")

    async def check_api_key(self) -> bool:
        """Check if the OpenRouter API key is valid by attempting a small request."""
        if not self.client:
            # Try to initialize if not already done
            if not await self._initialize_client():
                return False # Initialization failed

        try:
            # Attempt a simple, low-cost operation, e.g., list models (even if it returns 404/permission error, it validates the key/URL)
            # Or use a very small completion request
            await self.client.chat.completions.create(
                model=self.get_default_model(),
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0
            )
            return True
        except Exception as e:
            logger.warning(f"API key check failed for {self.provider_name}: {str(e)}", emoji_key="warning")
            return False

    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch available models and their details from the OpenRouter API."""
        # OpenRouter uses the standard OpenAI /models endpoint
        if not self.client:
            logger.warning("Cannot fetch models; OpenRouter client not initialized (likely missing API key).")
            return {}

        try:
            logger.debug("Fetching available models from OpenRouter...")
            # This call uses the standard client, which includes the API key
            # Note: httpx might handle the async call internally
            response = self.client.models.list() # Synchronous call within method, check if client handles async

            models_data = {}
            if isinstance(response, BaseModel):
                 # Handle Pydantic model response (common with newer OpenAI lib versions)
                 if hasattr(response, 'data') and isinstance(response.data, list):
                      model_list = response.data
                 else:
                      logger.warning("Unexpected response structure from models endpoint (Pydantic).")
                      model_list = []
            elif isinstance(response, dict):
                 # Handle dictionary response (older versions or direct httpx use)
                 model_list = response.get('data', [])
            else:
                 logger.warning(f"Unexpected response type from models endpoint: {type(response)}")
                 model_list = []

            for model_info_raw in model_list:
                 # Adapt based on actual response structure (OpenAI vs OpenRouter specifics)
                 if isinstance(model_info_raw, BaseModel):
                      # Access attributes if it's a Pydantic model
                      model_id = getattr(model_info_raw, 'id', None)
                      # Extract other relevant fields if needed (e.g., context_length from OpenRouter docs)
                      # Note: Standard OpenAI response might not have all OpenRouter fields directly
                 elif isinstance(model_info_raw, dict):
                      # Access keys if it's a dictionary
                      model_id = model_info_raw.get('id')
                 else:
                      model_id = None

                 if model_id:
                      # Store basic info for now; enhance with OpenRouter specifics if possible
                      models_data[model_id] = {'id': model_id} # Add more fields later

            logger.info(f"Found {len(models_data)} models available via OpenRouter.")
            # Optional: Fetch detailed pricing/context from OpenRouter site/docs if needed
            # and merge into models_data
            return models_data

        except Exception as e:
            logger.error(f"Failed to fetch available models from OpenRouter: {e}", exc_info=True)
            return {}

    def get_available_models(self) -> List[str]:
        """Return a list of available model names."""
        return list(self.available_models.keys())

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        return model_name in self.available_models

    async def create_completion(self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """Create a completion using the specified model."""
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized (likely missing API key).")
        if not self.is_model_available(model):
            # Fallback to default if provided model isn't listed? Or raise error?
            # Let's try the default model if the requested one isn't confirmed available.
            if self.default_model and self.is_model_available(self.default_model):
                 logger.warning(f"Model '{model}' not found in available list. Falling back to default '{self.default_model}'.")
                 model = self.default_model
            else:
                 # If even the default isn't available or set, raise error
                 raise ValueError(f"Model '{model}' is not available via OpenRouter according to fetched list, and no valid default model is set.")

        merged_kwargs = {**kwargs}
        # OpenRouter uses standard OpenAI params like max_tokens, temperature, etc.
        # Ensure essential params are passed
        if 'max_tokens' not in merged_kwargs:
            merged_kwargs['max_tokens'] = get_config().providers.openrouter.max_tokens or 1024 # Use config or default

        if stream:
            logger.debug(f"Creating stream completion: Model={model}, Params={merged_kwargs}")
            return self._stream_completion_generator(model, messages, **merged_kwargs)
        else:
            logger.debug(f"Creating completion: Model={model}, Params={merged_kwargs}")
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    **merged_kwargs
                )
                # Extract content based on OpenAI library version
                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        return choice.message.content or "" # Return empty string if content is None
                    elif hasattr(choice, 'delta') and hasattr(choice.delta, 'content'): # Should not happen for stream=False but check
                        return choice.delta.content or ""
                logger.warning("Could not extract content from OpenRouter response.")
                return "" # Return empty string if no content found
            except Exception as e:
                logger.error(f"OpenRouter completion failed: {e}", exc_info=True)
                raise RuntimeError(f"OpenRouter API call failed: {e}") from e

    async def _stream_completion_generator(self, model: str, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Async generator for streaming completions."""
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized (likely missing API key).")
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs
            )
            async for chunk in stream:
                # Extract content based on OpenAI library version
                content = ""
                if hasattr(chunk, 'choices') and chunk.choices:
                     choice = chunk.choices[0]
                     if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                          content = choice.delta.content
                     elif hasattr(choice, 'message') and hasattr(choice.message, 'content'): # Should not happen for stream=True
                          content = choice.message.content

                if content:
                     yield content
        except Exception as e:
            logger.error(f"OpenRouter stream completion failed: {e}", exc_info=True)
            # Depending on desired behavior, either raise or yield an error message
            # yield f"Error during stream: {e}"
            raise RuntimeError(f"OpenRouter API stream failed: {e}") from e

    # --- Cost Calculation (Needs OpenRouter Specific Data) ---
    def get_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Calculate the cost of a request based on OpenRouter pricing.

        Note: Requires loading detailed model pricing info, which is not
              done by default in fetch_available_models.
              This is a placeholder and needs enhancement.
        """
        # Placeholder: Need to fetch and store detailed pricing from OpenRouter
        # Example structure (needs actual data):
        openrouter_pricing = {
             # "model_id": {"prompt_cost_per_mtok": X, "completion_cost_per_mtok": Y},
             "openai/gpt-4o": {"prompt_cost_per_mtok": 5.0, "completion_cost_per_mtok": 15.0},
             "google/gemini-pro-1.5": {"prompt_cost_per_mtok": 3.5, "completion_cost_per_mtok": 10.5},
             "anthropic/claude-3-opus": {"prompt_cost_per_mtok": 15.0, "completion_cost_per_mtok": 75.0},
             # ... add more model costs from openrouter.ai/docs#models ...
        }

        model_cost = openrouter_pricing.get(model)
        if model_cost:
            prompt_cost = (prompt_tokens / 1_000_000) * model_cost.get("prompt_cost_per_mtok", 0)
            completion_cost = (completion_tokens / 1_000_000) * model_cost.get("completion_cost_per_mtok", 0)
            return prompt_cost + completion_cost
        else:
            logger.warning(f"Cost calculation not available for OpenRouter model: {model}")
            # Return None if cost cannot be calculated
            return None

    # --- Prompt Formatting --- #
    def format_prompt(self, messages: List[Dict[str, str]]) -> Any:
        """Use standard list of dictionaries format for OpenRouter (like OpenAI)."""
        # OpenRouter generally uses the same format as OpenAI
        return messages

# Make available via discovery
__all__ = ["OpenRouterProvider"]