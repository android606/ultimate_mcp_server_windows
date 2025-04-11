# llm_gateway/core/providers/openrouter.py
"""OpenRouter provider implementation."""
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from llm_gateway.constants import DEFAULT_MODELS, Provider
from llm_gateway.core.providers.base import BaseProvider, ModelResponse
from llm_gateway.utils import get_logger

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("llm_gateway.providers.openrouter")

class OpenRouterProvider(BaseProvider):
    """Provider implementation for OpenRouter API (using OpenAI-compatible interface)."""

    provider_name = Provider.OPENROUTER.value

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key (uses OPENROUTER_API_KEY env var if None)
            **kwargs: Additional options:
                - base_url (str): Override the default OpenRouter API base URL.
                - http_referer (str): Optional HTTP-Referer header.
                - x_title (str): Optional X-Title header.
        """
        super().__init__(api_key=api_key, **kwargs)
        # Use provided base_url or default, allow override via kwargs
        self.base_url = kwargs.get("base_url", os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
        self.http_referer = kwargs.get("http_referer", os.environ.get("OPENROUTER_HTTP_REFERER"))
        self.x_title = kwargs.get("x_title", os.environ.get("OPENROUTER_X_TITLE"))
        self.models_cache = None
        self.client = None

    def get_api_key_from_env(self) -> Optional[str]:
        """Get API key specifically for OpenRouter."""
        return os.environ.get("OPENROUTER_API_KEY")

    async def initialize(self) -> bool:
        """Initialize the OpenRouter client.

        Returns:
            bool: True if initialization was successful
        """
        if not self.api_key:
            logger.error(f"{self.provider_name} API key not found. Set OPENROUTER_API_KEY environment variable.", emoji_key="error")
            return False

        try:
            # Prepare default headers
            default_headers = {}
            if self.http_referer:
                default_headers["HTTP-Referer"] = self.http_referer
            if self.x_title:
                default_headers["X-Title"] = self.x_title

            # OpenRouter uses OpenAI-compatible API
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=default_headers if default_headers else None, # Pass headers if they exist
            )

            # Optional: Add a basic API check if desired, e.g., listing models (might fail if permission denied)
            # try:
            #     await self.client.models.list()
            # except Exception as check_err:
            #     logger.warning(f"Post-initialization check (listing models) failed for {self.provider_name}: {check_err}. Continuing...", emoji_key="warning")

            logger.success(
                f"{self.provider_name} provider initialized successfully",
                emoji_key="provider"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize {self.provider_name} provider: {str(e)}",
                emoji_key="error"
            )
            self.client = None # Ensure client is None on failure
            return False

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
            model: Model name (e.g., "openai/gpt-4o-mini", "google/gemini-flash-1.5")
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
            initialized = await self.initialize()
            if not initialized:
                raise RuntimeError(f"{self.provider_name} provider not initialized.")

        # Use default model if not specified
        model = model or self.get_default_model()

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
            model: Model name (e.g., "openai/gpt-4o-mini")
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
            initialized = await self.initialize()
            if not initialized:
                raise RuntimeError(f"{self.provider_name} provider not initialized.")

        model = model or self.get_default_model()
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
        if self.models_cache:
            return self.models_cache

        models = [
            {
                "id": "openai/gpt-4o-mini",
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

        self.models_cache = models
        logger.warning(f"{self.provider_name} model list is illustrative. Check OpenRouter for full details.", emoji_key="warning")
        return models

    def get_default_model(self) -> str:
        """Get the default OpenRouter model.

        Returns:
            Default model name (e.g., "openai/gpt-4o-mini")
        """
        # Allow override via environment variable
        default_model_env = os.environ.get("OPENROUTER_DEFAULT_MODEL")
        if default_model_env:
            return default_model_env

        # Fallback to constants
        return DEFAULT_MODELS.get(self.provider_name, "openai/gpt-4o-mini")

    async def check_api_key(self) -> bool:
        """Check if the OpenRouter API key is valid by attempting a small request."""
        if not self.client:
            # Try to initialize if not already done
            if not await self.initialize():
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