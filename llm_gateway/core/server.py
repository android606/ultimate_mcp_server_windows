"""Main MCP server implementation for LLM Gateway."""
import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from mcp.server.fastmcp import Context, FastMCP

from llm_gateway.config import config
from llm_gateway.constants import Provider, TaskType
from llm_gateway.core.providers.base import get_provider
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProviderStatus:
    """Status information for a provider."""
    enabled: bool
    available: bool
    api_key_configured: bool
    models: List[Dict[str, Any]]
    error: Optional[str] = None


class Gateway:
    """Main LLM Gateway implementation."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the gateway.
        
        Args:
            name: Name for the MCP server
        """
        self.name = name or config.server.name
        self.logger = get_logger(__name__)
        self.providers = {}
        self.provider_status = {}
        
        # Create MCP server
        self.mcp = FastMCP(self.name, lifespan=self._server_lifespan)
        
        # Register tools
        self._register_tools()
        
        # Register resources
        self._register_resources()
        
        self.logger.info(
            f"LLM Gateway '{self.name}' initialized",
            emoji_key="start"
        )
        
    @asynccontextmanager
    async def _server_lifespan(self, server: FastMCP):
        """Server lifespan context manager.
        
        Args:
            server: MCP server instance
            
        Yields:
            Dict containing initialized resources
        """
        self.logger.info(
            f"Starting LLM Gateway '{self.name}'",
            emoji_key="server"
        )
        
        # Initialize providers
        await self._initialize_providers()
        
        # Create lifespan context
        context = {
            "providers": self.providers,
            "provider_status": self.provider_status,
        }
        
        try:
            yield context
        finally:
            self.logger.info(
                f"Shutting down LLM Gateway '{self.name}'",
                emoji_key="server"
            )
    
    async def _initialize_providers(self):
        """Initialize all enabled providers."""
        self.logger.info(
            "Initializing LLM providers",
            emoji_key="provider"
        )
        
        # Get list of providers to initialize
        providers_to_init = []
        for provider_name in [p.value for p in Provider]:
            provider_config = getattr(config.providers, provider_name, None)
            if provider_config and provider_config.enabled:
                providers_to_init.append(provider_name)
                
        # Initialize providers in parallel
        init_tasks = []
        for provider_name in providers_to_init:
            task = asyncio.create_task(
                self._initialize_provider(provider_name),
                name=f"init-{provider_name}"
            )
            init_tasks.append(task)
            
        await asyncio.gather(*init_tasks)
        
        # Log initialization summary
        available_providers = [
            name for name, status in self.provider_status.items()
            if status.available
        ]
        
        self.logger.success(
            f"Providers initialized: {len(available_providers)}/{len(providers_to_init)} available",
            emoji_key="provider"
        )
    
    async def _initialize_provider(self, provider_name: str):
        """Initialize a single provider.
        
        Args:
            provider_name: Provider name
        """
        try:
            provider_config = getattr(config.providers, provider_name, None)
            
            # Check if API key is configured
            api_key_configured = bool(provider_config and provider_config.api_key)
            
            if not api_key_configured:
                self.provider_status[provider_name] = ProviderStatus(
                    enabled=True,
                    available=False,
                    api_key_configured=False,
                    models=[],
                    error="API key not configured"
                )
                self.logger.warning(
                    f"Provider {provider_name} not initialized: API key not configured",
                    emoji_key="warning"
                )
                return
                
            # Create provider instance
            provider = get_provider(provider_name)
            
            # Initialize provider
            available = await provider.initialize()
            
            if available:
                # Get available models
                models = await provider.list_models()
                
                # Store provider instance and status
                self.providers[provider_name] = provider
                self.provider_status[provider_name] = ProviderStatus(
                    enabled=True,
                    available=True,
                    api_key_configured=True,
                    models=models
                )
                
                self.logger.success(
                    f"Provider {provider_name} initialized successfully with {len(models)} models",
                    emoji_key="provider"
                )
            else:
                self.provider_status[provider_name] = ProviderStatus(
                    enabled=True,
                    available=False,
                    api_key_configured=True,
                    models=[],
                    error="Initialization failed"
                )
                
                self.logger.error(
                    f"Provider {provider_name} initialization failed",
                    emoji_key="error"
                )
                
        except Exception as e:
            self.provider_status[provider_name] = ProviderStatus(
                enabled=True,
                available=False,
                api_key_configured=False,
                models=[],
                error=str(e)
            )
            
            self.logger.error(
                f"Error initializing provider {provider_name}: {str(e)}",
                emoji_key="error"
            )
    
    def _register_tools(self):
        """Register MCP tools."""
        # Register providers tool group
        self._register_provider_tools()
        
        # Register document processing tools
        self._register_document_tools()
        
        # Register cost optimization tools
        self._register_cost_tools()
    
    def _register_provider_tools(self):
        """Register provider-related tools."""
        
        @self.mcp.tool()
        async def get_provider_status(ctx: Context) -> Dict[str, Any]:
            """
            Get status information for all providers.
            
            Returns:
                Dict containing provider status information
            """
            lifespan_ctx = ctx.request_context.lifespan_context
            provider_status = lifespan_ctx.get("provider_status", {})
            
            return {
                "providers": {
                    name: {
                        "enabled": status.enabled,
                        "available": status.available,
                        "api_key_configured": status.api_key_configured,
                        "error": status.error,
                        "models_count": len(status.models)
                    }
                    for name, status in provider_status.items()
                }
            }
        
        @self.mcp.tool()
        async def list_models(
            provider: Optional[str] = None,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """
            List available models from specified provider or all providers.
            
            Args:
                provider: Provider name (optional)
                
            Returns:
                Dict containing available models
            """
            lifespan_ctx = ctx.request_context.lifespan_context
            provider_status = lifespan_ctx.get("provider_status", {})
            
            models = {}
            
            if provider:
                # Check if provider is valid
                if provider not in provider_status:
                    return {"error": f"Invalid provider: {provider}"}
                    
                # Return models for specified provider
                status = provider_status[provider]
                models[provider] = status.models if status.available else []
            else:
                # Return models for all providers
                for name, status in provider_status.items():
                    models[name] = status.models if status.available else []
            
            return {"models": models}
        
        @self.mcp.tool()
        async def generate_completion(
            prompt: str,
            provider: str = "openai",
            model: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: float = 0.7,
            stream: bool = False,
            ctx: Context = None,
            **kwargs
        ) -> Dict[str, Any]:
            """
            Generate completion using specified LLM provider.
            
            Args:
                prompt: Text prompt to send to the model
                provider: LLM provider to use (openai, anthropic, deepseek, gemini)
                model: Optional model name (if not provided, default for provider will be used)
                max_tokens: Maximum tokens in completion
                temperature: Temperature parameter (0.0-1.0)
                stream: Whether to stream the response (not implemented yet)
                **kwargs: Additional model-specific parameters
                
            Returns:
                Dictionary containing completion text and usage statistics
            """
            lifespan_ctx = ctx.request_context.lifespan_context
            providers = lifespan_ctx.get("providers", {})
            
            # Check if provider is valid and available
            if provider not in providers:
                return {
                    "error": f"Provider '{provider}' not available. Valid options: {list(providers.keys())}"
                }
                
            provider_instance = providers[provider]
            
            # Generate completion
            result = await provider_instance.generate_completion(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Return result
            return result.to_dict()
    
    def _register_document_tools(self):
        """Register document processing tools."""
        # Placeholder for document processing tools
        # These will be implemented in the document.py module
        pass
    
    def _register_cost_tools(self):
        """Register cost optimization tools."""
        # Placeholder for cost optimization tools
        # These will be implemented in the optimization.py module
        pass
    
    def _register_resources(self):
        """Register MCP resources."""
        
        @self.mcp.resource("info://server")
        def get_server_info() -> Dict[str, Any]:
            """Get information about the LLM Gateway server."""
            return {
                "name": self.name,
                "version": "0.1.0",
                "description": "MCP server for accessing multiple LLM providers",
                "providers": [p.value for p in Provider],
            }
        
    def run(self):
        """Run the MCP server."""
        self.logger.info(
            f"Starting MCP server on {config.server.host}:{config.server.port}",
            emoji_key="server"
        )
        self.mcp.run()