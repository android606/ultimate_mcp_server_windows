"""Main server implementation for LLM Gateway."""
import asyncio
import logging
import logging.config
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from mcp.server.fastmcp import Context, FastMCP

import llm_gateway

# Import core specifically to set the global instance
import llm_gateway.core
from llm_gateway.config import get_config, load_config
from llm_gateway.constants import Provider
from llm_gateway.core.state_store import StateStore

# --- Import the trigger function directly instead of the whole module---
# from llm_gateway.tools.marqo_fused_search import trigger_dynamic_docstring_generation
# Import the function later when needed to avoid circular import
from llm_gateway.utils import get_logger
from llm_gateway.utils.logging import logger

# --- Define Logging Configuration Dictionary ---

LOG_FILE_PATH = "logs/llm_gateway.log"

# Ensure log directory exists before config is used
log_dir = os.path.dirname(LOG_FILE_PATH)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False, # Let Uvicorn's loggers pass through if needed
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
        "file": { # Formatter for file output
            "format": "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": { # Console handler - redirect to stderr
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",  # Changed from stdout to stderr
        },
        "access": { # Access log handler - redirect to stderr
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",  # Changed from stdout to stderr
        },
        "rich_console": { # Rich console handler
            "()": "llm_gateway.utils.logging.formatter.create_rich_console_handler",
            "stderr": True,  # Add this parameter to use stderr
        },
        "file": { # File handler
            "formatter": "file",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_FILE_PATH,
            "maxBytes": 2 * 1024 * 1024, # 2 MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "tools_file": { # Tools log file handler
            "formatter": "file",
            "class": "logging.FileHandler",
            "filename": "logs/direct_tools.log",
            "encoding": "utf-8",
        },
        "completions_file": { # Completions log file handler
            "formatter": "file",
            "class": "logging.FileHandler",
            "filename": "logs/direct_completions.log",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["rich_console"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO", "propagate": True}, # Propagate errors to root
        "uvicorn.access": {"handlers": ["access", "file"], "level": "INFO", "propagate": False},
        "llm_gateway": { # Our application's logger namespace
            "handlers": ["rich_console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "llm_gateway.tools": { # Tools-specific logger
            "handlers": ["tools_file"],
            "level": "DEBUG",
            "propagate": True, # Propagate to parent for console display
        },
        "llm_gateway.completions": { # Completions-specific logger
            "handlers": ["completions_file"],
            "level": "DEBUG",
            "propagate": True, # Propagate to parent for console display
        },
    },
    "root": { # Root logger configuration
        "level": "INFO",
        "handlers": ["rich_console", "file"], # Root catches logs not handled by specific loggers
    },
}

# DO NOT apply the config here - it will be applied by Uvicorn through log_config parameter

# Global server instance
_server_app = None
_gateway_instance = None

# Get loggers
tools_logger = get_logger("llm_gateway.tools")
completions_logger = get_logger("llm_gateway.completions")

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
    
    def __init__(
        self, 
        name: str = "main", 
        register_tools: bool = True,
        provider_exclusions: List[str] = None
    ):
        """Initialize Gateway.
        
        Args:
            name: Server name
            register_tools: Whether to register tools with the server
            provider_exclusions: List of provider names to exclude from initialization
        """
        self.name = name
        self.providers = {}
        self.provider_status = {}
        self.logger = get_logger(f"llm_gateway.{name}")
        self.event_handlers = {}
        self.provider_exclusions = provider_exclusions or []
        self.api_meta_tool = None # Initialize api_meta_tool attribute
        
        # Load configuration if not already loaded
        if get_config() is None:
            self.logger.info("Initializing Gateway: Loading configuration...")
            load_config()
        
        # Initialize logger
        self.logger.info(f"Initializing {self.name}...")
        
        # Create MCP server with host and port settings
        self.mcp = FastMCP(
            self.name,
            lifespan=self._server_lifespan,
            host=get_config().server.host,
            port=get_config().server.port,
            instructions=self.system_instructions,
            timeout=300,
            debug=True
        )
        
        # Initialize the state store
        persistence_dir = None
        if get_config() and hasattr(get_config(), 'state_persistence') and hasattr(get_config().state_persistence, 'dir'):
            persistence_dir = get_config().state_persistence.dir
        self.state_store = StateStore(persistence_dir)
        
        # Connect state store to MCP server
        self._init_mcp()
        
        # Register tools if requested
        if register_tools:
            self._register_tools()
            self._register_resources()
        
        self.logger.info(f"LLM Gateway '{self.name}' initialized")
    
    def log_tool_calls(self, func):
        """Decorator to log MCP tool calls."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            tool_name = func.__name__
            
            # Format parameters for logging
            args_str = ", ".join([repr(arg) for arg in args[1:] if arg is not None])
            kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items() if k != 'ctx'])
            params_str = ", ".join(filter(None, [args_str, kwargs_str]))
            
            # Log the request - only through tools_logger
            tools_logger.info(f"TOOL CALL: {tool_name}({params_str})")
            
            try:
                result = await func(*args, **kwargs)
                processing_time = time.time() - start_time
                
                # Format result for logging
                if isinstance(result, dict):
                    result_keys = list(result.keys())
                    result_summary = f"dict with keys: {result_keys}"
                else:
                    result_str = str(result)
                    result_summary = (result_str[:100] + '...') if len(result_str) > 100 else result_str
                
                # Log successful completion - only through tools_logger
                tools_logger.info(f"TOOL SUCCESS: {tool_name} completed in {processing_time:.2f}s - Result: {result_summary}")
                
                return result
            except Exception as e:
                processing_time = time.time() - start_time
                tools_logger.error(f"TOOL ERROR: {tool_name} failed after {processing_time:.2f}s: {str(e)}", exc_info=True)
                raise
        return wrapper
    
    @asynccontextmanager
    async def _server_lifespan(self, server: FastMCP):
        """Server lifespan context manager.
        
        Args:
            server: MCP server instance
            
        Yields:
            Dict containing initialized resources
        """
        self.logger.info(f"Starting LLM Gateway '{self.name}'")
        
        # Initialize providers
        await self._initialize_providers()

        # --- Trigger Dynamic Docstring Generation ---
        # This should run after config is loaded but before the server is fully ready
        # It checks cache and potentially calls an LLM.
        self.logger.info("Initiating dynamic docstring generation for Marqo tool...")
        try:
            # Import the function here to avoid circular imports
            from llm_gateway.tools.marqo_fused_search import trigger_dynamic_docstring_generation
            await trigger_dynamic_docstring_generation()
            self.logger.info("Dynamic docstring generation/loading complete.")
        except Exception as e:
            self.logger.error(f"Error during dynamic docstring generation startup task: {e}", exc_info=True)
        # ---------------------------------------------

        # --- Set the global instance variable --- 
        # Make the fully initialized instance accessible globally AFTER init
        llm_gateway.core._gateway_instance = self
        self.logger.info("Global gateway instance set.")
        # ----------------------------------------

        # Create lifespan context (still useful for framework calls)
        context = {
            "providers": self.providers,
            "provider_status": self.provider_status,
        }
        
        self.logger.info("Lifespan context initialized, MCP server ready to handle requests")
        
        try:
            # Import and call trigger_dynamic_docstring_generation again
            from llm_gateway.tools.marqo_fused_search import trigger_dynamic_docstring_generation
            await trigger_dynamic_docstring_generation()
            logger.info("Dynamic docstring generation/loading complete.")
            yield context
        finally:
            # --- Clear the global instance on shutdown --- 
            llm_gateway.core._gateway_instance = None
            self.logger.info("Global gateway instance cleared.")
            # -------------------------------------------
            self.logger.info(f"Shutting down LLM Gateway '{self.name}'")
    
    async def _initialize_providers(self):
        """Initialize all enabled providers based on the loaded config."""
        self.logger.info("Initializing LLM providers")

        cfg = get_config()
        providers_to_init = []

        # Determine which providers to initialize based SOLELY on the loaded config
        for provider_name in [p.value for p in Provider]:
            # Skip providers that are in the exclusion list
            if provider_name in self.provider_exclusions:
                self.logger.debug(f"Skipping provider {provider_name} (excluded)")
                continue
                
            provider_config = getattr(cfg.providers, provider_name, None)
            # Check if the provider is enabled AND has an API key configured in the loaded settings
            if provider_config and provider_config.enabled and provider_config.api_key:
                self.logger.debug(f"Found configured and enabled provider: {provider_name}")
                providers_to_init.append(provider_name)
            elif provider_config and provider_config.enabled:
                self.logger.warning(f"Provider {provider_name} is enabled but missing API key in config. Skipping.")
            # else: # Provider not found in config or not enabled
            #     self.logger.debug(f"Provider {provider_name} not configured or not enabled.")

        # Initialize providers in parallel
        init_tasks = [
            asyncio.create_task(
                self._initialize_provider(provider_name),
                name=f"init-{provider_name}"
            )
            for provider_name in providers_to_init
        ]

        if init_tasks:
            await asyncio.gather(*init_tasks)

        # Log initialization summary
        available_providers = [
            name for name, status in self.provider_status.items()
            if status.available
        ]
        self.logger.info(f"Providers initialized: {len(available_providers)}/{len(providers_to_init)} available")

    async def _initialize_provider(self, provider_name: str):
        """Initialize a single provider using ONLY the loaded configuration."""
        api_key = None
        api_key_configured = False
        provider_config = None

        try:
            cfg = get_config()
            provider_config = getattr(cfg.providers, provider_name, None)

            # Get API key ONLY from the loaded config object
            if provider_config and provider_config.api_key:
                api_key = provider_config.api_key
                api_key_configured = True
            else:
                # This case should ideally not be reached if checks in _initialize_providers are correct,
                # but handle defensively.
                self.logger.warning(f"Attempted to initialize {provider_name}, but API key not found in loaded config.")
                api_key_configured = False

            if not api_key_configured:
                # Record status for providers found in config but without a key
                if provider_config:
                     self.provider_status[provider_name] = ProviderStatus(
                        enabled=provider_config.enabled, # Reflects config setting
                        available=False,
                        api_key_configured=False,
                        models=[],
                        error="API key not found in loaded configuration"
                    )
                # Do not log the warning here again, just return
                return

            # --- API Key is configured, proceed with initialization ---
            self.logger.debug(f"Initializing provider {provider_name} with key from config.")

            # Import provider classes (consider moving imports outside the loop if performance is critical)
            from llm_gateway.core.providers.anthropic import AnthropicProvider
            from llm_gateway.core.providers.deepseek import DeepSeekProvider
            from llm_gateway.core.providers.gemini import GeminiProvider
            from llm_gateway.core.providers.grok import GrokProvider
            from llm_gateway.core.providers.openai import OpenAIProvider
            from llm_gateway.core.providers.openrouter import OpenRouterProvider

            providers = {
                Provider.OPENAI.value: OpenAIProvider,
                Provider.ANTHROPIC.value: AnthropicProvider,
                Provider.DEEPSEEK.value: DeepSeekProvider,
                Provider.GEMINI.value: GeminiProvider,
                Provider.OPENROUTER.value: OpenRouterProvider,
                Provider.GROK.value: GrokProvider,
            }

            provider_class = providers.get(provider_name)
            if not provider_class:
                raise ValueError(f"Invalid provider name mapping: {provider_name}")

            # Instantiate provider with the API key retrieved from the config (via decouple)
            # Ensure provider classes' __init__ expect 'api_key' as a keyword argument
            provider = provider_class(api_key=api_key)

            # Initialize provider (which should use the config passed)
            available = await provider.initialize()

            # Update status based on initialization result
            if available:
                models = await provider.list_models()
                self.providers[provider_name] = provider
                self.provider_status[provider_name] = ProviderStatus(
                    enabled=provider_config.enabled,
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
                    enabled=provider_config.enabled,
                    available=False,
                    api_key_configured=True, # Key was found, but init failed
                    models=[],
                    error="Initialization failed (check provider API status or logs)"
                )
                self.logger.error(
                    f"Provider {provider_name} initialization failed",
                    emoji_key="error"
                )

        except Exception as e:
            # Handle unexpected errors during initialization
            error_msg = f"Error initializing provider {provider_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Ensure status is updated even on exceptions
            enabled_status = provider_config.enabled if provider_config else False # Best guess
            self.provider_status[provider_name] = ProviderStatus(
                enabled=enabled_status,
                available=False,
                api_key_configured=api_key_configured, # Reflects if key was found before error
                models=[],
                error=error_msg
            )
    
    @property
    def system_instructions(self) -> str:
        """Return system-level instructions for LLMs on how to use the gateway.
        
        These instructions are intended to be included in the system prompt for LLMs
        that will be using the LLM Gateway tools, to help them understand the most
        effective ways to use the available capabilities.
        
        Returns:
            String containing formatted instructions
        """
        return """
# LLM Gateway Tool Usage Instructions
        
You have access to the LLM Gateway, which provides unified access to multiple language model
providers (OpenAI, Anthropic, etc.) through a standardized interface. Follow these instructions
to effectively use the gateway tools.

## Core Tool Categories

1. **Provider Tools**: Use these to discover available providers and models
   - `get_provider_status`: Check which providers are available
   - `list_models`: List models available from a specific provider

2. **Completion Tools**: Use these for text generation
   - `generate_completion`: Single-prompt text generation (non-streaming)
   - `chat_completion`: Multi-turn conversation with message history
   - `multi_completion`: Compare outputs from multiple providers/models

3. **Tournament Tools**: Use these to run competitions between models
   - `create_tournament`: Create and start a new tournament
   - `get_tournament_status`: Check tournament progress
   - `get_tournament_results`: Get detailed tournament results
   - `list_tournaments`: List all tournaments
   - `cancel_tournament`: Cancel a running tournament

## Best Practices

1. **Provider Selection**:
   - Always check provider availability with `get_provider_status` before use
   - Verify model availability with `list_models` before using specific models

2. **Error Handling**:
   - All tools include error handling in their responses
   - Check for the presence of an "error" field in responses
   - If an error occurs, adapt your approach based on the error message

3. **Efficient Usage**:
   - Use cached tools when repeatedly calling the same function with identical parameters
   - For long-running operations like tournaments, poll status periodically

4. **Tool Selection Guidelines**:
   - For single-turn text generation → `generate_completion`
   - For conversation-based interactions → `chat_completion`
   - For comparing outputs across models → `multi_completion`
   - For evaluating model performance → Tournament tools

## Additional Resources

For more detailed information and examples, access these MCP resources:
- `info://server`: Basic server information
- `info://tools`: Overview of available tools
- `provider://{provider_name}`: Details about a specific provider
- `guide://llm`: Comprehensive usage guide for LLMs
- `guide://error-handling`: Detailed error handling guidance
- `examples://workflows`: Detailed examples of common workflows
- `examples://completions`: Examples of different completion types
- `examples://tournaments`: Guidance on tournament configuration and analysis

Remember to use appropriate error handling and follow the documented parameter formats
for each tool. All providers may not be available at all times, so always check status
first and be prepared to adapt to available providers.
"""
        
    def _register_tools(self):
        """Register MCP tools."""
        # Import here to avoid circular dependency
        from llm_gateway.tools import register_all_tools
        
        self.logger.info("Registering core tools...")
        # Echo tool
        @self.mcp.tool()
        @self.log_tool_calls
        async def echo(message: str, ctx: Context = None) -> Dict[str, Any]:
            """
            Echo back the message for testing MCP connectivity.
            
            Args:
                message: The message to echo back
                
            Returns:
                Dictionary containing the echoed message
            """
            self.logger.info(f"Echo tool called with message: {message}")
            return {"message": message}

        self.logger.info("Calling register_all_tools to register all tools...")
        register_all_tools(self.mcp)

    def _register_resources(self):
        """Register MCP resources."""
        
        @self.mcp.resource("info://server")
        def get_server_info() -> Dict[str, Any]:
            """
            Get information about the LLM Gateway server.
            
            This resource provides basic metadata about the LLM Gateway server instance,
            including its name, version, and supported providers. Use this resource to
            discover server capabilities and version information.
            
            Resource URI: info://server
            
            Returns:
                Dictionary containing server information:
                - name: Name of the LLM Gateway server
                - version: Version of the LLM Gateway server
                - description: Brief description of server functionality
                - providers: List of supported LLM provider names
                
            Example:
                {
                    "name": "LLM Gateway",
                    "version": "0.1.0",
                    "description": "MCP server for accessing multiple LLM providers",
                    "providers": ["openai", "anthropic", "deepseek", "gemini"]
                }
                
            Usage:
                This resource is useful for clients to verify server identity, check compatibility,
                and discover basic capabilities. For detailed provider status, use the
                get_provider_status tool instead.
            """
            return {
                "name": self.name,
                "version": "0.1.0",
                "description": "MCP server for accessing multiple LLM providers",
                "providers": [p.value for p in Provider],
            }
            
        @self.mcp.resource("info://tools")
        def get_tools_info() -> Dict[str, Any]:
            """
            Get information about available LLM Gateway tools.
            
            This resource provides a descriptive overview of the tools available in the
            LLM Gateway, organized by category. Use this resource to understand which
            tools are available and how they're organized.
            
            Resource URI: info://tools
            
            Returns:
                Dictionary containing tools information organized by category:
                - provider_tools: Tools for interacting with LLM providers
                - completion_tools: Tools for text generation and completion
                - tournament_tools: Tools for running model tournaments
                - document_tools: Tools for document processing
                
            Example:
                {
                    "provider_tools": {
                        "description": "Tools for accessing and managing LLM providers",
                        "tools": ["get_provider_status", "list_models"]
                    },
                    "completion_tools": {
                        "description": "Tools for text generation and completion",
                        "tools": ["generate_completion", "chat_completion", "multi_completion"]
                    },
                    "tournament_tools": {
                        "description": "Tools for running and managing model tournaments",
                        "tools": ["create_tournament", "list_tournaments", "get_tournament_status", 
                                 "get_tournament_results", "cancel_tournament"]
                    }
                }
                
            Usage:
                Use this resource to understand the capabilities of the LLM Gateway and
                discover available tools. For detailed information about specific tools,
                use the MCP list_tools method.
            """
            return {
                "provider_tools": {
                    "description": "Tools for accessing and managing LLM providers",
                    "tools": ["get_provider_status", "list_models"]
                },
                "completion_tools": {
                    "description": "Tools for text generation and completion",
                    "tools": ["generate_completion", "chat_completion", "multi_completion"]
                },
                "tournament_tools": {
                    "description": "Tools for running and managing model tournaments",
                    "tools": ["create_tournament", "list_tournaments", "get_tournament_status", 
                             "get_tournament_results", "cancel_tournament"]
                },
                "document_tools": {
                    "description": "Tools for document processing (placeholder for future implementation)",
                    "tools": []
                }
            }
            
        @self.mcp.resource("guide://llm")
        def get_llm_guide() -> str:
            """
            Usage guide for LLMs using the LLM Gateway.
            
            This resource provides structured guidance specifically designed for LLMs to
            effectively use the tools and resources provided by the LLM Gateway. It includes
            recommended tool selection strategies, common usage patterns, and examples.
            
            Resource URI: guide://llm
            
            Returns:
                A detailed text guide with sections on tool selection, usage patterns,
                and example workflows.
            
            Usage:
                This resource is primarily intended to be included in context for LLMs
                that will be using the gateway tools, to help them understand how to
                effectively use the available capabilities.
            """
            return """
                # LLM Gateway Usage Guide for Language Models
                
                ## Overview
                
                The LLM Gateway provides a set of tools for accessing multiple language model providers
                (OpenAI, Anthropic, etc.) through a unified interface. This guide will help you understand
                how to effectively use these tools.
                
                ## Tool Selection Guidelines
                
                ### For Text Generation:
                
                1. For single-prompt text generation:
                   - Use `generate_completion` with a specific provider and model
                
                2. For multi-turn conversations:
                   - Use `chat_completion` with a list of message dictionaries
                
                3. For streaming responses (real-time text output):
                   - Use streaming tools in the CompletionTools class
                
                4. For comparing outputs across providers:
                   - Use `multi_completion` with a list of provider configurations
                
                ### For Provider Management:
                
                1. To check available providers:
                   - Use `get_provider_status` to see which providers are available
                
                2. To list available models:
                   - Use `list_models` to view models from all providers or a specific provider
                
                ### For Running Tournaments:
                
                1. To create a new tournament:
                   - Use `create_tournament` with a prompt and list of model IDs
                
                2. To check tournament status:
                   - Use `get_tournament_status` with a tournament ID
                
                3. To get detailed tournament results:
                   - Use `get_tournament_results` with a tournament ID
                
                ## Common Workflows
                
                ### Provider Selection Workflow:
                ```
                1. Call get_provider_status() to see available providers
                2. Call list_models(provider="openai") to see available models
                3. Call generate_completion(prompt="...", provider="openai", model="gpt-4o")
                ```
                
                ### Multi-Provider Comparison Workflow:
                ```
                1. Call multi_completion(
                      prompt="...",
                      providers=[
                          {"provider": "openai", "model": "gpt-4o"},
                          {"provider": "anthropic", "model": "claude-3-opus-20240229"}
                      ]
                   )
                2. Compare results from each provider
                ```
                
                ### Tournament Workflow:
                ```
                1. Call create_tournament(name="...", prompt="...", model_ids=["openai/gpt-4o", "anthropic/claude-3-opus"])
                2. Store the tournament_id from the response
                3. Call get_tournament_status(tournament_id="...") to monitor progress
                4. Once status is "COMPLETED", call get_tournament_results(tournament_id="...")
                ```
                
                ## Error Handling Best Practices
                
                1. Always check for "error" fields in tool responses
                2. Verify provider availability before attempting to use specific models
                3. For tournament tools, handle potential 404 errors for invalid tournament IDs
                
                ## Performance Considerations
                
                1. Most completion tools include token usage and cost metrics in their responses
                2. Use caching decorators for repetitive requests to save costs
                3. Consider using stream=True for long completions to improve user experience
            """
            
        @self.mcp.resource("provider://{{provider_name}}")
        def get_provider_info(provider_name: str) -> Dict[str, Any]:
            """
            Get detailed information about a specific LLM provider.
            
            This resource provides comprehensive information about a specific provider,
            including its capabilities, available models, and configuration status.
            
            Resource URI template: provider://{provider_name}
            
            Args:
                provider_name: Name of the provider to retrieve information for
                              (e.g., "openai", "anthropic", "gemini")
                              
            Returns:
                Dictionary containing detailed provider information:
                - name: Provider name
                - status: Current status (enabled, available, etc.)
                - capabilities: List of supported capabilities
                - models: List of available models and their details
                - config: Current configuration settings (with sensitive info redacted)
                
            Example:
                {
                    "name": "openai",
                    "status": {
                        "enabled": true,
                        "available": true,
                        "api_key_configured": true,
                        "error": null
                    },
                    "capabilities": ["chat", "completion", "embeddings", "vision"],
                    "models": [
                        {
                            "id": "gpt-4o",
                            "name": "GPT-4o",
                            "context_window": 128000,
                            "features": ["chat", "completion", "vision"]
                        },
                        # More models...
                    ],
                    "config": {
                        "base_url": "https://api.openai.com/v1",
                        "timeout_seconds": 30,
                        "default_model": "gpt-4.1-mini"
                    }
                }
                
            Error Handling:
                If the provider doesn't exist or isn't configured, returns an appropriate
                error message in the response.
                
            Usage:
                Use this resource to get detailed information about a specific provider
                before using its models for completions or other operations.
            """
            # Check if provider exists in status dictionary
            provider_status = self.provider_status.get(provider_name)
            if not provider_status:
                return {
                    "name": provider_name,
                    "error": f"Provider '{provider_name}' not found or not configured",
                    "status": {
                        "enabled": False,
                        "available": False,
                        "api_key_configured": False
                    },
                    "models": []
                }
                
            # Get provider instance if available
            provider_instance = self.providers.get(provider_name)
                
            # Build capability list based on provider name
            capabilities = []
            if provider_name in [Provider.OPENAI.value, Provider.ANTHROPIC.value, Provider.GEMINI.value]:
                capabilities = ["chat", "completion"]
                
            if provider_name == Provider.OPENAI.value:
                capabilities.extend(["embeddings", "vision", "image_generation"])
            elif provider_name == Provider.ANTHROPIC.value:
                capabilities.extend(["vision"])
                
            # Return provider details
            return {
                "name": provider_name,
                "status": {
                    "enabled": provider_status.enabled,
                    "available": provider_status.available,
                    "api_key_configured": provider_status.api_key_configured,
                    "error": provider_status.error
                },
                "capabilities": capabilities,
                "models": provider_status.models,
                "config": {
                    # Include non-sensitive config info
                    "default_model": provider_instance.default_model if provider_instance else None,
                    "timeout_seconds": 30  # Example default
                }
            }
            
        @self.mcp.resource("guide://error-handling")
        def get_error_handling_guide() -> Dict[str, Any]:
            """
            Get comprehensive guidance on handling errors from LLM Gateway tools.
            
            This resource provides detailed information about common error patterns,
            error handling strategies, and recovery approaches for each tool in the
            LLM Gateway. It helps LLMs understand how to gracefully handle and recover
            from various error conditions.
            
            Resource URI: guide://error-handling
            
            Returns:
                Dictionary containing error handling guidance organized by tool type:
                - provider_tools: Error handling for provider-related tools
                - completion_tools: Error handling for completion tools
                - tournament_tools: Error handling for tournament tools
                
            Usage:
                This resource helps LLMs implement robust error handling when using
                the LLM Gateway tools, improving the resilience of their interactions.
            """
            return {
                "general_principles": {
                    "error_detection": {
                        "description": "How to detect errors in tool responses",
                        "patterns": [
                            "Check for an 'error' field in the response dictionary",
                            "Look for status codes in error messages (e.g., 404, 500)",
                            "Check for empty or null results where data is expected",
                            "Look for 'warning' fields that may indicate partial success"
                        ]
                    },
                    "error_recovery": {
                        "description": "General strategies for recovering from errors",
                        "strategies": [
                            "Retry with different parameters when appropriate",
                            "Fallback to alternative tools or providers",
                            "Gracefully degrade functionality when optimal path is unavailable",
                            "Clearly communicate errors to users with context and suggestions"
                        ]
                    }
                },
                
                "provider_tools": {
                    "get_provider_status": {
                        "common_errors": [
                            {
                                "error": "Server context not available",
                                "cause": "The server may not be fully initialized",
                                "handling": "Wait and retry or report server initialization issue"
                            },
                            {
                                "error": "No providers are currently configured",
                                "cause": "No LLM providers are enabled or initialization is incomplete",
                                "handling": "Proceed with caution and check if specific providers are required"
                            }
                        ],
                        "recovery_strategies": [
                            "If no providers are available, clearly inform the user of limited capabilities",
                            "If specific providers are unavailable, suggest alternatives based on task requirements"
                        ]
                    },
                    
                    "list_models": {
                        "common_errors": [
                            {
                                "error": "Invalid provider",
                                "cause": "Specified provider name doesn't exist or isn't configured",
                                "handling": "Use valid providers from the error message's 'valid_providers' field"
                            },
                            {
                                "warning": "Provider is configured but not available",
                                "cause": "Provider API key issues or service connectivity problems",
                                "handling": "Use an alternative provider or inform user of limited options"
                            }
                        ],
                        "recovery_strategies": [
                            "When provider is invalid, fall back to listing all available providers",
                            "When models list is empty, suggest using the default model or another provider"
                        ]
                    }
                },
                
                "completion_tools": {
                    "generate_completion": {
                        "common_errors": [
                            {
                                "error": "Provider not available",
                                "cause": "Specified provider doesn't exist or isn't configured",
                                "handling": "Switch to an available provider (check with get_provider_status)"
                            },
                            {
                                "error": "Failed to initialize provider",
                                "cause": "API key configuration or network issues",
                                "handling": "Try another provider or check provider status"
                            },
                            {
                                "error": "Completion generation failed",
                                "cause": "Provider API errors, rate limits, or invalid parameters",
                                "handling": "Retry with different parameters or use another provider"
                            }
                        ],
                        "recovery_strategies": [
                            "Use multi_completion to try multiple providers simultaneously",
                            "Progressively reduce complexity (max_tokens, simplify prompt) if facing limits",
                            "Fall back to more reliable models if specialized ones are unavailable"
                        ]
                    },
                    
                    "multi_completion": {
                        "common_errors": [
                            {
                                "error": "Invalid providers format",
                                "cause": "Providers parameter is not a list of provider configurations",
                                "handling": "Correct the format to a list of dictionaries with provider info"
                            },
                            {
                                "partial_failure": "Some providers failed",
                                "cause": "Indicated by successful_count < total_providers",
                                "handling": "Use the successful results and analyze error fields for failed ones"
                            }
                        ],
                        "recovery_strategies": [
                            "Focus on successful completions even if some providers failed",
                            "Check each provider's 'success' field to identify which ones worked",
                            "If timeout occurs, consider increasing the timeout parameter or reducing providers"
                        ]
                    }
                },
                
                "tournament_tools": {
                    "create_tournament": {
                        "common_errors": [
                            {
                                "error": "Invalid input",
                                "cause": "Missing required fields or validation errors",
                                "handling": "Check all required parameters are provided with valid values"
                            },
                            {
                                "error": "Failed to start tournament execution",
                                "cause": "Server resource constraints or initialization errors",
                                "handling": "Retry with fewer rounds or models, or try again later"
                            }
                        ],
                        "recovery_strategies": [
                            "Verify model IDs are valid before creating tournament",
                            "Start with simple tournaments to validate functionality before complex ones",
                            "Use error message details to correct specific input problems"
                        ]
                    },
                    
                    "get_tournament_status": {
                        "common_errors": [
                            {
                                "error": "Tournament not found",
                                "cause": "Invalid tournament ID or tournament was deleted",
                                "handling": "Verify tournament ID or use list_tournaments to see available tournaments"
                            },
                            {
                                "error": "Invalid tournament ID format",
                                "cause": "Tournament ID is not a string or is empty",
                                "handling": "Ensure tournament ID is a valid string matching the expected format"
                            }
                        ],
                        "recovery_strategies": [
                            "When tournament not found, list all tournaments to find valid ones",
                            "If tournament status is FAILED, check error_message for details",
                            "Implement polling with backoff for monitoring long-running tournaments"
                        ]
                    }
                },
                
                "error_pattern_examples": {
                    "retry_with_fallback": {
                        "description": "Retry with fallback to another provider",
                        "example": """
                            # Try primary provider
                            result = generate_completion(prompt="...", provider="openai", model="gpt-4o")
                            
                            # Check for errors and fall back if needed
                            if "error" in result:
                                logger.warning(f"Primary provider failed: {result['error']}")
                                # Fall back to alternative provider
                                result = generate_completion(prompt="...", provider="anthropic", model="claude-3-opus-20240229")
                        """
                    },
                    "validation_before_call": {
                        "description": "Validate parameters before making tool calls",
                        "example": """
                            # Get available providers first
                            provider_status = get_provider_status()
                            
                            # Check if requested provider is available
                            requested_provider = "openai"
                            if requested_provider not in provider_status["providers"] or not provider_status["providers"][requested_provider]["available"]:
                                # Fall back to any available provider
                                available_providers = [p for p, status in provider_status["providers"].items() if status["available"]]
                                if available_providers:
                                    requested_provider = available_providers[0]
                                else:
                                    return {"error": "No LLM providers are available"}
                        """
                    }
                }
            }

        @self.mcp.resource("examples://workflows")
        def get_workflow_examples() -> Dict[str, Any]:
            """
            Get comprehensive examples of multi-tool workflows.
            
            This resource provides detailed, executable examples showing how to combine
            multiple tools into common workflows. These examples demonstrate best practices
            for tool sequencing, error handling, and result processing.
            
            Resource URI: examples://workflows
            
            Returns:
                Dictionary containing workflow examples organized by scenario:
                - basic_provider_selection: Example of selecting a provider and model
                - model_comparison: Example of comparing outputs across providers
                - tournaments: Example of creating and monitoring a tournament
                - advanced_chat: Example of a multi-turn conversation with system prompts
                
            Usage:
                These examples are designed to be used as reference by LLMs to understand
                how to combine multiple tools in the LLM Gateway to accomplish common tasks.
                Each example includes expected outputs to help understand the flow.
            """
            return {
                "basic_provider_selection": {
                    "description": "Selecting a provider and model for text generation",
                    "steps": [
                        {
                            "step": 1,
                            "tool": "get_provider_status",
                            "parameters": {},
                            "purpose": "Check which providers are available",
                            "example_output": {
                                "providers": {
                                    "openai": {"available": True, "models_count": 12},
                                    "anthropic": {"available": True, "models_count": 6}
                                }
                            }
                        },
                        {
                            "step": 2,
                            "tool": "list_models",
                            "parameters": {"provider": "openai"},
                            "purpose": "Get available models for the selected provider",
                            "example_output": {
                                "models": {
                                    "openai": [
                                        {"id": "gpt-4o", "name": "GPT-4o", "features": ["chat", "completion"]}
                                    ]
                                }
                            }
                        },
                        {
                            "step": 3,
                            "tool": "generate_completion",
                            "parameters": {
                                "prompt": "Explain quantum computing in simple terms",
                                "provider": "openai",
                                "model": "gpt-4o",
                                "temperature": 0.7
                            },
                            "purpose": "Generate text with the selected provider and model",
                            "example_output": {
                                "text": "Quantum computing is like...",
                                "model": "gpt-4o",
                                "provider": "openai",
                                "tokens": {"input": 8, "output": 150, "total": 158},
                                "cost": 0.000123
                            }
                        }
                    ],
                    "error_handling": [
                        "If get_provider_status shows provider unavailable, try a different provider",
                        "If list_models returns empty list, select a different provider",
                        "If generate_completion returns an error, check the error message for guidance"
                    ]
                },
                
                "model_comparison": {
                    "description": "Comparing multiple models on the same task",
                    "steps": [
                        {
                            "step": 1,
                            "tool": "multi_completion",
                            "parameters": {
                                "prompt": "Write a haiku about programming",
                                "providers": [
                                    {"provider": "openai", "model": "gpt-4o"},
                                    {"provider": "anthropic", "model": "claude-3-opus-20240229"}
                                ],
                                "temperature": 0.7
                            },
                            "purpose": "Generate completions from multiple providers simultaneously",
                            "example_output": {
                                "results": {
                                    "openai/gpt-4o": {
                                        "success": True,
                                        "text": "Code flows like water\nBugs emerge from the depths\nPatience brings order",
                                        "model": "gpt-4o"
                                    },
                                    "anthropic/claude-3-opus-20240229": {
                                        "success": True,
                                        "text": "Fingers dance on keys\nLogic blooms in silent thought\nPrograms come alive",
                                        "model": "claude-3-opus-20240229"
                                    }
                                },
                                "successful_count": 2,
                                "total_providers": 2
                            }
                        },
                        {
                            "step": 2,
                            "suggestion": "Compare the results for quality, style, and adherence to the haiku format"
                        }
                    ],
                    "error_handling": [
                        "Check successful_count vs total_providers to see if all providers succeeded",
                        "For each provider, check the success field to determine if it completed successfully",
                        "If a provider failed, look at its error field for details"
                    ]
                },
                
                "tournaments": {
                    "description": "Creating and monitoring a multi-model tournament",
                    "steps": [
                        {
                            "step": 1,
                            "tool": "create_tournament",
                            "parameters": {
                                "name": "Sorting Algorithm Tournament",
                                "prompt": "Implement a quicksort algorithm in Python that handles duplicates efficiently",
                                "model_ids": ["openai/gpt-4o", "anthropic/claude-3-opus-20240229"],
                                "rounds": 3,
                                "tournament_type": "code"
                            },
                            "purpose": "Create a new tournament comparing multiple models",
                            "example_output": {
                                "tournament_id": "tour_abc123xyz789",
                                "status": "PENDING"
                            }
                        },
                        {
                            "step": 2,
                            "tool": "get_tournament_status",
                            "parameters": {
                                "tournament_id": "tour_abc123xyz789"
                            },
                            "purpose": "Check if the tournament has started running",
                            "example_output": {
                                "tournament_id": "tour_abc123xyz789",
                                "status": "RUNNING",
                                "current_round": 1,
                                "total_rounds": 3
                            }
                        },
                        {
                            "step": 3,
                            "suggestion": "Wait for the tournament to complete",
                            "purpose": "Tournaments run asynchronously and may take time to complete"
                        },
                        {
                            "step": 4,
                            "tool": "get_tournament_results",
                            "parameters": {
                                "tournament_id": "tour_abc123xyz789"
                            },
                            "purpose": "Retrieve full results once the tournament is complete",
                            "example_output": {
                                "tournament_id": "tour_abc123xyz789",
                                "status": "COMPLETED",
                                "rounds_data": [
                                    {
                                        "round_number": 1,
                                        "model_outputs": {
                                            "openai/gpt-4o": "def quicksort(arr): ...",
                                            "anthropic/claude-3-opus-20240229": "def quicksort(arr): ..."
                                        },
                                        "scores": {
                                            "openai/gpt-4o": 0.85,
                                            "anthropic/claude-3-opus-20240229": 0.92
                                        }
                                    }
                                    # Additional rounds would be here in a real response
                                ]
                            }
                        }
                    ],
                    "error_handling": [
                        "If create_tournament fails, check the error message for missing or invalid parameters",
                        "If get_tournament_status returns an error, verify the tournament_id is correct",
                        "If tournament status is FAILED, check the error_message field for details"
                    ]
                },
                
                "advanced_chat": {
                    "description": "Multi-turn conversation with system prompt and context",
                    "steps": [
                        {
                            "step": 1,
                            "tool": "chat_completion",
                            "parameters": {
                                "messages": [
                                    {"role": "user", "content": "Hello, can you help me with Python?"}
                                ],
                                "provider": "anthropic",
                                "model": "claude-3-opus-20240229",
                                "system_prompt": "You are an expert Python tutor. Provide concise, helpful answers with code examples when appropriate.",
                                "temperature": 0.5
                            },
                            "purpose": "Start a conversation with a system prompt for context",
                            "example_output": {
                                "text": "Hello! I'd be happy to help you with Python. What specific aspect are you interested in learning about?",
                                "model": "claude-3-opus-20240229",
                                "provider": "anthropic"
                            }
                        },
                        {
                            "step": 2,
                            "tool": "chat_completion",
                            "parameters": {
                                "messages": [
                                    {"role": "user", "content": "Hello, can you help me with Python?"},
                                    {"role": "assistant", "content": "Hello! I'd be happy to help you with Python. What specific aspect are you interested in learning about?"},
                                    {"role": "user", "content": "How do I write a function that checks if a string is a palindrome?"}
                                ],
                                "provider": "anthropic",
                                "model": "claude-3-opus-20240229",
                                "system_prompt": "You are an expert Python tutor. Provide concise, helpful answers with code examples when appropriate.",
                                "temperature": 0.5
                            },
                            "purpose": "Continue the conversation by including the full message history",
                            "example_output": {
                                "text": "Here's a simple function to check if a string is a palindrome in Python:\n\n```python\ndef is_palindrome(s):\n    # Remove spaces and convert to lowercase for more flexible matching\n    s = s.lower().replace(' ', '')\n    # Compare the string with its reverse\n    return s == s[::-1]\n\n# Examples\nprint(is_palindrome('racecar'))  # True\nprint(is_palindrome('hello'))    # False\nprint(is_palindrome('A man a plan a canal Panama'))  # True\n```\n\nThis function works by:\n1. Converting the string to lowercase and removing spaces\n2. Checking if the processed string equals its reverse (using slice notation `[::-1]`)\n\nIs there anything specific about this solution you'd like me to explain further?",
                                "model": "claude-3-opus-20240229",
                                "provider": "anthropic"
                            }
                        }
                    ],
                    "error_handling": [
                        "Always include the full conversation history in the messages array",
                        "Ensure each message has both 'role' and 'content' fields",
                        "If using system_prompt, ensure it's appropriate for the provider"
                    ]
                }
            }

        @self.mcp.resource("examples://completions")
        def get_completion_examples() -> Dict[str, Any]:
            """
            Get examples of different completion types and when to use them.
            
            This resource provides detailed examples of different completion tools available
            in the LLM Gateway, along with guidance on when to use each type. It helps with
            selecting the most appropriate completion tool for different scenarios.
            
            Resource URI: examples://completions
            
            Returns:
                Dictionary containing completion examples organized by type:
                - standard_completion: When to use generate_completion
                - chat_completion: When to use chat_completion
                - streaming_completion: When to use stream_completion
                - multi_provider: When to use multi_completion
                
            Usage:
                This resource helps LLMs understand the appropriate completion tool
                to use for different scenarios, with concrete examples and use cases.
            """
            return {
                "standard_completion": {
                    "tool": "generate_completion",
                    "description": "Single-turn text generation without streaming",
                    "best_for": [
                        "Simple, one-off text generation tasks",
                        "When you need a complete response at once",
                        "When you don't need conversation history"
                    ],
                    "example": {
                        "request": {
                            "prompt": "Explain the concept of quantum entanglement in simple terms",
                            "provider": "openai",
                            "model": "gpt-4o",
                            "temperature": 0.7
                        },
                        "response": {
                            "text": "Quantum entanglement is like having two magic coins...",
                            "model": "gpt-4o",
                            "provider": "openai",
                            "tokens": {"input": 10, "output": 150, "total": 160},
                            "cost": 0.00032,
                            "processing_time": 2.1
                        }
                    }
                },
                
                "chat_completion": {
                    "tool": "chat_completion",
                    "description": "Multi-turn conversation with message history",
                    "best_for": [
                        "Maintaining conversation context across multiple turns",
                        "When dialogue history matters for the response",
                        "When using system prompts to guide assistant behavior"
                    ],
                    "example": {
                        "request": {
                            "messages": [
                                {"role": "user", "content": "What's the capital of France?"},
                                {"role": "assistant", "content": "The capital of France is Paris."},
                                {"role": "user", "content": "And what's its population?"}
                            ],
                            "provider": "anthropic",
                            "model": "claude-3-opus-20240229",
                            "system_prompt": "You are a helpful geography assistant."
                        },
                        "response": {
                            "text": "The population of Paris is approximately 2.1 million people in the city proper...",
                            "model": "claude-3-opus-20240229",
                            "provider": "anthropic",
                            "tokens": {"input": 62, "output": 48, "total": 110},
                            "cost": 0.00055,
                            "processing_time": 1.8
                        }
                    }
                },
                
                "streaming_completion": {
                    "tool": "stream_completion",
                    "description": "Generates text in smaller chunks as a stream",
                    "best_for": [
                        "When you need to show incremental progress to users",
                        "For real-time display of model outputs",
                        "Long-form content generation where waiting for the full response would be too long"
                    ],
                    "example": {
                        "request": {
                            "prompt": "Write a short story about a robot learning to paint",
                            "provider": "openai",
                            "model": "gpt-4o"
                        },
                        "response_chunks": [
                            {
                                "text": "In the year 2150, ",
                                "chunk_index": 1,
                                "provider": "openai",
                                "model": "gpt-4o",
                                "finished": False
                            },
                            {
                                "text": "a maintenance robot named ARIA-7 was assigned to",
                                "chunk_index": 2,
                                "provider": "openai",
                                "model": "gpt-4o",
                                "finished": False
                            },
                            {
                                "text": "",
                                "chunk_index": 25,
                                "provider": "openai",
                                "full_text": "In the year 2150, a maintenance robot named ARIA-7 was assigned to...",
                                "processing_time": 8.2,
                                "finished": True
                            }
                        ]
                    }
                },
                
                "multi_provider": {
                    "tool": "multi_completion",
                    "description": "Get completions from multiple providers simultaneously",
                    "best_for": [
                        "Comparing outputs from different models",
                        "Finding consensus among multiple models",
                        "Fallback scenarios where one provider might fail",
                        "Benchmarking different providers on the same task"
                    ],
                    "example": {
                        "request": {
                            "prompt": "Provide three tips for sustainable gardening",
                            "providers": [
                                {"provider": "openai", "model": "gpt-4o"},
                                {"provider": "anthropic", "model": "claude-3-opus-20240229"}
                            ]
                        },
                        "response": {
                            "results": {
                                "openai/gpt-4o": {
                                    "provider_key": "openai/gpt-4o",
                                    "success": True,
                                    "text": "1. Use compost instead of chemical fertilizers...",
                                    "model": "gpt-4o"
                                },
                                "anthropic/claude-3-opus-20240229": {
                                    "provider_key": "anthropic/claude-3-opus-20240229",
                                    "success": True,
                                    "text": "1. Implement water conservation techniques...",
                                    "model": "claude-3-opus-20240229"
                                }
                            },
                            "successful_count": 2,
                            "total_providers": 2,
                            "processing_time": 3.5
                        }
                    }
                }
            }

        @self.mcp.resource("examples://tournaments")
        def get_tournament_examples() -> Dict[str, Any]:
            """
            Get detailed examples and guidance for running LLM tournaments.
            
            This resource provides comprehensive examples and guidance for creating,
            monitoring, and analyzing LLM tournaments. It includes detailed information
            about tournament configuration, interpreting results, and best practices.
            
            Resource URI: examples://tournaments
            
            Returns:
                Dictionary containing tournament examples and guidance:
                - tournament_types: Different types of tournaments and their uses
                - configuration_guide: Guidance on how to configure tournaments
                - analysis_guide: How to interpret tournament results
                - example_tournaments: Complete examples of different tournament configurations
                
            Usage:
                This resource helps LLMs understand how to effectively use the tournament
                tools, with guidance on configuration, execution, and analysis.
            """
            return {
                "tournament_types": {
                    "code": {
                        "description": "Tournaments where models compete on coding tasks",
                        "ideal_for": [
                            "Algorithm implementation challenges",
                            "Debugging exercises",
                            "Code optimization problems",
                            "Comparing models' coding abilities"
                        ],
                        "evaluation_criteria": [
                            "Code correctness",
                            "Efficiency",
                            "Readability",
                            "Error handling"
                        ]
                    },
                    # Other tournament types could be added in the future
                },
                
                "configuration_guide": {
                    "model_selection": {
                        "description": "Guidelines for selecting models to include in tournaments",
                        "recommendations": [
                            "Include models from different providers for diverse approaches",
                            "Compare models within the same family (e.g., different Claude versions)",
                            "Consider including both specialized and general models",
                            "Ensure all models can handle the task complexity"
                        ]
                    },
                    "rounds": {
                        "description": "How to determine the appropriate number of rounds",
                        "recommendations": [
                            "Start with 3 rounds for most tournaments",
                            "Use more rounds (5+) for more complex or nuanced tasks",
                            "Consider that each round increases total runtime and cost",
                            "Each round gives models a chance to refine their solutions"
                        ]
                    },
                    "prompt_design": {
                        "description": "Best practices for tournament prompt design",
                        "recommendations": [
                            "Be specific about the problem requirements",
                            "Clearly define evaluation criteria",
                            "Specify output format expectations",
                            "Consider including test cases",
                            "Avoid ambiguous or underspecified requirements"
                        ]
                    }
                },
                
                "analysis_guide": {
                    "score_interpretation": {
                        "description": "How to interpret model scores in tournament results",
                        "guidance": [
                            "Scores are normalized to a 0-1 scale (1 being perfect)",
                            "Consider relative scores between models rather than absolute values",
                            "Look for consistency across rounds",
                            "Consider output quality even when scores are similar"
                        ]
                    },
                    "output_analysis": {
                        "description": "How to analyze model outputs from tournaments",
                        "guidance": [
                            "Compare approaches used by different models",
                            "Look for patterns in errors or limitations",
                            "Identify unique strengths of different providers",
                            "Consider both the score and actual output quality"
                        ]
                    }
                },
                
                "example_tournaments": {
                    "algorithm_implementation": {
                        "name": "Binary Search Algorithm",
                        "prompt": "Implement a binary search algorithm in Python that can search for an element in a sorted array. Include proper error handling, documentation, and test cases.",
                        "model_ids": ["openai/gpt-4o", "anthropic/claude-3-opus-20240229"],
                        "rounds": 3,
                        "tournament_type": "code",
                        "explanation": "This tournament tests the models' ability to implement a standard algorithm with proper error handling and testing."
                    },
                    "code_optimization": {
                        "name": "String Processing Optimization",
                        "prompt": "Optimize the following Python function to process large strings more efficiently: def find_substring_occurrences(text, pattern): return [i for i in range(len(text)) if text[i:i+len(pattern)] == pattern]",
                        "model_ids": ["openai/gpt-4o", "anthropic/claude-3-opus-20240229", "anthropic/claude-3-sonnet-20240229"],
                        "rounds": 4,
                        "tournament_type": "code",
                        "explanation": "This tournament compares models' ability to recognize and implement optimization opportunities in existing code."
                    }
                },
                
                "workflow_examples": {
                    "basic_tournament": {
                        "description": "A simple tournament workflow from creation to result analysis",
                        "steps": [
                            {
                                "step": 1,
                                "description": "Create the tournament",
                                "code": "tournament_id = create_tournament(name='Sorting Algorithm Challenge', prompt='Implement an efficient sorting algorithm...', model_ids=['openai/gpt-4o', 'anthropic/claude-3-opus-20240229'], rounds=3, tournament_type='code')"
                            },
                            {
                                "step": 2,
                                "description": "Poll for tournament status",
                                "code": "status = get_tournament_status(tournament_id)['status']\nwhile status in ['PENDING', 'RUNNING']:\n    time.sleep(30)  # Check every 30 seconds\n    status = get_tournament_status(tournament_id)['status']"
                            },
                            {
                                "step": 3,
                                "description": "Retrieve and analyze results",
                                "code": "results = get_tournament_results(tournament_id)\nwinner = max(results['final_scores'].items(), key=lambda x: x[1])[0]\noutputs = {model_id: results['rounds_data'][-1]['model_outputs'][model_id] for model_id in results['config']['model_ids']}"
                            }
                        ]
                    }
                }
            }

    def _init_mcp(self):
        # Existing MCP initialization
        # ...
        
        # Attach state store to MCP
        if hasattr(self, 'mcp') and hasattr(self, 'state_store'):
            self.mcp.state_store = self.state_store
            
        # ... rest of MCP initialization ...

def create_server() -> FastAPI:
    """Create and configure the FastAPI server."""
    global _server_app
    
    # Check if server already exists
    if _server_app is not None:
        return _server_app
        
    # Initialize the gateway if not already created
    global _gateway_instance
    if not _gateway_instance:
        _gateway_instance = Gateway()
    
    # Use FastMCP's app directly instead of mounting it
    app = _gateway_instance.mcp.app
    
    # Add CORS middleware
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add health check endpoint
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "version": "0.1.0",
        }
    
    # Store the app instance
    _server_app = app
    
    return app

def start_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    log_level: Optional[str] = None,
    reload: bool = False,
    transport_mode: str = "stdio",  # Changed default from "sse" to "stdio"
    include_tools: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None,
) -> None:
    """Start the LLM Gateway Server using dictConfig for logging.
    
    Args:
        host: Host to bind to (default: from config)
        port: Port to listen on (default: from config)
        workers: Number of worker processes (default: from config)
        log_level: Log level (default: from config)
        reload: Whether to reload the server on code changes
        transport_mode: Transport mode to use ("sse" for HTTP or "stdio" for direct process communication)
        include_tools: List of tool names to include (default: include all)
        exclude_tools: List of tool names to exclude (takes precedence over include_tools)
    """
    server_host = host or get_config().server.host
    server_port = port or get_config().server.port
    server_workers = workers or get_config().server.workers
    
    # Get the current config and update tool registration settings
    cfg = get_config()
    if include_tools or exclude_tools:
        cfg.tool_registration.filter_enabled = True
        
    if include_tools:
        cfg.tool_registration.included_tools = include_tools
        
    if exclude_tools:
        cfg.tool_registration.excluded_tools = exclude_tools
    
    # Validate transport_mode
    if transport_mode not in ["sse", "stdio"]:
        raise ValueError(f"Invalid transport_mode: {transport_mode}. Must be 'sse' or 'stdio'")
    
    # Determine final log level from the provided parameter or fallback to INFO
    final_log_level = (log_level or "INFO").upper()

    # Update LOGGING_CONFIG with the final level
    LOGGING_CONFIG["root"]["level"] = final_log_level
    LOGGING_CONFIG["loggers"]["llm_gateway"]["level"] = final_log_level
    LOGGING_CONFIG["loggers"]["llm_gateway.tools"]["level"] = final_log_level
    LOGGING_CONFIG["loggers"]["llm_gateway.completions"]["level"] = final_log_level
    
    # Set Uvicorn access level based on final level
    LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = final_log_level if final_log_level != "CRITICAL" else "CRITICAL"
    
    # Ensure Uvicorn base/error logs are at least INFO unless final level is DEBUG
    uvicorn_base_level = "INFO" if final_log_level not in ["DEBUG"] else "DEBUG"
    LOGGING_CONFIG["loggers"]["uvicorn"]["level"] = uvicorn_base_level
    LOGGING_CONFIG["loggers"]["uvicorn.error"]["level"] = uvicorn_base_level

    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Initialize the gateway if not already created
    global _gateway_instance
    if not _gateway_instance:
        # Create gateway with tool filtering based on config
        cfg = get_config()
        _gateway_instance = Gateway(register_tools=True)
    
    # Log startup info to stderr instead of using logging directly
    print("Starting LLM Gateway server", file=sys.stderr)
    print(f"Host: {server_host}", file=sys.stderr)
    print(f"Port: {server_port}", file=sys.stderr)
    print(f"Workers: {server_workers}", file=sys.stderr)
    print(f"Log level: {final_log_level}", file=sys.stderr)
    print(f"Transport mode: {transport_mode}", file=sys.stderr)
    
    # Log tool filtering info if enabled
    if cfg.tool_registration.filter_enabled:
        if cfg.tool_registration.included_tools:
            print(f"Including tools: {', '.join(cfg.tool_registration.included_tools)}", file=sys.stderr)
        if cfg.tool_registration.excluded_tools:
            print(f"Excluding tools: {', '.join(cfg.tool_registration.excluded_tools)}", file=sys.stderr)
    
    if transport_mode == "sse":
        # Run in SSE mode (HTTP server)
        import uvicorn
        
        # Get the SSE app from FastMCP
        app = _gateway_instance.mcp.sse_app()
        
        # Log SSE endpoint
        print(f"SSE endpoint available at: http://{server_host}:{server_port}/sse", file=sys.stderr)
        
        # Run the app with uvicorn
        uvicorn.run(
            app,
            host=server_host,
            port=server_port,
            log_config=LOGGING_CONFIG,
            log_level=final_log_level.lower(),
        )
    else:
        # Run in stdio mode (direct process communication)
        print("Running in stdio mode for direct process communication", file=sys.stderr)
        _gateway_instance.mcp.run()