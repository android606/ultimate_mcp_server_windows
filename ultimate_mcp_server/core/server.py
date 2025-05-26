"""Main server implementation for Ultimate MCP Server."""
import asyncio
import logging
import logging.config
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from mcp.server.fastmcp import Context, FastMCP

import ultimate_mcp_server

# Import core specifically to set the global instance
import ultimate_mcp_server.core
from ultimate_mcp_server.config import get_config, load_config
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.state_store import StateStore
from ultimate_mcp_server.graceful_shutdown import (
    create_quiet_server,
    enable_quiet_shutdown,
    register_shutdown_handler,
)
from ultimate_mcp_server.tools.smart_browser import (
    _ensure_initialized as smart_browser_ensure_initialized,
)
from ultimate_mcp_server.tools.smart_browser import (
    shutdown as smart_browser_shutdown,
)
from ultimate_mcp_server.tools.sql_databases import initialize_sql_tools, shutdown_sql_tools

# --- Import the trigger function directly instead of the whole module---
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.logging import logger

# --- Define Logging Configuration Dictionary ---

LOG_FILE_PATH = "logs/ultimate_mcp_server.log"

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
            "()": "ultimate_mcp_server.utils.logging.formatter.create_rich_console_handler",
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
        "ultimate_mcp_server": { # Our application's logger namespace
            "handlers": ["rich_console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "ultimate_mcp_server.tools": { # Tools-specific logger
            "handlers": ["tools_file"],
            "level": "DEBUG",
            "propagate": True, # Propagate to parent for console display
        },
        "ultimate_mcp_server.completions": { # Completions-specific logger
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
tools_logger = get_logger("ultimate_mcp_server.tools")
completions_logger = get_logger("ultimate_mcp_server.completions")

@dataclass
class ProviderStatus:
    """
    Structured representation of an LLM provider's configuration and availability status.
    
    This dataclass encapsulates all essential status information about a language model
    provider in the Ultimate MCP Server. It's used to track the state of each provider,
    including whether it's properly configured, successfully initialized, and what models
    it offers. This information is vital for:
    
    1. Displaying provider status to clients via API endpoints
    2. Making runtime decisions about provider availability
    3. Debugging provider configuration and connectivity issues
    4. Resource listings and capability discovery
    
    The status is typically maintained in the Gateway's provider_status dictionary,
    with provider names as keys and ProviderStatus instances as values.
    
    Attributes:
        enabled: Whether the provider is enabled in the configuration.
                This reflects the user's intent, not actual availability.
        available: Whether the provider is successfully initialized and ready for use.
                  This is determined by runtime checks during server initialization.
        api_key_configured: Whether a valid API key was found for this provider.
                           A provider might be enabled but have no API key configured.
        models: List of available models from this provider, with each model represented
               as a dictionary containing model ID, name, and capabilities.
        error: Error message explaining why a provider is unavailable, or None if
              the provider initialized successfully or hasn't been initialized yet.
    """
    enabled: bool
    available: bool
    api_key_configured: bool
    models: List[Dict[str, Any]]
    error: Optional[str] = None

class Gateway:
    """
    Main Ultimate MCP Server implementation and central orchestrator.
    
    The Gateway class serves as the core of the Ultimate MCP Server, providing a unified
    interface to multiple LLM providers (OpenAI, Anthropic, etc.) and implementing the
    Model Control Protocol (MCP). It manages provider connections, tool registration,
    state persistence, and request handling.
    
    Key responsibilities:
    - Initializing and managing connections to LLM providers
    - Registering and exposing tools for model interaction
    - Providing consistent error handling and logging
    - Managing state persistence across requests
    - Exposing resources (guides, examples, reference info) for models
    - Implementing the MCP protocol for standardized model interaction
    
    The Gateway is designed to be instantiated once per server instance and serves
    as the central hub for all model interactions. It can be accessed globally through
    the ultimate_mcp_server.core._gateway_instance reference.
    """
    
    def __init__(
        self, 
        name: str = "main", 
        register_tools: bool = True,
        provider_exclusions: List[str] = None,
        load_all_tools: bool = False  # Remove result_serialization_mode
    ):
        """
        Initialize the MCP Gateway with configured providers and tools.
        
        This constructor sets up the complete MCP Gateway environment, including:
        - Loading configuration from environment variables and config files
        - Setting up logging infrastructure
        - Initializing the MCP server framework
        - Creating a state store for persistence
        - Registering tools and resources based on configuration
        
        The initialization process is designed to be flexible, allowing for customization
        through the provided parameters and the configuration system. Provider initialization
        is deferred until server startup to ensure proper async handling.
        
        Args:
            name: Server instance name, used for logging and identification purposes.
                 Default is "main".
            register_tools: Whether to register standard MCP tools with the server.
                           If False, only the minimal core functionality will be available.
                           Default is True.
            provider_exclusions: List of provider names to exclude from initialization.
                                This allows selectively disabling specific providers
                                regardless of their configuration status.
                                Default is None (no exclusions).
            load_all_tools: If True, load all available tools. If False (default),
                           load only the defined 'Base Toolset'.
        """
        self.name = name
        self.providers = {}
        self.provider_status = {}
        self.logger = get_logger(f"ultimate_mcp_server.{name}")
        self.event_handlers = {}
        self.provider_exclusions = provider_exclusions or []
        self.api_meta_tool = None # Initialize api_meta_tool attribute
        self.load_all_tools = load_all_tools  # Store the flag
        
        # Load configuration if not already loaded
        if get_config() is None:
            self.logger.info("Initializing Gateway: Loading configuration...")
            load_config()
        
        # Initialize logger
        self.logger.info(f"Initializing {self.name}...")
        
        # Set MCP protocol version to 2025-03-25
        import os
        os.environ["MCP_PROTOCOL_VERSION"] = "2025-03-25"
        
        # Create MCP server with host and port settings
        self.mcp = FastMCP(
            self.name,
            lifespan=self._server_lifespan,
            host=get_config().server.host,
            port=get_config().server.port,
            instructions=self.system_instructions,
            timeout=300,
            debug=True,
            server_version="2025-03-25"  # Use protocol version 2025-03-25
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
            self._register_tools(load_all=self.load_all_tools)
            self._register_resources()
        
        self.logger.info(f"Ultimate MCP Server '{self.name}' initialized")
    
    def log_tool_calls(self, func):
        """
        Decorator to log MCP tool calls with detailed timing and result information.
        
        This decorator wraps MCP tool functions to provide consistent logging of:
        - Tool name and parameters at invocation time
        - Execution time for performance tracking
        - Success or failure status
        - Summarized results or error information
        
        The decorator ensures that all tool calls are logged to a dedicated tools logger,
        which helps with diagnostics, debugging, and monitoring of tool usage patterns.
        Successful calls include timing information and a brief summary of the result,
        while failed calls include exception details.
        
        Args:
            func: The async function to wrap with logging. This should be a tool function
                 registered with the MCP server that will be called by models.
                
        Returns:
            A wrapped async function that performs the same operations as the original
            but with added logging before and after execution.
            
        Note:
            This decorator is automatically applied to all functions registered as tools
            via the @mcp.tool() decorator in the _register_tools method, so it doesn't
            need to be applied manually in most cases.
        """
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
        """
        Async context manager managing the server lifecycle during startup and shutdown.
        
        This method implements the lifespan protocol used by FastMCP (based on ASGI) to:
        1. Perform startup initialization before the server begins accepting requests
        2. Clean up resources when the server is shutting down
        3. Make shared context available to request handlers during the server's lifetime
        
        During startup, this method:
        - Initializes all configured LLM providers
        - Triggers dynamic docstring generation for tools that need it
        - Sets the global Gateway instance for access from other components
        - Prepares a shared context dictionary for use by request handlers
        
        During shutdown, it:
        - Clears the global Gateway instance reference
        - Handles any necessary cleanup of resources
        
        The lifespan context is active throughout the entire server runtime, from
        startup until shutdown is initiated.
        
        Args:
            server: The FastMCP server instance that's starting up, which provides
                   the framework context for the lifespan.
            
        Yields:
            Dict containing initialized resources that will be available to all
            request handlers during the server's lifetime.
            
        Note:
            This method is called automatically by the FastMCP framework during
            server startup and is not intended to be called directly.
        """
        self.logger.info(f"Starting Ultimate MCP Server '{self.name}'")
        
        # Initialize providers
        await self._initialize_providers()

        try:
            await initialize_sql_tools()
            self.logger.info("SQL tools state initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQL tools state: {e}", exc_info=True)

        # --- OPTIONAL: Pre-initialize SmartBrowser ---
        try:
            self.logger.info("Pre-initializing Smart Browser components...")
            # Call the imported initialization function
            await smart_browser_ensure_initialized()
            self.logger.info("Smart Browser successfully pre-initialized.")
        except Exception as e:
            # Log warning but don't stop server startup if pre-init fails
            self.logger.warning(f"Could not pre-initialize Smart Browser: {e}", exc_info=True)
        # ---------------------------------------------------------------------

        # --- Trigger Dynamic Docstring Generation ---
        # This should run after config is loaded but before the server is fully ready
        # It checks cache and potentially calls an LLM.
        self.logger.info("Initiating dynamic docstring generation for Marqo tool...")
        try:
            # Import the function here to avoid circular imports
            from ultimate_mcp_server.tools.marqo_fused_search import (
                trigger_dynamic_docstring_generation,
            )
            await trigger_dynamic_docstring_generation()
            self.logger.info("Dynamic docstring generation/loading complete.")
        except Exception as e:
            self.logger.error(f"Error during dynamic docstring generation startup task: {e}", exc_info=True)
        # ---------------------------------------------

        # --- Set the global instance variable --- 
        # Make the fully initialized instance accessible globally AFTER init
        ultimate_mcp_server.core._gateway_instance = self
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
            from ultimate_mcp_server.tools.marqo_fused_search import (
                trigger_dynamic_docstring_generation,
            )
            await trigger_dynamic_docstring_generation()
            logger.info("Dynamic docstring generation/loading complete.")
            yield context
        finally:
            try:
                # --- Shutdown SQL Tools State ---
                await shutdown_sql_tools()
                self.logger.info("SQL tools state shut down.")
            except Exception as e:
                self.logger.error(f"Failed to shut down SQL tools state: {e}", exc_info=True)
        
            # 2. Shutdown Smart Browser explicitly
            try:
                self.logger.info("Initiating explicit Smart Browser shutdown...")
                await smart_browser_shutdown() # Call the imported function
                self.logger.info("Smart Browser shutdown completed successfully.")
            except Exception as e:
                logger.error(f"Error during explicit Smart Browser shutdown: {e}", exc_info=True)
                        
            # --- Clear the global instance on shutdown --- 
            ultimate_mcp_server.core._gateway_instance = None
            self.logger.info("Global gateway instance cleared.")
            # -------------------------------------------
            self.logger.info(f"Shutting down Ultimate MCP Server '{self.name}'")
    
    async def _initialize_providers(self):
        """
        Initialize all enabled LLM providers based on the loaded configuration.
        
        This asynchronous method performs the following steps:
        1. Identifies which providers are enabled and properly configured with API keys
        2. Skips providers that are in the exclusion list (specified at Gateway creation)
        3. Initializes each valid provider in parallel using asyncio tasks
        4. Updates the provider_status dictionary with the initialization results
        
        The method uses a defensive approach, handling cases where:
        - A provider is enabled but missing API keys
        - Configuration is incomplete or inconsistent
        - Initialization errors occur with specific providers
        
        After initialization, the Gateway will have a populated providers dictionary
        with available provider instances, and a comprehensive provider_status dictionary
        with status information for all providers (including those that failed to initialize).
        
        This method is automatically called during server startup and is not intended
        to be called directly by users of the Gateway class.
        
        Raises:
            No exceptions are propagated from this method. All provider initialization
            errors are caught, logged, and reflected in the provider_status dictionary.
        """
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
            # Special exception for Ollama: it doesn't require an API key since it runs locally
            if provider_name == Provider.OLLAMA.value and provider_config and provider_config.enabled:
                self.logger.debug(f"Found configured and enabled provider: {provider_name} (API key not required)")
                providers_to_init.append(provider_name)
            # Check if the provider is enabled AND has an API key configured in the loaded settings
            elif provider_config and provider_config.enabled and provider_config.api_key:
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
        """
        Initialize a single LLM provider with its API key and configuration.
        
        This method is responsible for initializing an individual provider by:
        1. Retrieving the provider's configuration and API key
        2. Importing the appropriate provider class
        3. Instantiating the provider with the configured API key
        4. Calling the provider's initialize method to establish connectivity
        5. Recording the provider's status (including available models)
        
        The method handles errors gracefully, ensuring that exceptions during any
        stage of initialization are caught, logged, and reflected in the provider's
        status rather than propagated up the call stack.
        
        Args:
            provider_name: Name of the provider to initialize, matching a value
                          in the Provider enum (e.g., "openai", "anthropic").
                          
        Returns:
            None. Results are stored in the Gateway's providers and provider_status
            dictionaries rather than returned directly.
            
        Note:
            This method is called by _initialize_providers during server startup
            and is not intended to be called directly by users of the Gateway class.
        """
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
            # Special case for Ollama: doesn't require an API key
            elif provider_name == Provider.OLLAMA.value and provider_config:
                api_key = None
                api_key_configured = True
                self.logger.debug("Initializing Ollama provider without API key (not required)")
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

            # Import PROVIDER_REGISTRY to use centralized provider registry
            from ultimate_mcp_server.core.providers import PROVIDER_REGISTRY

            # Use the registry instead of hardcoded providers dictionary
            provider_class = PROVIDER_REGISTRY.get(provider_name)
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
        """
        Return comprehensive system-level instructions for LLMs on how to use the gateway.
        
        This property generates detailed instructions that are injected into the system prompt
        for LLMs using the Gateway. These instructions serve as a guide for LLMs to effectively
        utilize the available tools and capabilities, helping them understand:
        
        - The categories of available tools and their purposes
        - Best practices for provider and model selection
        - Error handling strategies and patterns
        - Recommendations for efficient and appropriate tool usage
        - Guidelines for choosing the right tool for specific tasks
        
        The instructions are designed to be clear and actionable, helping LLMs make
        informed decisions about when and how to use different components of the
        Ultimate MCP Server. They're structured in a hierarchical format with sections
        covering core categories, best practices, and additional resources.
        
        Returns:
            A formatted string containing detailed instructions for LLMs on how to
            effectively use the Gateway's tools and capabilities. These instructions
            are automatically included in the system prompt for all LLM interactions.
        """
        # Tool loading message can be adjusted based on self.load_all_tools if needed
        tool_loading_info = "all available tools" if self.load_all_tools else "the Base Toolset"
        
        return f"""
# Ultimate MCP Server Tool Usage Instructions
        
You have access to the Ultimate MCP Server, which provides unified access to multiple language model
providers (OpenAI, Anthropic, etc.) through a standardized interface. This server instance has loaded {tool_loading_info}. 
Follow these instructions to effectively use the available tools.

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
- `provider://{{provider_name}}`: Details about a specific provider
- `guide://llm`: Comprehensive usage guide for LLMs
- `guide://error-handling`: Detailed error handling guidance
- `examples://workflows`: Detailed examples of common workflows
- `examples://completions`: Examples of different completion types
- `examples://tournaments`: Guidance on tournament configuration and analysis

Remember to use appropriate error handling and follow the documented parameter formats
for each tool. All providers may not be available at all times, so always check status
first and be prepared to adapt to available providers.
"""
        
    def _register_tools(self, load_all: bool = False):
        """
        Register all MCP tools with the server instance.
        
        This internal method sets up all available tools in the Ultimate MCP Server,
        making them accessible to LLMs through the MCP protocol. It handles:
        
        1. Setting up the basic echo tool for connectivity testing
        2. Conditionally calling the register_all_tools function to set up either
           the 'Base Toolset' or all specialized tools based on the `load_all` flag.
        
        The registration process wraps each tool function with logging functionality
        via the log_tool_calls decorator, ensuring consistent logging behavior across
        all tools. This provides valuable diagnostic information during tool execution.
        
        All registered tools become available through the MCP interface and can be
        discovered and used by LLMs interacting with the server.
        
        Args:
            load_all: If True, register all tools. If False, register only the base set.
        
        Note:
            This method is called automatically during Gateway initialization when
            register_tools=True (the default) and is not intended to be called directly.
        """
        # Import here to avoid circular dependency
        from ultimate_mcp_server.tools import register_all_tools
        
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

        # Define our base toolset - use function names not module names
        base_toolset = [

            # Completion tools
            "generate_completion", 
            "chat_completion", 
            "multi_completion",
            # "stream_completion", # Not that useful for MCP
            
            # Provider tools
            "get_provider_status", 
            "list_models",
            
            # Filesystem tools
            "read_file",
            "read_multiple_files",
            "write_file",
            "edit_file",
            "create_directory",
            "list_directory",
            "directory_tree",
            "move_file",
            "search_files",
            "get_file_info",
            "list_allowed_directories",
            "get_unique_filepath",
            
            # Optimization tools
            "estimate_cost", 
            "compare_models", 
            "recommend_model",
            
            # Local text tools
            "run_ripgrep", 
            "run_awk", 
            "run_sed", 
            "run_jq",
            
            # Search tools
            "marqo_fused_search",
            
            # SmartBrowser class methods
            "search",
            "download",
            "download_site_pdfs",
            "collect_documentation",
            "run_macro",
            "autopilot",

            # SQL class methods
            "manage_database",
            "execute_sql",
            "explore_database",
            "access_audit_log",
            
            # Document processing class methods
            "convert_document",
            "chunk_document",
            "clean_and_format_text_as_markdown",
            "batch_format_texts",
            "optimize_markdown_formatting",
            "generate_qa_pairs",
            "summarize_document",
            "ocr_image",
            "enhance_ocr_text",
            "analyze_pdf_structure",
            "extract_tables",
            "process_document_batch",
            
            # Python sandbox class methods
            "execute_python",
            "repl_python"

        ]
        
        # Conditionally register tools based on load_all flag
        if load_all:
            self.logger.info("Calling register_all_tools to register ALL available tools...")
            register_all_tools(self.mcp)
        else:
            self.logger.info("Calling register_all_tools to register only the BASE toolset...")
            # Check if tool_registration filter is enabled in config
            cfg = get_config()
            if cfg.tool_registration.filter_enabled:
                # If filtering is already enabled, respect that configuration
                self.logger.info("Tool filtering is enabled - using config filter settings")
                register_all_tools(self.mcp)
            else:
                # Otherwise, set up filtering for base toolset
                cfg.tool_registration.filter_enabled = True
                cfg.tool_registration.included_tools = base_toolset
                self.logger.info(f"Registering base toolset: {', '.join(base_toolset)}")
                register_all_tools(self.mcp)
        
        # After tools are registered, save the tool names to a file for the tools estimator script
        try:
            import json

            from ultimate_mcp_server.tools import STANDALONE_TOOL_FUNCTIONS
            
            # Get tools from STANDALONE_TOOL_FUNCTIONS plus class-based tools
            all_tool_names = []
            
            # Add standalone tool function names
            for tool_func in STANDALONE_TOOL_FUNCTIONS:
                if hasattr(tool_func, "__name__"):
                    all_tool_names.append(tool_func.__name__)
            
            # Add echo tool
            all_tool_names.append("echo")
            
            # Write to file
            with open("tools_list.json", "w") as f:
                json.dump(all_tool_names, f, indent=2)
                
            self.logger.info(f"Wrote {len(all_tool_names)} tool names to tools_list.json for context estimator")
        except Exception as e:
            self.logger.warning(f"Failed to write tool names to file: {str(e)}")

    def _register_resources(self):
        """
        Register all MCP resources with the server instance.
        
        This internal method registers standard MCP resources that provide static
        information and guidance to LLMs using the Ultimate MCP Server. Resources differ
        from tools in that they:
        
        1. Provide static reference information rather than interactive functionality
        2. Are accessed via URI-like identifiers (e.g., "info://server", "guide://llm")
        3. Don't require API calls or external services to generate their responses
        
        Registered resources include:
        - Server and tool information (info:// resources)
        - Provider details (provider:// resources)
        - Usage guides and tutorials (guide:// resources)
        - Example workflows and usage patterns (examples:// resources)
        
        These resources serve as a knowledge base for LLMs to better understand how to
        effectively use the available tools and follow best practices. They help reduce
        the need for extensive contextual information in prompts by making reference
        material available on-demand through the MCP protocol.
        
        Note:
            This method is called automatically during Gateway initialization when
            register_tools=True (the default) and is not intended to be called directly.
        """
        
        @self.mcp.resource("info://server")
        def get_server_info() -> Dict[str, Any]:
            """
            Get information about the Ultimate MCP Server server.
            
            This resource provides basic metadata about the Ultimate MCP Server server instance,
            including its name, version, and supported providers. Use this resource to
            discover server capabilities and version information.
            
            Resource URI: info://server
            
            Returns:
                Dictionary containing server information:
                - name: Name of the Ultimate MCP Server server
                - version: Version of the Ultimate MCP Server server
                - description: Brief description of server functionality
                - providers: List of supported LLM provider names
                
            Example:
                {
                    "name": "Ultimate MCP Server",
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
            Get information about available Ultimate MCP Server tools.
            
            This resource provides a descriptive overview of the tools available in the
            Ultimate MCP Server, organized by category. Use this resource to understand which
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
                Use this resource to understand the capabilities of the Ultimate MCP Server and
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
            Usage guide for LLMs using the Ultimate MCP Server.
            
            This resource provides structured guidance specifically designed for LLMs to
            effectively use the tools and resources provided by the Ultimate MCP Server. It includes
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
                # Ultimate MCP Server Usage Guide for Language Models
                
                ## Overview
                
                The Ultimate MCP Server provides a set of tools for accessing multiple language model providers
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
            Get comprehensive guidance on handling errors from Ultimate MCP Server tools.
            
            This resource provides detailed information about common error patterns,
            error handling strategies, and recovery approaches for each tool in the
            Ultimate MCP Server. It helps LLMs understand how to gracefully handle and recover
            from various error conditions.
            
            Resource URI: guide://error-handling
            
            Returns:
                Dictionary containing error handling guidance organized by tool type:
                - provider_tools: Error handling for provider-related tools
                - completion_tools: Error handling for completion tools
                - tournament_tools: Error handling for tournament tools
                
            Usage:
                This resource helps LLMs implement robust error handling when using
                the Ultimate MCP Server tools, improving the resilience of their interactions.
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
                how to combine multiple tools in the Ultimate MCP Server to accomplish common tasks.
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
            in the Ultimate MCP Server, along with guidance on when to use each type. It helps with
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
    """
    Create and configure the FastAPI server instance for the Ultimate MCP Server.
    
    This function serves as the main entry point for setting up the HTTP server
    component of the Ultimate MCP Server using FastAPI. It handles:
    
    1. Singleton management - ensuring only one server instance exists
    2. Gateway initialization - creating the Gateway if not already instantiated
    3. CORS configuration - setting up Cross-Origin Resource Sharing middleware
    4. Health check endpoints - adding utility endpoints for monitoring
    
    The function follows a singleton pattern, returning an existing server instance
    if one has already been created, or creating a new one if needed. This ensures
    consistent server state across multiple calls.
    
    The server instance created by this function can be used with ASGI servers like
    Uvicorn to serve HTTP requests, or with the FastMCP transport modes for more
    specialized communication patterns.
    
    Returns:
        FastAPI: Configured FastAPI application instance ready for deployment.
        The returned application has CORS middleware and health endpoints configured.
        
    Example:
        ```python
        # Create and run the server with Uvicorn
        app = create_server()
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
        ```
    """
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
    
    # Add custom endpoints for UMS Explorer and database access
    from pathlib import Path

    from fastapi.responses import FileResponse, JSONResponse
    
    # Get the project root directory (where storage/ is located)
    project_root = Path(__file__).parent.parent.parent
    tools_dir = project_root / "ultimate_mcp_server" / "tools"
    storage_dir = project_root / "storage"
    
    # Add custom endpoint for UMS Explorer HTML file
    @app.get("/tools/ums_explorer.html")
    async def serve_ums_explorer():
        """Serve the UMS Explorer HTML file."""
        html_path = tools_dir / "ums_explorer.html"
        if html_path.exists():
            # Don't set filename to avoid Content-Disposition: attachment header
            return FileResponse(
                path=str(html_path),
                media_type="text/html"
            )
        else:
            return JSONResponse(
                {"error": "UMS Explorer HTML file not found"},
                status_code=404
            )
    
    # Add custom endpoint for database file
    @app.get("/storage/unified_agent_memory.db")
    async def serve_database():
        """Serve the unified agent memory database file."""
        db_path = storage_dir / "unified_agent_memory.db"
        if db_path.exists():
            return FileResponse(
                path=str(db_path),
                media_type="application/x-sqlite3",
                filename="unified_agent_memory.db"
            )
        else:
            return JSONResponse(
                {"error": "Database file not found"},
                status_code=404
            )
    
    # Add health check endpoint
    @app.get("/health")
    async def health():
        """
        Health check endpoint for monitoring server status.
        
        This endpoint provides a simple way to verify that the server is running and
        responsive. It can be used by load balancers, monitoring systems, or client
        applications to check if the server is operational.
        
        Returns:
            dict: A simple status object containing:
                - status: "ok" if the server is healthy
                - version: Current server version string
        
        Example response:
            ```json
            {
                "status": "ok",
                "version": "0.1.0"
            }
            ```
        """
        return {
            "status": "ok",
            "version": "0.1.0",
        }
    
    # Add UMS Explorer endpoint
    @app.get("/ums-explorer")
    async def ums_explorer():
        """
        UMS Explorer web interface.
        
        Redirects to the UMS Explorer HTML interface for viewing and analyzing
        the Unified Memory System database through a web browser.
        """
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/tools/ums_explorer.html")
    
    # ===== UMS EXPLORER API ENDPOINTS =====
    
    import json
    import math
    import sqlite3
    from typing import Optional

    from fastapi import HTTPException
    
    # Database configuration for UMS Explorer
    DATABASE_PATH = str(storage_dir / "unified_agent_memory.db")
    
    def get_db_connection():
        """Get database connection for UMS Explorer API"""
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    # Helper functions for UMS Explorer API
    def calculate_state_complexity(state_data: dict) -> float:
        """Calculate complexity score for a cognitive state"""
        if not state_data:
            return 0.0
        
        # Count different types of components
        component_count = len(state_data.keys())
        
        # Calculate nested depth
        max_depth = calculate_dict_depth(state_data)
        
        # Count total values
        total_values = count_dict_values(state_data)
        
        # Normalize to 0-100 scale
        complexity = min(100, (component_count * 5) + (max_depth * 10) + (total_values * 0.5))
        return round(complexity, 2)

    def calculate_dict_depth(d: dict, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionary"""
        if not isinstance(d, dict):
            return current_depth
        
        if not d:
            return current_depth
        
        return max(calculate_dict_depth(v, current_depth + 1) for v in d.values())

    def count_dict_values(d: dict) -> int:
        """Count total number of values in nested dictionary"""
        count = 0
        for v in d.values():
            if isinstance(v, dict):
                count += count_dict_values(v)
            elif isinstance(v, list):
                count += len(v)
            else:
                count += 1
        return count

    def compute_state_diff(state1: dict, state2: dict) -> dict:
        """Compute difference between two cognitive states"""
        diff_result = {
            'added': {},
            'removed': {},
            'modified': {},
            'magnitude': 0.0
        }
        
        all_keys = set(state1.keys()) | set(state2.keys())
        changes = 0
        total_keys = len(all_keys)
        
        for key in all_keys:
            if key not in state1:
                diff_result['added'][key] = state2[key]
                changes += 1
            elif key not in state2:
                diff_result['removed'][key] = state1[key]
                changes += 1
            elif state1[key] != state2[key]:
                diff_result['modified'][key] = {
                    'before': state1[key],
                    'after': state2[key]
                }
                changes += 1
        
        # Calculate magnitude as percentage of changed keys
        if total_keys > 0:
            diff_result['magnitude'] = (changes / total_keys) * 100
        
        return diff_result

    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    # === COGNITIVE STATES API ===
    @app.get("/api/cognitive-states")
    async def get_cognitive_states(
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
        pattern_type: Optional[str] = None
    ):
        """Get cognitive states with optional filtering"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Base query
            query = """
                SELECT 
                    cs.*,
                    w.title as workflow_title,
                    COUNT(DISTINCT m.memory_id) as memory_count,
                    COUNT(DISTINCT a.action_id) as action_count
                FROM cognitive_timeline_states cs
                LEFT JOIN workflows w ON cs.workflow_id = w.workflow_id
                LEFT JOIN memories m ON cs.workflow_id = m.workflow_id
                LEFT JOIN actions a ON cs.workflow_id = a.workflow_id
                WHERE 1=1
            """
            params = []
            
            if start_time:
                query += " AND cs.timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND cs.timestamp <= ?"
                params.append(end_time)
            
            if pattern_type:
                query += " AND cs.state_type = ?"
                params.append(pattern_type)
            
            query += """
                GROUP BY cs.state_id
                ORDER BY cs.timestamp DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            states = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
            
            # Enhance states with additional metadata
            enhanced_states = []
            for state in states:
                # Parse state_data if it's JSON
                try:
                    state_data = json.loads(state.get('state_data', '{}'))
                except Exception:
                    state_data = {}
                
                enhanced_state = {
                    **state,
                    'state_data': state_data,
                    'formatted_timestamp': datetime.fromtimestamp(state['timestamp']).isoformat(),
                    'age_minutes': (datetime.now().timestamp() - state['timestamp']) / 60,
                    'complexity_score': calculate_state_complexity(state_data),
                    'change_magnitude': 0  # Will be calculated with diffs
                }
                enhanced_states.append(enhanced_state)
            
            # Calculate change magnitudes by comparing adjacent states
            for i in range(len(enhanced_states) - 1):
                current_state = enhanced_states[i]
                previous_state = enhanced_states[i + 1]  # Ordered DESC, so next is previous
                
                diff_result = compute_state_diff(previous_state['state_data'], current_state['state_data'])
                current_state['change_magnitude'] = diff_result.get('magnitude', 0)
            
            conn.close()
            
            return {
                "states": enhanced_states,
                "total": len(enhanced_states),
                "has_more": len(enhanced_states) == limit
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/cognitive-states/timeline")
    async def get_cognitive_timeline(
        hours: int = 24,
        granularity: str = "hour"  # hour, minute, second
    ):
        """Get cognitive state timeline data optimized for visualization"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            since_timestamp = datetime.now().timestamp() - (hours * 3600)
            
            # First try to get from cognitive_timeline_states table
            try:
                cursor.execute("""
                    SELECT 
                        state_id,
                        timestamp,
                        state_type,
                        state_data,
                        workflow_id,
                        description,
                        ROW_NUMBER() OVER (ORDER BY timestamp) as sequence_number
                    FROM cognitive_timeline_states 
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                """, (since_timestamp,))
                
                states = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # If we found states, process them
                if states:
                    timeline_data = []
                    for i, state in enumerate(states):
                        try:
                            state_data = json.loads(state.get('state_data', '{}'))
                        except Exception:
                            state_data = {}
                        
                        # Calculate change magnitude from previous state
                        change_magnitude = 0
                        if i > 0:
                            prev_state_data = json.loads(states[i-1].get('state_data', '{}'))
                            diff_result = compute_state_diff(prev_state_data, state_data)
                            change_magnitude = diff_result.get('magnitude', 0)
                        
                        timeline_item = {
                            'state_id': state['state_id'],
                            'timestamp': state['timestamp'],
                            'formatted_time': datetime.fromtimestamp(state['timestamp']).isoformat(),
                            'state_type': state['state_type'],
                            'workflow_id': state['workflow_id'],
                            'description': state['description'],
                            'sequence_number': state['sequence_number'],
                            'complexity_score': calculate_state_complexity(state_data),
                            'change_magnitude': change_magnitude,
                        }
                        timeline_data.append(timeline_item)
                    
                    conn.close()
                    
                    return {
                        'timeline_data': timeline_data,
                        'total_states': len(timeline_data),
                        'time_range_hours': hours,
                        'granularity': granularity,
                        'summary_stats': {
                            'avg_complexity': sum(item['complexity_score'] for item in timeline_data) / len(timeline_data) if timeline_data else 0,
                            'total_transitions': len(timeline_data) - 1 if len(timeline_data) > 1 else 0,
                            'max_change_magnitude': max((item['change_magnitude'] for item in timeline_data), default=0)
                        }
                    }
                    
            except sqlite3.OperationalError:
                # Table doesn't exist, fall back to memories table
                pass
            
            # Fallback: Create timeline from memories table
            cursor.execute("""
                SELECT 
                    memory_id,
                    memory_type,
                    content,
                    importance,
                    confidence,
                    created_at,
                    workflow_id,
                    ROW_NUMBER() OVER (ORDER BY created_at) as sequence_number
                FROM memories 
                WHERE memory_type IN ('thought', 'reasoning', 'analysis', 'plan', 'goal')
                AND created_at >= ?
                ORDER BY created_at ASC
                LIMIT 100
            """, (since_timestamp,))
            
            memories = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
            
            # Convert memories to timeline format
            timeline_data = []
            for i, memory in enumerate(memories):
                # Calculate change magnitude based on content differences
                change_magnitude = 0
                if i > 0:
                    prev_content = memories[i-1].get('content', '')
                    curr_content = memory.get('content', '')
                    # Simple content-based change calculation
                    if prev_content and curr_content:
                        change_magnitude = min(100, abs(len(curr_content) - len(prev_content)) / max(len(prev_content), 1) * 100)
                
                timeline_item = {
                    'state_id': memory['memory_id'],
                    'timestamp': memory['created_at'],
                    'formatted_time': datetime.fromtimestamp(memory['created_at']).isoformat(),
                    'state_type': memory['memory_type'],
                    'workflow_id': memory['workflow_id'],
                    'description': memory['content'][:100] + ('...' if len(memory.get('content', '')) > 100 else ''),
                    'sequence_number': memory['sequence_number'],
                    'complexity_score': (memory.get('importance', 5) * 10),  # Convert 1-10 to 0-100 scale
                    'change_magnitude': change_magnitude,
                }
                timeline_data.append(timeline_item)
            
            conn.close()
            
            return {
                'timeline_data': timeline_data,
                'total_states': len(timeline_data),
                'time_range_hours': hours,
                'granularity': granularity,
                'summary_stats': {
                    'avg_complexity': sum(item['complexity_score'] for item in timeline_data) / len(timeline_data) if timeline_data else 0,
                    'total_transitions': len(timeline_data) - 1 if len(timeline_data) > 1 else 0,
                    'max_change_magnitude': max((item['change_magnitude'] for item in timeline_data), default=0)
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    # === ARTIFACTS API ===
    @app.get("/api/artifacts")
    async def get_artifacts(
        artifact_type: Optional[str] = None,
        workflow_id: Optional[str] = None,
        tags: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0
    ):
        """Get artifacts with filtering and search"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Base query
            query = """
                SELECT 
                    a.*,
                    w.title as workflow_title,
                    COUNT(ar.target_artifact_id) as relationship_count,
                    COUNT(versions.artifact_id) as version_count
                FROM artifacts a
                LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
                LEFT JOIN artifact_relationships ar ON a.artifact_id = ar.source_artifact_id
                LEFT JOIN artifacts versions ON a.artifact_id = versions.parent_artifact_id
                WHERE 1=1
            """
            params = []
            
            if artifact_type:
                query += " AND a.artifact_type = ?"
                params.append(artifact_type)
            
            if workflow_id:
                query += " AND a.workflow_id = ?"
                params.append(workflow_id)
            
            if tags:
                query += " AND a.tags LIKE ?"
                params.append(f"%{tags}%")
            
            if search:
                query += " AND (a.name LIKE ? OR a.description LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])
            
            query += f"""
                GROUP BY a.artifact_id
                ORDER BY a.{sort_by} {'DESC' if sort_order == 'desc' else 'ASC'}
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            artifacts = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
            
            # Enhance artifacts with metadata
            for artifact in artifacts:
                # Parse tags and metadata
                try:
                    artifact['tags'] = json.loads(artifact.get('tags', '[]')) if artifact.get('tags') else []
                    artifact['metadata'] = json.loads(artifact.get('metadata', '{}')) if artifact.get('metadata') else {}
                except Exception:
                    artifact['tags'] = []
                    artifact['metadata'] = {}
                
                # Add computed fields
                artifact['formatted_created_at'] = datetime.fromtimestamp(artifact['created_at']).isoformat()
                artifact['formatted_updated_at'] = datetime.fromtimestamp(artifact['updated_at']).isoformat()
                artifact['age_days'] = (datetime.now().timestamp() - artifact['created_at']) / 86400
                artifact['file_size_human'] = format_file_size(artifact.get('file_size', 0))
            
            conn.close()
            
            return {
                "artifacts": artifacts,
                "total": len(artifacts),
                "has_more": len(artifacts) == limit
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/artifacts/stats")
    async def get_artifact_stats():
        """Get artifact statistics"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Total counts by type
            cursor.execute("""
                SELECT 
                    artifact_type,
                    COUNT(*) as count,
                    AVG(importance) as avg_importance,
                    SUM(file_size) as total_size,
                    MAX(access_count) as max_access_count
                FROM artifacts 
                GROUP BY artifact_type
            """)
            
            type_stats = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
            
            # Overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_artifacts,
                    COUNT(DISTINCT workflow_id) as unique_workflows,
                    AVG(importance) as avg_importance,
                    SUM(file_size) as total_file_size,
                    SUM(access_count) as total_access_count
                FROM artifacts
            """)
            
            overall_stats = dict(zip([d[0] for d in cursor.description], cursor.fetchone(), strict=False))
            
            conn.close()
            
            return {
                'overall': {
                    **overall_stats,
                    'total_file_size_human': format_file_size(overall_stats.get('total_file_size', 0))
                },
                'by_type': type_stats,
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    
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
    load_all_tools: bool = False,  # Added: Flag to control tool loading
) -> None:
    """
    Start the Ultimate MCP Server with configurable settings.
    
    This function serves as the main entry point for starting the Ultimate MCP Server
    in either SSE (HTTP server) or stdio (direct process communication) mode. It handles
    complete server initialization including:
    
    1. Configuration loading and parameter validation
    2. Logging setup with proper levels and formatting
    3. Gateway instantiation with tool registration
    4. Transport mode selection and server startup
    
    The function provides flexibility in server configuration through parameters that
    override settings from the configuration file, allowing for quick adjustments without
    modifying configuration files. It also supports tool filtering, enabling selective
    registration of specific tools.
    
    Args:
        host: Hostname or IP address to bind the server to (e.g., "localhost", "0.0.0.0").
             If None, uses the value from the configuration file.
        port: TCP port for the server to listen on when in SSE mode.
             If None, uses the value from the configuration file.
        workers: Number of worker processes to spawn for handling requests.
                Higher values improve concurrency but increase resource usage.
                If None, uses the value from the configuration file.
        log_level: Logging verbosity level. One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
                  If None, uses the value from the configuration file.
        reload: Whether to automatically reload the server when code changes are detected.
               Useful during development but not recommended for production.
        transport_mode: Communication mode for the server. Options:
                      - "stdio": Run using standard input/output for direct process communication (default)
                      - "sse": Run as an HTTP server with Server-Sent Events for streaming
        include_tools: Optional list of specific tool names to include in registration.
                      If provided, only these tools will be registered unless they are
                      also in exclude_tools. If None, all tools are included by default.
        exclude_tools: Optional list of tool names to exclude from registration.
                      These tools will not be registered even if they are also in include_tools.
        load_all_tools: If True, load all available tools. If False (default), load only the base set.
                      
    Raises:
        ValueError: If transport_mode is not one of the valid options.
        ConfigurationError: If there are critical errors in the server configuration.
        
    Note:
        This function does not return as it initiates the server event loop, which
        runs until interrupted (e.g., by a SIGINT signal). In SSE mode, it starts 
        a Uvicorn server; in stdio mode, it runs the FastMCP stdio handler.
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
    LOGGING_CONFIG["loggers"]["ultimate_mcp_server"]["level"] = final_log_level
    LOGGING_CONFIG["loggers"]["ultimate_mcp_server.tools"]["level"] = final_log_level
    LOGGING_CONFIG["loggers"]["ultimate_mcp_server.completions"]["level"] = final_log_level
    
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
        _gateway_instance = Gateway(
            name=cfg.server.name, 
            register_tools=True,
            load_all_tools=load_all_tools  # Pass the flag to Gateway
        )
    
    # Log startup info to stderr instead of using logging directly
    print("Starting Ultimate MCP Server server", file=sys.stderr)
    print(f"Host: {server_host}", file=sys.stderr)
    print(f"Port: {server_port}", file=sys.stderr)
    print(f"Workers: {server_workers}", file=sys.stderr)
    print(f"Log level: {final_log_level}", file=sys.stderr)
    print(f"Transport mode: {transport_mode}", file=sys.stderr)
    
    # Log tool loading strategy
    if load_all_tools:
        print("Tool Loading: ALL available tools", file=sys.stderr)
    else:
        print("Tool Loading: Base Toolset Only", file=sys.stderr)
        base_toolset = ["completion", "filesystem", "optimization", "provider", "local_text", "search"]
        print(f"  (Includes: {', '.join(base_toolset)})", file=sys.stderr)

    # Log tool filtering info if enabled
    if cfg.tool_registration.filter_enabled:
        if cfg.tool_registration.included_tools:
            print(f"Including tools: {', '.join(cfg.tool_registration.included_tools)}", file=sys.stderr)
        if cfg.tool_registration.excluded_tools:
            print(f"Excluding tools: {', '.join(cfg.tool_registration.excluded_tools)}", file=sys.stderr)
    
    if transport_mode == "sse":
        # Run in SSE mode (HTTP server)
        import os
        import subprocess
        import threading
        import time

        import uvicorn
        
        # Set up a function to run the tool context estimator after the server starts
        def run_tool_context_estimator():
            # Wait a bit for the server to start up
            time.sleep(5)
            try:
                # Ensure tools_list.json exists
                if not os.path.exists("tools_list.json"):
                    print("\n--- Tool Context Window Analysis ---", file=sys.stderr)
                    print("Error: tools_list.json not found. Tool registration may have failed.", file=sys.stderr)
                    print("The tool context estimator will run with limited functionality.", file=sys.stderr)
                    print("-" * 40, file=sys.stderr)
                
                # Run the tool context estimator script with the --quiet flag
                # Use python -m instead of direct script execution to avoid import issues
                result = subprocess.run(
                    ["python", "-m", "mcp_tool_context_estimator", "--quiet"],
                    capture_output=True,
                    text=True
                )
                # Output the results to stderr
                if result.stdout:
                    print("\n--- Tool Context Window Analysis ---", file=sys.stderr)
                    print(result.stdout, file=sys.stderr)
                    print("-" * 40, file=sys.stderr)
                # Check if there was an error
                if result.returncode != 0:
                    print("\n--- Tool Context Estimator Error ---", file=sys.stderr)
                    print("Failed to run mcp_tool_context_estimator.py - likely due to an error.", file=sys.stderr)
                    print("Error output:", file=sys.stderr)
                    print(result.stderr, file=sys.stderr)
                    print("-" * 40, file=sys.stderr)
            except Exception as e:
                print(f"\nError running tool context estimator: {str(e)}", file=sys.stderr)
                print("Check if mcp_tool_context_estimator.py exists and is executable.", file=sys.stderr)
        
        # Start the tool context estimator in a separate thread
        if os.path.exists("mcp_tool_context_estimator.py"):
            threading.Thread(target=run_tool_context_estimator, daemon=True).start()
        
        # Setup graceful shutdown
        logger = logging.getLogger("ultimate_mcp_server.server")
        
        # Configure graceful shutdown with error suppression
        enable_quiet_shutdown()
        
        # Create a shutdown handler for gateway cleanup
        async def cleanup_resources():
            """Performs cleanup for various components during shutdown."""
            
            # First attempt quick tasks then long tasks with timeouts
            print("Cleaning up Gateway instance and associated resources...", file=sys.stderr)
            
            # Shutdown SQL Tools with timeout
            try:
                await asyncio.wait_for(shutdown_sql_tools(), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown
            
            # Shutdown Connection Manager with timeout
            try:
                from ultimate_mcp_server.tools.sql_databases import _connection_manager
                await asyncio.wait_for(_connection_manager.shutdown(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown
            
            # Shutdown Smart Browser with timeout
            try:
                await asyncio.wait_for(smart_browser_shutdown(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown
        
        # Register the cleanup function with the graceful shutdown system
        register_shutdown_handler(cleanup_resources)
        
        # Get the SSE app from FastMCP
        app = _gateway_instance.mcp.sse_app()
        print(f"[DEBUG] SSE app type: {type(app)}", file=sys.stderr)
        print(f"[DEBUG] SSE app routes before adding custom routes: {len(app.routes) if hasattr(app, 'routes') else 'unknown'}", file=sys.stderr)
        
        # Add static file serving for UMS Explorer and database access to SSE app
        from pathlib import Path

        from starlette.responses import FileResponse
        
        # Get the project root directory (where storage/ is located)
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "ultimate_mcp_server" / "tools"
        storage_dir = project_root / "storage"
        
        # Add custom endpoint for UMS Explorer HTML file instead of mounting tools directory
        async def serve_ums_explorer(request):
            """Serve the UMS Explorer HTML file."""
            html_path = tools_dir / "ums_explorer.html"
            if html_path.exists():
                # Don't set filename to avoid Content-Disposition: attachment header
                return FileResponse(
                    path=str(html_path),
                    media_type="text/html"
                )
            else:
                from starlette.responses import JSONResponse
                return JSONResponse(
                    {"error": "UMS Explorer HTML file not found"},
                    status_code=404
                )
        
        app.add_route("/tools/ums_explorer.html", serve_ums_explorer, methods=["GET"])
        print("[DEBUG] Registered UMS Explorer endpoint: /tools/ums_explorer.html", file=sys.stderr)
        print(f"[DEBUG] Tools directory path: {tools_dir}", file=sys.stderr)
        print(f"[DEBUG] UMS Explorer file exists: {(tools_dir / 'ums_explorer.html').exists()}", file=sys.stderr)
        
        # Add custom endpoint for database file instead of mounting storage directory
        async def serve_database(request):
            """Serve the unified agent memory database file."""
            db_path = storage_dir / "unified_agent_memory.db"
            print(f"[DEBUG] Database endpoint called. Looking for: {db_path}", file=sys.stderr)
            print(f"[DEBUG] Database file exists: {db_path.exists()}", file=sys.stderr)
            if db_path.exists():
                print(f"[DEBUG] Serving database file: {db_path}", file=sys.stderr)
                return FileResponse(
                    path=str(db_path),
                    media_type="application/x-sqlite3",
                    filename="unified_agent_memory.db"
                )
            else:
                print(f"[DEBUG] Database file not found at: {db_path}", file=sys.stderr)
                from starlette.responses import JSONResponse
                return JSONResponse(
                    {"error": f"Database file not found at {db_path}"},
                    status_code=404
                )
        
        app.add_route("/storage/unified_agent_memory.db", serve_database, methods=["GET"])
        print("[DEBUG] Registered database endpoint: /storage/unified_agent_memory.db", file=sys.stderr)
        print(f"[DEBUG] Storage directory path: {storage_dir}", file=sys.stderr)
        print(f"[DEBUG] Storage directory exists: {storage_dir.exists()}", file=sys.stderr)
        
        # Add UMS Explorer endpoint to SSE app
        from starlette.responses import RedirectResponse
        
        async def ums_explorer(request):
            """UMS Explorer web interface."""
            return RedirectResponse(url="/tools/ums_explorer.html")
        
        app.add_route("/ums-explorer", ums_explorer, methods=["GET"])
        
        # ===== RESTORE ALL MISSING UMS EXPLORER API ENDPOINTS =====
        
        # First, add all the helper functions that support the API endpoints
        def get_action_status_indicator(status: str, execution_time: float) -> dict:
            """Get status indicator with color and icon for action status"""
            indicators = {
                'running': {'color': 'blue', 'icon': 'play', 'label': 'Running'},
                'executing': {'color': 'blue', 'icon': 'cpu', 'label': 'Executing'},
                'in_progress': {'color': 'orange', 'icon': 'clock', 'label': 'In Progress'},
                'completed': {'color': 'green', 'icon': 'check', 'label': 'Completed'},
                'failed': {'color': 'red', 'icon': 'x', 'label': 'Failed'},
                'cancelled': {'color': 'gray', 'icon': 'stop', 'label': 'Cancelled'},
                'timeout': {'color': 'yellow', 'icon': 'timer-off', 'label': 'Timeout'}
            }
            
            indicator = indicators.get(status, {'color': 'gray', 'icon': 'help', 'label': 'Unknown'})
            
            # Add urgency flag for long-running actions
            if status in ['running', 'executing', 'in_progress'] and execution_time > 120:  # 2 minutes
                indicator['urgency'] = 'high'
            elif status in ['running', 'executing', 'in_progress'] and execution_time > 60:  # 1 minute
                indicator['urgency'] = 'medium'
            else:
                indicator['urgency'] = 'low'
            
            return indicator

        def categorize_action_performance(execution_time: float, estimated_duration: float) -> str:
            """Categorize action performance based on execution time vs estimate"""
            if estimated_duration <= 0:
                return 'unknown'
            
            ratio = execution_time / estimated_duration
            
            if ratio <= 0.5:
                return 'excellent'
            elif ratio <= 0.8:
                return 'good'
            elif ratio <= 1.2:
                return 'acceptable'
            elif ratio <= 2.0:
                return 'slow'
            else:
                return 'very_slow'

        def get_action_resource_usage(action_id: str) -> dict:
            """Get resource usage for an action (placeholder implementation)"""
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'network_io': 0.0,
                'disk_io': 0.0
            }

        def estimate_wait_time(position: int, queue: list) -> float:
            """Estimate wait time based on queue position and historical data"""
            if position == 0:
                return 0.0
            avg_action_time = 30.0
            return position * avg_action_time

        def get_priority_label(priority: int) -> str:
            """Get human-readable priority label"""
            if priority <= 1:
                return 'Critical'
            elif priority <= 3:
                return 'High'
            elif priority <= 5:
                return 'Normal'
            elif priority <= 7:
                return 'Low'
            else:
                return 'Very Low'

        def calculate_action_performance_score(action: dict) -> float:
            """Calculate performance score for a completed action"""
            if action['status'] != 'completed':
                return 0.0
            
            execution_time = action.get('execution_duration', 0)
            if execution_time <= 0:
                return 100.0
            
            if execution_time <= 5:
                return 100.0
            elif execution_time <= 15:
                return 90.0
            elif execution_time <= 30:
                return 80.0
            elif execution_time <= 60:
                return 70.0
            elif execution_time <= 120:
                return 60.0
            else:
                return max(50.0, 100.0 - (execution_time / 10))

        def calculate_efficiency_rating(execution_time: float, result_size: int) -> str:
            """Calculate efficiency rating based on time and output"""
            if execution_time <= 0:
                return 'unknown'
            
            efficiency_score = result_size / execution_time if execution_time > 0 else 0
            
            if efficiency_score >= 100:
                return 'excellent'
            elif efficiency_score >= 50:
                return 'good'
            elif efficiency_score >= 20:
                return 'fair'
            else:
                return 'poor'

        def calculate_performance_summary(actions: list) -> dict:
            """Calculate performance summary from action history"""
            if not actions:
                return {
                    'avg_score': 0.0,
                    'top_performer': None,
                    'worst_performer': None,
                    'efficiency_distribution': {}
                }
            
            scores = [a.get('performance_score', 0) for a in actions]
            avg_score = sum(scores) / len(scores)
            
            best_action = max(actions, key=lambda a: a.get('performance_score', 0))
            worst_action = min(actions, key=lambda a: a.get('performance_score', 0))
            
            from collections import Counter
            efficiency_counts = Counter(a.get('efficiency_rating', 'unknown') for a in actions)
            
            return {
                'avg_score': round(avg_score, 2),
                'top_performer': {
                    'tool_name': best_action.get('tool_name', ''),
                    'score': best_action.get('performance_score', 0)
                },
                'worst_performer': {
                    'tool_name': worst_action.get('tool_name', ''),
                    'score': worst_action.get('performance_score', 0)
                },
                'efficiency_distribution': dict(efficiency_counts)
            }

        def generate_performance_insights(overall_stats: dict, tool_stats: list, hourly_metrics: list) -> list:
            """Generate actionable performance insights"""
            insights = []
            
            success_rate = (overall_stats.get('successful_actions', 0) / overall_stats.get('total_actions', 1)) * 100
            if success_rate < 80:
                insights.append({
                    'type': 'warning',
                    'title': 'Low Success Rate',
                    'message': f'Current success rate is {success_rate:.1f}%. Consider investigating failing tools.',
                    'severity': 'high'
                })
            
            if tool_stats:
                slowest_tool = max(tool_stats, key=lambda t: t.get('avg_duration', 0))
                if slowest_tool.get('avg_duration', 0) > 60:
                    insights.append({
                        'type': 'info',
                        'title': 'Performance Optimization',
                        'message': f'{slowest_tool["tool_name"]} is taking {slowest_tool["avg_duration"]:.1f}s on average. Consider optimization.',
                        'severity': 'medium'
                    })
            
            if hourly_metrics:
                peak_hour = max(hourly_metrics, key=lambda h: h.get('action_count', 0))
                insights.append({
                    'type': 'info',
                    'title': 'Peak Usage',
                    'message': f'Peak usage occurs at {peak_hour["hour"]}:00 with {peak_hour["action_count"]} actions.',
                    'severity': 'low'
                })
            
            return insights

        def calculate_tool_reliability_score(tool_stats: dict) -> float:
            """Calculate reliability score for a tool"""
            total_calls = tool_stats.get('total_calls', 0)
            successful_calls = tool_stats.get('successful_calls', 0)
            
            if total_calls == 0:
                return 0.0
            
            success_rate = successful_calls / total_calls
            volume_factor = min(1.0, total_calls / 100)
            
            return round(success_rate * volume_factor * 100, 2)

        def categorize_tool_performance(avg_execution_time: float) -> str:
            """Categorize tool performance based on average execution time"""
            if avg_execution_time is None:
                return 'unknown'
            
            if avg_execution_time <= 5:
                return 'fast'
            elif avg_execution_time <= 15:
                return 'normal'
            elif avg_execution_time <= 30:
                return 'slow'
            else:
                return 'very_slow'

        # Additional cognitive state helper functions
        def extract_key_state_components(state_data: dict) -> list:
            """Extract key components from state data for visualization"""
            components = []
            
            for key, value in state_data.items():
                component = {
                    'name': key,
                    'type': type(value).__name__,
                    'summary': str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                }
                
                if isinstance(value, dict):
                    component['count'] = len(value)
                elif isinstance(value, list):
                    component['count'] = len(value)
                
                components.append(component)
            
            return components[:10]

        def generate_state_tags(state_data: dict, state_type: str) -> list:
            """Generate descriptive tags for a cognitive state"""
            tags = [state_type]
            
            if 'goals' in state_data:
                tags.append('goal-oriented')
            
            if 'working_memory' in state_data:
                tags.append('memory-active')
            
            if 'error' in str(state_data).lower():
                tags.append('error-state')
            
            if 'decision' in str(state_data).lower():
                tags.append('decision-point')
            
            return tags

        def generate_timeline_segments(timeline_data: list, granularity: str, hours: int) -> list:
            """Generate timeline segments for visualization"""
            if not timeline_data:
                return []
            
            start_time = min(item['timestamp'] for item in timeline_data)
            end_time = max(item['timestamp'] for item in timeline_data)
            
            if granularity == 'minute':
                segment_duration = 60
            elif granularity == 'hour':
                segment_duration = 3600
            else:
                segment_duration = 1
            
            segments = []
            current_time = start_time
            
            while current_time < end_time:
                segment_end = current_time + segment_duration
                
                segment_states = [
                    item for item in timeline_data
                    if current_time <= item['timestamp'] < segment_end
                ]
                
                if segment_states:
                    avg_complexity = sum(s['complexity_score'] for s in segment_states) / len(segment_states)
                    max_change = max(s['change_magnitude'] for s in segment_states)
                    
                    from collections import Counter
                    segments.append({
                        'start_time': current_time,
                        'end_time': segment_end,
                        'state_count': len(segment_states),
                        'avg_complexity': avg_complexity,
                        'max_change_magnitude': max_change,
                        'dominant_type': Counter(s['state_type'] for s in segment_states).most_common(1)[0][0]
                    })
                
                current_time = segment_end
            
            return segments

        def calculate_timeline_stats(timeline_data: list) -> dict:
            """Calculate summary statistics for timeline data"""
            if not timeline_data:
                return {}
            
            complexities = [item['complexity_score'] for item in timeline_data]
            changes = [item['change_magnitude'] for item in timeline_data if item['change_magnitude'] > 0]
            
            from collections import Counter
            state_types = Counter(item['state_type'] for item in timeline_data)
            
            return {
                'avg_complexity': sum(complexities) / len(complexities),
                'max_complexity': max(complexities),
                'avg_change_magnitude': sum(changes) / len(changes) if changes else 0,
                'max_change_magnitude': max(changes) if changes else 0,
                'most_common_type': state_types.most_common(1)[0][0] if state_types else None,
                'type_distribution': dict(state_types),
                'total_duration_hours': (timeline_data[-1]['timestamp'] - timeline_data[0]['timestamp']) / 3600
            }

        # More cognitive state helper functions
        def calculate_state_complexity(state_data: dict) -> float:
            """Calculate complexity score for a cognitive state"""
            if not state_data:
                return 0.0
            
            component_count = len(state_data.keys())
            max_depth = calculate_dict_depth(state_data)
            total_values = count_dict_values(state_data)
            
            complexity = min(100, (component_count * 5) + (max_depth * 10) + (total_values * 0.5))
            return round(complexity, 2)

        def calculate_dict_depth(d: dict, current_depth: int = 0) -> int:
            """Calculate maximum depth of nested dictionary"""
            if not isinstance(d, dict):
                return current_depth
            
            if not d:
                return current_depth
            
            return max(calculate_dict_depth(v, current_depth + 1) for v in d.values())

        def count_dict_values(d: dict) -> int:
            """Count total number of values in nested dictionary"""
            count = 0
            for v in d.values():
                if isinstance(v, dict):
                    count += count_dict_values(v)
                elif isinstance(v, list):
                    count += len(v)
                else:
                    count += 1
            return count

        def compute_state_diff(state1: dict, state2: dict) -> dict:
            """Compute difference between two cognitive states"""
            diff_result = {
                'added': {},
                'removed': {},
                'modified': {},
                'magnitude': 0.0
            }
            
            all_keys = set(state1.keys()) | set(state2.keys())
            changes = 0
            total_keys = len(all_keys)
            
            for key in all_keys:
                if key not in state1:
                    diff_result['added'][key] = state2[key]
                    changes += 1
                elif key not in state2:
                    diff_result['removed'][key] = state1[key]
                    changes += 1
                elif state1[key] != state2[key]:
                    diff_result['modified'][key] = {
                        'before': state1[key],
                        'after': state2[key]
                    }
                    changes += 1
            
            if total_keys > 0:
                diff_result['magnitude'] = (changes / total_keys) * 100
            
            return diff_result

        def format_file_size(size_bytes: int) -> str:
            """Format file size in human readable format"""
            if size_bytes == 0:
                return "0 B"
            
            import math
            size_names = ["B", "KB", "MB", "GB", "TB"]
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return f"{s} {size_names[i]}"

        # Pattern analysis functions
        def find_cognitive_patterns(states: list, min_length: int, similarity_threshold: float) -> list:
            """Find recurring patterns in cognitive states"""
            patterns = []
            
            from collections import defaultdict
            type_sequences = defaultdict(list)
            for state in states:
                type_sequences[state['state_type']].append(state)
            
            for state_type, sequence in type_sequences.items():
                if len(sequence) >= min_length * 2:
                    for length in range(min_length, len(sequence) // 2 + 1):
                        for start in range(len(sequence) - length * 2 + 1):
                            subseq1 = sequence[start:start + length]
                            subseq2 = sequence[start + length:start + length * 2]
                            
                            similarity = calculate_sequence_similarity(subseq1, subseq2)
                            if similarity >= similarity_threshold:
                                patterns.append({
                                    'type': f'repeating_{state_type}',
                                    'length': length,
                                    'similarity': similarity,
                                    'occurrences': 2,
                                    'first_occurrence': subseq1[0]['timestamp'],
                                    'pattern_description': f"Repeating {state_type} sequence of {length} states"
                                })
            
            return sorted(patterns, key=lambda p: p['similarity'], reverse=True)

        def calculate_sequence_similarity(seq1: list, seq2: list) -> float:
            """Calculate similarity between two state sequences"""
            if len(seq1) != len(seq2):
                return 0.0
            
            total_similarity = 0.0
            for s1, s2 in zip(seq1, seq2, strict=False):
                state_sim = calculate_single_state_similarity(s1, s2)
                total_similarity += state_sim
            
            return total_similarity / len(seq1)

        def calculate_single_state_similarity(state1: dict, state2: dict) -> float:
            """Calculate similarity between two individual states"""
            data1 = state1.get('state_data', {})
            data2 = state2.get('state_data', {})
            
            if not data1 and not data2:
                return 1.0
            
            if not data1 or not data2:
                return 0.0
            
            keys1 = set(data1.keys())
            keys2 = set(data2.keys())
            key_similarity = len(keys1 & keys2) / len(keys1 | keys2) if keys1 | keys2 else 1.0
            
            common_keys = keys1 & keys2
            value_similarity = 0.0
            if common_keys:
                matching_values = sum(1 for key in common_keys if data1[key] == data2[key])
                value_similarity = matching_values / len(common_keys)
            
            return (key_similarity + value_similarity) / 2

        def analyze_state_transitions(states: list) -> list:
            """Analyze transitions between cognitive states"""
            from collections import defaultdict
            transitions = defaultdict(int)
            
            for i in range(len(states) - 1):
                current_type = states[i]['state_type']
                next_type = states[i + 1]['state_type']
                transition = f"{current_type} → {next_type}"
                transitions[transition] += 1
            
            sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
            
            return [
                {
                    'transition': transition,
                    'count': count,
                    'percentage': (count / (len(states) - 1)) * 100 if len(states) > 1 else 0
                }
                for transition, count in sorted_transitions
            ]

        def detect_cognitive_anomalies(states: list) -> list:
            """Detect anomalous cognitive states"""
            anomalies = []
            
            if len(states) < 3:
                return anomalies
            
            complexities = [calculate_state_complexity(s.get('state_data', {})) for s in states]
            avg_complexity = sum(complexities) / len(complexities)
            std_complexity = (sum((c - avg_complexity) ** 2 for c in complexities) / len(complexities)) ** 0.5
            
            for i, state in enumerate(states):
                complexity = complexities[i]
                z_score = (complexity - avg_complexity) / std_complexity if std_complexity > 0 else 0
                
                if abs(z_score) > 2:
                    anomalies.append({
                        'state_id': state['state_id'],
                        'timestamp': state['timestamp'],
                        'anomaly_type': 'complexity_outlier',
                        'z_score': z_score,
                        'description': f"Unusual complexity: {complexity:.1f} (avg: {avg_complexity:.1f})",
                        'severity': 'high' if abs(z_score) > 3 else 'medium'
                    })
            
            return anomalies

        # ===== EXTENDED COGNITIVE STATES API ENDPOINTS =====
        
        # Add required imports and helper functions
        import json
        from datetime import datetime
        
        def get_db_connection():
            """Get database connection for UMS Explorer API"""
            import sqlite3
            # Use the storage_dir that's already defined in this context
            database_path = str(storage_dir / "unified_agent_memory.db")
            conn = sqlite3.connect(database_path)
            conn.row_factory = sqlite3.Row
            return conn
        
        async def api_cognitive_state_detail(request):
            """Get detailed cognitive state by ID"""
            state_id = request.path_params['state_id']
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        cs.*,
                        w.title as workflow_title,
                        w.goal as workflow_goal
                    FROM cognitive_timeline_states cs
                    LEFT JOIN workflows w ON cs.workflow_id = w.workflow_id
                    WHERE cs.state_id = ?
                """, (state_id,))
                
                row = cursor.fetchone()
                if not row:
                    return JSONResponse({"error": "Cognitive state not found"}, status_code=404)
                
                columns = [description[0] for description in cursor.description]
                state = dict(zip(columns, row, strict=False))
                
                # Parse state data
                try:
                    state['state_data'] = json.loads(state.get('state_data', '{}'))
                except Exception:
                    state['state_data'] = {}
                
                # Get associated memories and actions
                cursor.execute("""
                    SELECT memory_id, memory_type, content, importance, created_at
                    FROM memories 
                    WHERE workflow_id = ? 
                    ORDER BY created_at DESC
                    LIMIT 20
                """, (state.get('workflow_id', ''),))
                
                memories = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                cursor.execute("""
                    SELECT action_id, action_type, tool_name, status, started_at
                    FROM actions 
                    WHERE workflow_id = ? 
                    ORDER BY started_at DESC
                    LIMIT 20
                """, (state.get('workflow_id', ''),))
                
                actions = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                conn.close()
                
                return JSONResponse({
                    **state,
                    'memories': memories,
                    'actions': actions,
                    'formatted_timestamp': datetime.fromtimestamp(state['timestamp']).isoformat(),
                    'complexity_score': calculate_state_complexity(state['state_data'])
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_cognitive_patterns(request):
            """Analyze recurring cognitive patterns"""
            query_params = request.query_params
            lookback_hours = int(query_params.get('lookback_hours', 24))
            min_pattern_length = int(query_params.get('min_pattern_length', 3))
            similarity_threshold = float(query_params.get('similarity_threshold', 0.7))
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (lookback_hours * 3600)
                cursor.execute("""
                    SELECT state_id, timestamp, state_type, state_data, workflow_id
                    FROM cognitive_timeline_states 
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                """, (since_timestamp,))
                
                states = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Parse state data
                for state in states:
                    try:
                        state['state_data'] = json.loads(state.get('state_data', '{}'))
                    except Exception:
                        state['state_data'] = {}
                
                # Analyze patterns
                patterns = find_cognitive_patterns(states, min_pattern_length, similarity_threshold)
                transitions = analyze_state_transitions(states)
                anomalies = detect_cognitive_anomalies(states)
                
                conn.close()
                
                return JSONResponse({
                    'total_states': len(states),
                    'time_range_hours': lookback_hours,
                    'patterns': patterns,
                    'transitions': transitions,
                    'anomalies': anomalies,
                    'summary': {
                        'pattern_count': len(patterns),
                        'most_common_transition': transitions[0] if transitions else None,
                        'anomaly_count': len(anomalies)
                    }
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        # ===== ACTION MONITOR API ENDPOINTS =====
        
        async def api_running_actions(request):
            """Get currently executing actions with real-time status"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        a.*,
                        w.title as workflow_title,
                        (unixepoch() - a.started_at) as execution_time,
                        CASE 
                            WHEN a.tool_data IS NOT NULL THEN json_extract(a.tool_data, '$.estimated_duration')
                            ELSE NULL 
                        END as estimated_duration
                    FROM actions a
                    LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
                    WHERE a.status IN ('running', 'executing', 'in_progress')
                    ORDER BY a.started_at ASC
                """)
                
                columns = [description[0] for description in cursor.description]
                running_actions = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
                
                # Enhance with real-time metrics
                enhanced_actions = []
                for action in running_actions:
                    try:
                        tool_data = json.loads(action.get('tool_data', '{}'))
                    except Exception:
                        tool_data = {}
                    
                    execution_time = action.get('execution_time', 0)
                    estimated_duration = action.get('estimated_duration') or 30
                    progress_percentage = min(95, (execution_time / estimated_duration) * 100) if estimated_duration > 0 else 0
                    
                    enhanced_action = {
                        **action,
                        'tool_data': tool_data,
                        'execution_time_seconds': execution_time,
                        'progress_percentage': progress_percentage,
                        'status_indicator': get_action_status_indicator(action['status'], execution_time),
                        'performance_category': categorize_action_performance(execution_time, estimated_duration),
                        'resource_usage': get_action_resource_usage(action['action_id']),
                        'formatted_start_time': datetime.fromtimestamp(action['started_at']).isoformat()
                    }
                    enhanced_actions.append(enhanced_action)
                
                conn.close()
                
                return JSONResponse({
                    'running_actions': enhanced_actions,
                    'total_running': len(enhanced_actions),
                    'avg_execution_time': sum(a['execution_time_seconds'] for a in enhanced_actions) / len(enhanced_actions) if enhanced_actions else 0,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_action_queue(request):
            """Get queued actions waiting for execution"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        a.*,
                        w.title as workflow_title,
                        (unixepoch() - a.created_at) as queue_time,
                        CASE 
                            WHEN a.tool_data IS NOT NULL THEN json_extract(a.tool_data, '$.priority')
                            ELSE 5 
                        END as priority
                    FROM actions a
                    LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
                    WHERE a.status IN ('queued', 'pending', 'waiting')
                    ORDER BY priority ASC, a.created_at ASC
                """)
                
                columns = [description[0] for description in cursor.description]
                queued_actions = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
                
                # Enhance queue data
                enhanced_queue = []
                for i, action in enumerate(queued_actions):
                    try:
                        tool_data = json.loads(action.get('tool_data', '{}'))
                    except Exception:
                        tool_data = {}
                    
                    enhanced_action = {
                        **action,
                        'tool_data': tool_data,
                        'queue_position': i + 1,
                        'queue_time_seconds': action.get('queue_time', 0),
                        'estimated_wait_time': estimate_wait_time(i, queued_actions),
                        'priority_label': get_priority_label(action.get('priority', 5)),
                        'formatted_queue_time': datetime.fromtimestamp(action['created_at']).isoformat()
                    }
                    enhanced_queue.append(enhanced_action)
                
                conn.close()
                
                return JSONResponse({
                    'queued_actions': enhanced_queue,
                    'total_queued': len(enhanced_queue),
                    'avg_queue_time': sum(a['queue_time_seconds'] for a in enhanced_queue) / len(enhanced_queue) if enhanced_queue else 0,
                    'next_action': enhanced_queue[0] if enhanced_queue else None,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        # ===== MORE MISSING API ENDPOINTS =====
        
        async def api_action_history(request):
            """Get completed actions with performance metrics"""
            query_params = request.query_params
            limit = int(query_params.get('limit', 50))
            offset = int(query_params.get('offset', 0))
            status_filter = query_params.get('status_filter')
            tool_filter = query_params.get('tool_filter')
            hours_back = int(query_params.get('hours_back', 24))
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
                
                query = """
                    SELECT 
                        a.*,
                        w.title as workflow_title,
                        (a.completed_at - a.started_at) as execution_duration,
                        CASE 
                            WHEN a.tool_data IS NOT NULL THEN json_extract(a.tool_data, '$.result_size')
                            ELSE 0 
                        END as result_size
                    FROM actions a
                    LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
                    WHERE a.status IN ('completed', 'failed', 'cancelled', 'timeout')
                    AND a.completed_at >= ?
                """
                params = [since_timestamp]
                
                if status_filter:
                    query += " AND a.status = ?"
                    params.append(status_filter)
                
                if tool_filter:
                    query += " AND a.tool_name = ?"
                    params.append(tool_filter)
                
                query += " ORDER BY a.completed_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                completed_actions = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
                
                # Calculate performance metrics
                enhanced_history = []
                for action in completed_actions:
                    try:
                        tool_data = json.loads(action.get('tool_data', '{}'))
                        result_data = json.loads(action.get('result', '{}'))
                    except Exception:
                        tool_data = {}
                        result_data = {}
                    
                    execution_duration = action.get('execution_duration', 0)
                    
                    enhanced_action = {
                        **action,
                        'tool_data': tool_data,
                        'result_data': result_data,
                        'execution_duration_seconds': execution_duration,
                        'performance_score': calculate_action_performance_score(action),
                        'efficiency_rating': calculate_efficiency_rating(execution_duration, action.get('result_size', 0)),
                        'success_rate_impact': 1 if action['status'] == 'completed' else 0,
                        'formatted_start_time': datetime.fromtimestamp(action['started_at']).isoformat(),
                        'formatted_completion_time': datetime.fromtimestamp(action['completed_at']).isoformat() if action['completed_at'] else None
                    }
                    enhanced_history.append(enhanced_action)
                
                conn.close()
                
                # Calculate aggregate metrics
                total_actions = len(enhanced_history)
                successful_actions = len([a for a in enhanced_history if a['status'] == 'completed'])
                avg_duration = sum(a['execution_duration_seconds'] for a in enhanced_history) / total_actions if total_actions > 0 else 0
                
                return JSONResponse({
                    'action_history': enhanced_history,
                    'total_actions': total_actions,
                    'success_rate': (successful_actions / total_actions * 100) if total_actions > 0 else 0,
                    'avg_execution_time': avg_duration,
                    'performance_summary': calculate_performance_summary(enhanced_history),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_action_metrics(request):
            """Get comprehensive action execution metrics and analytics"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Get metrics for last 24 hours
                since_timestamp = datetime.now().timestamp() - (24 * 3600)
                
                # Overall statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_actions,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_actions,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_actions,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration
                    FROM actions 
                    WHERE created_at >= ?
                """, (since_timestamp,))
                
                overall_stats = dict(zip([d[0] for d in cursor.description], cursor.fetchone(), strict=False))
                
                # Tool usage statistics
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as usage_count,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration
                    FROM actions 
                    WHERE created_at >= ?
                    GROUP BY tool_name
                    ORDER BY usage_count DESC
                """, (since_timestamp,))
                
                tool_stats = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Performance distribution over time (hourly)
                cursor.execute("""
                    SELECT 
                        strftime('%H', datetime(started_at, 'unixepoch')) as hour,
                        COUNT(*) as action_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count
                    FROM actions 
                    WHERE started_at >= ?
                    GROUP BY hour
                    ORDER BY hour
                """, (since_timestamp,))
                
                hourly_metrics = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                conn.close()
                
                # Calculate derived metrics
                success_rate = (overall_stats['successful_actions'] / overall_stats['total_actions'] * 100) if overall_stats['total_actions'] > 0 else 0
                
                return JSONResponse({
                    'overall_metrics': {
                        **overall_stats,
                        'success_rate_percentage': success_rate,
                        'failure_rate_percentage': 100 - success_rate,
                        'avg_duration_seconds': overall_stats['avg_duration'] or 0
                    },
                    'tool_usage_stats': tool_stats,
                    'hourly_performance': hourly_metrics,
                    'performance_insights': generate_performance_insights(overall_stats, tool_stats, hourly_metrics),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_tools_usage(request):
            """Get detailed tool usage statistics with performance breakdown"""
            query_params = request.query_params
            hours_back = int(query_params.get('hours_back', 24))
            include_performance = query_params.get('include_performance', 'true').lower() == 'true'  # noqa: F841
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
                
                # Comprehensive tool statistics
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as total_calls,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_calls,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_calls,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_execution_time,
                        MIN(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as min_execution_time,
                        MAX(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as max_execution_time,
                        COUNT(DISTINCT workflow_id) as unique_workflows,
                        MAX(started_at) as last_used
                    FROM actions 
                    WHERE created_at >= ?
                    GROUP BY tool_name
                    ORDER BY total_calls DESC
                """, (since_timestamp,))
                
                tool_statistics = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Enhance with calculated metrics
                enhanced_tool_stats = []
                for tool in tool_statistics:
                    success_rate = (tool['successful_calls'] / tool['total_calls'] * 100) if tool['total_calls'] > 0 else 0
                    
                    enhanced_tool = {
                        **tool,
                        'success_rate_percentage': success_rate,
                        'reliability_score': calculate_tool_reliability_score(tool),
                        'performance_category': categorize_tool_performance(tool['avg_execution_time']),
                        'usage_trend': 'increasing',  # This would be calculated with historical data
                        'last_used_formatted': datetime.fromtimestamp(tool['last_used']).isoformat()
                    }
                    enhanced_tool_stats.append(enhanced_tool)
                
                conn.close()
                
                return JSONResponse({
                    'tool_statistics': enhanced_tool_stats,
                    'summary': {
                        'total_tools_used': len(enhanced_tool_stats),
                        'most_used_tool': enhanced_tool_stats[0]['tool_name'] if enhanced_tool_stats else None,
                        'highest_success_rate': max(t['success_rate_percentage'] for t in enhanced_tool_stats) if enhanced_tool_stats else 0,
                        'avg_tool_performance': sum(t['avg_execution_time'] or 0 for t in enhanced_tool_stats) / len(enhanced_tool_stats) if enhanced_tool_stats else 0
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_cognitive_compare(request):
            """Compare two cognitive states and return detailed diff"""
            try:
                body = await request.json()
                state_id_1 = body.get('state_id_1')
                state_id_2 = body.get('state_id_2')
                
                if not state_id_1 or not state_id_2:
                    return JSONResponse({"error": "Both state IDs required"}, status_code=400)
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Get both states
                cursor.execute("SELECT * FROM cognitive_timeline_states WHERE state_id IN (?, ?)", (state_id_1, state_id_2))
                states = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                if len(states) != 2:
                    return JSONResponse({"error": "One or both states not found"}, status_code=404)
                
                # Parse state data
                for state in states:
                    try:
                        state['state_data'] = json.loads(state.get('state_data', '{}'))
                    except Exception:
                        state['state_data'] = {}
                
                # Order states by timestamp
                states.sort(key=lambda s: s['timestamp'])
                state_1, state_2 = states
                
                # Compute comprehensive diff
                diff_result = compute_state_diff(state_1.get('state_data', {}), state_2.get('state_data', {}))
                
                conn.close()
                
                return JSONResponse({
                    'state_1': {
                        'state_id': state_1['state_id'],
                        'timestamp': state_1['timestamp'],
                        'formatted_timestamp': datetime.fromtimestamp(state_1['timestamp']).isoformat()
                    },
                    'state_2': {
                        'state_id': state_2['state_id'],
                        'timestamp': state_2['timestamp'],
                        'formatted_timestamp': datetime.fromtimestamp(state_2['timestamp']).isoformat()
                    },
                    'time_diff_minutes': (state_2['timestamp'] - state_1['timestamp']) / 60,
                    'diff': diff_result
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        # Register all the API endpoints
        app.add_route("/api/cognitive-states/{state_id}", api_cognitive_state_detail, methods=["GET"])
        app.add_route("/api/cognitive-states/patterns", api_cognitive_patterns, methods=["GET"])
        app.add_route("/api/cognitive-states/compare", api_cognitive_compare, methods=["POST"])
        app.add_route("/api/actions/running", api_running_actions, methods=["GET"])
        app.add_route("/api/actions/queue", api_action_queue, methods=["GET"])
        app.add_route("/api/actions/history", api_action_history, methods=["GET"])
        app.add_route("/api/actions/metrics", api_action_metrics, methods=["GET"])
        app.add_route("/api/tools/usage", api_tools_usage, methods=["GET"])
        
        # ===== WORKFLOW PERFORMANCE PROFILER ENDPOINTS =====
        
        async def api_performance_overview(request):
            """Get comprehensive performance overview with metrics and trends"""
            try:
                query_params = request.query_params
                hours_back = int(query_params.get('hours_back', 24))
                granularity = query_params.get('granularity', 'hour')  # hour, minute, day
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
                
                # Overall performance metrics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_actions,
                        COUNT(DISTINCT workflow_id) as active_workflows,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_execution_time,
                        MIN(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as min_execution_time,
                        MAX(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as max_execution_time,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_actions,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_actions,
                        COUNT(DISTINCT tool_name) as tools_used
                    FROM actions 
                    WHERE started_at >= ?
                """, (since_timestamp,))
                
                overview_result = cursor.fetchone()
                overview_stats = dict(zip([d[0] for d in cursor.description], overview_result, strict=False)) if overview_result else {}
                
                # Calculate performance metrics
                success_rate = (overview_stats.get('successful_actions', 0) / max(1, overview_stats.get('total_actions', 1))) * 100
                throughput = overview_stats.get('total_actions', 0) / max(1, hours_back)
                
                # Performance timeline
                if granularity == 'hour':
                    time_format = "strftime('%Y-%m-%d %H:00:00', datetime(started_at, 'unixepoch'))"
                elif granularity == 'minute':
                    time_format = "strftime('%Y-%m-%d %H:%M:00', datetime(started_at, 'unixepoch'))"
                else:  # day
                    time_format = "strftime('%Y-%m-%d', datetime(started_at, 'unixepoch'))"
                
                cursor.execute(f"""
                    SELECT 
                        {time_format} as time_bucket,
                        COUNT(*) as action_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_count,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count,
                        COUNT(DISTINCT workflow_id) as workflow_count
                    FROM actions 
                    WHERE started_at >= ?
                    GROUP BY {time_format}
                    ORDER BY time_bucket
                """, (since_timestamp,))
                
                timeline_data = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Resource utilization by tool
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as usage_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count,
                        MAX(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as max_duration
                    FROM actions 
                    WHERE started_at >= ?
                    GROUP BY tool_name
                    ORDER BY usage_count DESC
                """, (since_timestamp,))
                
                tool_utilization = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Top bottlenecks (slowest operations)
                cursor.execute("""
                    SELECT 
                        tool_name,
                        workflow_id,
                        action_id,
                        started_at,
                        completed_at,
                        (completed_at - started_at) as duration,
                        status,
                        reasoning
                    FROM actions 
                    WHERE started_at >= ? AND completed_at IS NOT NULL
                    ORDER BY duration DESC
                    LIMIT 10
                """, (since_timestamp,))
                
                bottlenecks = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                conn.close()
                
                return JSONResponse({
                    'overview': {
                        **overview_stats,
                        'success_rate_percentage': success_rate,
                        'throughput_per_hour': throughput,
                        'error_rate_percentage': 100 - success_rate,
                        'avg_workflow_size': overview_stats.get('total_actions', 0) / max(1, overview_stats.get('active_workflows', 1))
                    },
                    'timeline': timeline_data,
                    'tool_utilization': tool_utilization,
                    'bottlenecks': bottlenecks,
                    'analysis_period': {
                        'hours_back': hours_back,
                        'granularity': granularity,
                        'start_time': since_timestamp,
                        'end_time': datetime.now().timestamp()
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_performance_bottlenecks(request):
            """Identify and analyze performance bottlenecks with detailed insights"""
            try:
                query_params = request.query_params
                hours_back = int(query_params.get('hours_back', 24))
                min_duration = float(query_params.get('min_duration', 1.0))  # minimum seconds to consider
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
                
                # Identify bottlenecks by tool
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as total_calls,
                        AVG(completed_at - started_at) as avg_duration,
                        MAX(completed_at - started_at) as max_duration,
                        MIN(completed_at - started_at) as min_duration,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY completed_at - started_at) as p95_duration,
                        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY completed_at - started_at) as p99_duration,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failure_count,
                        SUM(completed_at - started_at) as total_time_spent
                    FROM actions 
                    WHERE started_at >= ? AND completed_at IS NOT NULL AND (completed_at - started_at) >= ?
                    GROUP BY tool_name
                    ORDER BY avg_duration DESC
                """, (since_timestamp, min_duration))
                
                tool_bottlenecks = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Identify workflow bottlenecks
                cursor.execute("""
                    SELECT 
                        w.workflow_id,
                        w.title,
                        COUNT(a.action_id) as action_count,
                        AVG(a.completed_at - a.started_at) as avg_action_duration,
                        MAX(a.completed_at - a.started_at) as max_action_duration,
                        SUM(a.completed_at - a.started_at) as total_workflow_time,
                        MIN(a.started_at) as workflow_start,
                        MAX(a.completed_at) as workflow_end,
                        (MAX(a.completed_at) - MIN(a.started_at)) as total_elapsed_time
                    FROM workflows w
                    JOIN actions a ON w.workflow_id = a.workflow_id
                    WHERE a.started_at >= ? AND a.completed_at IS NOT NULL
                    GROUP BY w.workflow_id, w.title
                    HAVING COUNT(a.action_id) > 1
                    ORDER BY total_workflow_time DESC
                    LIMIT 20
                """, (since_timestamp,))
                
                workflow_bottlenecks = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Calculate parallelization opportunities
                cursor.execute("""
                    SELECT 
                        workflow_id,
                        COUNT(*) as sequential_actions,
                        SUM(completed_at - started_at) as total_sequential_time,
                        (MAX(completed_at) - MIN(started_at)) as actual_elapsed_time
                    FROM actions 
                    WHERE started_at >= ? AND completed_at IS NOT NULL
                    GROUP BY workflow_id
                    HAVING COUNT(*) > 2
                """, (since_timestamp,))
                
                parallelization_data = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Calculate potential time savings through parallelization
                parallelization_opportunities = []
                for data in parallelization_data:
                    potential_savings = data['total_sequential_time'] - data['actual_elapsed_time']
                    if potential_savings > 0:
                        parallelization_opportunities.append({
                            **data,
                            'potential_time_savings': potential_savings,
                            'parallelization_efficiency': (data['actual_elapsed_time'] / data['total_sequential_time']) * 100,
                            'optimization_score': min(10, potential_savings / data['actual_elapsed_time'] * 10)
                        })
                
                # Resource contention analysis
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as concurrent_usage,
                        AVG(completed_at - started_at) as avg_duration_under_contention
                    FROM actions a1
                    WHERE started_at >= ? AND EXISTS (
                        SELECT 1 FROM actions a2 
                        WHERE a2.tool_name = a1.tool_name 
                        AND a2.action_id != a1.action_id
                        AND a2.started_at <= a1.completed_at 
                        AND a2.completed_at >= a1.started_at
                    )
                    GROUP BY tool_name
                    ORDER BY concurrent_usage DESC
                """, (since_timestamp,))
                
                resource_contention = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                conn.close()
                
                # Generate optimization recommendations
                recommendations = []
                
                # Tool-based recommendations
                for tool in tool_bottlenecks[:5]:
                    if tool['avg_duration'] > 10:  # More than 10 seconds average
                        recommendations.append({
                            'type': 'tool_optimization',
                            'priority': 'high' if tool['avg_duration'] > 30 else 'medium',
                            'title': f"Optimize {tool['tool_name']} performance",
                            'description': f"Tool {tool['tool_name']} has high average duration of {tool['avg_duration']:.2f}s",
                            'impact': f"Could save ~{tool['total_time_spent'] * 0.3:.2f}s per execution period",
                            'actions': [
                                'Review tool implementation for optimization opportunities',
                                'Consider caching strategies for repeated operations',
                                'Evaluate if tool can be replaced with faster alternative'
                            ]
                        })
                
                # Parallelization recommendations
                for opp in sorted(parallelization_opportunities, key=lambda x: x['potential_time_savings'], reverse=True)[:3]:
                    recommendations.append({
                        'type': 'parallelization',
                        'priority': 'high' if opp['potential_time_savings'] > 20 else 'medium',
                        'title': f"Parallelize workflow {opp['workflow_id']}",
                        'description': f"Workflow could save {opp['potential_time_savings']:.2f}s through parallel execution",
                        'impact': f"Up to {opp['parallelization_efficiency']:.1f}% efficiency improvement",
                        'actions': [
                            'Analyze action dependencies to identify parallelizable segments',
                            'Implement async execution where possible',
                            'Consider workflow restructuring for better parallelization'
                        ]
                    })
                
                return JSONResponse({
                    'tool_bottlenecks': tool_bottlenecks,
                    'workflow_bottlenecks': workflow_bottlenecks,
                    'parallelization_opportunities': parallelization_opportunities,
                    'resource_contention': resource_contention,
                    'recommendations': recommendations,
                    'analysis_summary': {
                        'total_bottlenecks_identified': len(tool_bottlenecks) + len(workflow_bottlenecks),
                        'highest_impact_tool': tool_bottlenecks[0]['tool_name'] if tool_bottlenecks else None,
                        'avg_tool_duration': sum(t['avg_duration'] for t in tool_bottlenecks) / len(tool_bottlenecks) if tool_bottlenecks else 0,
                        'parallelization_potential': len(parallelization_opportunities)
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_performance_flame_graph(request):
            """Generate flame graph data for workflow performance visualization"""
            try:
                query_params = request.query_params
                workflow_id = query_params.get('workflow_id')
                hours_back = int(query_params.get('hours_back', 24))
                
                if not workflow_id:
                    return JSONResponse({"error": "workflow_id parameter required"}, status_code=400)
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
                
                # Get workflow actions with timing data
                cursor.execute("""
                    SELECT 
                        action_id,
                        tool_name,
                        started_at,
                        completed_at,
                        (completed_at - started_at) as duration,
                        status,
                        reasoning,
                        summary,
                        dependency_path
                    FROM actions 
                    WHERE workflow_id = ? AND started_at >= ?
                    ORDER BY started_at
                """, (workflow_id, since_timestamp))
                
                actions = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                if not actions:
                    return JSONResponse({"error": "No actions found for workflow"}, status_code=404)
                
                # Build flame graph structure
                flame_graph_data = {
                    'name': f'Workflow {workflow_id}',
                    'value': sum(action.get('duration', 0) for action in actions if action.get('duration')),
                    'children': []
                }
                
                # Group actions by tool for flame graph hierarchy
                tool_groups = {}
                for action in actions:
                    tool_name = action.get('tool_name', 'unknown')
                    if tool_name not in tool_groups:
                        tool_groups[tool_name] = []
                    tool_groups[tool_name].append(action)
                
                # Build hierarchical structure
                for tool_name, tool_actions in tool_groups.items():
                    tool_duration = sum(action.get('duration', 0) for action in tool_actions if action.get('duration'))
                    
                    tool_node = {
                        'name': tool_name,
                        'value': tool_duration,
                        'children': []
                    }
                    
                    # Add individual actions as children
                    for action in tool_actions:
                        if action.get('duration'):
                            action_node = {
                                'name': f"Action {action['action_id']}",
                                'value': action['duration'],
                                'action_id': action['action_id'],
                                'status': action.get('status'),
                                'reasoning': action.get('reasoning', ''),
                                'started_at': action.get('started_at'),
                                'completed_at': action.get('completed_at')
                            }
                            tool_node['children'].append(action_node)
                    
                    flame_graph_data['children'].append(tool_node)
                
                # Calculate performance metrics
                total_duration = sum(action.get('duration', 0) for action in actions if action.get('duration'))
                workflow_start = min(action['started_at'] for action in actions if action.get('started_at'))
                workflow_end = max(action['completed_at'] for action in actions if action.get('completed_at'))
                wall_clock_time = workflow_end - workflow_start if workflow_end and workflow_start else 0
                
                # Parallelization efficiency
                parallelization_efficiency = (wall_clock_time / total_duration * 100) if total_duration > 0 else 0
                
                # Critical path analysis
                critical_path = []
                current_time = workflow_start
                
                while current_time < workflow_end:
                    # Find action that was running at current_time and ends latest
                    running_actions = [
                        a for a in actions 
                        if a.get('started_at', 0) <= current_time and a.get('completed_at', 0) > current_time
                    ]
                    
                    if running_actions:
                        # Find the action that ends latest (most critical)
                        critical_action = max(running_actions, key=lambda x: x.get('completed_at', 0))
                        if critical_action not in critical_path:
                            critical_path.append({
                                'action_id': critical_action['action_id'],
                                'tool_name': critical_action.get('tool_name'),
                                'duration': critical_action.get('duration'),
                                'start_time': critical_action.get('started_at'),
                                'end_time': critical_action.get('completed_at')
                            })
                        current_time = critical_action.get('completed_at', current_time + 1)
                    else:
                        current_time += 1
                
                conn.close()
                
                return JSONResponse({
                    'flame_graph': flame_graph_data,
                    'metrics': {
                        'total_actions': len(actions),
                        'total_cpu_time': total_duration,
                        'wall_clock_time': wall_clock_time,
                        'parallelization_efficiency': parallelization_efficiency,
                        'avg_action_duration': total_duration / len(actions) if actions else 0,
                        'workflow_start': workflow_start,
                        'workflow_end': workflow_end
                    },
                    'critical_path': critical_path,
                    'analysis': {
                        'bottleneck_tool': max(tool_groups.keys(), key=lambda t: sum(a.get('duration', 0) for a in tool_groups[t])) if tool_groups else None,
                        'parallelization_potential': max(0, total_duration - wall_clock_time),
                        'optimization_score': min(10, parallelization_efficiency / 10)
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_performance_trends(request):
            """Analyze performance trends and patterns over time"""
            try:
                query_params = request.query_params
                days_back = int(query_params.get('days_back', 7))
                query_params.get('metric', 'duration')  # duration, success_rate, throughput
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (days_back * 24 * 3600)
                
                # Daily trends
                cursor.execute("""
                    SELECT 
                        DATE(datetime(started_at, 'unixepoch')) as date,
                        COUNT(*) as action_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_actions,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_actions,
                        COUNT(DISTINCT workflow_id) as workflow_count,
                        COUNT(DISTINCT tool_name) as tool_count
                    FROM actions 
                    WHERE started_at >= ?
                    GROUP BY DATE(datetime(started_at, 'unixepoch'))
                    ORDER BY date
                """, (since_timestamp,))
                
                daily_trends = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Calculate derived metrics
                for day in daily_trends:
                    day['success_rate'] = (day['successful_actions'] / max(1, day['action_count'])) * 100
                    day['throughput'] = day['action_count'] / 24  # actions per hour
                    day['error_rate'] = (day['failed_actions'] / max(1, day['action_count'])) * 100
                
                # Tool performance trends
                cursor.execute("""
                    SELECT 
                        tool_name,
                        DATE(datetime(started_at, 'unixepoch')) as date,
                        COUNT(*) as usage_count,
                        AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count
                    FROM actions 
                    WHERE started_at >= ?
                    GROUP BY tool_name, DATE(datetime(started_at, 'unixepoch'))
                    ORDER BY tool_name, date
                """, (since_timestamp,))
                
                tool_trends = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Workflow complexity trends
                cursor.execute("""
                    SELECT 
                        DATE(datetime(started_at, 'unixepoch')) as date,
                        workflow_id,
                        COUNT(*) as action_count,
                        SUM(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as total_duration,
                        (MAX(completed_at) - MIN(started_at)) as elapsed_time
                    FROM actions 
                    WHERE started_at >= ? AND workflow_id IS NOT NULL
                    GROUP BY DATE(datetime(started_at, 'unixepoch')), workflow_id
                    ORDER BY date, workflow_id
                """, (since_timestamp,))
                
                workflow_complexity = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                # Calculate trend analysis
                if len(daily_trends) >= 2:
                    # Performance trend (improving, degrading, stable)
                    recent_avg = sum(d['avg_duration'] or 0 for d in daily_trends[-3:]) / min(3, len(daily_trends))
                    earlier_avg = sum(d['avg_duration'] or 0 for d in daily_trends[:3]) / min(3, len(daily_trends))
                    
                    if recent_avg > earlier_avg * 1.1:
                        performance_trend = 'degrading'
                    elif recent_avg < earlier_avg * 0.9:
                        performance_trend = 'improving'
                    else:
                        performance_trend = 'stable'
                    
                    # Success rate trend
                    recent_success = sum(d['success_rate'] for d in daily_trends[-3:]) / min(3, len(daily_trends))
                    earlier_success = sum(d['success_rate'] for d in daily_trends[:3]) / min(3, len(daily_trends))
                    
                    success_trend = 'improving' if recent_success > earlier_success else 'degrading' if recent_success < earlier_success else 'stable'
                else:
                    performance_trend = 'insufficient_data'
                    success_trend = 'insufficient_data'
                
                # Identify performance patterns
                patterns = []
                
                # Weekly pattern detection
                if len(daily_trends) >= 7:
                    weekend_performance = [d for d in daily_trends if datetime.strptime(d['date'], '%Y-%m-%d').weekday() >= 5]
                    weekday_performance = [d for d in daily_trends if datetime.strptime(d['date'], '%Y-%m-%d').weekday() < 5]
                    
                    if weekend_performance and weekday_performance:
                        weekend_avg = sum(d['avg_duration'] or 0 for d in weekend_performance) / len(weekend_performance)
                        weekday_avg = sum(d['avg_duration'] or 0 for d in weekday_performance) / len(weekday_performance)
                        
                        if abs(weekend_avg - weekday_avg) > weekday_avg * 0.2:
                            patterns.append({
                                'type': 'weekly_pattern',
                                'description': f"Performance varies significantly between weekdays ({weekday_avg:.2f}s) and weekends ({weekend_avg:.2f}s)",
                                'impact': 'medium',
                                'recommendation': 'Consider different optimization strategies for weekend vs weekday operations'
                            })
                
                # Anomaly detection (simple outlier detection)
                if daily_trends:
                    durations = [d['avg_duration'] or 0 for d in daily_trends]
                    mean_duration = sum(durations) / len(durations)
                    
                    outliers = [d for d in daily_trends if abs((d['avg_duration'] or 0) - mean_duration) > mean_duration * 0.5]
                    
                    for outlier in outliers:
                        patterns.append({
                            'type': 'performance_anomaly',
                            'date': outlier['date'],
                            'description': f"Unusual performance on {outlier['date']}: {outlier['avg_duration']:.2f}s vs normal {mean_duration:.2f}s",
                            'impact': 'high' if abs((outlier['avg_duration'] or 0) - mean_duration) > mean_duration else 'medium',
                            'recommendation': 'Investigate system conditions and workload on this date'
                        })
                
                conn.close()
                
                return JSONResponse({
                    'daily_trends': daily_trends,
                    'tool_trends': tool_trends,
                    'workflow_complexity': workflow_complexity,
                    'trend_analysis': {
                        'performance_trend': performance_trend,
                        'success_trend': success_trend,
                        'data_points': len(daily_trends),
                        'analysis_period_days': days_back
                    },
                    'patterns': patterns,
                    'insights': {
                        'best_performing_day': max(daily_trends, key=lambda x: x['success_rate']) if daily_trends else None,
                        'worst_performing_day': min(daily_trends, key=lambda x: x['success_rate']) if daily_trends else None,
                        'peak_throughput_day': max(daily_trends, key=lambda x: x['throughput']) if daily_trends else None,
                        'avg_daily_actions': sum(d['action_count'] for d in daily_trends) / len(daily_trends) if daily_trends else 0
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_performance_recommendations(request):
            """Generate actionable performance optimization recommendations"""
            try:
                query_params = request.query_params
                hours_back = int(query_params.get('hours_back', 24))
                priority_filter = query_params.get('priority', 'all')  # all, high, medium, low
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
                
                recommendations = []
                
                # Analyze slow tools
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as usage_count,
                        AVG(completed_at - started_at) as avg_duration,
                        MAX(completed_at - started_at) as max_duration,
                        SUM(completed_at - started_at) as total_time
                    FROM actions 
                    WHERE started_at >= ? AND completed_at IS NOT NULL
                    GROUP BY tool_name
                    HAVING avg_duration > 5
                    ORDER BY total_time DESC
                """, (since_timestamp,))
                
                slow_tools = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                for tool in slow_tools[:5]:
                    impact_score = tool['total_time'] / 3600  # hours of time spent
                    priority = 'high' if impact_score > 1 else 'medium' if impact_score > 0.5 else 'low'
                    
                    recommendations.append({
                        'id': f"optimize_tool_{tool['tool_name']}",
                        'type': 'tool_optimization',
                        'priority': priority,
                        'title': f"Optimize {tool['tool_name']} performance",
                        'description': f"Tool consumes {tool['total_time']:.1f}s total execution time with {tool['avg_duration']:.2f}s average",
                        'impact_estimate': {
                            'time_savings_potential': tool['total_time'] * 0.3,  # Assume 30% improvement possible
                            'affected_actions': tool['usage_count'],
                            'cost_benefit_ratio': impact_score
                        },
                        'implementation_steps': [
                            f"Profile {tool['tool_name']} execution to identify bottlenecks",
                            "Consider caching frequently used data",
                            "Optimize database queries if applicable",
                            "Evaluate alternative implementations or libraries"
                        ],
                        'estimated_effort': 'medium',
                        'prerequisites': ['Development environment setup', 'Performance profiling tools'],
                        'metrics_to_track': [
                            'Average execution time',
                            'P95 execution time',
                            'Tool success rate',
                            'Resource utilization'
                        ]
                    })
                
                # Analyze workflow parallelization opportunities
                cursor.execute("""
                    SELECT 
                        workflow_id,
                        COUNT(*) as action_count,
                        SUM(completed_at - started_at) as total_sequential_time,
                        (MAX(completed_at) - MIN(started_at)) as actual_elapsed_time
                    FROM actions 
                    WHERE started_at >= ? AND completed_at IS NOT NULL AND workflow_id IS NOT NULL
                    GROUP BY workflow_id
                    HAVING action_count > 3 AND total_sequential_time > actual_elapsed_time * 1.5
                    ORDER BY (total_sequential_time - actual_elapsed_time) DESC
                """, (since_timestamp,))
                
                parallelization_opps = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                for opp in parallelization_opps[:3]:
                    time_savings = opp['total_sequential_time'] - opp['actual_elapsed_time']
                    priority = 'high' if time_savings > 30 else 'medium'
                    
                    recommendations.append({
                        'id': f"parallelize_workflow_{opp['workflow_id']}",
                        'type': 'parallelization',
                        'priority': priority,
                        'title': f"Parallelize workflow {opp['workflow_id']}",
                        'description': f"Workflow could save {time_savings:.2f}s through better parallelization",
                        'impact_estimate': {
                            'time_savings_potential': time_savings,
                            'efficiency_improvement': (time_savings / opp['total_sequential_time']) * 100,
                            'affected_workflows': 1
                        },
                        'implementation_steps': [
                            "Analyze action dependencies in the workflow",
                            "Identify independent action sequences",
                            "Implement async execution patterns",
                            "Add proper synchronization points"
                        ],
                        'estimated_effort': 'high',
                        'prerequisites': ['Workflow dependency analysis', 'Async execution framework'],
                        'metrics_to_track': [
                            'Workflow end-to-end time',
                            'Action parallelization ratio',
                            'Resource utilization efficiency'
                        ]
                    })
                
                # Analyze error patterns
                cursor.execute("""
                    SELECT 
                        tool_name,
                        COUNT(*) as error_count,
                        COUNT(*) * 100.0 / (
                            SELECT COUNT(*) FROM actions a2 
                            WHERE a2.tool_name = actions.tool_name AND a2.started_at >= ?
                        ) as error_rate
                    FROM actions 
                    WHERE started_at >= ? AND status = 'failed'
                    GROUP BY tool_name
                    HAVING error_rate > 5
                    ORDER BY error_rate DESC
                """, (since_timestamp, since_timestamp))
                
                error_prone_tools = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                for tool in error_prone_tools[:3]:
                    priority = 'high' if tool['error_rate'] > 20 else 'medium'
                    
                    recommendations.append({
                        'id': f"improve_reliability_{tool['tool_name']}",
                        'type': 'reliability_improvement',
                        'priority': priority,
                        'title': f"Improve {tool['tool_name']} reliability",
                        'description': f"Tool has {tool['error_rate']:.1f}% failure rate ({tool['error_count']} failures)",
                        'impact_estimate': {
                            'reliability_improvement': tool['error_rate'],
                            'affected_actions': tool['error_count'],
                            'user_experience_impact': 'high'
                        },
                        'implementation_steps': [
                            "Analyze failure patterns and root causes",
                            "Implement better error handling and retries",
                            "Add input validation and sanitization",
                            "Improve tool documentation and usage examples"
                        ],
                        'estimated_effort': 'medium',
                        'prerequisites': ['Error logging analysis', 'Tool source code access'],
                        'metrics_to_track': [
                            'Tool failure rate',
                            'Time to recovery',
                            'User satisfaction scores'
                        ]
                    })
                
                # Filter recommendations by priority if requested
                if priority_filter != 'all':
                    recommendations = [r for r in recommendations if r['priority'] == priority_filter]
                
                # Sort by impact and priority
                priority_order = {'high': 3, 'medium': 2, 'low': 1}
                recommendations.sort(key=lambda x: (
                    priority_order.get(x['priority'], 0),
                    x['impact_estimate'].get('time_savings_potential', 0)
                ), reverse=True)
                
                conn.close()
                
                return JSONResponse({
                    'recommendations': recommendations,
                    'summary': {
                        'total_recommendations': len(recommendations),
                        'high_priority': len([r for r in recommendations if r['priority'] == 'high']),
                        'medium_priority': len([r for r in recommendations if r['priority'] == 'medium']),
                        'low_priority': len([r for r in recommendations if r['priority'] == 'low']),
                        'estimated_total_savings': sum(r['impact_estimate'].get('time_savings_potential', 0) for r in recommendations),
                        'analysis_period_hours': hours_back
                    },
                    'implementation_roadmap': {
                        'quick_wins': [r for r in recommendations if r['estimated_effort'] == 'low' and r['priority'] == 'high'],
                        'major_improvements': [r for r in recommendations if r['estimated_effort'] == 'high' and r['priority'] == 'high'],
                        'maintenance_tasks': [r for r in recommendations if r['priority'] == 'low']
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        # Register performance profiler endpoints
        app.add_route("/api/performance/overview", api_performance_overview, methods=["GET"])
        app.add_route("/api/performance/bottlenecks", api_performance_bottlenecks, methods=["GET"])
        app.add_route("/api/performance/flame-graph", api_performance_flame_graph, methods=["GET"])
        app.add_route("/api/performance/trends", api_performance_trends, methods=["GET"])
        app.add_route("/api/performance/recommendations", api_performance_recommendations, methods=["GET"])
        
        # ===== ADD MISSING ENDPOINTS EXPECTED BY FRONTEND =====
        
        # Memory Quality Inspector API
        async def api_memory_quality_stats(request):
            """Get memory quality statistics and overview"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Overall memory stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_memories,
                        COUNT(DISTINCT workflow_id) as unique_workflows,
                        AVG(importance) as avg_importance,
                        SUM(CASE WHEN importance < 3 THEN 1 ELSE 0 END) as low_importance_count,
                        COUNT(DISTINCT memory_type) as memory_types
                    FROM memories
                """)
                
                overall_result = cursor.fetchone()
                if not overall_result:
                    overall_stats = {
                        'total_memories': 0,
                        'unique_workflows': 0,
                        'avg_importance': 0.0,
                        'low_importance_count': 0,
                        'memory_types': 0
                    }
                else:
                    overall_stats = dict(zip([d[0] for d in cursor.description], overall_result, strict=False))
                
                # Find potential duplicates (simplified)
                cursor.execute("""
                    SELECT 
                        COUNT(*) as potential_duplicates
                    FROM (
                        SELECT content, COUNT(*) as cnt
                        FROM memories 
                        WHERE content IS NOT NULL 
                        GROUP BY content 
                        HAVING cnt > 1
                    )
                """)
                
                duplicate_stats = cursor.fetchone()
                potential_duplicates = duplicate_stats[0] if duplicate_stats and len(duplicate_stats) > 0 else 0
                
                # Find orphaned memories (no workflow)
                cursor.execute("""
                    SELECT COUNT(*) as orphaned_count
                    FROM memories m
                    LEFT JOIN workflows w ON m.workflow_id = w.workflow_id
                    WHERE w.workflow_id IS NULL
                """)
                
                orphaned_result = cursor.fetchone()
                orphaned_count = orphaned_result[0] if orphaned_result and len(orphaned_result) > 0 else 0
                
                # Find stale memories (older than 30 days)
                thirty_days_ago = datetime.now().timestamp() - (30 * 24 * 3600)
                cursor.execute("""
                    SELECT COUNT(*) as stale_count
                    FROM memories
                    WHERE created_at < ?
                """, (thirty_days_ago,))
                
                stale_result = cursor.fetchone()
                stale_count = stale_result[0] if stale_result and len(stale_result) > 0 else 0
                
                conn.close()
                
                quality_score = max(0, 100 - (potential_duplicates * 2) - (orphaned_count * 3) - (stale_count * 0.1))
                
                return JSONResponse({
                    'overall_stats': overall_stats,
                    'quality_issues': {
                        'potential_duplicates': potential_duplicates,
                        'orphaned_memories': orphaned_count,
                        'stale_memories': stale_count,
                        'low_importance_memories': overall_stats.get('low_importance_count', 0)
                    },
                    'quality_score': round(quality_score, 1),
                    'recommendations': [
                        'Review duplicate memories for consolidation',
                        'Assign orphaned memories to workflows',
                        'Archive or delete stale memories',
                        'Update importance scores for better prioritization'
                    ],
                    'last_analysis': datetime.now().isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_memory_quality_quick_scan(request):
            """Perform quick scan for memory quality issues"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Quick scan for immediate issues
                issues = []
                
                # Check for duplicate content
                cursor.execute("""
                    SELECT content, COUNT(*) as count, GROUP_CONCAT(memory_id) as memory_ids
                    FROM memories 
                    WHERE content IS NOT NULL AND LENGTH(content) > 10
                    GROUP BY content 
                    HAVING count > 1
                    LIMIT 10
                """)
                
                duplicates = cursor.fetchall()
                for dup in duplicates:
                    issues.append({
                        'type': 'duplicate_content',
                        'severity': 'medium',
                        'description': f'Found {dup[1]} memories with identical content',
                        'affected_memories': dup[2].split(',') if dup[2] else [],
                        'preview': dup[0][:100] + '...' if len(dup[0]) > 100 else dup[0]
                    })
                
                # Check for orphaned memories
                cursor.execute("""
                    SELECT m.memory_id, m.content
                    FROM memories m
                    LEFT JOIN workflows w ON m.workflow_id = w.workflow_id
                    WHERE w.workflow_id IS NULL
                    LIMIT 5
                """)
                
                orphaned = cursor.fetchall()
                for orphan in orphaned:
                    issues.append({
                        'type': 'orphaned_memory',
                        'severity': 'low',
                        'description': 'Memory not associated with any workflow',
                        'memory_id': orphan[0],
                        'preview': orphan[1][:100] + '...' if orphan[1] and len(orphan[1]) > 100 else orphan[1]
                    })
                
                # Get total memory count for score calculation
                cursor.execute("SELECT COUNT(*) FROM memories")
                total_memories_result = cursor.fetchone()
                total_memories = total_memories_result[0] if total_memories_result else 0
                
                # Calculate a simple quality score
                issue_count = len(issues)
                overall_score = max(60, 100 - (issue_count * 10))  # Simple scoring
                
                conn.close()
                
                return JSONResponse({
                    'success': True,
                    'quick_metrics': {
                        'overall_score': overall_score,
                        'total_memories': total_memories,
                        'issues_found': issue_count,
                        'scan_time': datetime.now().isoformat()
                    },
                    'issues': issues,
                    'scan_completed': True,
                    'next_scan_recommended': (datetime.now() + timedelta(hours=24)).isoformat()
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_memory_quality_analyze(request):
            """Perform comprehensive memory quality analysis"""
            try:
                body = await request.json()
                analysis_type = body.get('analysis_type', 'comprehensive')
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                analysis_results = {
                    'analysis_type': analysis_type,
                    'started_at': datetime.now().isoformat(),
                    'issues_found': 0,
                    'issues': [],
                    'recommendations': [],
                    'statistics': {}
                }
                
                if analysis_type in ['comprehensive', 'duplicates']:
                    # Analyze duplicates
                    cursor.execute("""
                        SELECT content, COUNT(*) as count, GROUP_CONCAT(memory_id) as memory_ids,
                               AVG(importance) as avg_importance
                        FROM memories 
                        WHERE content IS NOT NULL AND LENGTH(content) > 20
                        GROUP BY content 
                        HAVING count > 1
                    """)
                    
                    for row in cursor.fetchall():
                        analysis_results['issues'].append({
                            'issue_id': f"dup_{len(analysis_results['issues'])}",
                            'issue_type': 'duplicate',
                            'title': f'Duplicate Content ({row[1]} copies)',
                            'description': f'Found {row[1]} memories with identical content',
                            'severity': 'medium' if row[1] > 2 else 'low',
                            'auto_fixable': True,
                            'memory_ids': row[2].split(','),
                            'avg_importance': row[3],
                            'content_preview': row[0][:200] + '...' if len(row[0]) > 200 else row[0],
                            'recommendation': 'Merge duplicate memories into a single entry'
                        })
                
                if analysis_type in ['comprehensive', 'orphaned']:
                    # Analyze orphaned memories
                    cursor.execute("""
                        SELECT COUNT(*) as orphaned_count
                        FROM memories m
                        LEFT JOIN workflows w ON m.workflow_id = w.workflow_id
                        WHERE w.workflow_id IS NULL
                    """)
                    
                    orphaned_result = cursor.fetchone()
                    orphaned_count = orphaned_result[0] if orphaned_result else 0
                    if orphaned_count > 0:
                        analysis_results['issues'].append({
                            'issue_id': f"orph_{len(analysis_results['issues'])}",
                            'issue_type': 'orphaned',
                            'title': f'Orphaned Memories ({orphaned_count} found)',
                            'description': f'{orphaned_count} memories are not associated with any workflow',
                            'severity': 'medium',
                            'auto_fixable': False,
                            'count': orphaned_count,
                            'recommendation': 'Assign memories to appropriate workflows or archive if not needed'
                        })
                
                # Get total memory count for stats
                cursor.execute("SELECT COUNT(*) FROM memories")
                total_memories_result = cursor.fetchone()
                total_memories = total_memories_result[0] if total_memories_result else 0
                
                analysis_results['issues_found'] = len(analysis_results['issues'])
                analysis_results['overall_score'] = max(50, 100 - (analysis_results['issues_found'] * 15))
                analysis_results['total_memories'] = total_memories
                
                # Generate recommendations
                if len(analysis_results['issues']) == 0:
                    analysis_results['recommendations'].append('No major quality issues found!')
                else:
                    analysis_results['recommendations'].extend([
                        'Consider merging duplicate memories',
                        'Assign orphaned memories to appropriate workflows',
                        'Review low-importance memories for archival'
                    ])
                
                conn.close()
                
                analysis_results['completed_at'] = datetime.now().isoformat()
                
                return JSONResponse({
                    'success': True,
                    'analysis': analysis_results
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_memory_quality_duplicates(request):
            """Get detailed duplicate memory analysis"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT content, COUNT(*) as count, GROUP_CONCAT(memory_id) as memory_ids,
                           MIN(created_at) as first_created, MAX(created_at) as last_created,
                           AVG(importance) as avg_importance
                    FROM memories 
                    WHERE content IS NOT NULL AND LENGTH(content) > 10
                    GROUP BY content 
                    HAVING count > 1
                    ORDER BY count DESC
                """)
                
                duplicate_groups = []
                for row in cursor.fetchall():
                    memory_ids = row[2].split(',')
                    
                    # Get detailed info for each memory in the group
                    memory_details = []
                    for memory_id in memory_ids:
                        cursor.execute("""
                            SELECT memory_id, workflow_id, memory_type, importance, created_at
                            FROM memories WHERE memory_id = ?
                        """, (memory_id,))
                        
                        detail = cursor.fetchone()
                        if detail:
                            memory_details.append(dict(zip([d[0] for d in cursor.description], detail, strict=False)))
                    
                    duplicate_groups.append({
                        'cluster_id': f"dup_cluster_{len(duplicate_groups)}",
                        'content_preview': row[0][:200] + '...' if len(row[0]) > 200 else row[0],
                        'duplicate_count': row[1],
                        'memory_ids': memory_ids,
                        'primary_memory_id': memory_ids[0] if memory_ids else None,
                        'memory_details': memory_details,
                        'first_created': row[3],
                        'last_created': row[4],
                        'avg_importance': row[5],
                        'recommendation': 'merge' if row[1] > 2 else 'review'
                    })
                
                conn.close()
                
                return JSONResponse({
                    'success': True,
                    'clusters': duplicate_groups,
                    'duplicate_groups': duplicate_groups,
                    'total_groups': len(duplicate_groups),
                    'total_duplicates': sum(group['duplicate_count'] for group in duplicate_groups)
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_memory_quality_orphaned(request):
            """Get orphaned memories (not associated with workflows)"""
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT m.memory_id, m.content, m.memory_type, m.importance, m.created_at
                    FROM memories m
                    LEFT JOIN workflows w ON m.workflow_id = w.workflow_id
                    WHERE w.workflow_id IS NULL
                    ORDER BY m.created_at DESC
                """)
                
                orphaned_memories = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                conn.close()
                
                return JSONResponse({
                    'success': True,
                    'orphaned_memories': orphaned_memories,
                    'total_orphaned': len(orphaned_memories),
                    'recommendation': 'Assign to appropriate workflows or archive if no longer needed'
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_memory_quality_bulk_execute(request):
            """Execute bulk operations on memories"""
            try:
                body = await request.json()
                operation_type = body.get('operation_type')
                memory_ids = body.get('memory_ids', [])
                
                if not memory_ids:
                    return JSONResponse({"error": "No memory IDs provided"}, status_code=400)
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                results = {
                    'operation_type': operation_type,
                    'memory_ids': memory_ids,
                    'success_count': 0,
                    'error_count': 0,
                    'errors': []
                }
                
                placeholders = ','.join(['?' for _ in memory_ids])
                
                if operation_type == 'delete':
                    try:
                        cursor.execute(f"DELETE FROM memories WHERE memory_id IN ({placeholders})", memory_ids)
                        results['success_count'] = cursor.rowcount
                    except Exception as e:
                        results['error_count'] = len(memory_ids)
                        results['errors'].append(str(e))
                
                elif operation_type == 'archive':
                    # Add metadata to mark as archived
                    try:
                        cursor.execute(f"""
                            UPDATE memories 
                            SET metadata = json_set(COALESCE(metadata, '{{}}'), '$.archived', 'true', '$.archived_at', ?)
                            WHERE memory_id IN ({placeholders})
                        """, [datetime.now().isoformat()] + memory_ids)
                        results['success_count'] = cursor.rowcount
                    except Exception as e:
                        results['error_count'] = len(memory_ids)
                        results['errors'].append(str(e))
                
                elif operation_type == 'merge':
                    # For merge operations, keep the first memory and delete others
                    if len(memory_ids) > 1:
                        try:
                            # Keep the first memory, delete the rest
                            cursor.execute(f"DELETE FROM memories WHERE memory_id IN ({','.join(['?' for _ in memory_ids[1:]])})", memory_ids[1:])
                            results['success_count'] = len(memory_ids) - 1
                            results['merged_into'] = memory_ids[0]
                        except Exception as e:
                            results['error_count'] = len(memory_ids)
                            results['errors'].append(str(e))
                
                # Commit changes
                conn.commit()
                conn.close()
                
                results['success'] = results['error_count'] == 0
                results['message'] = f"Operation completed: {results['success_count']} succeeded, {results['error_count']} failed"
                
                return JSONResponse(results)
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_memory_quality_bulk_preview(request):
            """Preview bulk operations before execution"""
            try:
                body = await request.json()
                operation_type = body.get('operation_type')
                memory_ids = body.get('memory_ids', [])
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Get memory details for preview
                placeholders = ','.join(['?' for _ in memory_ids])
                cursor.execute(f"""
                    SELECT memory_id, content, memory_type, importance, workflow_id
                    FROM memories 
                    WHERE memory_id IN ({placeholders})
                """, memory_ids)
                
                memories = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
                
                preview = {
                    'success': True,
                    'operation_type': operation_type,
                    'affected_memories': memories,
                    'total_affected': len(memories),
                    'preview_description': f'This will {operation_type} {len(memories)} memories'
                }
                
                if operation_type == 'merge' and len(memories) > 1:
                    preview['merge_target'] = memories[0]
                    preview['will_be_deleted'] = memories[1:]
                
                conn.close()
                
                return JSONResponse({'success': True, 'operation': preview})
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        # Workflow Scheduling API
        async def api_workflow_schedule(request):
            """Schedule workflow execution"""
            try:
                workflow_id = request.path_params['workflow_id']
                body = await request.json()
                
                # This is a placeholder implementation
                schedule_data = {
                    'workflow_id': workflow_id,
                    'scheduled_at': body.get('scheduled_at'),
                    'priority': body.get('priority', 5),
                    'status': 'scheduled',
                    'created_at': datetime.now().isoformat()
                }
                
                # In a real implementation, this would integrate with a task scheduler
                return JSONResponse({
                    'success': True,
                    'schedule_id': f"sched_{workflow_id}_{int(datetime.now().timestamp())}",
                    'message': 'Workflow scheduled successfully',
                    'schedule_data': schedule_data
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        # Cognitive State Restoration API
        async def api_cognitive_state_restore(request):
            """Restore a cognitive state"""
            try:
                state_id = request.path_params['state_id']
                body = await request.json()
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Get the state to restore
                cursor.execute("SELECT * FROM cognitive_timeline_states WHERE state_id = ?", (state_id,))
                state = cursor.fetchone()
                
                if not state:
                    return JSONResponse({"error": "Cognitive state not found"}, status_code=404)
                
                # Create a new state entry for the restoration
                restore_data = {
                    'state_id': state_id,
                    'restore_mode': body.get('restore_mode', 'full'),
                    'restored_at': datetime.now().isoformat(),
                    'original_timestamp': state[1] if state else None  # timestamp column
                }
                
                # In a real implementation, this would restore the actual cognitive state
                conn.close()
                
                return JSONResponse({
                    'success': True,
                    'message': 'Cognitive state restoration initiated',
                    'restore_data': restore_data
                })
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        # Artifact Download API
        async def api_artifact_download(request):
            """Download an artifact"""
            try:
                artifact_id = request.path_params['artifact_id']
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
                artifact = cursor.fetchone()
                
                if not artifact:
                    return JSONResponse({"error": "Artifact not found"}, status_code=404)
                
                # For now, return the artifact data as JSON
                # In a real implementation, this would serve the actual file
                artifact_dict = dict(zip([d[0] for d in cursor.description], artifact, strict=False))
                
                conn.close()
                
                # Return as downloadable JSON for now
                import json

                from starlette.responses import Response
                
                content = json.dumps(artifact_dict, indent=2)
                filename = f"{artifact_dict.get('name', 'artifact')}.json"
                
                return Response(
                    content=content,
                    media_type='application/json',
                    headers={
                        'Content-Disposition': f'attachment; filename="{filename}"'
                    }
                )
                
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        # Register the new endpoints
        app.add_route("/api/memory-quality/stats", api_memory_quality_stats, methods=["GET"])
        app.add_route("/api/memory-quality/quick-scan", api_memory_quality_quick_scan, methods=["GET"])
        app.add_route("/api/memory-quality/analyze", api_memory_quality_analyze, methods=["POST"])
        app.add_route("/api/memory-quality/duplicates", api_memory_quality_duplicates, methods=["GET"])
        app.add_route("/api/memory-quality/orphaned", api_memory_quality_orphaned, methods=["GET"])
        app.add_route("/api/memory-quality/bulk-execute", api_memory_quality_bulk_execute, methods=["POST"])
        app.add_route("/api/memory-quality/bulk-preview", api_memory_quality_bulk_preview, methods=["POST"])
        app.add_route("/api/workflows/{workflow_id}/schedule", api_workflow_schedule, methods=["POST"])
        app.add_route("/api/cognitive-states/{state_id}/restore", api_cognitive_state_restore, methods=["POST"])
        app.add_route("/api/artifacts/{artifact_id}/download", api_artifact_download, methods=["GET"])
        
        print("[DEBUG] Added comprehensive UMS Explorer API endpoints")
        
        # ===== ADD OPENAPI SCHEMA ENDPOINT FOR SWAGGER DOCS =====
        from starlette.responses import JSONResponse
        
        async def openapi_schema(request):
            """Generate comprehensive OpenAPI schema for the UMS Explorer API endpoints"""
            openapi_schema = {
                "openapi": "3.0.0",
                "info": {
                    "title": "Ultimate MCP Server - UMS Explorer API",
                    "version": "1.0.0",
                    "description": """
# Ultimate MCP Server - UMS Explorer API

The UMS Explorer API provides comprehensive access to the Unified Memory System (UMS) database for exploring cognitive states, actions, artifacts, and system performance metrics.

## Overview

This API is designed for:
- **Cognitive State Analysis**: Track and analyze the evolution of cognitive states over time
- **Action Monitoring**: Monitor running, queued, and completed actions with performance metrics
- **Artifact Management**: Explore and manage system artifacts and their relationships
- **System Analytics**: Get insights into tool usage, performance patterns, and system health

## Authentication

Currently, this API does not require authentication but is intended for internal system use.

## Rate Limiting

No rate limiting is currently implemented, but reasonable usage is expected.
                    """,
                    "contact": {
                        "name": "Ultimate MCP Server",
                        "url": "https://github.com/your-repo/ultimate-mcp-server"
                    }
                },
                "servers": [
                    {
                        "url": "http://localhost:8013",
                        "description": "Local development server"
                    }
                ],
                "tags": [
                    {
                        "name": "Cognitive States",
                        "description": "Endpoints for exploring and analyzing cognitive states and their evolution"
                    },
                    {
                        "name": "Action Monitor", 
                        "description": "Endpoints for monitoring action execution, queues, and performance metrics"
                    },
                    {
                        "name": "Artifacts",
                        "description": "Endpoints for managing artifacts and their relationships"
                    },
                    {
                        "name": "System Analytics",
                        "description": "Endpoints for system-wide analytics and performance insights"
                    },
                    {
                        "name": "Memory Quality",
                        "description": "Endpoints for analyzing and managing memory quality, including duplicates, orphaned memories, and quality metrics"
                    },
                    {
                        "name": "Working Memory",
                        "description": "Endpoints for managing working memory pool, focus modes, and memory optimization"
                    },
                    {
                        "name": "Workflow Management",
                        "description": "Endpoints for scheduling and managing workflow execution"
                    },
                    {
                        "name": "Performance Profiler",
                        "description": "Comprehensive workflow performance analysis with bottleneck identification, flame graphs, and optimization recommendations"
                    },
                    {
                        "name": "Health & Utilities",
                        "description": "System health checks and utility endpoints"
                    }
                ],
                "paths": {
                    "/api/cognitive-states": {
                        "get": {
                            "tags": ["Cognitive States"],
                            "summary": "List cognitive states with filtering",
                            "description": """
Retrieve a paginated list of cognitive states from the timeline with optional filtering capabilities.

This endpoint allows you to explore the cognitive state history with various filters:
- **Time Range**: Filter by start and end timestamps
- **State Type**: Filter by specific cognitive state types
- **Pagination**: Control the number of results and offset

**Features:**
- Enhanced metadata including complexity scores and change magnitudes
- Formatted timestamps for easier consumption
- Associated workflow information
- Memory and action counts for context
                            """,
                            "parameters": [
                                {
                                    "name": "start_time", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "number"}, 
                                    "description": "Unix timestamp to filter states created after this time",
                                    "example": 1703980800
                                },
                                {
                                    "name": "end_time", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "number"}, 
                                    "description": "Unix timestamp to filter states created before this time",
                                    "example": 1704067200
                                },
                                {
                                    "name": "limit", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 100, "minimum": 1, "maximum": 1000}, 
                                    "description": "Maximum number of states to return",
                                    "example": 50
                                },
                                {
                                    "name": "offset", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 0, "minimum": 0}, 
                                    "description": "Number of states to skip for pagination",
                                    "example": 0
                                },
                                {
                                    "name": "pattern_type", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "string"}, 
                                    "description": "Filter by specific cognitive state type",
                                    "example": "decision_point"
                                }
                            ],
                            "responses": {
                                "200": {
                                    "description": "Successfully retrieved cognitive states",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "states": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "state_id": {"type": "string"},
                                                                "timestamp": {"type": "number"},
                                                                "formatted_timestamp": {"type": "string"},
                                                                "state_type": {"type": "string"},
                                                                "description": {"type": "string"},
                                                                "workflow_id": {"type": "string"},
                                                                "workflow_title": {"type": "string"},
                                                                "complexity_score": {"type": "number"},
                                                                "change_magnitude": {"type": "number"},
                                                                "age_minutes": {"type": "number"},
                                                                "memory_count": {"type": "integer"},
                                                                "action_count": {"type": "integer"}
                                                            }
                                                        }
                                                    },
                                                    "total": {"type": "integer"},
                                                    "has_more": {"type": "boolean"}
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/cognitive-states/timeline": {
                        "get": {
                            "tags": ["Cognitive States"],
                            "summary": "Get cognitive state timeline for visualization",
                            "description": """
Retrieve cognitive state timeline data optimized for visualization and temporal analysis.

This endpoint provides:
- **Chronological ordering** of cognitive states
- **Complexity scoring** for each state
- **Change magnitude calculation** between consecutive states
- **Sequence numbering** for timeline positioning

Perfect for building timeline visualizations, trend analysis, and understanding cognitive state evolution patterns.
                            """,
                            "parameters": [
                                {
                                    "name": "hours", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 24, "minimum": 1, "maximum": 168}, 
                                    "description": "Number of hours back to retrieve timeline data",
                                    "example": 24
                                },
                                {
                                    "name": "granularity", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {
                                        "type": "string", 
                                        "enum": ["second", "minute", "hour"], 
                                        "default": "hour"
                                    }, 
                                    "description": "Timeline granularity for data aggregation",
                                    "example": "hour"
                                }
                            ],
                            "responses": {
                                "200": {
                                    "description": "Timeline data successfully retrieved"
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/cognitive-states/{state_id}": {
                        "get": {
                            "tags": ["Cognitive States"],
                            "summary": "Get detailed cognitive state information",
                            "description": """
Retrieve comprehensive details about a specific cognitive state including:

- **Full state data** with parsed JSON content
- **Associated memories** linked to the same workflow
- **Related actions** that occurred in the same context
- **Workflow information** for broader context
- **Complexity analysis** and formatted timestamps

This endpoint is ideal for deep-dive analysis of specific cognitive states.
                            """,
                            "parameters": [
                                {
                                    "name": "state_id", 
                                    "in": "path", 
                                    "required": True, 
                                    "schema": {"type": "string"}, 
                                    "description": "Unique identifier of the cognitive state",
                                    "example": "state_abc123xyz789"
                                }
                            ],
                            "responses": {
                                "200": {"description": "Detailed cognitive state information"},
                                "404": {"description": "Cognitive state not found"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/cognitive-states/patterns": {
                        "get": {
                            "tags": ["Cognitive States"],
                            "summary": "Analyze cognitive patterns and anomalies",
                            "description": """
Perform advanced pattern analysis on cognitive states to identify:

- **Recurring patterns** in state sequences
- **State transitions** and their frequencies
- **Anomalous states** that deviate from normal patterns
- **Pattern similarity analysis** with configurable thresholds

This endpoint uses sophisticated algorithms to detect meaningful patterns in cognitive state evolution.
                            """,
                            "parameters": [
                                {
                                    "name": "lookback_hours", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 24, "minimum": 1, "maximum": 720}, 
                                    "description": "Hours to look back for pattern analysis",
                                    "example": 48
                                },
                                {
                                    "name": "min_pattern_length", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 3, "minimum": 2, "maximum": 20}, 
                                    "description": "Minimum length of patterns to detect",
                                    "example": 3
                                },
                                {
                                    "name": "similarity_threshold", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "number", "default": 0.7, "minimum": 0.1, "maximum": 1.0}, 
                                    "description": "Similarity threshold for pattern matching (0.0-1.0)",
                                    "example": 0.8
                                }
                            ],
                            "responses": {
                                "200": {"description": "Pattern analysis results with transitions and anomalies"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/cognitive-states/compare": {
                        "post": {
                            "tags": ["Cognitive States"],
                            "summary": "Compare two cognitive states",
                            "description": """
Perform detailed comparison between two cognitive states to understand:

- **Structural differences** in state data
- **Added, removed, and modified** components
- **Change magnitude** calculation
- **Time differential** between states

Perfect for understanding how cognitive states evolve and what changes between specific points in time.
                            """,
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "required": ["state_id_1", "state_id_2"],
                                            "properties": {
                                                "state_id_1": {
                                                    "type": "string",
                                                    "description": "First cognitive state ID for comparison"
                                                },
                                                "state_id_2": {
                                                    "type": "string", 
                                                    "description": "Second cognitive state ID for comparison"
                                                }
                                            }
                                        },
                                        "example": {
                                            "state_id_1": "state_abc123",
                                            "state_id_2": "state_xyz789"
                                        }
                                    }
                                }
                            },
                            "responses": {
                                "200": {"description": "Detailed comparison results"},
                                "400": {"description": "Invalid request - both state IDs required"},
                                "404": {"description": "One or both states not found"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/actions/running": {
                        "get": {
                            "tags": ["Action Monitor"],
                            "summary": "Get currently executing actions",
                            "description": """
Monitor actions that are currently executing with real-time status information:

- **Execution progress** with percentage completion estimates
- **Performance categorization** (excellent, good, slow, etc.)
- **Resource usage indicators** (placeholder for future implementation)
- **Status indicators** with urgency levels
- **Estimated duration** vs actual execution time

Ideal for monitoring system activity and identifying long-running or problematic actions.
                            """,
                            "responses": {
                                "200": {
                                    "description": "List of currently running actions with real-time metrics"
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/actions/queue": {
                        "get": {
                            "tags": ["Action Monitor"],
                            "summary": "Get queued actions waiting for execution",
                            "description": """
Monitor the action execution queue to understand:

- **Queue position** for each waiting action
- **Priority levels** with human-readable labels
- **Estimated wait times** based on queue position
- **Queue time** (how long actions have been waiting)

Essential for understanding system load and execution priorities.
                            """,
                            "responses": {
                                "200": {"description": "List of queued actions with wait time estimates"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/actions/history": {
                        "get": {
                            "tags": ["Action Monitor"],
                            "summary": "Get completed actions with performance metrics",
                            "description": """
Analyze historical action execution data with comprehensive performance metrics:

- **Execution duration** and performance scoring
- **Success/failure rates** and efficiency ratings
- **Tool-specific filtering** and status filtering
- **Aggregate performance metrics** and trends

Perfect for performance analysis, debugging, and system optimization.
                            """,
                            "parameters": [
                                {
                                    "name": "limit", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500}, 
                                    "description": "Maximum number of actions to return"
                                },
                                {
                                    "name": "offset", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 0, "minimum": 0}, 
                                    "description": "Number of actions to skip for pagination"
                                },
                                {
                                    "name": "status_filter", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {
                                        "type": "string",
                                        "enum": ["completed", "failed", "cancelled", "timeout"]
                                    }, 
                                    "description": "Filter by action completion status"
                                },
                                {
                                    "name": "tool_filter", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "string"}, 
                                    "description": "Filter by specific tool name"
                                },
                                {
                                    "name": "hours_back", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 24, "minimum": 1, "maximum": 720}, 
                                    "description": "Hours back to search for completed actions"
                                }
                            ],
                            "responses": {
                                "200": {"description": "Historical actions with performance analysis"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/actions/metrics": {
                        "get": {
                            "tags": ["Action Monitor"],
                            "summary": "Get comprehensive action execution metrics",
                            "description": """
Retrieve system-wide action execution analytics including:

- **Overall success/failure rates** for the past 24 hours
- **Tool usage statistics** with performance breakdowns
- **Hourly performance distribution** showing usage patterns
- **Performance insights** with actionable recommendations

This endpoint provides executive-level insights into system performance and health.
                            """,
                            "responses": {
                                "200": {"description": "Comprehensive action execution metrics and analytics"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/tools/usage": {
                        "get": {
                            "tags": ["System Analytics"],
                            "summary": "Get detailed tool usage statistics",
                            "description": """
Analyze tool usage patterns and performance with detailed breakdown:

- **Usage frequency** and success rates per tool
- **Performance categorization** (fast, normal, slow)
- **Reliability scoring** based on volume and success rate
- **Execution time statistics** (min, max, average)
- **Usage trends** and workflow distribution

Essential for understanding which tools are most/least reliable and performant.
                            """,
                            "parameters": [
                                {
                                    "name": "hours_back", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 24, "minimum": 1, "maximum": 720}, 
                                    "description": "Hours back to analyze tool usage"
                                },
                                {
                                    "name": "include_performance", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "boolean", "default": True}, 
                                    "description": "Include detailed performance breakdown"
                                }
                            ],
                            "responses": {
                                "200": {"description": "Detailed tool usage statistics with performance metrics"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/artifacts": {
                        "get": {
                            "tags": ["Artifacts"],
                            "summary": "List artifacts with filtering and search",
                            "description": """
Explore system artifacts with comprehensive filtering and search capabilities:

- **Type-based filtering** for specific artifact categories
- **Workflow association** to see artifacts by workflow
- **Tag-based search** for categorized artifacts
- **Full-text search** across names and descriptions
- **Sorting options** with configurable order

Includes relationship counts, version information, and human-readable metadata.
                            """,
                            "parameters": [
                                {
                                    "name": "artifact_type", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "string"}, 
                                    "description": "Filter by specific artifact type",
                                    "example": "document"
                                },
                                {
                                    "name": "workflow_id", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "string"}, 
                                    "description": "Filter by workflow ID",
                                    "example": "workflow_abc123"
                                },
                                {
                                    "name": "tags", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "string"}, 
                                    "description": "Search within artifact tags",
                                    "example": "important"
                                },
                                {
                                    "name": "search", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "string"}, 
                                    "description": "Full-text search in names and descriptions",
                                    "example": "analysis report"
                                },
                                {
                                    "name": "sort_by", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {
                                        "type": "string", 
                                        "enum": ["created_at", "updated_at", "name", "importance", "access_count"], 
                                        "default": "created_at"
                                    }, 
                                    "description": "Field to sort results by"
                                },
                                {
                                    "name": "sort_order", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {
                                        "type": "string", 
                                        "enum": ["asc", "desc"], 
                                        "default": "desc"
                                    }, 
                                    "description": "Sort order direction"
                                },
                                {
                                    "name": "limit", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200}, 
                                    "description": "Maximum number of artifacts to return"
                                },
                                {
                                    "name": "offset", 
                                    "in": "query", 
                                    "required": False,
                                    "schema": {"type": "integer", "default": 0, "minimum": 0}, 
                                    "description": "Number of artifacts to skip for pagination"
                                }
                            ],
                            "responses": {
                                "200": {"description": "List of artifacts with metadata and relationships"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/artifacts/stats": {
                        "get": {
                            "tags": ["Artifacts"],
                            "summary": "Get artifact statistics and analytics",
                            "description": """
Retrieve comprehensive statistics about system artifacts including:

- **Overall counts** and storage usage
- **Type-based breakdown** with metrics per artifact type
- **Importance scoring** averages and distributions
- **Access patterns** and usage statistics

Perfect for understanding artifact distribution and usage patterns across the system.
                            """,
                            "responses": {
                                "200": {"description": "Comprehensive artifact statistics and analytics"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/memory-quality/stats": {
                        "get": {
                            "tags": ["Memory Quality"],
                            "summary": "Get memory quality statistics and overview",
                            "description": """
Retrieve comprehensive memory quality statistics including:

- **Overall memory counts** and quality metrics
- **Quality score** based on identified issues
- **Issue breakdown** by type (duplicates, orphaned, stale)
- **Quality recommendations** for improvement

This endpoint provides a high-level overview of the memory system's health and quality.
                            """,
                            "responses": {
                                "200": {
                                    "description": "Memory quality statistics successfully retrieved",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "overall_stats": {
                                                        "type": "object",
                                                        "properties": {
                                                            "total_memories": {"type": "integer"},
                                                            "unique_workflows": {"type": "integer"},
                                                            "avg_importance": {"type": "number"},
                                                            "low_importance_count": {"type": "integer"},
                                                            "memory_types": {"type": "integer"}
                                                        }
                                                    },
                                                    "quality_issues": {
                                                        "type": "object",
                                                        "properties": {
                                                            "potential_duplicates": {"type": "integer"},
                                                            "orphaned_memories": {"type": "integer"},
                                                            "stale_memories": {"type": "integer"},
                                                            "low_importance_memories": {"type": "integer"}
                                                        }
                                                    },
                                                    "quality_score": {"type": "number"},
                                                    "recommendations": {
                                                        "type": "array",
                                                        "items": {"type": "string"}
                                                    },
                                                    "last_analysis": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/memory-quality/quick-scan": {
                        "get": {
                            "tags": ["Memory Quality"],
                            "summary": "Perform quick memory quality scan",
                            "description": """
Execute a fast scan of the memory system to identify immediate quality issues:

- **Duplicate detection** for identical content
- **Orphaned memory identification** 
- **Quality scoring** based on found issues
- **Issue categorization** by severity and type

This endpoint provides rapid feedback on memory quality without the overhead of a full analysis.
                            """,
                            "responses": {
                                "200": {
                                    "description": "Quick scan completed successfully",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "success": {"type": "boolean"},
                                                    "quick_metrics": {
                                                        "type": "object",
                                                        "properties": {
                                                            "overall_score": {"type": "number"},
                                                            "total_memories": {"type": "integer"},
                                                            "issues_found": {"type": "integer"},
                                                            "scan_time": {"type": "string"}
                                                        }
                                                    },
                                                    "issues": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "type": {"type": "string"},
                                                                "severity": {"type": "string"},
                                                                "description": {"type": "string"},
                                                                "preview": {"type": "string"}
                                                            }
                                                        }
                                                    },
                                                    "scan_completed": {"type": "boolean"},
                                                    "next_scan_recommended": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/memory-quality/analyze": {
                        "post": {
                            "tags": ["Memory Quality"],
                            "summary": "Perform comprehensive memory quality analysis",
                            "description": """
Execute a detailed analysis of memory quality with configurable options:

- **Comprehensive issue detection** across all categories
- **Detailed issue metadata** with auto-fix recommendations
- **Quality scoring** and trend analysis
- **Actionable recommendations** for quality improvement

This endpoint provides deep insights into memory quality and specific actionable issues.
                            """,
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "analysis_type": {
                                                    "type": "string",
                                                    "enum": ["comprehensive", "duplicates", "orphaned"],
                                                    "default": "comprehensive",
                                                    "description": "Type of analysis to perform"
                                                }
                                            }
                                        },
                                        "example": {"analysis_type": "comprehensive"}
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Analysis completed successfully",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "success": {"type": "boolean"},
                                                    "analysis": {
                                                        "type": "object",
                                                        "properties": {
                                                            "analysis_type": {"type": "string"},
                                                            "issues_found": {"type": "integer"},
                                                            "overall_score": {"type": "number"},
                                                            "total_memories": {"type": "integer"},
                                                            "issues": {
                                                                "type": "array",
                                                                "items": {
                                                                    "type": "object",
                                                                    "properties": {
                                                                        "issue_id": {"type": "string"},
                                                                        "issue_type": {"type": "string"},
                                                                        "title": {"type": "string"},
                                                                        "description": {"type": "string"},
                                                                        "severity": {"type": "string"},
                                                                        "auto_fixable": {"type": "boolean"},
                                                                        "recommendation": {"type": "string"}
                                                                    }
                                                                }
                                                            },
                                                            "recommendations": {
                                                                "type": "array",
                                                                "items": {"type": "string"}
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/memory-quality/duplicates": {
                        "get": {
                            "tags": ["Memory Quality"],
                            "summary": "Get detailed duplicate memory analysis",
                            "description": """
Retrieve comprehensive information about duplicate memories:

- **Duplicate clusters** with identical content
- **Memory details** for each duplicate group
- **Merge recommendations** based on duplicate count
- **Temporal analysis** of when duplicates were created

Essential for understanding and resolving memory duplication issues.
                            """,
                            "responses": {
                                "200": {
                                    "description": "Duplicate analysis successfully retrieved",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "success": {"type": "boolean"},
                                                    "clusters": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "cluster_id": {"type": "string"},
                                                                "duplicate_count": {"type": "integer"},
                                                                "memory_ids": {
                                                                    "type": "array",
                                                                    "items": {"type": "string"}
                                                                },
                                                                "primary_memory_id": {"type": "string"},
                                                                "content_preview": {"type": "string"},
                                                                "recommendation": {"type": "string"}
                                                            }
                                                        }
                                                    },
                                                    "total_groups": {"type": "integer"},
                                                    "total_duplicates": {"type": "integer"}
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/memory-quality/orphaned": {
                        "get": {
                            "tags": ["Memory Quality"],
                            "summary": "Get orphaned memories not associated with workflows",
                            "description": """
Retrieve memories that are not associated with any workflow:

- **Orphaned memory details** including content and metadata
- **Creation timestamps** for temporal analysis
- **Importance scoring** to prioritize action
- **Assignment recommendations** for workflow integration

Critical for maintaining memory system organization and preventing data loss.
                            """,
                            "responses": {
                                "200": {
                                    "description": "Orphaned memories successfully retrieved",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "success": {"type": "boolean"},
                                                    "orphaned_memories": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "memory_id": {"type": "string"},
                                                                "content": {"type": "string"},
                                                                "memory_type": {"type": "string"},
                                                                "importance": {"type": "number"},
                                                                "created_at": {"type": "number"}
                                                            }
                                                        }
                                                    },
                                                    "total_orphaned": {"type": "integer"},
                                                    "recommendation": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/memory-quality/bulk-execute": {
                        "post": {
                            "tags": ["Memory Quality"],
                            "summary": "Execute bulk operations on memories",
                            "description": """
Perform bulk operations on multiple memories:

- **Merge operations** for duplicate consolidation
- **Archive operations** for stale memory management
- **Delete operations** for cleanup
- **Progress tracking** and error reporting

Enables efficient bulk management of memory quality issues.
                            """,
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "required": ["operation_type", "memory_ids"],
                                            "properties": {
                                                "operation_type": {
                                                    "type": "string",
                                                    "enum": ["merge", "archive", "delete"],
                                                    "description": "Type of bulk operation to perform"
                                                },
                                                "memory_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "description": "List of memory IDs to operate on"
                                                },
                                                "target_memory_id": {
                                                    "type": "string",
                                                    "description": "Target memory ID for merge operations"
                                                }
                                            }
                                        },
                                        "example": {
                                            "operation_type": "merge",
                                            "memory_ids": ["mem_123", "mem_456", "mem_789"],
                                            "target_memory_id": "mem_123"
                                        }
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Bulk operation completed",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "success": {"type": "boolean"},
                                                    "operation_type": {"type": "string"},
                                                    "success_count": {"type": "integer"},
                                                    "error_count": {"type": "integer"},
                                                    "message": {"type": "string"},
                                                    "errors": {
                                                        "type": "array",
                                                        "items": {"type": "string"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "400": {"description": "Invalid request parameters"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/memory-quality/bulk-preview": {
                        "post": {
                            "tags": ["Memory Quality"],
                            "summary": "Preview bulk operations before execution",
                            "description": """
Preview the effects of bulk operations before executing them:

- **Operation impact preview** with affected memories
- **Risk assessment** for destructive operations
- **Merge target selection** for duplicate operations
- **Cost estimation** for large operations

Essential for safe bulk operations and preventing accidental data loss.
                            """,
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "required": ["operation_type", "memory_ids"],
                                            "properties": {
                                                "operation_type": {
                                                    "type": "string",
                                                    "enum": ["merge", "archive", "delete"],
                                                    "description": "Type of bulk operation to preview"
                                                },
                                                "memory_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "description": "List of memory IDs to preview operation for"
                                                }
                                            }
                                        },
                                        "example": {
                                            "operation_type": "merge",
                                            "memory_ids": ["mem_123", "mem_456", "mem_789"]
                                        }
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Preview generated successfully",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "success": {"type": "boolean"},
                                                    "operation": {
                                                        "type": "object",
                                                        "properties": {
                                                            "operation_type": {"type": "string"},
                                                            "total_affected": {"type": "integer"},
                                                            "preview_description": {"type": "string"},
                                                            "affected_memories": {
                                                                "type": "array",
                                                                "items": {
                                                                    "type": "object",
                                                                    "properties": {
                                                                        "memory_id": {"type": "string"},
                                                                        "content": {"type": "string"},
                                                                        "importance": {"type": "number"}
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "400": {"description": "Invalid request parameters"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/workflows/{workflow_id}/schedule": {
                        "post": {
                            "tags": ["Workflow Management"],
                            "summary": "Schedule workflow execution",
                            "description": """
Schedule a workflow for future execution with configurable priority and timing:

- **Workflow scheduling** with specific timing
- **Priority management** for execution order
- **Status tracking** for scheduled workflows
- **Integration** with workflow execution system

Essential for orchestrating complex multi-step processes and time-based automation.
                            """,
                            "parameters": [
                                {
                                    "name": "workflow_id",
                                    "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"},
                                    "description": "Unique identifier of the workflow to schedule",
                                    "example": "workflow_abc123"
                                }
                            ],
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "scheduled_at": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                    "description": "ISO timestamp for when to execute the workflow"
                                                },
                                                "priority": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                    "maximum": 10,
                                                    "default": 5,
                                                    "description": "Execution priority (1=highest, 10=lowest)"
                                                }
                                            }
                                        },
                                        "example": {
                                            "scheduled_at": "2024-01-01T12:00:00Z",
                                            "priority": 3
                                        }
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Workflow scheduled successfully",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "success": {"type": "boolean"},
                                                    "schedule_id": {"type": "string"},
                                                    "message": {"type": "string"},
                                                    "schedule_data": {
                                                        "type": "object",
                                                        "properties": {
                                                            "workflow_id": {"type": "string"},
                                                            "scheduled_at": {"type": "string"},
                                                            "priority": {"type": "integer"},
                                                            "status": {"type": "string"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "400": {"description": "Invalid request parameters"},
                                "404": {"description": "Workflow not found"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/cognitive-states/{state_id}/restore": {
                        "post": {
                            "tags": ["Cognitive States"],
                            "summary": "Restore a previous cognitive state",
                            "description": """
Restore the system to a previous cognitive state for analysis or recovery:

- **State restoration** with configurable restore modes
- **Temporal analysis** by reverting to specific points in time
- **Recovery mechanisms** for problematic state transitions
- **Research capabilities** for understanding state evolution

Critical for debugging cognitive state issues and temporal analysis of system behavior.
                            """,
                            "parameters": [
                                {
                                    "name": "state_id",
                                    "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"},
                                    "description": "Unique identifier of the cognitive state to restore",
                                    "example": "state_abc123xyz789"
                                }
                            ],
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "restore_mode": {
                                                    "type": "string",
                                                    "enum": ["full", "partial", "snapshot"],
                                                    "default": "full",
                                                    "description": "Type of restoration to perform"
                                                }
                                            }
                                        },
                                        "example": {"restore_mode": "full"}
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Cognitive state restoration initiated",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "success": {"type": "boolean"},
                                                    "message": {"type": "string"},
                                                    "restore_data": {
                                                        "type": "object",
                                                        "properties": {
                                                            "state_id": {"type": "string"},
                                                            "restore_mode": {"type": "string"},
                                                            "restored_at": {"type": "string"},
                                                            "original_timestamp": {"type": "number"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "400": {"description": "Invalid request parameters"},
                                "404": {"description": "Cognitive state not found"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/artifacts/{artifact_id}/download": {
                        "get": {
                            "tags": ["Artifacts"],
                            "summary": "Download artifact file or data",
                            "description": """
Download the raw file or data associated with an artifact:

- **File download** with proper content types
- **Metadata preservation** in download headers
- **Access logging** for audit trails
- **Format handling** for different artifact types

Essential for accessing artifact content outside the UMS Explorer interface.
                            """,
                            "parameters": [
                                {
                                    "name": "artifact_id",
                                    "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"},
                                    "description": "Unique identifier of the artifact to download",
                                    "example": "artifact_abc123"
                                }
                            ],
                            "responses": {
                                "200": {
                                    "description": "Artifact file downloaded successfully",
                                    "content": {
                                        "application/octet-stream": {
                                            "schema": {
                                                "type": "string",
                                                "format": "binary"
                                            }
                                        },
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "description": "JSON representation of artifact data"
                                            }
                                        }
                                    }
                                },
                                "404": {"description": "Artifact not found"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/health": {
                        "get": {
                            "tags": ["Health & Utilities"],
                            "summary": "Health check endpoint",
                            "description": """
Check the health and operational status of the Ultimate MCP Server:

- **Server status** verification
- **Service availability** confirmation
- **Version information** for compatibility checks
- **Load balancer integration** support

Standard health check endpoint for monitoring systems and operational dashboards.
                            """,
                            "responses": {
                                "200": {
                                    "description": "Server is healthy and operational",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "status": {
                                                        "type": "string",
                                                        "enum": ["ok"],
                                                        "description": "Health status indicator"
                                                    },
                                                    "version": {
                                                        "type": "string",
                                                        "description": "Server version string"
                                                    }
                                                }
                                            },
                                            "example": {
                                                "status": "ok",
                                                "version": "0.1.0"
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Server health check failed"}
                            }
                        }
                    },
                    "/api/performance/overview": {
                        "get": {
                            "tags": ["Performance Profiler"],
                            "summary": "Get comprehensive performance overview with metrics and trends",
                            "description": """
Retrieve comprehensive workflow performance overview including:

- **Real-time performance metrics** with execution time analysis
- **Timeline visualization data** with configurable granularity
- **Tool utilization statistics** and performance breakdowns
- **Current bottlenecks** identification with severity indicators
- **Throughput analysis** and success rate metrics

Perfect for monitoring overall system performance and identifying optimization opportunities.
                            """,
                            "parameters": [
                                {
                                    "name": "hours_back",
                                    "in": "query",
                                    "required": False,
                                    "schema": {"type": "integer", "default": 24, "minimum": 1, "maximum": 720},
                                    "description": "Number of hours back to analyze performance data",
                                    "example": 24
                                },
                                {
                                    "name": "granularity",
                                    "in": "query",
                                    "required": False,
                                    "schema": {
                                        "type": "string",
                                        "enum": ["minute", "hour", "day"],
                                        "default": "hour"
                                    },
                                    "description": "Time granularity for timeline data aggregation",
                                    "example": "hour"
                                }
                            ],
                            "responses": {
                                "200": {
                                    "description": "Performance overview data with metrics and timeline",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "overview": {
                                                        "type": "object",
                                                        "properties": {
                                                            "avg_execution_time": {"type": "number"},
                                                            "throughput_per_hour": {"type": "number"},
                                                            "success_rate_percentage": {"type": "number"},
                                                            "active_workflows": {"type": "integer"},
                                                            "total_actions": {"type": "integer"}
                                                        }
                                                    },
                                                    "timeline": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "time_bucket": {"type": "string"},
                                                                "action_count": {"type": "integer"},
                                                                "avg_duration": {"type": "number"},
                                                                "successful_count": {"type": "integer"},
                                                                "failed_count": {"type": "integer"}
                                                            }
                                                        }
                                                    },
                                                    "tool_utilization": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "tool_name": {"type": "string"},
                                                                "usage_count": {"type": "integer"},
                                                                "avg_duration": {"type": "number"},
                                                                "success_count": {"type": "integer"}
                                                            }
                                                        }
                                                    },
                                                    "bottlenecks": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "tool_name": {"type": "string"},
                                                                "duration": {"type": "number"},
                                                                "action_id": {"type": "string"},
                                                                "status": {"type": "string"}
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/performance/bottlenecks": {
                        "get": {
                            "tags": ["Performance Profiler"],
                            "summary": "Identify and analyze performance bottlenecks with detailed insights",
                            "description": """
Perform comprehensive bottleneck analysis including:

- **Tool performance analysis** with percentile breakdowns (P95, P99)
- **Workflow efficiency scoring** and parallelization opportunities
- **Resource contention detection** and conflict analysis
- **Optimization recommendations** with impact estimates
- **Critical path identification** for workflow optimization

Advanced algorithms identify bottlenecks using statistical analysis and provide actionable insights.
                            """,
                            "parameters": [
                                {
                                    "name": "hours_back",
                                    "in": "query",
                                    "required": False,
                                    "schema": {"type": "integer", "default": 24, "minimum": 1, "maximum": 720},
                                    "description": "Hours back to analyze for bottlenecks"
                                },
                                {
                                    "name": "min_duration",
                                    "in": "query",
                                    "required": False,
                                    "schema": {"type": "number", "default": 1.0, "minimum": 0.1},
                                    "description": "Minimum execution duration (seconds) to consider as potential bottleneck"
                                }
                            ],
                            "responses": {
                                "200": {
                                    "description": "Comprehensive bottleneck analysis with optimization opportunities",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "tool_bottlenecks": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "tool_name": {"type": "string"},
                                                                "avg_duration": {"type": "number"},
                                                                "p95_duration": {"type": "number"},
                                                                "p99_duration": {"type": "number"},
                                                                "total_calls": {"type": "integer"},
                                                                "failure_count": {"type": "integer"}
                                                            }
                                                        }
                                                    },
                                                    "workflow_bottlenecks": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "workflow_id": {"type": "string"},
                                                                "total_workflow_time": {"type": "number"},
                                                                "total_elapsed_time": {"type": "number"},
                                                                "action_count": {"type": "integer"}
                                                            }
                                                        }
                                                    },
                                                    "parallelization_opportunities": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "workflow_id": {"type": "string"},
                                                                "potential_time_savings": {"type": "number"},
                                                                "parallelization_efficiency": {"type": "number"},
                                                                "optimization_score": {"type": "number"}
                                                            }
                                                        }
                                                    },
                                                    "recommendations": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "type": {"type": "string"},
                                                                "priority": {"type": "string"},
                                                                "title": {"type": "string"},
                                                                "description": {"type": "string"},
                                                                "impact": {"type": "string"}
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/performance/flame-graph": {
                        "get": {
                            "tags": ["Performance Profiler"],
                            "summary": "Generate flame graph data for workflow performance visualization",
                            "description": """
Generate hierarchical flame graph data for detailed workflow performance analysis:

- **Interactive flame graph structure** showing execution hierarchy
- **Critical path analysis** highlighting the longest dependency chain
- **Tool-level performance breakdown** with execution times
- **Parallelization efficiency metrics** and optimization scores
- **Execution timeline analysis** with CPU vs wall-clock time

Industry-standard flame graph visualization for profiling workflow execution patterns.
                            """,
                            "parameters": [
                                {
                                    "name": "workflow_id",
                                    "in": "query",
                                    "required": True,
                                    "schema": {"type": "string"},
                                    "description": "Workflow ID to generate flame graph for",
                                    "example": "workflow_abc123"
                                },
                                {
                                    "name": "hours_back",
                                    "in": "query",
                                    "required": False,
                                    "schema": {"type": "integer", "default": 24, "minimum": 1, "maximum": 720},
                                    "description": "Hours back to search for workflow execution data"
                                }
                            ],
                            "responses": {
                                "200": {
                                    "description": "Flame graph data with performance metrics and critical path",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "flame_graph": {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {"type": "string"},
                                                            "value": {"type": "number"},
                                                            "children": {
                                                                "type": "array",
                                                                "items": {"type": "object"}
                                                            }
                                                        }
                                                    },
                                                    "metrics": {
                                                        "type": "object",
                                                        "properties": {
                                                            "total_actions": {"type": "integer"},
                                                            "total_cpu_time": {"type": "number"},
                                                            "wall_clock_time": {"type": "number"},
                                                            "parallelization_efficiency": {"type": "number"}
                                                        }
                                                    },
                                                    "critical_path": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "action_id": {"type": "string"},
                                                                "tool_name": {"type": "string"},
                                                                "duration": {"type": "number"}
                                                            }
                                                        }
                                                    },
                                                    "analysis": {
                                                        "type": "object",
                                                        "properties": {
                                                            "bottleneck_tool": {"type": "string"},
                                                            "parallelization_potential": {"type": "number"},
                                                            "optimization_score": {"type": "number"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "400": {"description": "Missing required workflow_id parameter"},
                                "404": {"description": "No actions found for specified workflow"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/performance/trends": {
                        "get": {
                            "tags": ["Performance Profiler"],
                            "summary": "Analyze performance trends and patterns over time",
                            "description": """
Comprehensive trend analysis for long-term performance monitoring:

- **Daily performance trends** with configurable time periods
- **Pattern detection algorithms** identifying weekly patterns and anomalies
- **Trend classification** (improving, degrading, stable) with confidence scores
- **Performance insights** with contextual explanations
- **Comparative analysis** showing best/worst performing periods

Advanced analytics help identify performance degradation and optimization opportunities over time.
                            """,
                            "parameters": [
                                {
                                    "name": "days_back",
                                    "in": "query",
                                    "required": False,
                                    "schema": {"type": "integer", "default": 7, "minimum": 1, "maximum": 90},
                                    "description": "Number of days back to analyze trends"
                                },
                                {
                                    "name": "metric",
                                    "in": "query",
                                    "required": False,
                                    "schema": {
                                        "type": "string",
                                        "enum": ["duration", "success_rate", "throughput"],
                                        "default": "duration"
                                    },
                                    "description": "Primary metric to analyze for trends"
                                }
                            ],
                            "responses": {
                                "200": {
                                    "description": "Performance trends with pattern analysis and insights",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "daily_trends": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "date": {"type": "string"},
                                                                "avg_duration": {"type": "number"},
                                                                "success_rate": {"type": "number"},
                                                                "throughput": {"type": "number"},
                                                                "action_count": {"type": "integer"}
                                                            }
                                                        }
                                                    },
                                                    "trend_analysis": {
                                                        "type": "object",
                                                        "properties": {
                                                            "performance_trend": {
                                                                "type": "string",
                                                                "enum": ["improving", "degrading", "stable", "insufficient_data"]
                                                            },
                                                            "success_trend": {
                                                                "type": "string",
                                                                "enum": ["improving", "degrading", "stable", "insufficient_data"]
                                                            },
                                                            "data_points": {"type": "integer"}
                                                        }
                                                    },
                                                    "patterns": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "type": {"type": "string"},
                                                                "description": {"type": "string"},
                                                                "impact": {"type": "string"},
                                                                "recommendation": {"type": "string"}
                                                            }
                                                        }
                                                    },
                                                    "insights": {
                                                        "type": "object",
                                                        "properties": {
                                                            "best_performing_day": {"type": "object"},
                                                            "worst_performing_day": {"type": "object"},
                                                            "avg_daily_actions": {"type": "number"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/performance/recommendations": {
                        "get": {
                            "tags": ["Performance Profiler"],
                            "summary": "Generate actionable performance optimization recommendations",
                            "description": """
AI-powered optimization recommendations engine providing:

- **Prioritized recommendations** with impact and effort scoring
- **Implementation roadmaps** categorized by complexity and impact
- **Detailed implementation steps** with prerequisites and metrics
- **Cost-benefit analysis** with quantified impact estimates
- **Progress tracking guidance** with success metrics

Smart recommendation system analyzes performance data to provide actionable optimization strategies.
                            """,
                            "parameters": [
                                {
                                    "name": "hours_back",
                                    "in": "query",
                                    "required": False,
                                    "schema": {"type": "integer", "default": 24, "minimum": 1, "maximum": 720},
                                    "description": "Hours back to analyze for recommendations"
                                },
                                {
                                    "name": "priority",
                                    "in": "query",
                                    "required": False,
                                    "schema": {
                                        "type": "string",
                                        "enum": ["all", "high", "medium", "low"],
                                        "default": "all"
                                    },
                                    "description": "Filter recommendations by priority level"
                                }
                            ],
                            "responses": {
                                "200": {
                                    "description": "Comprehensive optimization recommendations with implementation guidance",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "recommendations": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "id": {"type": "string"},
                                                                "type": {"type": "string"},
                                                                "priority": {"type": "string"},
                                                                "title": {"type": "string"},
                                                                "description": {"type": "string"},
                                                                "impact_estimate": {
                                                                    "type": "object",
                                                                    "properties": {
                                                                        "time_savings_potential": {"type": "number"},
                                                                        "affected_actions": {"type": "integer"},
                                                                        "cost_benefit_ratio": {"type": "number"}
                                                                    }
                                                                },
                                                                "implementation_steps": {
                                                                    "type": "array",
                                                                    "items": {"type": "string"}
                                                                },
                                                                "estimated_effort": {"type": "string"},
                                                                "prerequisites": {
                                                                    "type": "array",
                                                                    "items": {"type": "string"}
                                                                },
                                                                "metrics_to_track": {
                                                                    "type": "array",
                                                                    "items": {"type": "string"}
                                                                }
                                                            }
                                                        }
                                                    },
                                                    "summary": {
                                                        "type": "object",
                                                        "properties": {
                                                            "total_recommendations": {"type": "integer"},
                                                            "high_priority": {"type": "integer"},
                                                            "medium_priority": {"type": "integer"},
                                                            "low_priority": {"type": "integer"},
                                                            "estimated_total_savings": {"type": "number"}
                                                        }
                                                    },
                                                    "implementation_roadmap": {
                                                        "type": "object",
                                                        "properties": {
                                                            "quick_wins": {
                                                                "type": "array",
                                                                "items": {"type": "object"}
                                                            },
                                                            "major_improvements": {
                                                                "type": "array",
                                                                "items": {"type": "object"}
                                                            },
                                                            "maintenance_tasks": {
                                                                "type": "array",
                                                                "items": {"type": "object"}
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/working-memory/status": {
                        "get": {
                            "tags": ["Working Memory"],
                            "summary": "Get working memory system status",
                            "description": """
Retrieve the current status and configuration of the working memory system:

- **Pool utilization** and capacity metrics
- **Focus mode** status and configuration
- **Optimization statistics** and performance data
- **Memory distribution** across different categories

Essential for monitoring working memory health and performance optimization.
                            """,
                            "responses": {
                                "200": {
                                    "description": "Working memory status retrieved successfully",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "initialized": {"type": "boolean"},
                                                    "total_capacity": {"type": "integer"},
                                                    "current_size": {"type": "integer"},
                                                    "utilization_percentage": {"type": "number"},
                                                    "focus_mode": {
                                                        "type": "object",
                                                        "properties": {
                                                            "enabled": {"type": "boolean"},
                                                            "focus_keywords": {
                                                                "type": "array",
                                                                "items": {"type": "string"}
                                                            }
                                                        }
                                                    },
                                                    "performance_metrics": {
                                                        "type": "object",
                                                        "properties": {
                                                            "avg_relevance_score": {"type": "number"},
                                                            "optimization_suggestions": {"type": "integer"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "500": {"description": "Internal server error"}
                            }
                        }
                    },
                    "/api/working-memory/initialize": {
                        "post": {
                            "tags": ["Working Memory"],
                            "summary": "Initialize working memory system",
                            "description": """
Initialize or reinitialize the working memory system with specific configuration:

- **System initialization** with capacity settings
- **Configuration setup** for optimization parameters
- **Pool preparation** for memory operations
- **Performance tuning** based on usage patterns

Required before other working memory operations can be performed effectively.
                            """,
                            "requestBody": {
                                "required": False,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "capacity": {
                                                    "type": "integer",
                                                    "minimum": 10,
                                                    "maximum": 1000,
                                                    "default": 100,
                                                    "description": "Maximum number of memories in working pool"
                                                },
                                                "focus_threshold": {
                                                    "type": "number",
                                                    "minimum": 0.0,
                                                    "maximum": 1.0,
                                                    "default": 0.7,
                                                    "description": "Relevance threshold for focus mode"
                                                }
                                            }
                                        },
                                        "example": {
                                            "capacity": 150,
                                            "focus_threshold": 0.8
                                        }
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Working memory initialized successfully",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "success": {"type": "boolean"},
                                                    "message": {"type": "string"},
                                                    "configuration": {
                                                        "type": "object",
                                                        "properties": {
                                                            "capacity": {"type": "integer"},
                                                            "focus_threshold": {"type": "number"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "400": {"description": "Invalid configuration parameters"},
                                "500": {"description": "Internal server error"}
                            }
                        }
                    }
                },
                "components": {
                    "schemas": {
                        "Error": {
                            "type": "object",
                            "properties": {
                                "error": {
                                    "type": "string",
                                    "description": "Error message describing what went wrong"
                                }
                            }
                        }
                    }
                }
            }
            return JSONResponse(openapi_schema)
        
        # Add OpenAPI endpoints
        app.add_route("/openapi.json", openapi_schema, methods=["GET"])
        
        # Add Swagger UI endpoint
        async def swagger_ui(request):
            """Swagger UI for API documentation"""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Ultimate MCP Server API Documentation</title>
                <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
                <style>
                    html {
                        box-sizing: border-box;
                        overflow: -moz-scrollbars-vertical;
                        overflow-y: scroll;
                    }
                    *, *:before, *:after {
                        box-sizing: inherit;
                    }
                    body {
                        margin:0;
                        background: #fafafa;
                    }
                </style>
            </head>
            <body>
                <div id="swagger-ui"></div>
                <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
                <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
                <script>
                window.onload = function() {
                    const ui = SwaggerUIBundle({
                        url: '/openapi.json',
                        dom_id: '#swagger-ui',
                        deepLinking: true,
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIStandalonePreset
                        ],
                        plugins: [
                            SwaggerUIBundle.plugins.DownloadUrl
                        ],
                        layout: "StandaloneLayout"
                    });
                };
                </script>
            </body>
            </html>
            """
            from starlette.responses import HTMLResponse
            return HTMLResponse(content=html_content)
        
        app.add_route("/docs", swagger_ui, methods=["GET"])
        app.add_route("/api/docs", swagger_ui, methods=["GET"])  # Alternative endpoint
        
        print("[DEBUG] Added OpenAPI schema and Swagger UI endpoints", file=sys.stderr)
        print("[DEBUG] Swagger docs available at: /docs and /api/docs", file=sys.stderr)
        
        # Add root endpoint to the SSE app for MCP discovery
        
        async def root_endpoint(request):
            """Root endpoint for MCP server discovery"""
            response = JSONResponse({
                "type": "mcp-server",
                "version": "1.0.0",
                "transport": "sse",
                "endpoint": "/sse"
            })
            response.headers["X-MCP-Server"] = "true"
            response.headers["X-MCP-Version"] = "1.0.0"
            response.headers["X-MCP-Transport"] = "sse"
            return response
            
        # Add the root endpoint to the app using add_route
        app.add_route("/", root_endpoint)
        
        # Log SSE endpoint
        print(f"SSE endpoint available at: http://{server_host}:{server_port}/sse", file=sys.stderr)
        
        # Use our custom quiet Uvicorn server for silent shutdown
        config = uvicorn.Config(
            app,
            host=server_host,
            port=server_port,
            log_config=LOGGING_CONFIG,
            log_level=final_log_level.lower(),
        )
        
        server = create_quiet_server(config)
        server.run()
    else: # stdio mode
        # --- Stdio Mode Execution ---
        logger.info("Running in stdio mode...")
        
        # Create a shutdown handler for stdio mode cleanup  
        async def cleanup_resources():
            """Performs cleanup for various components during shutdown."""
            
            print("Cleaning up Gateway instance and associated resources...", file=sys.stderr)
            
            # Shutdown SQL Tools with timeout
            try:
                await asyncio.wait_for(shutdown_sql_tools(), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown
            
            # Shutdown Connection Manager with timeout
            try:
                from ultimate_mcp_server.tools.sql_databases import _connection_manager
                await asyncio.wait_for(_connection_manager.shutdown(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown
            
            # Shutdown Smart Browser with timeout
            try:
                await asyncio.wait_for(smart_browser_shutdown(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Suppress errors during shutdown
        
        # Configure graceful shutdown with error suppression
        enable_quiet_shutdown()
        
        # Register the same cleanup function for stdio mode
        register_shutdown_handler(cleanup_resources)
        
        try:
            # Run the FastMCP stdio loop - this will block until interrupted
            _gateway_instance.mcp.run()
        except (KeyboardInterrupt, SystemExit):
            # Normal shutdown - handled by graceful shutdown system
            pass
        except Exception:
            # Any other error - also handled by graceful shutdown
            pass
        # --- End Stdio Mode ---

    # --- Post-Server Exit ---
    logger.info("Server loop exited.")