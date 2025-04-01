"""Main server implementation for LLM Gateway."""
import asyncio
import logging
import logging.config
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from llm_gateway.config import get_config
from llm_gateway.constants import Provider
from llm_gateway.core.models.tournament import (
    CancelTournamentInput,
    CancelTournamentOutput,
    CreateTournamentInput,
    CreateTournamentOutput,
    GetTournamentResultsInput,
    GetTournamentStatusInput,
    GetTournamentStatusOutput,
    TournamentBasicInfo,
    TournamentData,
    TournamentStatus,
)
from llm_gateway.core.providers.base import get_provider
from llm_gateway.core.tournaments.manager import tournament_manager
from llm_gateway.utils.logging import logger
from llm_gateway.utils.logging.logger import get_logger

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
        "default": { # Console handler
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "access": { # Access log handler
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "rich_console": { # Rich console handler
            "()": "llm_gateway.utils.logging.formatter.create_rich_console_handler",
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
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the gateway.
        
        Args:
            name: Name for the MCP server
        """
        # Get the current config
        cfg = get_config()
        
        # Set up name
        self.name = name or cfg.server.name
        
        # Initialize logger
        self.logger = logger
        
        self.logger.info(f"Initializing {self.name}...")
        
        # Initialize provider tracking
        self.providers = {}
        self.provider_status = {}
        
        # Create MCP server with host and port settings
        self.mcp = FastMCP(
            self.name,
            lifespan=self._server_lifespan,
            host=cfg.server.host,
            port=cfg.server.port
        )
        
        # Register tools
        self._register_tools()
        
        # Register resources
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
        
        # Create lifespan context
        context = {
            "providers": self.providers,
            "provider_status": self.provider_status,
        }
        
        try:
            yield context
        finally:
            self.logger.info(f"Shutting down LLM Gateway '{self.name}'")
    
    async def _initialize_providers(self):
        """Initialize all enabled providers."""
        self.logger.info("Initializing LLM providers")
        
        # Get the current config
        cfg = get_config()
        
        # Get list of providers to initialize from config
        providers_to_init = []
        for provider_name in [p.value for p in Provider]:
            provider_config = getattr(cfg, 'providers', {}).get(provider_name, None)
            if provider_config and getattr(provider_config, 'enabled', False):
                providers_to_init.append(provider_name)
                
        # Add providers that have environment variables set
        env_var_map = {
            Provider.OPENAI.value: "OPENAI_API_KEY",
            Provider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
            Provider.DEEPSEEK.value: "DEEPSEEK_API_KEY",
            Provider.GEMINI.value: "GEMINI_API_KEY",
        }
        
        for provider_name, env_var in env_var_map.items():
            if provider_name not in providers_to_init and os.environ.get(env_var):
                self.logger.info(f"Adding provider {provider_name} from environment variable {env_var}")
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
        
        self.logger.info(f"Providers initialized: {len(available_providers)}/{len(providers_to_init)} available")
    
    async def _initialize_provider(self, provider_name: str):
        """Initialize a single provider.
        
        Args:
            provider_name: Provider name
        """
        try:
            # Create provider instance (will automatically use env vars if needed)
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
                
                self.logger.info(f"Provider {provider_name} initialized successfully with {len(models)} models")
            else:
                self.provider_status[provider_name] = ProviderStatus(
                    enabled=True,
                    available=False,
                    api_key_configured=True,
                    models=[],
                    error="Initialization failed"
                )
                
                self.logger.error(f"Provider {provider_name} initialization failed")
                
        except Exception as e:
            self.provider_status[provider_name] = ProviderStatus(
                enabled=True,
                available=False,
                api_key_configured=False,
                models=[],
                error=str(e)
            )
            
            self.logger.error(f"Error initializing provider {provider_name}: {str(e)}")
    
    def _register_tools(self):
        """Register MCP tools."""
        # Register providers tool group
        self._register_provider_tools()
        
        # Register document processing tools
        self._register_document_tools()
        
        # Register cost optimization tools
        self._register_cost_tools()
        
        # Register Tournament Tools
        self._register_tournament_tools()
    
    def _register_provider_tools(self):
        """Register provider-related tools."""
        
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
            """
            # Log the completion request - only through completions_logger
            prompt_summary = prompt[:50] + '...' if len(prompt) > 50 else prompt
            completions_logger.info(f"COMPLETION REQUEST - Provider: {provider}, Model: {model}")
            completions_logger.debug(f"Prompt: {prompt_summary}")
            
            # Get the provider instance
            provider_instance = self.providers.get(provider)
            if not provider_instance:
                error_msg = f"Provider '{provider}' not available. Valid options: {list(self.providers.keys())}"
                completions_logger.error(f"ERROR: {error_msg}")
                return {"error": error_msg}
            
            # Generate completion
            try:
                start_time = time.time()
                result = await provider_instance.generate_completion(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                processing_time = time.time() - start_time
                
                # Log success - only through completions_logger
                result_summary = result.text[:100] + '...' if len(result.text) > 100 else result.text
                completions_logger.info(
                    f"COMPLETION SUCCESS - Provider: {provider}, Model: {result.model}, "
                    f"Tokens: input={result.input_tokens}, output={result.output_tokens}, "
                    f"cost=${result.cost:.6f}, Processing time: {processing_time:.2f}s"
                )
                completions_logger.debug(f"Text: {result_summary}")
                
                return result.to_dict()
            except Exception as e:
                completions_logger.error(f"COMPLETION ERROR: {str(e)}", exc_info=True)
                raise
            
        @self.mcp.tool()
        @self.log_tool_calls
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
        @self.log_tool_calls
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
        
        self.logger.info("Registered provider tools")
    
    def _register_document_tools(self):
        """Register document processing tools."""
        # Placeholder for document tools
        # Example:
        # @self.mcp.tool()
        # async def summarize_document(...): ...
        
        self.logger.info("Registered document tools")
    
    def _register_cost_tools(self):
        """Register cost optimization tools."""
        # Placeholder for cost tools
        # Example:
        # @self.mcp.tool()
        # async def estimate_completion_cost(...): ...
        
        self.logger.info("Registered cost tools")

    # --- Tournament Tools Registration --- 
    def _register_tournament_tools(self) -> None:
        """Registers the tournament tools with MCP."""
        # Access each property to trigger the @self.mcp.tool decorator within them
        _ = self.create_tournament
        _ = self.get_tournament_status
        _ = self.list_tournaments
        _ = self.get_tournament_results
        _ = self.cancel_tournament
        self.logger.info("Registered tournament tools")

    @property
    def create_tournament(self):
        """Creates a new tournament and schedules it for background execution."""
        @self.mcp.tool(
            name="create_tournament",
            description="Creates and starts a new coding tournament based on a prompt and model configurations."
        )
        @self.log_tool_calls
        async def create_tournament_impl(
            name: str,
            prompt: str,
            model_ids: List[str],
            rounds: int = 3,
            tournament_type: str = "code",
            extraction_model: Optional[str] = None,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """Creates a new tournament and schedules it for background execution."""
            # Use logger consistently
            self.logger.info(f"Received request to create tournament: {name}")
            try:
                # Convert to input model
                input_data = CreateTournamentInput(
                    name=name,
                    prompt=prompt,
                    model_ids=model_ids,
                    rounds=rounds,
                    tournament_type=tournament_type,
                    extraction_model=extraction_model
                )
                
                tournament = tournament_manager.create_tournament(input_data)
                if not tournament:
                    raise ToolError("Failed to create tournament entry.")
                
                # Start execution in the background using the manager's asyncio fallback
                # We explicitly pass None for background_tasks as we are in the MCP context
                self.logger.info("Calling start_tournament_execution (using asyncio)")
                success = tournament_manager.start_tournament_execution(
                    tournament_id=tournament.tournament_id 
                    # No background_tasks argument needed anymore
                )
                
                if not success:
                    # If scheduling failed, the manager should have updated the status
                    self.logger.error(f"Failed to schedule background execution for tournament {tournament.tournament_id}")
                    # Attempt to load the potentially updated state
                    updated_tournament = tournament_manager.get_tournament(tournament.tournament_id)
                    error_msg = updated_tournament.error_message if updated_tournament else "Failed to schedule execution." # Updated field name
                    raise ToolError(f"Failed to start tournament execution: {error_msg}")
                    
                self.logger.info(f"Tournament {tournament.tournament_id} ({tournament.name}) created and background execution started.")
                # Return current status (likely PENDING or RUNNING)
                output = CreateTournamentOutput(
                    tournament_id=tournament.tournament_id, 
                    status=tournament.status
                )
                return output.dict()

            except ValueError as ve:
                self.logger.warning(f"Validation error creating tournament: {ve}")
                raise ToolError(f"Invalid input: {ve}") from ve
            except Exception as e:
                self.logger.error(f"Error creating tournament: {e}", exc_info=True)
                raise ToolError(f"An unexpected error occurred: {e}") from e
                
        return create_tournament_impl

    @property
    def get_tournament_status(self):
        """Retrieves the current status of a specific tournament."""
        @self.mcp.tool(
            name="get_tournament_status",
            description="Retrieves the current status and basic details of a specific tournament."
        )
        @self.log_tool_calls
        async def get_tournament_status_impl(
            tournament_id: str,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """Retrieves the current status of a specific tournament."""
            self.logger.debug(f"Getting status for tournament: {tournament_id}")
            try:
                # Convert to input model for validation
                input_data = GetTournamentStatusInput(tournament_id=tournament_id)
                
                tournament = tournament_manager.get_tournament(input_data.tournament_id)
                if not tournament:
                    raise ToolError(status_code=404, detail=f"Tournament not found: {tournament_id}")

                # Prepare the output using the Pydantic model
                output = GetTournamentStatusOutput(
                    tournament_id=tournament.tournament_id,
                    name=tournament.name,
                    tournament_type=tournament.config.tournament_type,
                    status=tournament.status,
                    current_round=tournament.current_round,
                    total_rounds=tournament.config.rounds,
                    created_at=tournament.created_at,
                    updated_at=tournament.updated_at,
                    error_message=tournament.error_message
                )
                return output.dict()
            except Exception as e:
                self.logger.error(f"Error getting tournament status for {tournament_id}: {e}", exc_info=True)
                raise ToolError(status_code=500, detail="Internal server error retrieving tournament status.") from e
        
        return get_tournament_status_impl

    @property
    def list_tournaments(self):
        """Lists all tournaments known to the manager."""
        @self.mcp.tool(
            name="list_tournaments",
            description="Lists all recorded tournaments with basic status information."
        )
        @self.log_tool_calls
        async def list_tournaments_impl(ctx: Context = None) -> List[Dict[str, Any]]:
            """Lists all tournaments known to the manager."""
            self.logger.debug("Listing all tournaments")
            try:
                # Get raw tournament data
                tournament_list_dicts = tournament_manager.list_tournaments()
                
                # Validate and convert each dict to the Pydantic model, then back to dict
                tournaments = [
                    TournamentBasicInfo(**t_dict).dict() for t_dict in tournament_list_dicts
                ]
                return tournaments
            except Exception as e:
                self.logger.error(f"Error listing tournaments: {e}", exc_info=True)
                raise ToolError(status_code=500, detail="Internal server error listing tournaments.") from e
        
        return list_tournaments_impl

    @property
    def get_tournament_results(self):
        """Retrieves the full TournamentData for a given tournament ID."""
        @self.mcp.tool(
            name="get_tournament_results",
            description="Retrieves the full results and configuration for a specific tournament, including all round data."
        )
        @self.log_tool_calls
        async def get_tournament_results_impl(
            tournament_id: str,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """Retrieves the full TournamentData for a given tournament ID."""
            self.logger.debug(f"Getting full results for tournament: {tournament_id}")
            try:
                # Convert to input model for validation
                input_data = GetTournamentResultsInput(tournament_id=tournament_id)
                
                # Use force_reload=True to ensure we get the latest from disk
                tournament: TournamentData = tournament_manager.get_tournament(input_data.tournament_id, force_reload=True)
                if not tournament:
                    raise ToolError(status_code=404, detail=f"Tournament not found: {tournament_id}")
                
                # The tournament is already a TournamentData instance
                return tournament.dict()
            except ToolError: # Re-raise specific ToolErrors
                raise
            except Exception as e:
                self.logger.error(f"Error getting tournament results for {tournament_id}: {e}", exc_info=True)
                raise ToolError(status_code=500, detail="Internal server error retrieving tournament results.") from e
        
        return get_tournament_results_impl

    @property
    def cancel_tournament(self):
        """Attempts to cancel a tournament by updating its status."""
        @self.mcp.tool(
            name="cancel_tournament",
            description="Attempts to cancel a running or pending tournament."
        )
        @self.log_tool_calls
        async def cancel_tournament_impl(
            tournament_id: str,
            ctx: Context = None
        ) -> Dict[str, Any]:
            """Attempts to cancel a tournament by updating its status."""
            self.logger.info(f"Received request to cancel tournament: {tournament_id}")
            try:
                # Convert to input model for validation
                input_data = CancelTournamentInput(tournament_id=tournament_id)
                
                success, message = tournament_manager.cancel_tournament(input_data.tournament_id)
                
                # Fetch the final status after cancellation attempt
                final_status = TournamentStatus.CANCELLED # Assume cancelled unless fetch fails
                updated_tournament = tournament_manager.get_tournament(input_data.tournament_id)
                if updated_tournament:
                    final_status = updated_tournament.status
                else:
                    # If tournament disappeared after cancel attempt (shouldn't happen) 
                    # or wasn't found initially
                    if message == "Tournament not found.": 
                        raise ToolError(status_code=404, detail=message)
                    else:
                        # If cancel succeeded but we can't fetch status, still report success
                        self.logger.warning(f"Could not fetch final status for tournament {tournament_id} after cancellation attempt.")
                
                if success:
                    self.logger.info(f"Cancellation request for tournament {tournament_id} processed. Final Status: {final_status}. Message: {message}")
                else:
                    self.logger.warning(f"Cancellation failed for tournament {tournament_id}. Final Status: {final_status}. Reason: {message}")
                    # Raise tool error if cancellation failed for reasons other than not found/already done
                    if final_status not in [TournamentStatus.CANCELLED, TournamentStatus.COMPLETED, TournamentStatus.FAILED]:
                        raise ToolError(status_code=400, detail=message)

                # Use the output model for consistency
                output = CancelTournamentOutput(
                    tournament_id=input_data.tournament_id,
                    status=final_status,
                    message=message
                )
                return output.dict()
            except ToolError: # Re-raise specific ToolErrors
                raise
            except Exception as e:
                self.logger.error(f"Error cancelling tournament {tournament_id}: {e}", exc_info=True)
                raise ToolError(status_code=500, detail=f"Internal server error cancelling tournament: {e}") from e
        
        return cancel_tournament_impl

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server lifespan context manager.
    
    Args:
        app: FastAPI application instance
    """
    logger.info("Starting LLM Gateway server")
    
    try:
        # Get the global gateway instance - don't recreate it
        global _gateway_instance
        if not _gateway_instance:
            _gateway_instance = Gateway()
            
        # Server is ready
        logger.info("LLM Gateway server ready")
        
        yield
        
    finally:
        logger.info("Shutting down LLM Gateway server")


def create_server() -> FastAPI:
    """Create and configure the FastAPI server.
    
    Returns:
        Configured FastAPI application
    """
    global _server_app
    
    # Check if server already exists
    if _server_app is not None:
        return _server_app
        
    # Create FastAPI app
    app = FastAPI(
        title="LLM Gateway Server",
        description="LLM Gateway Server for accessing multiple LLM providers",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # From config in a real app
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize the gateway if not already created
    global _gateway_instance
    if not _gateway_instance:
        _gateway_instance = Gateway()
    
    # Mount the MCP app to the FastAPI app
    app.mount("/mcp", _gateway_instance.mcp.sse_app())
    
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
) -> None:
    """Start the LLM Gateway Server using dictConfig for logging."""
    server_host = host or get_config().server.host
    server_port = port or get_config().server.port
    server_workers = workers or get_config().server.workers
    
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

    # Log startup info using the standard logger BEFORE Uvicorn takes over
    logging.info(f"Preparing to start Uvicorn on {server_host}:{server_port}...")

    # Start server with Uvicorn using the dictConfig
    import uvicorn
    uvicorn.run(
        "llm_gateway.core.server:create_server",  # Use factory pattern
        host=server_host,
        port=server_port,
        workers=server_workers,
        log_config=LOGGING_CONFIG,  # Pass the config dict
        reload=reload,
        factory=True,
    )