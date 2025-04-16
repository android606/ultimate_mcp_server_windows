"""
Configuration management for LLM Gateway MCP Server.

Handles loading, validation, and access to configuration settings
from environment variables and config files.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from decouple import Config as DecoupleConfig
from decouple import RepositoryEnv, UndefinedValueError
from pydantic import BaseModel, Field, ValidationError, field_validator

# from pydantic_settings import BaseSettings, SettingsConfigDict # Removed BaseSettings

# --- Decouple Config Instance ---
# This will read from .env file and environment variables
decouple_config = DecoupleConfig(RepositoryEnv('.env'))
# --------------------------------

# Default configuration file paths (Adapt as needed)
DEFAULT_CONFIG_PATHS = [
    "./gateway_config.yaml",
    "./gateway_config.yml",
    "./gateway_config.json",
    "~/.config/llm_gateway/config.yaml",
    "~/.llm_gateway.yaml",
]

# Environment variable prefix (still potentially useful for non-secret env vars)
ENV_PREFIX = "GATEWAY_"

# Global configuration instance
_config = None

# Basic logger for config loading issues before full logging is set up
config_logger = logging.getLogger("llm_gateway.config")
handler = logging.StreamHandler(sys.stderr)
if not config_logger.hasHandlers():
    config_logger.addHandler(handler)
    config_logger.setLevel(logging.INFO)


class ServerConfig(BaseModel):
    """Server configuration settings."""
    name: str = Field("LLM Gateway MCP Server", description="Name of the server")
    host: str = Field("127.0.0.1", description="Host to bind the server to")
    port: int = Field(8013, description="Port to bind the server to") # Default port changed
    workers: int = Field(1, description="Number of worker processes")
    debug: bool = Field(False, description="Enable debug mode (affects reload)")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="CORS allowed origins") # Use default_factory for mutable defaults
    log_level: str = Field("info", description="Logging level (debug, info, warning, error, critical)")
    version: str = Field("0.1.0", description="Server version (from config, not package)")

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed = ['debug', 'info', 'warning', 'error', 'critical']
        level_lower = v.lower()
        if level_lower not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return level_lower

class CacheConfig(BaseModel):
    """Cache configuration settings."""
    enabled: bool = Field(True, description="Whether caching is enabled")
    ttl: int = Field(3600, description="Time-to-live for cache entries in seconds")
    max_entries: int = Field(10000, description="Maximum number of entries to store in cache")
    directory: Optional[str] = Field(None, description="Directory for cache persistence")
    fuzzy_match: bool = Field(True, description="Whether to use fuzzy matching for cache keys")

class ProviderConfig(BaseModel):
    """Configuration for a specific provider."""
    enabled: bool = Field(True, description="Whether the provider is enabled")
    api_key: Optional[str] = Field(None, description="API key for the provider (loaded via decouple)") # Updated description
    base_url: Optional[str] = Field(None, description="Base URL for API requests (loaded via decouple/file)") # Updated description
    organization: Optional[str] = Field(None, description="Organization identifier (loaded via decouple/file)") # Updated description
    default_model: Optional[str] = Field(None, description="Default model to use (loaded via decouple/file)") # Updated description
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for completions")
    timeout: Optional[float] = Field(30.0, description="Timeout for API requests in seconds")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters (loaded via decouple/file)") # Updated description

class ProvidersConfig(BaseModel):
    """Configuration for all LLM providers."""
    openai: ProviderConfig = Field(default_factory=ProviderConfig, description="OpenAI provider configuration")
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig, description="Anthropic provider configuration")
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig, description="DeepSeek provider configuration")
    gemini: ProviderConfig = Field(default_factory=ProviderConfig, description="Gemini provider configuration")
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig, description="OpenRouter provider configuration")
    grok: ProviderConfig = Field(default_factory=ProviderConfig, description="Grok (xAI) provider configuration")

class FilesystemProtectionConfig(BaseModel):
    """Configuration for filesystem protection heuristics."""
    enabled: bool = Field(True, description="Enable protection checks for this operation")
    max_files_threshold: int = Field(50, description="Trigger detailed check above this many files")
    datetime_stddev_threshold_sec: float = Field(60 * 60 * 24 * 30, description="Timestamp variance threshold (seconds)")
    file_type_variance_threshold: int = Field(5, description="File extension variance threshold")
    max_stat_errors_pct: float = Field(10.0, description="Max percentage of failed stat calls allowed during check")

class FilesystemConfig(BaseModel):
    """Configuration for filesystem tools."""
    allowed_directories: List[str] = Field(default_factory=list, description="List of absolute paths allowed for access")
    file_deletion_protection: FilesystemProtectionConfig = Field(default_factory=FilesystemProtectionConfig, description="Settings for deletion protection heuristics")
    file_modification_protection: FilesystemProtectionConfig = Field(default_factory=FilesystemProtectionConfig, description="Settings for modification protection heuristics (placeholder)")
    default_encoding: str = Field("utf-8", description="Default encoding for text file operations")
    max_read_size_bytes: int = Field(100 * 1024 * 1024, description="Maximum size for reading files") # 100MB example

class AgentMemoryConfig(BaseModel):
    """Configuration for Cognitive and Agent Memory tool."""
    db_path: str = Field("unified_agent_memory.db", description="Path to the agent memory SQLite database")
    max_working_memory_size: int = Field(20, description="Maximum number of items in working memory")
    memory_decay_rate: float = Field(0.01, description="Decay rate for memory relevance per hour")
    importance_boost_factor: float = Field(1.5, description="Multiplier for explicitly marked important memories")
    similarity_threshold: float = Field(0.75, description="Default threshold for semantic similarity search")

class ToolRegistrationConfig(BaseModel):
    """Configuration for tool registration."""
    filter_enabled: bool = Field(False, description="Whether to filter which tools are registered")
    included_tools: List[str] = Field(default_factory=list, description="List of tool names to include (empty means include all)")
    excluded_tools: List[str] = Field(default_factory=list, description="List of tool names to exclude (takes precedence over included_tools)")

class GatewayConfig(BaseModel): # Inherit from BaseModel now
    """Main LLM Gateway configuration model."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig)
    agent_memory: AgentMemoryConfig = Field(default_factory=AgentMemoryConfig) # Added agent memory
    tool_registration: ToolRegistrationConfig = Field(default_factory=ToolRegistrationConfig) # Added tool registration config

    storage_directory: str = Field("./storage", description="Directory for persistent storage")
    log_directory: str = Field("./logs", description="Directory for log files")
    prompt_templates_directory: str = Field("./prompt_templates", description="Directory containing prompt templates") # Added prompt dir

def expand_path(path: str) -> str:
    """Expand user and variables in path."""
    expanded = os.path.expanduser(path)
    expanded = os.path.expandvars(expanded)
    return os.path.abspath(expanded)

def find_config_file() -> Optional[str]:
    """Find the first available configuration file from default paths."""
    for path in DEFAULT_CONFIG_PATHS:
        try:
            expanded_path = expand_path(path)
            if os.path.isfile(expanded_path):
                config_logger.debug(f"Found config file: {expanded_path}")
                return expanded_path
        except Exception as e:
            config_logger.debug(f"Could not check path {path}: {e}")
    config_logger.debug("No default config file found.")
    return None

def load_config_from_file(path: str) -> Dict[str, Any]:
    """Load configuration from a file (YAML or JSON)."""
    path = expand_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config_logger.debug(f"Loading configuration from file: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith(('.yaml', '.yml')):
                config_data = yaml.safe_load(f)
            elif path.endswith('.json'):
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path}")
            return config_data if config_data is not None else {}
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid format in {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading {path}: {e}") from e

def load_config(
    config_file_path: Optional[str] = None,
    load_default_files: bool = True,
) -> GatewayConfig:
    """Load configuration from defaults, file, and inject keys/settings via decouple.

    Priority: .env/Env Vars (via decouple) > Config File > Pydantic Defaults

    Args:
        config_file_path: Explicit path to a config file.
        load_default_files: Whether to search for default config files.

    Returns:
        Validated GatewayConfig object with settings applied.
    """
    global _config
    file_config_data = {}

    # 1. Find and load config file (if specified or found)
    chosen_file_path = None
    if config_file_path:
        chosen_file_path = expand_path(config_file_path)
    elif load_default_files:
        chosen_file_path = find_config_file()

    if chosen_file_path and os.path.isfile(chosen_file_path):
        try:
            file_config_data = load_config_from_file(chosen_file_path)
            config_logger.info(f"Loaded base configuration from: {chosen_file_path}")
        except Exception as e:
            config_logger.warning(f"Could not load config file {chosen_file_path}: {e}")
            if config_file_path:
                raise ValueError(f"Failed to load specified config: {chosen_file_path}") from e
    elif config_file_path:
         raise FileNotFoundError(f"Specified configuration file not found: {config_file_path}")

    # 2. Initialize GatewayConfig from Pydantic defaults and file data
    try:
        loaded_config = GatewayConfig.model_validate(file_config_data)
    except ValidationError as e:
        config_logger.error("Configuration validation failed during file/default loading:")
        config_logger.error(str(e))
        config_logger.warning("Falling back to default configuration before applying env vars.")
        loaded_config = GatewayConfig() # Fallback to defaults

    # 3. Use decouple to load/override settings from .env/environment variables
    #    Decouple handles checking env vars and .env file automatically.

    # --- Load Provider API Keys ---
    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "grok": "GROK_API_KEY",
    }
    for provider_name, env_var in provider_key_map.items():
        provider_conf = getattr(loaded_config.providers, provider_name, None)
        if provider_conf:
            api_key_from_env = decouple_config.get(env_var, default=None)
            if api_key_from_env:
                if provider_conf.api_key and provider_conf.api_key != api_key_from_env:
                    config_logger.debug(f"Overriding API key for {provider_name} from env/'.env'.")
                elif not provider_conf.api_key:
                    config_logger.debug(f"Setting API key for {provider_name} from env/'.env'.")
                provider_conf.api_key = api_key_from_env

    # --- Load other Provider settings (base_url, default_model, org, specific headers) ---
    # Example for OpenRouter specific headers
    openrouter_conf = loaded_config.providers.openrouter
    try:
        # Use get() to avoid UndefinedValueError if not set
        http_referer = decouple_config.get('GATEWAY_OPENROUTER_HTTP_REFERER', default=None)
        x_title = decouple_config.get('GATEWAY_OPENROUTER_X_TITLE', default=None)
        if http_referer:
            openrouter_conf.additional_params['http_referer'] = http_referer
            config_logger.debug("Setting OpenRouter http_referer from env/'.env'.")
        if x_title:
            openrouter_conf.additional_params['x_title'] = x_title
            config_logger.debug("Setting OpenRouter x_title from env/'.env'.")
    except Exception as e: # Catch potential decouple issues
        config_logger.warning(f"Could not load optional OpenRouter headers from env: {e}")

    # Example for generic provider settings like base_url, default_model, organization
    for provider_name in ["openai", "anthropic", "deepseek", "gemini", "openrouter", "grok"]:
        provider_conf = getattr(loaded_config.providers, provider_name, None)
        if provider_conf:
            p_name_upper = provider_name.upper()
            try:
                base_url_env = decouple_config.get(f"GATEWAY_{p_name_upper}_BASE_URL", default=None)
                if base_url_env:
                    provider_conf.base_url = base_url_env
                    config_logger.debug(f"Setting {provider_name} base_url from env/'.env'.")

                default_model_env = decouple_config.get(f"GATEWAY_{p_name_upper}_DEFAULT_MODEL", default=None)
                if default_model_env:
                    provider_conf.default_model = default_model_env
                    config_logger.debug(f"Setting {provider_name} default_model from env/'.env'.")

                org_env = decouple_config.get(f"GATEWAY_{p_name_upper}_ORGANIZATION", default=None)
                if org_env:
                    provider_conf.organization = org_env
                    config_logger.debug(f"Setting {provider_name} organization from env/'.env'.")

            except Exception as e:
                 config_logger.warning(f"Could not load optional settings for provider {provider_name} from env: {e}")


    # --- Load Server Port ---
    try:
        server_port_env = decouple_config.get('GATEWAY_SERVER_PORT', default=None)
        if server_port_env is not None:
            loaded_config.server.port = decouple_config('GATEWAY_SERVER_PORT', cast=int)
            config_logger.debug(f"Overriding server port from env: {loaded_config.server.port}")
    except (ValueError, UndefinedValueError) as e:
        config_logger.warning(f"Invalid or missing GATEWAY_SERVER_PORT env var: {e}. Using default/file value.")

    # --- Load Filesystem Allowed Directories ---
    allowed_dirs_env_var = "GATEWAY__FILESYSTEM__ALLOWED_DIRECTORIES"
    try:
        allowed_dirs_env_value_str = decouple_config.get(allowed_dirs_env_var, default=None)
        if allowed_dirs_env_value_str is not None:
            try:
                allowed_dirs_from_env = json.loads(allowed_dirs_env_value_str)
                if isinstance(allowed_dirs_from_env, list):
                    if loaded_config.filesystem.allowed_directories:
                        config_logger.debug(f"Overriding filesystem.allowed_directories from env var {allowed_dirs_env_var}.")
                    else:
                        config_logger.debug(f"Setting filesystem.allowed_directories from env var {allowed_dirs_env_var}.")
                    loaded_config.filesystem.allowed_directories = allowed_dirs_from_env
                else:
                     config_logger.warning(f"Env var {allowed_dirs_env_var} did not contain a valid JSON list. Value ignored.")
            except json.JSONDecodeError:
                config_logger.warning(f"Failed to parse JSON from env var {allowed_dirs_env_var}. Value: '{allowed_dirs_env_value_str}'. Ignoring env var.")
    except Exception as e:
        config_logger.error(f"Error processing env var {allowed_dirs_env_var}: {e}", exc_info=True)

    # --- Load Agent Memory Settings ---
    try:
        loaded_config.agent_memory.db_path = decouple_config('GATEWAY_AGENT_MEMORY_DB_PATH', default=loaded_config.agent_memory.db_path)
        loaded_config.agent_memory.max_working_memory_size = decouple_config('GATEWAY_AGENT_MEMORY_MAX_WORKING_SIZE', default=loaded_config.agent_memory.max_working_memory_size, cast=int)
        loaded_config.agent_memory.memory_decay_rate = decouple_config('GATEWAY_AGENT_MEMORY_DECAY_RATE', default=loaded_config.agent_memory.memory_decay_rate, cast=float)
        loaded_config.agent_memory.importance_boost_factor = decouple_config('GATEWAY_AGENT_MEMORY_IMPORTANCE_BOOST', default=loaded_config.agent_memory.importance_boost_factor, cast=float)
        loaded_config.agent_memory.similarity_threshold = decouple_config('GATEWAY_AGENT_MEMORY_SIMILARITY_THRESHOLD', default=loaded_config.agent_memory.similarity_threshold, cast=float)
        config_logger.debug("Loaded agent memory settings from env/'.env' or defaults.")
    except (ValueError, UndefinedValueError) as e:
         config_logger.warning(f"Issue loading agent memory settings from env: {e}. Using Pydantic defaults.")
    except Exception as e:
        config_logger.error(f"Unexpected error loading agent memory settings: {e}", exc_info=True)


    # --- Load Prompt Templates Directory ---
    try:
        loaded_config.prompt_templates_directory = decouple_config('GATEWAY_PROMPT_TEMPLATES_DIR', default=loaded_config.prompt_templates_directory)
        config_logger.debug(f"Set prompt templates directory: {loaded_config.prompt_templates_directory}")
    except Exception as e:
        config_logger.warning(f"Could not load prompt templates directory from env: {e}")


    # --- Load Cache Directory ---
    try:
        cache_dir_env = decouple_config('GATEWAY_CACHE_DIR', default=None) # Changed env var name for clarity
        if cache_dir_env:
             loaded_config.cache.directory = cache_dir_env
             config_logger.debug(f"Set cache directory from env: {loaded_config.cache.directory}")
    except Exception as e:
         config_logger.warning(f"Could not load cache directory from env: {e}")


    # --- Expand paths ---
    try:
        # Expand core directories
        loaded_config.storage_directory = expand_path(loaded_config.storage_directory)
        loaded_config.log_directory = expand_path(loaded_config.log_directory)
        loaded_config.prompt_templates_directory = expand_path(loaded_config.prompt_templates_directory) # Expand new dir

        # Expand cache directory if set
        if loaded_config.cache.directory:
            loaded_config.cache.directory = expand_path(loaded_config.cache.directory)

        # Expand agent memory DB path (assuming it's a relative path)
        # Check if it's already absolute to avoid issues
        if not os.path.isabs(loaded_config.agent_memory.db_path):
            # Place it relative to storage_directory by default? Or workspace root? Let's choose storage.
            db_in_storage = Path(loaded_config.storage_directory) / loaded_config.agent_memory.db_path
            loaded_config.agent_memory.db_path = str(db_in_storage.resolve())
            config_logger.debug(f"Expanded agent memory db path to: {loaded_config.agent_memory.db_path}")

        # Expand allowed filesystem directories
        expanded_allowed_dirs = []
        for d in loaded_config.filesystem.allowed_directories:
             if isinstance(d, str):
                  expanded_allowed_dirs.append(expand_path(d))
             else:
                  config_logger.warning(f"Ignoring non-string entry in allowed_directories: {d!r}")
        loaded_config.filesystem.allowed_directories = expanded_allowed_dirs
    except Exception as e:
        config_logger.error(f"Error expanding configured paths: {e}", exc_info=True)

    # --- Ensure critical directories exist ---
    try:
        # Use pathlib for consistency
        Path(loaded_config.storage_directory).mkdir(parents=True, exist_ok=True)
        Path(loaded_config.log_directory).mkdir(parents=True, exist_ok=True)
        Path(loaded_config.prompt_templates_directory).mkdir(parents=True, exist_ok=True) # Ensure prompt dir exists

        if loaded_config.cache.enabled and loaded_config.cache.directory:
             Path(loaded_config.cache.directory).mkdir(parents=True, exist_ok=True)

        # Ensure Agent Memory DB directory exists
        db_dir = Path(loaded_config.agent_memory.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    except OSError as e:
        config_logger.error(f"Failed to create necessary directories: {e}")

    _config = loaded_config
    config_logger.debug(f"Effective allowed directories: {loaded_config.filesystem.allowed_directories}")
    config_logger.debug(f"Effective Agent Memory DB path: {loaded_config.agent_memory.db_path}")
    config_logger.debug(f"Effective Prompt Templates directory: {loaded_config.prompt_templates_directory}")
    return _config

def get_config() -> GatewayConfig:
    """Get the globally loaded configuration using decouple.

    Loads the configuration if it hasn't been loaded yet. Reloads if
    GATEWAY_FORCE_CONFIG_RELOAD=true is set in the environment/.env.

    Returns:
        The GatewayConfig instance.

    Raises:
        RuntimeError: If configuration cannot be loaded.
    """
    global _config
    # Use decouple directly here for the reload flag check
    force_reload = decouple_config.get("GATEWAY_FORCE_CONFIG_RELOAD", default='false').lower() == 'true'

    if _config is None or force_reload:
        try:
            _config = load_config() # load_config now handles internal state update
        except Exception as e:
            config_logger.critical(f"Failed to load configuration: {e}", exc_info=True)
            raise RuntimeError("Configuration could not be loaded.") from e

    if _config is None: # Should not happen if load_config succeeded or raised
        raise RuntimeError("Configuration is None after loading attempt.")

    return _config


def get_config_as_dict() -> Dict[str, Any]:
    """Get the current configuration as a dictionary."""
    config_obj = get_config()
    return config_obj.model_dump()
