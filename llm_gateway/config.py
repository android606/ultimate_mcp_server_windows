"""
Configuration management for LLM Gateway MCP Server.

Handles loading, validation, and access to configuration settings
from environment variables and config files.
"""
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default configuration file paths (Adapt as needed)
DEFAULT_CONFIG_PATHS = [
    "./gateway_config.yaml",
    "./gateway_config.yml",
    "./gateway_config.json",
    "~/.config/llm_gateway/config.yaml",
    "~/.llm_gateway.yaml",
]

# Environment variable prefix
ENV_PREFIX = "GATEWAY_"

# Global configuration instance
_config = None

# Basic logger for config loading issues before full logging is set up
config_logger = logging.getLogger("llm_gateway.config")
handler = logging.StreamHandler(sys.stderr)  # Explicitly use stderr
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
    # Add Pydantic v2 version field if desired (not strictly needed here)
    version: str = Field("0.1.0", description="Server version (from config, not package)") # Added for consistency

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

class FilesystemProtectionConfig(BaseModel):
    """Configuration for filesystem protection heuristics."""
    enabled: bool = Field(False, description="Enable protection checks for this operation")
    max_files_threshold: int = Field(100, description="Trigger detailed check above this many files")
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

class GatewayConfig(BaseSettings): # Inherit from BaseSettings for env var loading
    """Main LLM Gateway configuration model."""
    # --- Pydantic v2 model_config ---
    model_config = SettingsConfigDict(
        env_nested_delimiter='__', # For nested env vars like GATEWAY_SERVER__PORT
        env_prefix=ENV_PREFIX, # Ensure Pydantic uses the correct prefix
        extra='allow', # Allow extra fields not explicitly defined
        env_file='.env',       # Specify .env file for BaseSettings
        env_file_encoding='utf-8' # Specify encoding
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    # providers: ProviderConfig = Field(default_factory=ProviderConfig) # Keep commented if not defined
    cache: CacheConfig = Field(default_factory=CacheConfig)
    filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig) # +++ Add filesystem field +++

    storage_directory: str = Field("./storage", description="Directory for persistent storage")
    log_directory: str = Field("./logs", description="Directory for log files")

    # Allow extra fields for flexibility or future expansion - handled by model_config extra='allow'
    # extra: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration") # No longer needed

def expand_path(path: str) -> str:
    """Expand user and variables in path."""
    expanded = os.path.expanduser(path)
    expanded = os.path.expandvars(expanded)
    # Ensure it's absolute *after* expansion
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
    config_logger.debug("No default config file found in standard locations.")
    return None

def load_config_from_file(path: str) -> Dict[str, Any]:
    """Load configuration from a file (YAML or JSON)."""
    path = expand_path(path) # Ensure path is expanded before check
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    config_logger.debug(f"Loading configuration from file: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f: # Specify encoding
            if path.endswith(('.yaml', '.yml')):
                config_data = yaml.safe_load(f)
            elif path.endswith('.json'):
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path}. Use .yaml or .json.")
            return config_data if config_data is not None else {} # Return empty dict if file is empty
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid format in configuration file {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading configuration file {path}: {e}") from e


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables prefixed with GATEWAY_."""
    config = {}
    for key, value in os.environ.items():
        if key.startswith(ENV_PREFIX):
            # Remove prefix and split by double underscore for nesting
            key_parts = key[len(ENV_PREFIX):].lower().split('__')
            
            # Basic type conversion (consider more robust parsing if needed)
            parsed_value: Any
            try:
                if value.lower() in ['true', 'false']:
                    parsed_value = value.lower() == 'true'
                elif value.isdigit():
                    parsed_value = int(value)
                else:
                    try:
                        parsed_value = float(value)
                    except ValueError:
                         # Try parsing as JSON list/dict if it looks like one
                         if (value.startswith('[') and value.endswith(']')) or \
                            (value.startswith('{') and value.endswith('}')):
                             try:
                                 parsed_value = json.loads(value)
                             except json.JSONDecodeError:
                                 parsed_value = value # Keep as string if JSON parse fails
                         else:
                             parsed_value = value # Default to string
            except Exception:
                 parsed_value = value # Fallback to string on any parsing error
            
            # Build nested dictionary
            d = config
            nesting_ok = True
            for i, part in enumerate(key_parts):
                if i == len(key_parts) - 1:
                    d[part] = parsed_value
                else:
                    d = d.setdefault(part, {})
                    if not isinstance(d, dict): # Handle overwrite conflict
                        config_logger.warning(f"Config conflict for env var {key}: cannot nest under non-dict.")
                        nesting_ok = False
                        break 
            if not nesting_ok: # Skip if nesting failed
                continue
                        
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merge_configs(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    config_file_path: Optional[str] = None,
    load_env: bool = True, # Kept for potential future use, but BaseSettings handles it
    load_default_files: bool = True,
) -> GatewayConfig:
    """Load configuration from defaults, file, and environment variables.

    Priority: Env Vars > Specific Config File > Default Config Files > Pydantic Defaults

    Args:
        config_file_path: Explicit path to a config file.
        load_env: Whether to load config from environment variables (handled by BaseSettings).
        load_default_files: Whether to search for default config files.

    Returns:
        Validated GatewayConfig object.
    """
    global _config

    # --- Configuration Loading Strategy ---
    # 1. Start with Pydantic defaults (defined in the models).
    # 2. Load from default files if enabled and found.
    # 3. Load from specific config file if provided (overrides defaults/default file).
    # 4. BaseSettings will automatically load from .env and environment variables (overriding previous layers).

    file_config_data = {}

    # Step 2 & 3: Load from files
    chosen_file_path = None
    if config_file_path: # Specific file takes precedence
        chosen_file_path = expand_path(config_file_path)
    elif load_default_files: # Otherwise, look for defaults
        chosen_file_path = find_config_file()

    if chosen_file_path and os.path.isfile(chosen_file_path):
        try:
            file_config_data = load_config_from_file(chosen_file_path)
        except Exception as e:
            config_logger.warning(f"Could not load config file {chosen_file_path}: {e}")
            # Decide if failure to load a specified file should be fatal
            if config_file_path:
                raise ValueError(f"Failed to load specified config: {chosen_file_path}") from e
            # Otherwise, continue without file config if it was just a default search
    elif config_file_path:
         # If a specific file was given but not found
         raise FileNotFoundError(f"Specified configuration file not found: {config_file_path}")


    # Step 4: Initialize BaseSettings. It loads from .env and environment vars automatically.
    # We pass the file_config_data to initialize the model *before* env vars are applied.
    try:
        # Initialize with file data, env vars will override during BaseSettings init
        # Pydantic v2: BaseSettings takes keyword arguments corresponding to fields
        # Need to ensure file_config_data keys match GatewayConfig fields
        loaded_config = GatewayConfig(**file_config_data)

    except ValidationError as e:
        # Simple string logging for the validation error object
        config_logger.error("Configuration validation failed. Details below:")
        config_logger.error(str(e))
        config_logger.warning("Returning default configuration due to validation errors.")
        loaded_config = GatewayConfig() # Fallback to defaults

    # Expand paths in the validated config AFTER loading
    loaded_config.storage_directory = expand_path(loaded_config.storage_directory)
    loaded_config.log_directory = expand_path(loaded_config.log_directory)
    if loaded_config.cache.directory:
        loaded_config.cache.directory = expand_path(loaded_config.cache.directory)

    # --- Expand allowed_directories paths ---
    expanded_allowed_dirs = []
    for d in loaded_config.filesystem.allowed_directories:
         try:
             if isinstance(d, str):
                  expanded_allowed_dirs.append(expand_path(d))
             else:
                  config_logger.warning(f"Ignoring non-string entry in allowed_directories: {d!r}")
         except Exception as e:
              config_logger.warning(f"Failed to expand path in allowed_directories '{d}': {e}")
    loaded_config.filesystem.allowed_directories = expanded_allowed_dirs
    # --- End expand allowed_directories ---


    # Ensure critical directories exist
    try:
        os.makedirs(loaded_config.storage_directory, exist_ok=True)
        os.makedirs(loaded_config.log_directory, exist_ok=True)
        if loaded_config.cache.enabled and loaded_config.cache.directory:
             os.makedirs(loaded_config.cache.directory, exist_ok=True)
    except OSError as e:
        config_logger.error(f"Failed to create necessary directories: {e}")
        # Consider if this should be fatal

    _config = loaded_config # Store globally
    config_logger.info("Configuration loaded successfully.")
    # Log the allowed directories specifically for debugging
    config_logger.debug(f"Effective allowed directories: {loaded_config.filesystem.allowed_directories}")
    return _config


def get_config() -> GatewayConfig:
    """Get the globally loaded configuration.

    Loads the configuration if it hasn't been loaded yet.

    Returns:
        The GatewayConfig instance.

    Raises:
        RuntimeError: If configuration has not been loaded and cannot be loaded.
    """
    global _config
    # Check if FORCE_RELOAD env var is set
    force_reload_env = "GATEWAY_FORCE_CONFIG_RELOAD"
    if os.environ.get(force_reload_env, "false").lower() == "true":
        config_logger.info(f"Forcing configuration reload due to {force_reload_env} env var.", emoji_key="cache")
        _config = None # Clear global config to trigger reload
        os.environ.pop(force_reload_env, None) # Consume the flag


    if _config is None:
        config_logger.info("Configuration not yet loaded or reload forced. Loading now...")
        try:
            # Load with default settings (searches files, loads env)
            load_config()
        except Exception as e:
            # Use standard error formatting
            config_logger.critical(f"Failed to load configuration on demand: {e}", exc_info=True)
            # Depending on application needs, either raise or return a default config
            # raise RuntimeError("Configuration could not be loaded.") from e
            _config = GatewayConfig() # Return default config on failure
            config_logger.warning("Returning default configuration due to load failure.")

    # Check _config *after* loading attempt
    if _config is None:
        # Ensure correct syntax for raise
        raise RuntimeError("Configuration is None after loading attempt.")

    return _config


def get_config_as_dict() -> Dict[str, Any]:
    """Get the current configuration as a dictionary.

    Returns:
        Configuration as a dictionary using Pydantic's model_dump
    """
    config_obj = get_config()
    return config_obj.model_dump()
def get_env() -> ServerConfig:
    """Get server environment configuration"""
    return get_config().server
