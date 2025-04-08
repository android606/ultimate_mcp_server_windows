"""
Configuration management for LLM Gateway MCP Server.

Handles loading, validation, and access to configuration settings
from environment variables and config files.
"""
import json
import logging  # Use standard logging for config loading messages
import os
import sys  # Add sys import
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, validator
from pydantic_settings import BaseSettings

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
config_logger.addHandler(handler)
config_logger.setLevel(logging.INFO)


class ServerConfig(BaseModel):
    """Server configuration settings."""
    name: str = Field("LLM Gateway MCP Server", description="Name of the server")
    host: str = Field("127.0.0.1", description="Host to bind the server to")
    port: int = Field(8013, description="Port to bind the server to") # Default port changed
    workers: int = Field(1, description="Number of worker processes")
    debug: bool = Field(False, description="Enable debug mode (affects reload)")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    log_level: str = Field("info", description="Logging level (debug, info, warning, error, critical)")
    # Add other server-specific settings here if needed
    
    @validator('log_level')
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

class GatewayConfig(BaseModel):
    """Main LLM Gateway configuration model."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    # providers: ProviderConfig = Field(default_factory=ProviderConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    # Example: Define base paths used by the application
    storage_directory: str = Field("./storage", description="Directory for persistent storage")
    log_directory: str = Field("./logs", description="Directory for log files")
    
    # Allow extra fields for flexibility or future expansion
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")

    class Config:
        env_nested_delimiter = '__' # For nested env vars like GATEWAY_SERVER__PORT
        env_prefix = ENV_PREFIX # Ensure Pydantic uses the correct prefix
        extra = 'allow' # Allow extra fields not explicitly defined


def expand_path(path: str) -> str:
    """Expand user and variables in path."""
    expanded = os.path.expanduser(path)
    expanded = os.path.expandvars(expanded)
    return os.path.abspath(expanded) # Return absolute path


def find_config_file() -> Optional[str]:
    """Find the first available configuration file from default paths."""
    for path in DEFAULT_CONFIG_PATHS:
        expanded_path = expand_path(path)
        if os.path.isfile(expanded_path):
            return expanded_path
    return None


def load_config_from_file(path: str) -> Dict[str, Any]:
    """Load configuration from a file (YAML or JSON)."""
    path = expand_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    config_logger.debug(f"Loading configuration from file: {path}")
    
    try:
        with open(path, 'r') as f:
            if path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f) or {}
            elif path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path}. Use .yaml or .json.")
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
    load_env: bool = True,
    load_default_files: bool = True,
) -> GatewayConfig:
    """Load configuration from defaults, file, and environment variables.
    
    Priority: Env Vars > Specific Config File > Default Config Files > Pydantic Defaults
    
    Args:
        config_file_path: Explicit path to a config file.
        load_env: Whether to load config from environment variables.
        load_default_files: Whether to search for default config files.
        
    Returns:
        Validated GatewayConfig object.
    """
    global _config
    
    # Start with empty dict (Pydantic defaults applied during validation)
    final_config_data = {}
    
    # 0. Load environment variables from .env file if it exists
    try:
        from decouple import Config as DecoupleConfig
        from decouple import RepositoryEnv
        
        env_file = ".env"
        if os.path.exists(env_file):
            config_logger.info(f"Loading environment variables from {env_file}")
            env_config = DecoupleConfig(RepositoryEnv(env_file))
            
            # Load API keys from .env into environment variables
            env_var_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY"]
            for key in env_var_keys:
                try:
                    value = env_config.get(key, default=None)
                    if value:
                        os.environ[key] = value
                        config_logger.debug(f"Loaded {key} from .env file")
                except Exception as e:
                    config_logger.debug(f"Could not load {key} from .env file: {e}")
    except ImportError:
        config_logger.debug("python-decouple not installed, skipping .env loading")
    except Exception as e:
        config_logger.warning(f"Error loading .env file: {e}")

    # 1. Load from default files if enabled and no specific file given
    if load_default_files and not config_file_path:
        default_file = find_config_file()
        if default_file:
            try:
                file_config = load_config_from_file(default_file)
                final_config_data = merge_configs(final_config_data, file_config)
            except Exception as e:
                config_logger.warning(f"Could not load default config file {default_file}: {e}")

    # 2. Load from specific config file if provided
    if config_file_path:
        try:
            specific_file_config = load_config_from_file(config_file_path)
            final_config_data = merge_configs(final_config_data, specific_file_config)
        except Exception as e:
            config_logger.error(f"Could not load specified config file {config_file_path}: {e}")
            # Decide whether to raise, exit, or continue with defaults/env
            raise ValueError(f"Failed to load specified config: {config_file_path}") from e

    # 3. Load from environment variables if enabled
    if load_env:
        env_config = load_config_from_env()
        final_config_data = merge_configs(final_config_data, env_config)
        
    # 4. Validate with Pydantic model
    try:
        loaded_config = GatewayConfig(**final_config_data)
    except ValidationError as e:
        # Simple string logging for the validation error object
        config_logger.error("Configuration validation failed. Details below:")
        config_logger.error(str(e))
        config_logger.warning("Returning default configuration due to validation errors.")
        loaded_config = GatewayConfig() # Fallback to defaults
        
    # Expand paths in the validated config
    loaded_config.storage_directory = expand_path(loaded_config.storage_directory)
    loaded_config.log_directory = expand_path(loaded_config.log_directory)
    # Expand other path-like fields if added
    
    # Ensure directories exist
    try:
        os.makedirs(loaded_config.storage_directory, exist_ok=True)
        os.makedirs(loaded_config.log_directory, exist_ok=True)
    except OSError as e:
        config_logger.error(f"Failed to create necessary directories: {e}")
        # Handle error appropriately - maybe raise? Or log and continue?

    _config = loaded_config # Store globally
    config_logger.info("Configuration loaded successfully.")
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
    if _config is None:
        config_logger.info("Configuration not yet loaded. Loading now...")
        try:
            # Load with default settings
            load_config() # Call the main loading function
        except Exception as e:
            # Use standard error formatting
            config_logger.critical(f"Failed to load configuration on demand: {e}")
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
        Configuration as a dictionary
    """
    config = get_config()
    return config.model_dump()

def get_env() -> ServerConfig:
    """Get server environment configuration"""
    return get_config().server

class Settings(BaseSettings):
    cache_dir: str = "./cache"
    cache_size_limit: int = 10_000_000  # 10MB
    cache_ttl: int = 3600  # 1 hour
    
    class Config:
        env_prefix = "LLM_GATEWAY_"
        
    # Add cache settings as a property to ensure backward compatibility
    @property
    def cache(self):
        return SimpleNamespace(
            enabled=True,
            ttl=self.cache_ttl,
            max_entries=1000,
            directory=self.cache_dir,
            fuzzy_match=True
        )

# Create a single instance of the settings
config = Settings()