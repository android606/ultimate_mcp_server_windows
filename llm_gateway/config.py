"""Central configuration management for LLM Gateway."""
from pathlib import Path
from typing import Any, Dict, Optional, Union

from decouple import Config as DecoupleConfig
from decouple import RepositoryEnv
from pydantic import BaseModel, Field

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Use python-decouple to load and parse environment variables
config_source = RepositoryEnv('.env')
env_config = DecoupleConfig(config_source)


def get_env(key, default=None, cast=None):
    """Get environment variable with fallback to default."""
    try:
        return env_config(key, default=default, cast=cast)
    except Exception:
        return default


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default=get_env("LOG_LEVEL", "INFO"))
    file: Optional[str] = Field(default=get_env("LOG_FILE", None))
    use_rich: bool = Field(default=get_env("USE_RICH_LOGGING", True, cast=bool))
    show_timestamps: bool = Field(default=True)
    emoji_enabled: bool = Field(default=True)
    

class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = Field(default=get_env("CACHE_ENABLED", True, cast=bool))
    ttl: int = Field(default=get_env("CACHE_TTL", 86400, cast=int))  # Default TTL in seconds (24 hours)
    directory: Optional[str] = Field(default=get_env("CACHE_DIR", None))
    max_entries: int = Field(default=get_env("CACHE_MAX_ENTRIES", 10000, cast=int))
    fuzzy_match: bool = Field(default=get_env("CACHE_FUZZY_MATCH", True, cast=bool))


class ProviderConfig(BaseModel):
    """Provider configuration."""
    enabled: bool = Field(default=True)
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    max_tokens: Optional[int] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class ProvidersConfig(BaseModel):
    """Configuration for all providers."""
    openai: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        enabled=True,
        api_key=get_env("OPENAI_API_KEY"),
        default_model=get_env("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
        max_tokens=get_env("OPENAI_MAX_TOKENS", 8192, cast=int),
    ))
    anthropic: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        enabled=True,
        api_key=get_env("ANTHROPIC_API_KEY"),
        default_model=get_env("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-haiku-latest"),
        max_tokens=get_env("ANTHROPIC_MAX_TOKENS", 200000, cast=int),
    ))
    deepseek: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        enabled=True,
        api_key=get_env("DEEPSEEK_API_KEY"),
        default_model=get_env("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
        max_tokens=get_env("DEEPSEEK_MAX_TOKENS", 8192, cast=int),
    ))
    gemini: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        enabled=True,
        api_key=get_env("GEMINI_API_KEY"),
        default_model=get_env("GEMINI_DEFAULT_MODEL", "gemini-2.0-flash-lite"),
        max_tokens=get_env("GEMINI_MAX_TOKENS", 8192, cast=int),
    ))


class ServerConfig(BaseModel):
    """Server configuration."""
    name: str = Field(default=get_env("SERVER_NAME", "LLM Gateway"))
    version: str = Field(default=get_env("SERVER_VERSION", "0.1.0"))
    port: int = Field(default=get_env("SERVER_PORT", 8000, cast=int))
    host: str = Field(default=get_env("SERVER_HOST", "127.0.0.1"))
    workers: int = Field(default=get_env("SERVER_WORKERS", 1, cast=int))
    debug: bool = Field(default=get_env("SERVER_DEBUG", False, cast=bool))


class Config(BaseModel):
    """Global configuration for LLM Gateway."""
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


# Global configuration instance
config = Config()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a file.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        Config: Loaded configuration
    """
    # Implement configuration loading logic
    # This is a placeholder - would need to implement actual loading
    return config


# Initialize the global configuration
config = load_config()