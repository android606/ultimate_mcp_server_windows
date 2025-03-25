"""Central configuration management for LLM Gateway."""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default=os.environ.get("LOG_LEVEL", "INFO"))
    file: Optional[str] = Field(default=os.environ.get("LOG_FILE", None))
    use_rich: bool = Field(default=os.environ.get("USE_RICH_LOGGING", "true").lower() == "true")
    show_timestamps: bool = Field(default=True)
    emoji_enabled: bool = Field(default=True)
    

class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = Field(default=os.environ.get("CACHE_ENABLED", "true").lower() == "true")
    ttl: int = Field(default=int(os.environ.get("CACHE_TTL", "86400")))  # 24 hours
    directory: Optional[str] = Field(default=os.environ.get("CACHE_DIR", None))
    max_entries: int = Field(default=int(os.environ.get("CACHE_MAX_ENTRIES", "10000")))
    fuzzy_match: bool = Field(default=os.environ.get("CACHE_FUZZY_MATCH", "true").lower() == "true")


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
        api_key=os.environ.get("OPENAI_API_KEY"),
        default_model=os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
        max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "8192")),
    ))
    anthropic: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        enabled=True,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        default_model=os.environ.get("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-haiku-latest"),
        max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "200000")),
    ))
    deepseek: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        enabled=True,
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        default_model=os.environ.get("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
        max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "8192")),
    ))
    gemini: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        enabled=True,
        api_key=os.environ.get("GEMINI_API_KEY"),
        default_model=os.environ.get("GEMINI_DEFAULT_MODEL", "gemini-2.0-flash-lite"),
        max_tokens=int(os.environ.get("GEMINI_MAX_TOKENS", "8192")),
    ))


class ServerConfig(BaseModel):
    """Server configuration."""
    name: str = Field(default=os.environ.get("SERVER_NAME", "LLM Gateway"))
    port: int = Field(default=int(os.environ.get("SERVER_PORT", "8000")))
    host: str = Field(default=os.environ.get("SERVER_HOST", "127.0.0.1"))
    workers: int = Field(default=int(os.environ.get("SERVER_WORKERS", "1")))
    debug: bool = Field(default=os.environ.get("SERVER_DEBUG", "false").lower() == "true")


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