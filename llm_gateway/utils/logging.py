"""Enhanced logging using Rich."""
import logging
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from llm_gateway.config import config
from llm_gateway.constants import EMOJI_MAP, LogLevel


# Create custom Rich theme
RICH_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "critical": "red reverse",
    "debug": "dim",
    "success": "green",
    "provider": "blue",
    "token": "bright_black",
    "cost": "yellow",
    "time": "bright_black"
})


# Create console
console = Console(theme=RICH_THEME, highlight=True)


class GatewayLogger:
    """Enhanced logger with Rich formatting and emojis."""
    
    def __init__(self, name: str, level: Union[str, int] = None):
        """Initialize the logger.
        
        Args:
            name: Logger name
            level: Log level
        """
        self.name = name
        self.emoji_enabled = config.logging.emoji_enabled
        
        # Set up Rich handler
        self.console = console
        rich_handler = RichHandler(
            console=self.console,
            rich_tracebacks=True,
            markup=True,
            show_time=config.logging.show_timestamps,
            show_path=False,
            enable_link_path=False,
        )
        
        # Set up file handler if configured
        handlers = [rich_handler]
        if config.logging.file:
            log_path = Path(config.logging.file)
            # Create directory if it doesn't exist
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        
        # Configure the logger
        self.logger = logging.getLogger(name)
        
        # Set level
        level = level or config.logging.level
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
        
        # Remove existing handlers
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            
        # Add our handlers
        for handler in handlers:
            self.logger.addHandler(handler)
    
    def _format_message(self, message: str, emoji_key: Optional[str] = None, **kwargs) -> str:
        """Format log message with emoji and optional metadata.
        
        Args:
            message: The log message
            emoji_key: Key for emoji lookup
            **kwargs: Additional context data to include
            
        Returns:
            Formatted message
        """
        # Add emoji if enabled and available
        emoji = ""
        if self.emoji_enabled and emoji_key and emoji_key in EMOJI_MAP:
            emoji = f"{EMOJI_MAP[emoji_key]} "
            
        # Format message
        formatted_message = f"{emoji}{message}"
        
        # Add any additional context as key=value pairs
        if kwargs:
            context_pairs = []
            for key, value in kwargs.items():
                if key == "cost" and isinstance(value, (int, float)):
                    # Format cost as currency
                    context_pairs.append(f"[cost]${value:.6f}[/cost]")
                elif key == "time" and isinstance(value, (int, float)):
                    # Format time in seconds
                    context_pairs.append(f"[time]{value:.2f}s[/time]")
                elif key == "tokens" and isinstance(value, (int, dict)):
                    # Format token count
                    if isinstance(value, dict):
                        tokens_str = f"in={value.get('input', 0)} out={value.get('output', 0)}"
                    else:
                        tokens_str = str(value)
                    context_pairs.append(f"[token]tokens: {tokens_str}[/token]")
                elif key == "provider":
                    # Format provider name with color
                    context_pairs.append(f"[provider]{value}[/provider]")
                else:
                    # General context
                    context_pairs.append(f"{key}={value}")
            
            if context_pairs:
                formatted_message = f"{formatted_message} " + " ".join(context_pairs)
                
        return formatted_message
    
    def debug(self, message: str, emoji_key: Optional[str] = "debug", **kwargs):
        """Log a debug message.
        
        Args:
            message: The log message
            emoji_key: Key for emoji lookup
            **kwargs: Additional context data to include
        """
        self.logger.debug(self._format_message(message, emoji_key, **kwargs))
    
    def info(self, message: str, emoji_key: Optional[str] = "info", **kwargs):
        """Log an info message.
        
        Args:
            message: The log message
            emoji_key: Key for emoji lookup
            **kwargs: Additional context data to include
        """
        self.logger.info(self._format_message(message, emoji_key, **kwargs))
    
    def warning(self, message: str, emoji_key: Optional[str] = "warning", **kwargs):
        """Log a warning message.
        
        Args:
            message: The log message
            emoji_key: Key for emoji lookup
            **kwargs: Additional context data to include
        """
        self.logger.warning(self._format_message(message, emoji_key, **kwargs))
    
    def error(self, message: str, emoji_key: Optional[str] = "error", **kwargs):
        """Log an error message.
        
        Args:
            message: The log message
            emoji_key: Key for emoji lookup
            **kwargs: Additional context data to include
        """
        self.logger.error(self._format_message(message, emoji_key, **kwargs))
    
    def critical(self, message: str, emoji_key: Optional[str] = "critical", **kwargs):
        """Log a critical message.
        
        Args:
            message: The log message
            emoji_key: Key for emoji lookup
            **kwargs: Additional context data to include
        """
        self.logger.critical(self._format_message(message, emoji_key, **kwargs))
    
    def success(self, message: str, emoji_key: Optional[str] = "success", **kwargs):
        """Log a success message (alias for info with success styling).
        
        Args:
            message: The log message
            emoji_key: Key for emoji lookup
            **kwargs: Additional context data to include
        """
        formatted = self._format_message(message, emoji_key, **kwargs)
        self.logger.info(f"[success]{formatted}[/success]")


@lru_cache(maxsize=32)
def get_logger(name: str) -> GatewayLogger:
    """Get a logger instance with caching.
    
    Args:
        name: Logger name
        
    Returns:
        GatewayLogger instance
    """
    return GatewayLogger(name)