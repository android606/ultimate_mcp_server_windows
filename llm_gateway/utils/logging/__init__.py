"""
Gateway Logging Package.

This package provides enhanced logging capabilities with rich formatting,
progress tracking, and console output for the Gateway system.
"""

import logging
import logging.handlers
from typing import Dict, Any, Optional, List

# Import Rich-based console
# Adjusted imports to be relative within the new structure
from .console import (
    console,
    create_progress,
    status,
    print_panel,
    print_syntax,
    print_table,
    print_tree,
    print_json,
    live_display,
    get_rich_console # Added missing import used in server.py LOGGING_CONFIG
)

# Import logger and related utilities
from .logger import (
    Logger,
    debug,
    info,
    success,
    warning,
    error,
    critical,
    section,
)

# Import emojis
from .emojis import (
    get_emoji,
    INFO,
    DEBUG,
    WARNING,
    ERROR,
    CRITICAL,
    SUCCESS,
    RUNNING,
    COMPLETED,
    FAILED,
)

# Import panels
from .panels import (
    HeaderPanel,
    ResultPanel,
    InfoPanel,
    WarningPanel,
    ErrorPanel,
    ToolOutputPanel,
    CodePanel,
    display_header,
    display_results,
    display_info,
    display_warning,
    display_error,
    display_tool_output,
    display_code,
)

# Import progress tracking
from .progress import (
    GatewayProgress,
    track,
)

# Import formatters and handlers
from .formatter import (
    GatewayLogRecord,
    SimpleLogFormatter,
    DetailedLogFormatter,
    RichLoggingHandler,
    create_rich_console_handler # Added missing import used in server.py LOGGING_CONFIG
)

# Create a global logger instance for importing
logger = Logger("llm_gateway")

# Removed configure_root_logger, initialize_logging, set_log_level functions
# Logging is now configured via dictConfig in main.py (or server.py equivalent)

def get_logger(name: str) -> Logger:
    """Get a logger for a specific component.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    # Use the new base name for sub-loggers if needed, or keep original logic
    # return Logger(f"llm_gateway.{name}") # Option 1: Prefix with base name
    return Logger(name) # Option 2: Keep original name logic

def capture_logs(level: Optional[str] = None) -> "LogCapture":
    """Create a context manager to capture logs.
    
    Args:
        level: Minimum log level to capture
        
    Returns:
        Log capture context manager
    """
    return LogCapture(level)

# Log capturing for testing
class LogCapture:
    """Context manager for capturing logs."""
    
    def __init__(self, level: Optional[str] = None):
        """Initialize the log capture.
        
        Args:
            level: Minimum log level to capture
        """
        self.level = level
        self.level_num = getattr(logging, self.level.upper(), 0) if self.level else 0
        self.logs: List[Dict[str, Any]] = []
        self.handler = self._create_handler()
    
    def _create_handler(self) -> logging.Handler:
        """Create a handler to capture logs.
        
        Returns:
            Log handler
        """
        class CaptureHandler(logging.Handler):
            def __init__(self, capture):
                super().__init__()
                self.capture = capture
            
            def emit(self, record):
                # Skip if record level is lower than minimum
                if record.levelno < self.capture.level_num:
                    return
                
                # Add log record to captured logs
                self.capture.logs.append({
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "name": record.name,
                    "time": record.created,
                    "file": record.pathname,
                    "line": record.lineno,
                })
        
        return CaptureHandler(self)
    
    def __enter__(self) -> "LogCapture":
        """Enter the context manager.
        
        Returns:
            Self
        """
        # Add handler to root logger
        # Use the project's logger name
        logging.getLogger("llm_gateway").addHandler(self.handler)
        # Consider adding to the absolute root logger as well if needed
        # logging.getLogger().addHandler(self.handler) 
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        # Remove handler from root logger
        logging.getLogger("llm_gateway").removeHandler(self.handler)
        # logging.getLogger().removeHandler(self.handler)
    
    def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get captured logs, optionally filtered by level.
        
        Args:
            level: Filter logs by level
            
        Returns:
            List of log records
        """
        if not level:
            return self.logs
        
        level_num = getattr(logging, level.upper(), 0)
        return [log for log in self.logs if getattr(logging, log["level"], 0) >= level_num]
    
    def get_messages(self, level: Optional[str] = None) -> List[str]:
        """Get captured log messages, optionally filtered by level.
        
        Args:
            level: Filter logs by level
            
        Returns:
            List of log messages
        """
        return [log["message"] for log in self.get_logs(level)]
    
    def contains(self, text: str, level: Optional[str] = None) -> bool:
        """Check if any log message contains the given text.
        
        Args:
            text: Text to search for
            level: Optional level filter
            
        Returns:
            True if text is found in any message
        """
        return any(text in msg for msg in self.get_messages(level))

__all__ = [
    # Console
    "console",
    "create_progress",
    "status",
    "print_panel",
    "print_syntax",
    "print_table",
    "print_tree",
    "print_json",
    "live_display",
    
    # Logger and utilities
    "logger",
    "Logger",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "critical",
    "section",
    "get_logger",
    "capture_logs",
    "LogCapture",
    
    # Emojis
    "get_emoji",
    "INFO",
    "DEBUG",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "SUCCESS",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    
    # Panels
    "HeaderPanel",
    "ResultPanel",
    "InfoPanel",
    "WarningPanel",
    "ErrorPanel",
    "ToolOutputPanel",
    "CodePanel",
    "display_header",
    "display_results",
    "display_info",
    "display_warning",
    "display_error",
    "display_tool_output",
    "display_code",
    
    # Progress tracking
    "GatewayProgress",
    "track",
    
    # Formatters and handlers
    "GatewayLogRecord",
    "SimpleLogFormatter",
    "DetailedLogFormatter",
    "RichLoggingHandler",
    "create_rich_console_handler",
] 