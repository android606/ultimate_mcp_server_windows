"""
Main Logger class for Gateway.

This module provides the central Logger class that integrates all Gateway logging
functionality with a beautiful, informative interface.
"""
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Console

# Use relative imports for utils within the same package
from .console import console
from .emojis import get_emoji
from .formatter import (
    DetailedLogFormatter,
    RichLoggingHandler,
    SimpleLogFormatter,
)
from .panels import (
    CodePanel,
    ErrorPanel,
    HeaderPanel,
    InfoPanel,
    ResultPanel,
    ToolOutputPanel,
    WarningPanel,
)
from .progress import GatewayProgress

# Set up standard Python logging with our custom handler
# Logging configuration is handled externally via dictConfig

class Logger:
    """Advanced logger for Gateway with rich formatting and progress tracking."""
    
    def __init__(
        self,
        name: str = "llm_gateway", # Default logger name changed
        console: Optional[Console] = None,
        level: str = "info",
        show_timestamps: bool = True,
        component: Optional[str] = None,
        capture_output: bool = False,
    ):
        """Initialize the logger.
        
        Args:
            name: Logger name
            console: Rich console to use
            level: Initial log level
            show_timestamps: Whether to show timestamps in logs
            component: Default component name
            capture_output: Whether to capture and store log output
        """
        self.name = name
        # Use provided console or get global console, defaulting to stderr console
        if console is not None:
            self.console = console
        else:
            global_console = globals().get("console")
            if global_console is not None:
                self.console = global_console
            else:
                self.console = Console(file=sys.stderr)
                
        self.level = level.lower()
        self.show_timestamps = show_timestamps
        self.component = component
        self.capture_output = capture_output
        
        # Create a standard Python logger
        self.python_logger = logging.getLogger(name)
        
        # Set up formatters
        self.simple_formatter = SimpleLogFormatter(show_time=show_timestamps, show_level=True, show_component=True)
        self.detailed_formatter = DetailedLogFormatter(show_time=show_timestamps, show_level=True, show_component=True)
        
        # Progress tracker
        self.progress = GatewayProgress(console=self.console)
        
        # Output capture if enabled
        self.captured_logs = [] if capture_output else None
        
        # Restore propagation to allow messages to reach root handlers
        # Make sure this is True so logs configured via dictConfig are passed up
        self.python_logger.propagate = True 
        
        # Set initial log level on the Python logger instance
        # Note: The effective level will be determined by the handler/root config
        self.set_level(level)
    
    def set_level(self, level: str) -> None:
        """Set the log level.
        
        Args:
            level: Log level (debug, info, warning, error, critical)
        """
        level = level.lower()
        self.level = level # Store the intended level for should_log checks
        
        # Map to Python logging levels
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        
        python_level = level_map.get(level, logging.INFO)
        # Set level on the logger itself. Handlers might have their own levels.
        self.python_logger.setLevel(python_level)
    
    def get_level(self) -> str:
        """Get the current log level.
        
        Returns:
            Current log level
        """
        # Return the Python logger's effective level
        effective_level_num = self.python_logger.getEffectiveLevel()
        level_map_rev = {
            logging.DEBUG: "debug",
            logging.INFO: "info",
            logging.WARNING: "warning",
            logging.ERROR: "error",
            logging.CRITICAL: "critical",
        }
        return level_map_rev.get(effective_level_num, "info")

    
    def should_log(self, level: str) -> bool:
        """Check if a message at the given level should be logged based on Python logger's effective level.
        
        Args:
            level: Log level to check
            
        Returns:
            Whether messages at this level should be logged
        """
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "success": logging.INFO, # Map success to info for level check
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        message_level_num = level_map.get(level.lower(), logging.INFO)
        return self.python_logger.isEnabledFor(message_level_num)

    
    def _log(
        self,
        level: str,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        emoji: Optional[str] = None,
        emoji_key: Optional[str] = None,  # Add emoji_key parameter
        context: Optional[Dict[str, Any]] = None,
        use_detailed_formatter: bool = False, # This arg seems unused now?
        exception_info: Optional[Union[bool, Tuple]] = None,
        stack_info: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Internal method to handle logging via the standard Python logging mechanism.
        
        Args:
            level: Log level
            message: Log message
            component: Gateway component (core, composite, analysis, etc.)
            operation: Operation being performed
            emoji: Custom emoji override
            emoji_key: Key to look up emoji from emoji map (alternative to emoji)
            context: Additional contextual data
            exception_info: Include exception info (True/False or tuple)
            stack_info: Include stack info
            extra: Dictionary passed as extra to logging framework
        """
        # Check if we should log at this level using standard Python logging check
        # No need for the custom should_log method here if using stdlib correctly
        
        # Map level name to Python level number
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "success": logging.INFO, # Log success as INFO
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        level_num = level_map.get(level.lower(), logging.INFO)

        if not self.python_logger.isEnabledFor(level_num):
            return
            
        # Use default component if not provided
        component = component or self.component
        
        # If emoji_key is provided, use it to determine emoji
        if emoji_key and not emoji:
            emoji = get_emoji("operation", emoji_key)
            if emoji == "❓":  # If operation emoji not found
                # Try level emojis
                from .emojis import LEVEL_EMOJIS
                emoji = LEVEL_EMOJIS.get(emoji_key, "❓")
        
        # Prepare 'extra' dict for LogRecord
        log_extra = {} if extra is None else extra.copy()  # Create a copy to avoid modifying the original
        
        # Remove any keys that conflict with built-in LogRecord attributes
        for reserved_key in ['message', 'asctime', 'exc_info', 'exc_text', 'lineno', 'funcName', 'created', 'levelname', 'levelno']:
            if reserved_key in log_extra:
                del log_extra[reserved_key]
                
        # Add our custom keys
        log_extra['component'] = component
        log_extra['operation'] = operation
        log_extra['custom_emoji'] = emoji
        log_extra['log_context'] = context # Use a different key to avoid collision
        log_extra['gateway_level'] = level # Pass the original level name if needed by formatter
        
        # Handle exception info
        exc_info = None
        if exception_info:
            if isinstance(exception_info, bool):
                exc_info = sys.exc_info()
            else:
                exc_info = exception_info # Assume it's a valid tuple

        # Log through Python's logging system
        self.python_logger.log(
            level=level_num,
            msg=message,
            exc_info=exc_info,
            stack_info=stack_info,
            extra=log_extra
        )
            
        # Capture if enabled
        if self.captured_logs is not None:
            self.captured_logs.append({
                "level": level,
                "message": message,
                "component": component,
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "context": context,
            })

    # --- Standard Logging Methods --- 

    def debug(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        emoji_key: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log a debug message."""
        self._log("debug", message, component, operation, context=context, emoji_key=emoji_key, extra=kwargs)

    def info(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        emoji_key: Optional[str] = None,
         **kwargs
    ) -> None:
        """Log an info message."""
        self._log("info", message, component, operation, context=context, emoji_key=emoji_key, extra=kwargs)

    def success(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        emoji_key: Optional[str] = None,
         **kwargs
    ) -> None:
        """Log a success message."""
        self._log("success", message, component, operation, context=context, emoji_key=emoji_key, extra=kwargs)

    def warning(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        emoji_key: Optional[str] = None,
        # details: Optional[List[str]] = None, # Details handled by panel methods
         **kwargs
    ) -> None:
        """Log a warning message."""
        self._log("warning", message, component, operation, context=context, emoji_key=emoji_key, extra=kwargs)

    def error(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        emoji_key: Optional[str] = None,
        # error_code: Optional[str] = None,
        # resolution_steps: Optional[List[str]] = None,
         **kwargs
    ) -> None:
        """Log an error message."""
        # Get the exception info tuple if an exception was provided
        exc_info = None
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)
        elif 'exc_info' in kwargs:
            exc_info = kwargs.pop('exc_info')  # Remove from kwargs to prevent conflicts
        
        self._log("error", message, component, operation, context=context, 
                 exception_info=exc_info, emoji_key=emoji_key, extra=kwargs)

    def critical(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        emoji_key: Optional[str] = None,
        # error_code: Optional[str] = None, # Pass via context or kwargs
         **kwargs
    ) -> None:
        """Log a critical message."""
        # Get the exception info tuple if an exception was provided
        exc_info = None
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)
        elif 'exc_info' in kwargs:
            exc_info = kwargs.pop('exc_info')  # Remove from kwargs to prevent conflicts
        
        self._log("critical", message, component, operation, context=context, 
                 exception_info=exc_info, emoji_key=emoji_key, extra=kwargs)

    # --- Rich Display Methods --- 
    # These methods use the console directly or generate renderables
    # They might bypass the standard logging flow, or log additionally

    def operation(
        self,
        operation: str,
        message: str,
        component: Optional[str] = None,
        level: str = "info",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log an operation-specific message.
        
        Args:
            operation: Operation name
            message: Log message
            component: Gateway component
            level: Log level (default: info)
            context: Additional context
            **kwargs: Extra fields for logging
        """
        self._log(level, message, component, operation, context=context, extra=kwargs)

    def tool(
        self,
        tool: str,
        command: str,
        output: str,
        status: str = "success",
        duration: Optional[float] = None,
        component: Optional[str] = None,
        **kwargs
    ) -> None:
        """Display formatted output from a tool.
        
        Args:
            tool: Name of the tool
            command: Command executed
            output: Tool output
            status: Execution status (success, error)
            duration: Execution duration in seconds
            component: Gateway component
            **kwargs: Extra fields for logging
        """
        # Optionally log the event
        log_level = "error" if status == "error" else "debug"
        log_message = f"Tool '{tool}' finished (status: {status})"
        log_context = {"command": command, "output_preview": output[:100] + "..." if len(output) > 100 else output}
        if duration is not None:
            log_context["duration_s"] = duration
        self._log(log_level, log_message, component, operation=f"tool.{tool}", context=log_context, extra=kwargs)

        # Display the panel directly on the console
        panel = ToolOutputPanel(tool, command, output, status, duration)
        self.console.print(panel)

    def code(
        self,
        code: str,
        language: str = "python",
        title: Optional[str] = None,
        line_numbers: bool = True,
        highlight_lines: Optional[List[int]] = None,
        message: Optional[str] = None,
        component: Optional[str] = None,
        level: str = "debug",
        **kwargs
    ) -> None:
        """Display a code block.

        Args:
            code: Code string
            language: Language for syntax highlighting
            title: Optional title for the panel
            line_numbers: Show line numbers
            highlight_lines: Lines to highlight
            message: Optional message to log alongside displaying the code
            component: Gateway component
            level: Log level for the optional message (default: debug)
            **kwargs: Extra fields for logging
        """
        if message:
            self._log(level, message, component, context={"code_preview": code[:100] + "..." if len(code) > 100 else code}, extra=kwargs)

        # Display the panel directly
        panel = CodePanel(code, language, title, line_numbers, highlight_lines)
        self.console.print(panel)

    def display_results(
        self,
        title: str,
        results: Union[List[Dict[str, Any]], Dict[str, Any]],
        status: str = "success",
        component: Optional[str] = None,
        show_count: bool = True,
        compact: bool = False,
        message: Optional[str] = None,
        level: str = "info",
        **kwargs
    ) -> None:
        """Display results in a formatted panel.

        Args:
            title: Panel title
            results: Results data
            status: Status (success, warning, error)
            component: Gateway component
            show_count: Show count in title
            compact: Use compact format
            message: Optional message to log
            level: Log level for the optional message (default: info)
            **kwargs: Extra fields for logging
        """
        if message:
            self._log(level, message, component, context={"result_count": len(results) if isinstance(results, list) else 1, "status": status}, extra=kwargs)
            
        # Display the panel directly
        panel = ResultPanel(title, results, status, component, show_count, compact)
        self.console.print(panel)

    def section(
        self,
        title: str,
        subtitle: Optional[str] = None,
        component: Optional[str] = None,
    ) -> None:
        """Display a section header.

        Args:
            title: Section title
            subtitle: Optional subtitle
            component: Gateway component
        """
        # This is purely presentational, doesn't log typically
        panel = HeaderPanel(title, subtitle, component=component)
        self.console.print(panel)

    def info_panel(
        self,
        title: str,
        content: Union[str, List[str], Dict[str, Any]],
        icon: Optional[str] = None,
        style: str = "info",
        component: Optional[str] = None,
    ) -> None:
        """Display an informational panel.

        Args:
            title: Panel title
            content: Panel content
            icon: Optional icon
            style: Panel style
            component: Gateway component
        """
        # Could log the title/content summary if desired
        # self._log("info", f"Displaying info panel: {title}", component)
        panel = InfoPanel(title, content, icon, style)
        self.console.print(panel)

    def warning_panel(
        self,
        title: Optional[str] = None,
        message: str = "",
        details: Optional[List[str]] = None,
        component: Optional[str] = None,
    ) -> None:
        """Display a warning panel.

        Args:
            title: Optional panel title
            message: Warning message
            details: Optional list of detail strings
            component: Gateway component
        """
        # Log the warning separately
        log_title = title if title else "Warning"
        self.warning(f"{log_title}: {message}", component, context={"details": details})

        # Display the panel directly
        panel = WarningPanel(title, message, details)
        self.console.print(panel)

    def error_panel(
        self,
        title: Optional[str] = None,
        message: str = "",
        details: Optional[str] = None,
        resolution_steps: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        component: Optional[str] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        """Display an error panel.

        Args:
            title: Optional panel title
            message: Error message
            details: Optional detail string (e.g., traceback)
            resolution_steps: Optional list of resolution steps
            error_code: Optional error code
            component: Gateway component
            exception: Associated exception (for logging traceback)
        """
        # Log the error separately
        log_title = title if title else "Error"
        log_context = {
            "details": details,
            "resolution": resolution_steps,
            "error_code": error_code,
        }
        self.error(f"{log_title}: {message}", component, context=log_context, exception=exception)

        # Display the panel directly
        panel = ErrorPanel(title, message, details, resolution_steps, error_code)
        self.console.print(panel)

    # --- Context Managers & Decorators --- 

    @contextmanager
    def time_operation(
        self,
        operation: str,
        component: Optional[str] = None,
        level: str = "info",
        start_message: Optional[str] = "Starting {operation}...",
        end_message: Optional[str] = "Finished {operation} in {duration:.2f}s",
        **kwargs
    ):
        """Context manager to time an operation and log start/end messages.
        
        Args:
            operation: Name of the operation
            component: Gateway component
            level: Log level for messages (default: info)
            start_message: Message format string for start log (can be None)
            end_message: Message format string for end log (can be None)
            **kwargs: Extra fields for logging
        
        Yields:
            None
        """
        start_time = time.monotonic()
        if start_message:
            self._log(level, start_message.format(operation=operation), component, operation, extra=kwargs)
            
        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            if end_message:
                self._log(level, end_message.format(operation=operation, duration=duration), component, operation, context={"duration_s": duration}, extra=kwargs)

    def track(
        self,
        iterable: Any,
        description: str,
        name: Optional[str] = None,
        total: Optional[int] = None,
        parent: Optional[str] = None,
        # Removed component - handled by logger instance
    ) -> Any:
        """Track progress over an iterable using the logger's progress tracker.
        
        Args:
            iterable: Iterable to track
            description: Description of the task
            name: Optional task name (defaults to description)
            total: Optional total number of items
            parent: Optional parent task name
            
        Returns:
            The iterable wrapped with progress tracking
        """
        return self.progress.track(iterable, description, name, total, parent)

    @contextmanager
    def task(
        self,
        description: str,
        name: Optional[str] = None,
        total: int = 100,
        parent: Optional[str] = None,
        # Removed component - handled by logger instance
        autostart: bool = True,
    ):
        """Context manager for a single task with progress tracking.
        
        Args:
            description: Description of the task
            name: Optional task name (defaults to description)
            total: Total steps/work units for the task
            parent: Optional parent task name
            autostart: Start the progress display immediately (default: True)
        
        Yields:
            GatewayProgress instance for updating the task
        """
        with self.progress.task(description, name, total, parent, autostart) as task_context:
             yield task_context

    @contextmanager
    def catch_and_log(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        reraise: bool = True,
        level: str = "error",
        message: str = "An error occurred during {operation}",
    ):
        """Context manager to catch exceptions and log them.
        
        Args:
            component: Component name
            operation: Operation name
            reraise: Whether to re-raise the exception after logging (default: True)
            level: Log level for the error (default: error)
            message: Message format string for the error log
        
        Yields:
            None
        """
        component = component or self.component
        operation = operation or "operation"
        try:
            yield
        except Exception:
            log_msg = message.format(operation=operation)
            self._log(level, log_msg, component, operation, exception_info=True)
            if reraise:
                raise

    def log_call(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        level: str = "debug",
        log_args: bool = True,
        log_result: bool = False,
        log_exceptions: bool = True,
    ):
        """Decorator to log function calls.
        
        Args:
            component: Component name
            operation: Operation name (defaults to function name)
            level: Log level for entry/exit messages (default: debug)
            log_args: Whether to log function arguments (default: True)
            log_result: Whether to log function result (default: False)
            log_exceptions: Whether to log exceptions (default: True)
        """
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Determine operation name
                op_name = operation or func.__name__
                comp_name = component or self.component
                
                # Log entry
                entry_msg = f"Entering {op_name}..."
                context = {}
                if log_args:
                    # Be careful logging args, could contain sensitive info or be large
                    try:
                        arg_repr = f"args={args!r}, kwargs={kwargs!r}"
                        context['args'] = arg_repr[:200] + '...' if len(arg_repr) > 200 else arg_repr
                    except Exception:
                        context['args'] = "<Could not represent args>"
                        
                self._log(level, entry_msg, comp_name, op_name, context=context)
                
                start_time = time.monotonic()
                try:
                    result = func(*args, **kwargs)
                    duration = time.monotonic() - start_time
                    
                    # Log exit
                    exit_msg = f"Exiting {op_name} (duration: {duration:.3f}s)"
                    exit_context = {"duration_s": duration}
                    if log_result:
                        try:
                            res_repr = repr(result)
                            exit_context['result'] = res_repr[:200] + '...' if len(res_repr) > 200 else res_repr
                        except Exception:
                           exit_context['result'] = "<Could not represent result>"
                            
                    self._log(level, exit_msg, comp_name, op_name, context=exit_context)
                    return result
                    
                except Exception as e:
                    duration = time.monotonic() - start_time
                    if log_exceptions:
                        exc_level = "error" # Always log exceptions as error?
                        exc_msg = f"Exception in {op_name} after {duration:.3f}s: {e}"
                        exc_context = {"duration_s": duration}
                        if log_args: # Include args context if available
                           exc_context.update(context)
                           
                        self._log(exc_level, exc_msg, comp_name, op_name, exception_info=True, context=exc_context)
                    raise
                    
            return wrapper
        return decorator

    # --- Startup/Shutdown Methods --- 

    def startup(
        self,
        version: str,
        component: Optional[str] = None,
        mode: str = "standard",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log server startup information.
        
        Args:
            version: Server version
            component: Component name (usually None for global startup)
            mode: Performance mode
            context: Additional startup context
            **kwargs: Extra fields for logging
        """
        message = f"Starting Server (Version: {version}, Mode: {mode})"
        emoji = get_emoji("system", "startup")
        self.info(message, component, operation="startup", emoji=emoji, context=context, **kwargs)

    def shutdown(
        self,
        component: Optional[str] = None,
        duration: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log server shutdown information.
        
        Args:
            component: Component name
            duration: Optional uptime duration
            context: Additional shutdown context
            **kwargs: Extra fields for logging
        """
        message = "Server Shutting Down"
        if duration is not None:
            message += f" (Uptime: {duration:.2f}s)"
        emoji = get_emoji("system", "shutdown")
        self.info(message, component, operation="shutdown", emoji=emoji, context=context, **kwargs)

# --- Global Convenience Functions --- 
# These use the global 'logger' instance created in __init__.py

# At the global level, declare logger as None initially
logger = None  

def get_logger(name: str) -> Logger:
    """Get a logger for a specific component.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    # Initialize the global logger if needed
    global logger
    if logger is None:
        logger = Logger(name)
    
    # Return a new logger with the requested name
    return Logger(name)

# Helper functions for global usage
def debug(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    emoji_key: Optional[str] = None,
    **kwargs
) -> None:
    """Forward to default logger's debug method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)
    
    logger.debug(message, component, operation, context, emoji_key=emoji_key, **kwargs)

def info(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    emoji_key: Optional[str] = None,
    **kwargs
) -> None:
    """Forward to default logger's info method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)
    
    logger.info(message, component, operation, context, emoji_key=emoji_key, **kwargs)

def success(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    emoji_key: Optional[str] = None,
    **kwargs
) -> None:
    """Forward to default logger's success method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)
    
    logger.success(message, component, operation, context, emoji_key=emoji_key, **kwargs)

def warning(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    emoji_key: Optional[str] = None,
    # details: Optional[List[str]] = None,
    **kwargs
) -> None:
    """Forward to default logger's warning method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)
    
    logger.warning(message, component, operation, context, emoji_key=emoji_key, **kwargs)

def error(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
    emoji_key: Optional[str] = None,
    # error_code: Optional[str] = None,
    # resolution_steps: Optional[List[str]] = None,
    **kwargs
) -> None:
    """Forward to default logger's error method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)
    
    # Handle exc_info specially to prevent conflicts
    exc_info = kwargs.pop('exc_info', None) if 'exc_info' in kwargs else None
    
    logger.error(message, component, operation, context, 
                exception=exception, emoji_key=emoji_key, 
                **{**kwargs, 'exc_info': exc_info} if exc_info is not None else kwargs)

def critical(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
    emoji_key: Optional[str] = None,
    # error_code: Optional[str] = None,
    **kwargs
) -> None:
    """Forward to default logger's critical method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)
    
    # Handle exc_info specially to prevent conflicts
    exc_info = kwargs.pop('exc_info', None) if 'exc_info' in kwargs else None
    
    logger.critical(message, component, operation, context, 
                   exception=exception, emoji_key=emoji_key, 
                   **{**kwargs, 'exc_info': exc_info} if exc_info is not None else kwargs)

def section(
    title: str,
    subtitle: Optional[str] = None,
    component: Optional[str] = None,
) -> None:
    """Display a section header using the global logger's console."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)
    
    logger.section(title, subtitle, component)

# Example Usage (if run directly)
if __name__ == '__main__':
    # Example of how the logger might be configured and used
    
    # Normally configuration happens via dictConfig in main entry point
    # For standalone testing, we can add a handler manually
    test_logger = Logger("test_logger", level="debug") # Create instance
    test_logger.python_logger.addHandler(RichLoggingHandler(console=console))
    # Need to prevent propagation if manually adding handler here for test
    test_logger.python_logger.propagate = False 
    
    test_logger.section("Initialization", "Setting up components")
    test_logger.startup(version="1.0.0", mode="test")
    
    test_logger.debug("This is a debug message", component="core", operation="setup")
    test_logger.info("This is an info message", component="api")
    test_logger.success("Operation completed successfully", component="worker", operation="process_data")
    test_logger.warning("Something looks suspicious", component="cache", context={"key": "user:123"})
    
    try:
        x = 1 / 0
    except ZeroDivisionError as e:
        test_logger.error("An error occurred", component="math", operation="divide", exception=e)
        
    test_logger.critical("System unstable!", component="core", context={"reason": "disk full"})

    test_logger.info_panel("Configuration", {"host": "localhost", "port": 8013}, component="core")
    test_logger.warning_panel("Cache Alert", "Cache nearing capacity", details=["Size: 95MB", "Limit: 100MB"], component="cache")
    test_logger.error_panel("DB Connection Failed", "Could not connect to database", details="Connection timed out after 5s", resolution_steps=["Check DB server status", "Verify credentials"], error_code="DB500", component="db")

    test_logger.tool("grep", "grep 'error' log.txt", "line 1: error found\nline 5: error processing", status="success", duration=0.5, component="analysis")
    test_logger.code("def hello():\n  print('Hello')", language="python", title="Example Code", component="docs")

    with test_logger.time_operation("long_process", component="worker"):
        time.sleep(0.5)
        
    with test_logger.task("Processing items", total=10) as p:
        for _i in range(10):
            time.sleep(0.05)
            p.update_task(p.current_task_id, advance=1) # Assuming task context provides task_id

    @test_logger.log_call(component="utils", log_result=True)
    def add_numbers(a, b):
        return a + b
    
    add_numbers(5, 3)
    
    test_logger.shutdown(duration=123.45)

__all__ = [
    "critical",
    "debug",
    "error",
    "get_logger",
    "info",
    "logger",  # Add logger to exported names
    "warning",
] 