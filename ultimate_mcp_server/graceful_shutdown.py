"""
Graceful shutdown utilities for Ultimate MCP Server.

This module provides utilities to handle signals and gracefully terminate
the application with minimal error outputs during shutdown.
"""

import asyncio
import logging
import signal
import sys
from functools import partial
from typing import Callable, List, Optional

logger = logging.getLogger("ultimate_mcp_server.shutdown")

# Track registered shutdown handlers and state
_shutdown_handlers: List[Callable] = []
_shutdown_in_progress = False
_original_stderr = None


class QuietExit(Exception):
    """Special exception to trigger a clean exit with minimal error output"""
    pass


def redirect_stderr_during_shutdown():
    """Redirect stderr to /dev/null or NUL during shutdown to prevent error spam"""
    global _original_stderr
    
    if _original_stderr is None:
        _original_stderr = sys.stderr
        try:
            # Use null device, which is OS-dependent
            null_device = '/dev/null' if sys.platform != 'win32' else 'NUL'
            sys.stderr = open(null_device, 'w')
            logger.info("Redirected stderr to prevent shutdown noise")
        except Exception as e:
            # If we can't redirect, restore the original stderr
            sys.stderr = _original_stderr
            _original_stderr = None
            logger.warning(f"Failed to redirect stderr: {e}")


def register_shutdown_handler(handler: Callable) -> None:
    """Register a function to be called during graceful shutdown.
    
    Args:
        handler: Async or sync callable to execute during shutdown
    """
    if handler not in _shutdown_handlers:
        _shutdown_handlers.append(handler)
        logger.debug(f"Registered shutdown handler: {handler.__name__}")


def remove_shutdown_handler(handler: Callable) -> None:
    """Remove a previously registered shutdown handler.
    
    Args:
        handler: Previously registered handler to remove
    """
    if handler in _shutdown_handlers:
        _shutdown_handlers.remove(handler)
        logger.debug(f"Removed shutdown handler: {handler.__name__}")


async def _execute_shutdown_handlers():
    """Execute all registered shutdown handlers with error handling"""
    for handler in _shutdown_handlers:
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()
        except Exception as e:
            logger.error(f"Error in shutdown handler {handler.__name__}: {e}")


async def _handle_shutdown(sig_name):
    """Handle shutdown signals gracefully"""
    global _shutdown_in_progress
    
    if _shutdown_in_progress:
        logger.warning(f"Received {sig_name} while shutdown in progress - forcing exit")
        sys.exit(1)
        
    _shutdown_in_progress = True
    
    logger.info(f"Received {sig_name} signal. Initiating graceful shutdown...")
    
    # Print a clear message to the console
    print("\n[Graceful Shutdown] Closing connections and cleaning up...", file=_original_stderr or sys.stderr)
    
    try:
        await _execute_shutdown_handlers()
        logger.info("Graceful shutdown completed successfully")
        print("[Graceful Shutdown] Done.", file=_original_stderr or sys.stderr)
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}")
    finally:
        # Swallow errors during final exit by raising our custom exception
        raise QuietExit()


def setup_signal_handlers(loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """Set up signal handlers for graceful shutdown
    
    Args:
        loop: Optional asyncio event loop to use for scheduling shutdown
              If not provided, the current running loop will be used
    """
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("Cannot set up signal handlers - no running asyncio loop")
            return
            
    # Define handlers for common termination signals
    for sig_name, sig_num in [('SIGINT', signal.SIGINT), ('SIGTERM', signal.SIGTERM)]:
        try:
            # Create partial with the signal name for better logging
            handler = partial(_handle_shutdown, sig_name)
            
            # Add signal handler to event loop
            loop.add_signal_handler(
                sig_num,
                lambda h=handler: asyncio.create_task(h())
            )
            logger.info(f"Registered {sig_name} handler for graceful shutdown")
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning(f"Could not set up {sig_name} handler via asyncio (Windows?)")
            # Fall back to traditional signal handling on Windows
            if sys.platform == 'win32':
                signal.signal(sig_num, lambda s, f, h=handler: asyncio.create_task(h()))
                logger.info(f"Registered {sig_name} handler via signal module (Windows)")


def handle_quiet_exit():
    """Add custom excepthook to handle QuietExit without traceback"""
    original_excepthook = sys.excepthook
    
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, QuietExit):
            # Exit cleanly without showing the exception traceback
            sys.exit(0)
        else:
            # For other exceptions, use the original excepthook
            original_excepthook(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = custom_excepthook 