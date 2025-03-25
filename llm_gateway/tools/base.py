"""Base tool classes and decorators for LLM Gateway."""
import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union

from mcp import Tool

from llm_gateway.config import config
from llm_gateway.services.cache import with_cache
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class BaseToolMetrics:
    """Metrics tracking for tool execution."""
    
    def __init__(self):
        """Initialize metrics tracking."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_duration = 0.0
        self.min_duration = float('inf')
        self.max_duration = 0.0
        self.total_tokens = 0
        self.total_cost = 0.0
        
    def record_call(
        self,
        success: bool,
        duration: float,
        tokens: Optional[int] = None,
        cost: Optional[float] = None
    ) -> None:
        """Record metrics for a tool call.
        
        Args:
            success: Whether the call was successful
            duration: Duration of the call in seconds
            tokens: Number of tokens used (if applicable)
            cost: Cost of the call (if applicable)
        """
        self.total_calls += 1
        
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        
        if tokens is not None:
            self.total_tokens += tokens
            
        if cost is not None:
            self.total_cost += cost
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        if self.total_calls == 0:
            return {
                "total_calls": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
                "total_tokens": 0,
                "total_cost": 0.0,
            }
            
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls,
            "average_duration": self.total_duration / self.total_calls,
            "min_duration": self.min_duration if self.min_duration != float('inf') else 0.0,
            "max_duration": self.max_duration,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }


class BaseTool:
    """Base class for all LLM Gateway tools."""
    
    tool_name: str = "base_tool"
    description: str = "Base tool class for LLM Gateway."
    
    def __init__(self, mcp_server):
        """Initialize the tool.
        
        Args:
            mcp_server: MCP server instance
        """
        # If mcp_server is a Gateway instance, get the MCP object
        self.mcp = mcp_server.mcp if hasattr(mcp_server, 'mcp') else mcp_server
        self.logger = get_logger(f"tool.{self.tool_name}")
        self.metrics = BaseToolMetrics()
        
        # Register tools
        self._register_tools()
        
    def _register_tools(self):
        """Register tools with MCP server.
        
        Override this method in subclasses to register specific tools.
        """
        pass
        
    async def _wrap_with_metrics(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Wrap a function call with metrics tracking.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If function call fails
        """
        start_time = time.time()
        success = False
        tokens = None
        cost = None
        
        try:
            # Call function
            result = await func(*args, **kwargs)
            
            # Extract metrics if available
            if isinstance(result, dict):
                if "tokens" in result and isinstance(result["tokens"], dict):
                    tokens = result["tokens"].get("total")
                elif "total_tokens" in result:
                    tokens = result["total_tokens"]
                    
                cost = result.get("cost")
                
            success = True
            return result
            
        except Exception as e:
            self.logger.error(
                f"Tool execution failed: {str(e)}",
                emoji_key="error",
                tool=self.tool_name,
                exc_info=True
            )
            raise
            
        finally:
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_call(
                success=success,
                duration=duration,
                tokens=tokens,
                cost=cost
            )


def with_tool_metrics(func):
    """Decorator to add metrics tracking to a tool function.
    
    Args:
        func: Tool function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Get context from kwargs
        ctx = kwargs.get('ctx')
        
        start_time = time.time()
        success = False
        tokens = None
        cost = None
        
        try:
            # Call original function
            result = await func(self, *args, **kwargs)
            
            # Extract metrics if available
            if isinstance(result, dict):
                if "tokens" in result and isinstance(result["tokens"], dict):
                    tokens = result["tokens"].get("total")
                elif "total_tokens" in result:
                    tokens = result["total_tokens"]
                    
                cost = result.get("cost")
                
            success = True
            return result
            
        except Exception as e:
            tool_name = getattr(self, 'tool_name', func.__name__)
            logger.error(
                f"Tool execution failed: {tool_name}: {str(e)}",
                emoji_key="error",
                tool=tool_name,
                exc_info=True
            )
            raise
            
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            # Log execution stats
            tool_name = getattr(self, 'tool_name', func.__name__)
            logger.debug(
                f"Tool execution: {tool_name} ({'success' if success else 'failed'})",
                emoji_key="tool" if success else "error",
                tool=tool_name,
                time=duration,
                cost=cost
            )
            
            # Update metrics if metrics object exists
            if hasattr(self, 'metrics'):
                self.metrics.record_call(
                    success=success,
                    duration=duration,
                    tokens=tokens,
                    cost=cost
                )
                
    return wrapper


def with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_exceptions: List[Type[Exception]] = None
):
    """Decorator to add retry logic to a tool function.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay by on each retry
        retry_exceptions: List of exception types to retry on (defaults to all)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = retry_delay
            
            for attempt in range(max_retries + 1):
                try:
                    # Call original function
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    # Only retry on specified exceptions
                    if retry_exceptions and not any(
                        isinstance(e, exc_type) for exc_type in retry_exceptions
                    ):
                        raise
                        
                    last_exception = e
                    
                    # Log retry attempt
                    if attempt < max_retries:
                        logger.warning(
                            f"Tool execution failed, retrying ({attempt+1}/{max_retries}): {str(e)}",
                            emoji_key="warning",
                            tool=func.__name__,
                            attempt=attempt+1,
                            max_retries=max_retries,
                            delay=delay
                        )
                        
                        # Wait before retrying
                        await asyncio.sleep(delay)
                        
                        # Increase delay for next retry
                        delay *= backoff_factor
                    else:
                        # Log final failure
                        logger.error(
                            f"Tool execution failed after {max_retries} retries: {str(e)}",
                            emoji_key="error",
                            tool=func.__name__,
                            exc_info=True
                        )
                        
            # If we get here, all retries failed
            raise last_exception
                
        return wrapper
    return decorator
    

def register_tool(mcp_server, name=None, description=None, cache_ttl=None):
    """Decorator to register a function as an MCP tool with optional caching.
    
    Args:
        mcp_server: MCP server instance
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        cache_ttl: Cache TTL in seconds (if None, no caching)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        # Get function name and docstring
        tool_name = name or func.__name__
        tool_description = description or inspect.getdoc(func) or ""
        
        # Add metrics tracking
        func_with_metrics = with_tool_metrics(func)
        
        # Add caching if specified
        if cache_ttl is not None:
            func_with_cache = with_cache(ttl=cache_ttl)(func_with_metrics)
        else:
            func_with_cache = func_with_metrics
        
        # Register with MCP
        mcp_server.tool(name=tool_name, description=tool_description)(func_with_cache)
        
        # Return original function for reference
        return func
        
    return decorator