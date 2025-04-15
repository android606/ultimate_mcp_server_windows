"""Base tool classes and decorators for LLM Gateway."""
import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union

try:
    from mcp import Tool
except ImportError:
    # Handle case where mcp might be available via different import
    try:
        from mcp.server.fastmcp import Tool
    except ImportError:
        Tool = None  # Tool will be provided by the mcp_server

from llm_gateway.exceptions import (
    ResourceError,
    ToolError,
    ToolExecutionError,
    ToolInputError,
    format_error_response,
)
from llm_gateway.services.cache import with_cache
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tools.base")


def tool(name=None, description=None):
    """Decorator to mark a method as an MCP tool.
    
    Args:
        name: Tool name (defaults to method name)
        description: Tool description (defaults to method docstring)
        
    Returns:
        Decorated method
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            return await func(self, *args, **kwargs)
        
        wrapper._tool = True
        wrapper._tool_name = name
        wrapper._tool_description = description
        
        return wrapper
    
    return decorator


def with_resource(resource_type, allow_creation=False, require_existence=True):
    """Decorator to standardize resource handling for tools.
    
    Args:
        resource_type: Type of resource being accessed (e.g., "document", "embedding", "database")
        allow_creation: Whether the tool is allowed to create new resources
        require_existence: Whether the resource must exist before the tool is called
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get resource ID from kwargs (common parameter names)
            resource_id = None
            for param_name in [f"{resource_type}_id", "id", "resource_id"]:
                if param_name in kwargs:
                    resource_id = kwargs[param_name]
                    break
            
            # Check if resource exists if required
            if require_existence and resource_id:
                # Get resource registry from MCP server
                resource_registry = getattr(self.mcp, "resources", None)
                if resource_registry is None:
                    logger.warning(
                        f"Resource registry not available, skipping existence check for {resource_type}/{resource_id}",
                        emoji_key="warning"
                    )
                else:
                    # Check if resource exists
                    exists = await resource_registry.exists(resource_type, resource_id)
                    if not exists:
                        raise ResourceError(
                            f"{resource_type.capitalize()} not found: {resource_id}",
                            resource_type=resource_type,
                            resource_id=resource_id
                        )
            
            # Call function
            result = await func(self, *args, **kwargs)
            
            # If the function returns a new resource ID, register it
            if allow_creation and isinstance(result, dict) and "resource_id" in result:
                new_resource_id = result["resource_id"]
                # Get resource registry from MCP server
                resource_registry = getattr(self.mcp, "resources", None)
                if resource_registry is not None:
                    # Register new resource
                    metadata = {
                        "created_at": time.time(),
                        "creator": kwargs.get("ctx", {}).get("user_id", "unknown"),
                        "resource_type": resource_type
                    }
                    
                    # Add other metadata from result if available
                    if "metadata" in result:
                        metadata.update(result["metadata"])
                    
                    await resource_registry.register(
                        resource_type, 
                        new_resource_id, 
                        metadata=metadata
                    )
                    
                    logger.info(
                        f"Registered new {resource_type}: {new_resource_id}",
                        emoji_key="resource",
                        resource_type=resource_type,
                        resource_id=new_resource_id
                    )
            
            return result
                
        # Add resource metadata to function
        wrapper._resource_type = resource_type
        wrapper._allow_creation = allow_creation
        wrapper._require_existence = require_existence
        
        return wrapper
    
    return decorator


class ResourceRegistry:
    """Registry for tracking resources used by tools."""
    
    def __init__(self, storage_backend=None):
        """Initialize the resource registry.
        
        Args:
            storage_backend: Backend for persistent storage (if None, in-memory only)
        """
        self.resources = {}
        self.storage = storage_backend
        self.logger = get_logger("llm_gateway.resources")
    
    async def register(self, resource_type, resource_id, metadata=None):
        """Register a resource in the registry.
        
        Args:
            resource_type: Type of resource (e.g., "document", "embedding")
            resource_id: Resource identifier
            metadata: Additional metadata about the resource
            
        Returns:
            True if registration was successful
        """
        # Initialize resource type if not exists
        if resource_type not in self.resources:
            self.resources[resource_type] = {}
        
        # Register resource
        self.resources[resource_type][resource_id] = {
            "id": resource_id,
            "type": resource_type,
            "metadata": metadata or {},
            "registered_at": time.time()
        }
        
        # Persist to storage backend if available
        if self.storage:
            try:
                await self.storage.save_resource(
                    resource_type, 
                    resource_id, 
                    self.resources[resource_type][resource_id]
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to persist resource {resource_type}/{resource_id}: {str(e)}",
                    emoji_key="error",
                    exc_info=True
                )
        
        return True
    
    async def exists(self, resource_type, resource_id):
        """Check if a resource exists in the registry.
        
        Args:
            resource_type: Type of resource
            resource_id: Resource identifier
            
        Returns:
            True if the resource exists
        """
        # Check in-memory registry first
        if resource_type in self.resources and resource_id in self.resources[resource_type]:
            return True
        
        # Check storage backend if available
        if self.storage:
            try:
                return await self.storage.resource_exists(resource_type, resource_id)
            except Exception as e:
                self.logger.error(
                    f"Failed to check resource existence {resource_type}/{resource_id}: {str(e)}",
                    emoji_key="error",
                    exc_info=True
                )
        
        return False
    
    async def get(self, resource_type, resource_id):
        """Get resource metadata from the registry.
        
        Args:
            resource_type: Type of resource
            resource_id: Resource identifier
            
        Returns:
            Resource metadata or None if not found
        """
        # Check in-memory registry first
        if resource_type in self.resources and resource_id in self.resources[resource_type]:
            return self.resources[resource_type][resource_id]
        
        # Check storage backend if available
        if self.storage:
            try:
                resource = await self.storage.get_resource(resource_type, resource_id)
                if resource:
                    # Cache in memory for future access
                    if resource_type not in self.resources:
                        self.resources[resource_type] = {}
                    self.resources[resource_type][resource_id] = resource
                    return resource
            except Exception as e:
                self.logger.error(
                    f"Failed to get resource {resource_type}/{resource_id}: {str(e)}",
                    emoji_key="error",
                    exc_info=True
                )
        
        return None
    
    async def list(self, resource_type, limit=100, offset=0, filters=None):
        """List resources of a specific type.
        
        Args:
            resource_type: Type of resource to list
            limit: Maximum number of resources to return
            offset: Offset for pagination
            filters: Dictionary of filters to apply
            
        Returns:
            List of resource metadata
        """
        result = []
        
        # Get from storage backend first if available
        if self.storage:
            try:
                resources = await self.storage.list_resources(
                    resource_type, 
                    limit=limit, 
                    offset=offset, 
                    filters=filters
                )
                
                # Cache in memory for future access
                if resources:
                    if resource_type not in self.resources:
                        self.resources[resource_type] = {}
                    
                    for resource in resources:
                        resource_id = resource.get("id")
                        if resource_id:
                            self.resources[resource_type][resource_id] = resource
                    
                    return resources
            except Exception as e:
                self.logger.error(
                    f"Failed to list resources of type {resource_type}: {str(e)}",
                    emoji_key="error",
                    exc_info=True
                )
        
        # Fallback to in-memory registry
        if resource_type in self.resources:
            # Apply filters if provided
            filtered_resources = self.resources[resource_type].values()
            if filters:
                for key, value in filters.items():
                    filtered_resources = [
                        r for r in filtered_resources 
                        if r.get("metadata", {}).get(key) == value
                    ]
            
            # Apply pagination
            result = list(filtered_resources)[offset:offset+limit]
        
        return result
    
    async def delete(self, resource_type, resource_id):
        """Delete a resource from the registry.
        
        Args:
            resource_type: Type of resource
            resource_id: Resource identifier
            
        Returns:
            True if deletion was successful
        """
        # Delete from in-memory registry
        if resource_type in self.resources and resource_id in self.resources[resource_type]:
            del self.resources[resource_type][resource_id]
        
        # Delete from storage backend if available
        if self.storage:
            try:
                return await self.storage.delete_resource(resource_type, resource_id)
            except Exception as e:
                self.logger.error(
                    f"Failed to delete resource {resource_type}/{resource_id}: {str(e)}",
                    emoji_key="error",
                    exc_info=True
                )
        
        return True


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
        
        # Initialize resource registry if not already available
        if not hasattr(self.mcp, "resources"):
            self.mcp.resources = ResourceRegistry()
        
    def _register_tools(self):
        """Register tools with MCP server.
        
        Override this method in subclasses to register specific tools.
        This method is no longer called by the base class constructor.
        Registration is now handled externally, e.g., in register_all_tools.
        """
        pass
        
    async def execute(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool with consistent interface.
        
        This method abstracts differences between FastMCP and Gateway interfaces.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            
        Returns:
            Tool execution result
        """
        # Handle different interfaces between FastMCP and Gateway
        if hasattr(self.mcp, "call_tool"):
            # FastMCP
            return await self.mcp.call_tool(tool_name, params)
        elif hasattr(self.mcp, "execute"):
            # Gateway
            return await self.mcp.execute(tool_name, params)
        else:
            # Fallback - try the direct call first
            try:
                return await self.mcp.call_tool(tool_name, params)
            except AttributeError as e:
                self.logger.error(
                    f"MCP server does not support call_tool or execute methods. Type: {type(self.mcp)}",
                    emoji_key="error"
                )
                raise ValueError(f"Unsupported MCP server type: {type(self.mcp)}") from e
        
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
    """Decorator to add metrics tracking to a tool function/method.
    Adapts to both standalone functions and class methods.
    Args: func: Tool function/method to decorate
    Returns: Decorated function/method
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Check if the first arg looks like a BaseTool instance
        self_obj = args[0] if args and isinstance(args[0], BaseTool) else None
        tool_name = getattr(self_obj, 'tool_name', func.__name__)

        start_time = time.time()
        success = False
        tokens = None
        cost = None
        result = None
        
        try:
            # Call original function, passing self_obj if it exists
            if self_obj:
                # Assumes if self_obj exists, it's the first positional arg expected by func
                result = await func(self_obj, *args[1:], **kwargs)
            else:
                # Pass only the args/kwargs received, assuming func is standalone
                result = await func(*args, **kwargs)
            
            # Extract metrics if available from result
            if isinstance(result, dict):
                if "tokens" in result and isinstance(result["tokens"], dict):
                    tokens = result["tokens"].get("total")
                elif "total_tokens" in result:
                    tokens = result["total_tokens"]
                cost = result.get("cost")
                
            success = True
            return result
            
        except Exception as e:
            logger.error(
                f"Tool execution failed: {tool_name}: {str(e)}",
                emoji_key="error",
                tool=tool_name,
                exc_info=True
            )
            raise # Re-raise exception for other handlers (like with_error_handling)
            
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            # Log execution stats
            logger.debug(
                f"Tool execution: {tool_name} ({'success' if success else 'failed'})",
                emoji_key="tool" if success else "error",
                tool=tool_name,
                time=duration,
                cost=cost
            )
            
            # Update metrics if we found a self object with a metrics attribute
            if self_obj and hasattr(self_obj, 'metrics'):
                self_obj.metrics.record_call(
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
    

def with_error_handling(func):
    """Decorator to add standardized error handling to a tool function/method.
    Adapts to both standalone functions and class methods.
    Args: func: Tool function/method to decorate
    Returns: Decorated function/method with error handling
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Check if the first arg looks like a BaseTool instance
        self_obj = args[0] if args and isinstance(args[0], BaseTool) else None
        # Determine tool_name based on instance or func name
        tool_name = getattr(self_obj, 'tool_name', func.__name__) 
        
        sig = inspect.signature(func)
        func_params = set(sig.parameters.keys())  # noqa: F841
        
        call_args = []
        call_kwargs = {}

        if self_obj:
            expected_params = list(sig.parameters.values())
            if expected_params and expected_params[0].name == 'self':
                call_args.append(self_obj)
        
        start_index = 1 if self_obj and call_args else 0
        call_args.extend(args[start_index:])

        # Pass all original kwargs through
        call_kwargs.update(kwargs)
            
        try:
            # Call original function with reconstructed args/kwargs
            # This version passes *all* kwargs received by the wrapper,
            # trusting FastMCP to pass the correct ones including 'ctx'.
            return await func(*call_args, **call_kwargs)
            
        except ToolError as e:
            # Already a tool error, log and return
            logger.error(
                f"Tool error in {tool_name}: {str(e)} ({e.error_code})",
                emoji_key="error",
                tool=tool_name,
                error_code=e.error_code,
                details=e.details
            )
            
            # Debug log the formatted error response
            error_response = format_error_response(e)
            logger.debug(f"Formatted error response for {tool_name}: {error_response}")
            
            # Return standardized error response
            return error_response
            
        except ValueError as e:
            # Convert ValueError to ToolInputError with more detailed information
            error = ToolInputError(
                f"Invalid input to {tool_name}: {str(e)}",
                details={
                    "tool_name": tool_name,
                    "exception_type": "ValueError",
                    "original_error": str(e)
                }
            )
            
            logger.error(
                f"Invalid input to {tool_name}: {str(e)}",
                emoji_key="error",
                tool=tool_name,
                error_code=error.error_code
            )
            
            # Return standardized error response
            return format_error_response(error)
            
        except Exception as e:
            # Create a more specific error message that includes the tool name
            specific_message = f"Execution error in {tool_name}: {str(e)}"
            
            # Convert to ToolExecutionError for other exceptions
            error = ToolExecutionError(
                specific_message,
                cause=e,
                details={
                    "tool_name": tool_name,
                    "exception_type": type(e).__name__,
                    "original_message": str(e)
                }
            )
            
            logger.error(
                specific_message,
                emoji_key="error",
                tool=tool_name,
                exc_info=True
            )
            
            # Return standardized error response
            return format_error_response(error)
                
    return wrapper


def register_tool(mcp_server, name=None, description=None, cache_ttl=None):
    """Register a function as an MCP tool.
    
    Args:
        mcp_server: MCP server instance
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        cache_ttl: Optional TTL for caching tool results
        
    Returns:
        Decorator function
    """
    def decorator(func):
        # Get function name and docstring
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"  # noqa: F841
        
        # Apply caching if specified
        if cache_ttl is not None:
            func = with_cache(ttl=cache_ttl)(func)
        
        # Apply error handling
        func = with_error_handling(func)
        
        # Register with MCP server
        mcp_server.tool(name=tool_name)(func)
        
        return func
    
    return decorator

def _get_json_schema_type(type_annotation):
    """Convert Python type annotation to JSON schema type information.
    
    Args:
        type_annotation: Type annotation from function signature
        
    Returns:
        Dictionary with JSON schema type information
    """
    import typing
    
    # Handle basic types
    if type_annotation is str:
        return {"type": "string"}
    elif type_annotation is int:
        return {"type": "integer"}
    elif type_annotation is float:
        return {"type": "number"}
    elif type_annotation is bool:
        return {"type": "boolean"}
    
    # Handle Optional types
    origin = typing.get_origin(type_annotation)
    args = typing.get_args(type_annotation)
    
    if origin is Union and type(None) in args:
        # Optional type - get the non-None type
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            inner_type = _get_json_schema_type(non_none_args[0])
            return inner_type
    
    # Handle lists
    if origin is list or origin is List:
        if args:
            item_type = _get_json_schema_type(args[0])
            return {
                "type": "array",
                "items": item_type
            }
        return {"type": "array"}
    
    # Handle dictionaries
    if origin is dict or origin is Dict:
        return {"type": "object"}
    
    # Default to object for complex types
    return {"type": "object"}

def with_state_management(namespace: str):
    """Decorator to provide state management to tool functions.
    
    Args:
        namespace: The namespace to use for state storage
        
    Returns:
        A decorator that injects state management functions into the tool
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get MCP server from context
            context = kwargs.get('ctx')
            if not context or not hasattr(context, 'mcp'):
                raise ValueError("Context with MCP server required")
            
            mcp = context.mcp
            if not hasattr(mcp, 'state_store'):
                raise ValueError("MCP server does not have a state store")
            
            # Add state accessors to kwargs
            kwargs['get_state'] = lambda key, default=None: mcp.state_store.get(namespace, key, default)
            kwargs['set_state'] = lambda key, value: mcp.state_store.set(namespace, key, value)
            kwargs['delete_state'] = lambda key: mcp.state_store.delete(namespace, key)
            
            return await func(*args, **kwargs)
        
        # Update signature to include context parameter if not already present
        sig = inspect.signature(func)
        if 'ctx' not in sig.parameters:
            wrapped_params = list(sig.parameters.values())
            wrapped_params.append(
                inspect.Parameter('ctx', inspect.Parameter.KEYWORD_ONLY, 
                                 annotation='Optional[Dict[str, Any]]', default=None)
            )
            wrapper.__signature__ = sig.replace(parameters=wrapped_params)
        
        return wrapper
    return decorator