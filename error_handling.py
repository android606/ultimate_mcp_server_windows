"""
Error handling patterns for MCP tools.

This module provides standardized error handling patterns for MCP tools,
making it easier for LLMs to understand and recover from errors.
"""
import functools
import inspect
import time
import traceback
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union


class ErrorType(str, Enum):
    """Types of errors that can occur in MCP tools."""
    
    VALIDATION_ERROR = "validation_error"  # Input validation failed
    EXECUTION_ERROR = "execution_error"    # Error during execution
    PERMISSION_ERROR = "permission_error"  # Insufficient permissions
    NOT_FOUND_ERROR = "not_found_error"    # Resource not found
    TIMEOUT_ERROR = "timeout_error"        # Operation timed out
    RATE_LIMIT_ERROR = "rate_limit_error"  # Rate limit exceeded
    EXTERNAL_ERROR = "external_error"      # Error in external service
    UNKNOWN_ERROR = "unknown_error"        # Unknown error


def format_error_response(
    error_type: Union[ErrorType, str],
    message: str,
    details: Optional[Dict[str, Any]] = None,
    retriable: bool = False,
    suggestions: Optional[list] = None
) -> Dict[str, Any]:
    """
    Format a standardized error response.
    
    Args:
        error_type: Type of error
        message: Human-readable error message
        details: Additional error details
        retriable: Whether the operation can be retried
        suggestions: List of suggestions for resolving the error
        
    Returns:
        Formatted error response
    """
    return {
        "success": False,
        "isError": True,  # MCP protocol flag
        "error": {
            "type": error_type if isinstance(error_type, str) else error_type.value,
            "message": message,
            "details": details or {},
            "retriable": retriable,
            "suggestions": suggestions or [],
            "timestamp": time.time()
        }
    }


def with_error_handling(func: Callable) -> Callable:
    """
    Decorator for consistent error handling in MCP tools.
    
    This decorator wraps tool functions to provide consistent error handling.
    It catches exceptions and formats them according to the standardized error format.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with error handling
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Call the original function
            return await func(*args, **kwargs)
        except Exception as e:
            # Get exception details
            exc_type = type(e).__name__
            exc_message = str(e)
            exc_traceback = traceback.format_exc()
            
            # Determine error type
            error_type = ErrorType.UNKNOWN_ERROR
            retriable = False
            
            # Map common exceptions to error types
            if exc_type in ("ValueError", "TypeError", "KeyError", "AttributeError"):
                error_type = ErrorType.VALIDATION_ERROR
                retriable = True
            elif exc_type in ("FileNotFoundError", "KeyError", "IndexError"):
                error_type = ErrorType.NOT_FOUND_ERROR
                retriable = False
            elif exc_type in ("PermissionError", "AccessError"):
                error_type = ErrorType.PERMISSION_ERROR
                retriable = False
            elif exc_type in ("TimeoutError"):
                error_type = ErrorType.TIMEOUT_ERROR
                retriable = True
            elif "rate limit" in exc_message.lower():
                error_type = ErrorType.RATE_LIMIT_ERROR
                retriable = True
            
            # Generate suggestions based on error type
            suggestions = []
            if error_type == ErrorType.VALIDATION_ERROR:
                suggestions = [
                    "Check that all required parameters are provided",
                    "Verify parameter types and formats",
                    "Ensure parameter values are within allowed ranges"
                ]
            elif error_type == ErrorType.NOT_FOUND_ERROR:
                suggestions = [
                    "Verify the resource ID or path exists",
                    "Check for typos in identifiers",
                    "Ensure the resource hasn't been deleted"
                ]
            elif error_type == ErrorType.RATE_LIMIT_ERROR:
                suggestions = [
                    "Wait before retrying the request",
                    "Reduce the frequency of requests",
                    "Implement backoff strategy for retries"
                ]
            
            # Format and return error response
            return format_error_response(
                error_type=error_type,
                message=exc_message,
                details={
                    "exception_type": exc_type,
                    "traceback": exc_traceback
                },
                retriable=retriable,
                suggestions=suggestions
            )
    
    return wrapper


def validate_inputs(**validators):
    """
    Decorator for validating tool inputs.
    
    Args:
        **validators: Functions to validate inputs, keyed by parameter name
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            
            # Build mapping of parameter names to values
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate inputs
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        # Run validation
                        validator(value)
                    except Exception as e:
                        # Return validation error
                        return format_error_response(
                            error_type=ErrorType.VALIDATION_ERROR,
                            message=f"Invalid value for parameter '{param_name}': {str(e)}",
                            details={
                                "parameter": param_name,
                                "value": str(value),
                                "constraint": str(validator.__doc__ or "")
                            },
                            retriable=True,
                            suggestions=[
                                f"Provide a valid value for '{param_name}'",
                                "Check the parameter constraints in the tool description"
                            ]
                        )
            
            # Call the original function if validation passes
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Example validators
def non_empty_string(value):
    """Value must be a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Value must be a non-empty string")

def positive_number(value):
    """Value must be a positive number."""
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("Value must be a positive number")

def in_range(min_val, max_val):
    """Create validator for range checking."""
    def validator(value):
        """Value must be between {min_val} and {max_val}."""
        if not isinstance(value, (int, float)) or value < min_val or value > max_val:
            raise ValueError(f"Value must be between {min_val} and {max_val}")
    return validator 