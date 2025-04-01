"""Parsing utilities for LLM Gateway.

This module provides utility functions for parsing and processing 
results from LLM Gateway operations that were previously defined in
example scripts but are now part of the library.
"""

import json
from typing import Any, Dict

from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("llm_gateway.utils.parsing")

def parse_result(result: Any) -> Dict[str, Any]:
    """Parse the result from a tool call into a usable dictionary.
    
    Handles various return types from MCP tools, including TextContent objects,
    list results, and direct dictionaries.
    
    Args:
        result: Result from an MCP tool call or provider operation
            
    Returns:
        Parsed dictionary containing the result data
    """
    try:
        # Handle TextContent object (which has a .text attribute)
        if hasattr(result, 'text'):
            try:
                # Try to parse the text as JSON
                return json.loads(result.text)
            except json.JSONDecodeError:
                # Return the raw text if not JSON
                return {"text": result.text}
                
        # Handle list result
        if isinstance(result, list):
            if result:
                first_item = result[0]
                if hasattr(first_item, 'text'):
                    try:
                        return json.loads(first_item.text)
                    except json.JSONDecodeError:
                        return {"text": first_item.text}
                else:
                    return first_item
            return {}
            
        # Handle dictionary directly
        if isinstance(result, dict):
            return result
            
        # Handle other potential types or return error
        else:
            return {"error": f"Unexpected result type: {type(result)}"}
        
    except Exception as e:
        logger.warning(f"Error parsing result: {str(e)}", emoji_key="warning")
        return {"error": f"Error parsing result: {str(e)}"}

def process_mcp_result(result: Any) -> Dict[str, Any]:
    """Process result from MCP tool call, handling both list and dictionary formats.
    
    This is a more user-friendly alias for parse_result that provides the same functionality.
    
    Args:
        result: Result from an MCP tool call or provider operation
            
    Returns:
        Processed dictionary containing the result data
    """
    return parse_result(result) 