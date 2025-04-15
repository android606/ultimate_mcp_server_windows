"""Parsing utilities for LLM Gateway.

This module provides utility functions for parsing and processing 
results from LLM Gateway operations that were previously defined in
example scripts but are now part of the library.
"""

import json
import re
from typing import Any, Dict

from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("llm_gateway.utils.parsing")

def extract_json_from_markdown(text: str) -> str:
    """Extracts a JSON string embedded within markdown code fences.

    Handles fences like ```json ... ``` or ``` ... ```.
    If no fences are found, returns the original string stripped of whitespace.

    Args:
        text: The input string possibly containing markdown-fenced JSON.

    Returns:
        The extracted JSON string or the stripped original string.
    """
    cleaned_text = text.strip()
    
    # Regex to find JSON content within ```json ... ``` or ``` ... ```
    # re.DOTALL makes . match newlines, re.IGNORECASE ignores case for 'json'
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        # Return the captured group (the content inside the fences), stripped
        return match.group(1).strip()
    else:
        # No fences found, return the original cleaned text
        return cleaned_text

def parse_result(result: Any) -> Dict[str, Any]:
    """Parse the result from a tool call into a usable dictionary.
    
    Handles various return types from MCP tools, including TextContent objects,
    list results, and direct dictionaries. Attempts to extract JSON from
    markdown code fences if present.
    
    Args:
        result: Result from an MCP tool call or provider operation
            
    Returns:
        Parsed dictionary containing the result data
    """
    try:
        text_to_parse = None
        # Handle TextContent object (which has a .text attribute)
        if hasattr(result, 'text'):
            text_to_parse = result.text
                
        # Handle list result
        elif isinstance(result, list):
            if result:
                first_item = result[0]
                if hasattr(first_item, 'text'):
                    text_to_parse = first_item.text
                else:
                    # If the first item isn't text, try returning it directly
                    if isinstance(first_item, dict):
                        return first_item
                    # Or handle other types as needed, maybe log a warning?
                    return {"warning": f"List item type not directly parseable: {type(first_item)}"}
            else: # Empty list
                return {}
            
        # Handle dictionary directly
        elif isinstance(result, dict):
            return result

        # Attempt to parse if we found text
        if text_to_parse is not None:
            # Extract potential JSON content from markdown fences
            json_str = extract_json_from_markdown(text_to_parse)

            # If no fences were found, json_str remains the original cleaned_text

            try:
                # Try to parse the potentially extracted/cleaned text as JSON
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Return the raw cleaned text if not JSON
                # Log the string that failed parsing for debugging
                logger.warning(f"Content could not be parsed as JSON: {json_str[:100]}...", emoji_key="warning")
                return {"text": json_str} # Return the cleaned (potentially extracted) text

        # Handle other potential types or return error if no text was found/parsed
        else:
            logger.warning(f"Unexpected result type or structure: {type(result)}", emoji_key="warning")
            return {"error": f"Unexpected result type or structure: {type(result)}"}
        
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