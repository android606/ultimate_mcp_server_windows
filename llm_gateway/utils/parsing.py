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

    Handles various markdown code block formats and edge cases:
    - Complete code blocks: ```json ... ``` or ``` ... ```
    - Alternative fence styles: ~~~json ... ~~~ 
    - Incomplete/truncated blocks with only opening fence
    - Multiple code blocks (chooses the first valid JSON)
    - Extensive JSON repair for common LLM output issues:
        - Unterminated strings
        - Trailing commas
        - Missing closing brackets
        - Unquoted keys
        - Truncated content

    If no valid JSON-like content is found in fences, returns the original string.

    Args:
        text: The input string possibly containing markdown-fenced JSON.

    Returns:
        The extracted JSON string or the stripped original string.
    """
    if not text:
        return ""
        
    cleaned_text = text.strip()
    possible_json_candidates = []
    
    # Try to find JSON inside complete code blocks with various fence styles
    # Look for backtick fences (most common)
    backtick_matches = re.finditer(r"```(?:json)?\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
    for match in backtick_matches:
        possible_json_candidates.append(match.group(1).strip())
    
    # Look for tilde fences (less common but valid in some markdown)
    tilde_matches = re.finditer(r"~~~(?:json)?\s*(.*?)\s*~~~", cleaned_text, re.DOTALL | re.IGNORECASE)
    for match in tilde_matches:
        possible_json_candidates.append(match.group(1).strip())
    
    # If no complete blocks found, check for blocks with only opening fence
    if not possible_json_candidates:
        # Try backtick opening fence
        backtick_start = re.search(r"```(?:json)?\s*", cleaned_text, re.IGNORECASE)
        if backtick_start:
            content_after_fence = cleaned_text[backtick_start.end():].strip()
            possible_json_candidates.append(content_after_fence)
        
        # Try tilde opening fence
        tilde_start = re.search(r"~~~(?:json)?\s*", cleaned_text, re.IGNORECASE)
        if tilde_start:
            content_after_fence = cleaned_text[tilde_start.end():].strip()
            possible_json_candidates.append(content_after_fence)
    
    # If still no candidates, add the original text as last resort
    if not possible_json_candidates:
        possible_json_candidates.append(cleaned_text)
    
    # Try each candidate, returning the first one that looks like valid JSON
    for candidate in possible_json_candidates:
        # Apply advanced JSON repair
        repaired = _repair_json(candidate)
        try:
            # Validate if it's actually parseable JSON
            json.loads(repaired)
            return repaired  # Return the first valid JSON
        except json.JSONDecodeError:
            # If repair didn't work, continue to the next candidate
            continue
    
    # If no candidate worked with regular repair, try more aggressive repair on the first candidate
    if possible_json_candidates:
        aggressive_repair = _repair_json(possible_json_candidates[0], aggressive=True)
        try:
            json.loads(aggressive_repair)
            return aggressive_repair
        except json.JSONDecodeError:
            # Return the best we can - the first candidate with basic cleaning
            # This will still fail in json.loads, but at least we tried
            return possible_json_candidates[0]
    
    # Absolute fallback - return the original text
    return cleaned_text

def _repair_json(text: str, aggressive=False) -> str:
    """Repair common JSON issues in LLM outputs.
    
    Args:
        text: The JSON-like string to repair
        aggressive: Whether to apply more aggressive repair techniques
        
    Returns:
        Repaired JSON string
    """
    if not text:
        return text
        
    # Step 1: Basic cleanup
    result = text.strip()
    
    # Quick check if it even remotely looks like JSON
    if not (result.startswith('{') or result.startswith('[')):
        return result
        
    # Step 2: Fix common issues
    
    # Fix trailing commas before closing brackets
    result = re.sub(r',\s*([\}\]])', r'\1', result)
    
    # Ensure property names are quoted
    result = re.sub(r'([{,]\s*)([a-zA-Z0-9_$]+)(\s*:)', r'\1"\2"\3', result)
    
    # If we're not in aggressive mode, return after basic fixes
    if not aggressive:
        return result
        
    # Step 3: Aggressive repairs for truncated/malformed JSON
    
    # Track opening/closing brackets to detect imbalance
    open_braces = result.count('{')
    close_braces = result.count('}')
    open_brackets = result.count('[')
    close_brackets = result.count(']')
    
    # Count quotes to check if we have an odd number (indicating unterminated strings)
    quote_count = result.count('"')
    if quote_count % 2 != 0:
        # We have an odd number of quotes, meaning at least one string is unterminated
        # This is a much more aggressive approach to fix strings
        
        # First, try to find all strings that are properly terminated
        proper_strings = []
        pos = 0
        in_string = False
        string_start = 0
        
        # This helps track properly formed strings and identify problematic ones
        while pos < len(result):
            if result[pos] == '"' and (pos == 0 or result[pos-1] != '\\'):
                if not in_string:
                    # Start of a string
                    in_string = True
                    string_start = pos
                else:
                    # End of a string
                    in_string = False
                    proper_strings.append((string_start, pos))
            pos += 1
        
        # If we're still in a string at the end, we found an unterminated string
        if in_string:
            # Force terminate it at the end
            result += '"'
    
    # Even more aggressive string fixing
    # This regexp looks for a quote followed by any characters not containing a quote
    # followed by a comma, closing brace, or bracket, without a quote in between
    # This indicates an unterminated string
    result = re.sub(r'"([^"]*?)(?=,|\s*[\]}]|$)', r'"\1"', result)
    
    # Fix cases where value might be truncated mid-word just before closing quote
    # If we find something that looks like it's in the middle of a string, terminate it
    result = re.sub(r'"([^"]+)(\s*[\]}]|,|$)', lambda m: 
        f'"{m.group(1)}"{"" if m.group(2).startswith(",") or m.group(2) in "]}," else m.group(2)}', 
        result)
    
    # Fix dangling quotes at the end of the string - these usually indicate a truncated string
    if result.rstrip().endswith('"'):
        # Add closing quote and appropriate structure depending on context
        result = result.rstrip() + '"'
        
        # Look at the previous few characters to determine if we need a comma or not
        context = result[-20:] if len(result) > 20 else result
        # If string ends with x": " it's likely a property name
        if re.search(r'"\s*:\s*"$', context):
            # Add a placeholder value and closing structure for the property
            result += "unknown"
            
    # Check for dangling property (property name with colon but no value)
    result = re.sub(r'"([^"]+)"\s*:(?!\s*["{[\w-])', r'"\1": null', result)
    
    # Add missing closing brackets/braces if needed
    if open_braces > close_braces:
        result += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        result += ']' * (open_brackets - close_brackets)
    
    # Handle truncated JSON structure - look for incomplete objects at the end
    # This is complex, but we'll try some common patterns
    
    # If JSON ends with a property name and colon but no value
    if re.search(r'"[^"]+"\s*:\s*$', result):
        result += 'null'
    
    # If JSON ends with a comma, it needs another value - add a null
    if re.search(r',\s*$', result):
        result += 'null'
        
    # If the JSON structure is fundamentally corrupted at the end (common in truncation)
    # Close any unclosed objects or arrays
    if not (result.endswith('}') or result.endswith(']') or result.endswith('"')):
        # Count unmatched opening brackets
        stack = []
        for char in result:
            if char in '{[':
                stack.append(char)
            elif char in '}]':
                if stack and ((stack[-1] == '{' and char == '}') or (stack[-1] == '[' and char == ']')):
                    stack.pop()
                    
        # Close any unclosed structures
        for bracket in reversed(stack):
            if bracket == '{':
                result += '}'
            elif bracket == '[':
                result += ']'
    
    # As a final safeguard, try to eval the JSON with a permissive parser
    # This won't fix deep structural issues but catches cases our regexes missed
    try:
        import simplejson
        simplejson.loads(result, parse_constant=lambda x: x)
    except (ImportError, simplejson.JSONDecodeError):
        try:
            # Try one last time with the more permissive custom JSON parser
            _scan_once = json.scanner.py_make_scanner(json.JSONDecoder())
            try:
                _scan_once(result, 0)
            except StopIteration:
                # Likely unterminated JSON - do one final pass of common fixups
                
                # Check for unterminated strings of various forms one more time
                if re.search(r'(?<!")"(?:[^"\\]|\\.)*[^"\\](?!")(?=,|\s*[\]}]|$)', result):
                    # Even more aggressive fixes, replacing with generic values
                    result = re.sub(r'(?<!")"(?:[^"\\]|\\.)*[^"\\](?!")(?=,|\s*[\]}]|$)', 
                                    r'"invalid_string"', result)
                
                # Ensure valid JSON-like structure
                if not (result.endswith('}') or result.endswith(']')):
                    if result.count('{') > result.count('}'):
                        result += '}'
                    if result.count('[') > result.count(']'):
                        result += ']'
            except Exception:
                # Something else is wrong, but we've tried our best
                pass
        except Exception:
            # We've done all we reasonably can
            pass
    
    return result

async def parse_result(result: Any) -> Dict[str, Any]:
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
            except json.JSONDecodeError as e:
                # If parsing fails, try LLM repair
                logger.warning(f"Initial JSON parsing failed: {e}. Attempting LLM repair...", emoji_key="warning")
                try:
                    # Local import to minimize scope/potential circular dependency issues
                    from llm_gateway.tools.completion import generate_completion
                    
                    repair_prompt = (
                        f"The following text is supposed to be JSON but failed parsing. "
                        f"Please extract the valid JSON data from it and return *only* the raw JSON string, nothing else. "
                        f"If it\'s impossible to extract valid JSON, return an empty JSON object {{}}. "
                        f"Problematic text:\n\n```\n{json_str}\n```"
                    )
                    
                    llm_repair_result = await generate_completion(
                        prompt=repair_prompt,
                        provider="openai", # Use openai as requested
                        model="gpt-4.1-mini", # Use gpt-4.1-mini as requested
                        temperature=0.0 # Be deterministic for extraction
                    )
                    
                    if llm_repair_result.get("success"):
                        llm_json_text = llm_repair_result.get("text", "")
                        try:
                            # Try parsing the LLM response
                            repaired_json = json.loads(llm_json_text)
                            logger.info("LLM repair successful.", emoji_key="success")
                            return repaired_json
                        except json.JSONDecodeError as llm_e:
                            logger.error(f"LLM repair attempt failed. LLM response could not be parsed as JSON: {llm_e}. LLM response: {llm_json_text[:100]}...", emoji_key="error")
                            return {"error": f"LLM repair failed: LLM response was not valid JSON ({llm_e})", "raw_content": json_str, "llm_response": llm_json_text}
                    else:
                        llm_error = llm_repair_result.get("error", "Unknown LLM error")
                        logger.error(f"LLM repair attempt failed: LLM call failed: {llm_error}", emoji_key="error")
                        return {"error": f"LLM repair failed: LLM call error ({llm_error})", "raw_content": json_str}
                        
                except Exception as repair_ex:
                    logger.error(f"LLM repair attempt failed with exception: {repair_ex}", emoji_key="error", exc_info=True)
                    return {"error": f"LLM repair failed with exception: {repair_ex}", "raw_content": json_str}

        # Handle other potential types or return error if no text was found/parsed
        else:
            logger.warning(f"Unexpected result type or structure: {type(result)}", emoji_key="warning")
            return {"error": f"Unexpected result type or structure: {type(result)}"}
        
    except Exception as e:
        logger.warning(f"Error parsing result: {str(e)}", emoji_key="warning")
        return {"error": f"Error parsing result: {str(e)}"}

async def process_mcp_result(result: Any) -> Dict[str, Any]:
    """Process result from MCP tool call, handling both list and dictionary formats.
    
    This is a more user-friendly alias for parse_result that provides the same functionality.
    
    Args:
        result: Result from an MCP tool call or provider operation
            
    Returns:
        Processed dictionary containing the result data
    """
    return await parse_result(result) 