"""Secure asynchronous filesystem tools for LLM Gateway.

This module provides secure asynchronous filesystem operations, including reading, writing,
and manipulating files and directories, with robust security controls to limit access.
"""
import os
import time
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
import difflib
import aiofiles
import aiofiles.os  # Import async os operations
from fnmatch import fnmatch # Keep sync fnmatch for pattern matching

from llm_gateway.tools.base import with_error_handling, with_tool_metrics
from llm_gateway.exceptions import ToolError, ToolInputError
from llm_gateway.utils import get_logger
from llm_gateway.config import get_config

logger = get_logger("llm_gateway.tools.filesystem")

# --- Configuration and Security ---

# get_allowed_directories remains synchronous as config reading is typically fast
def get_allowed_directories() -> List[str]:
    """Get allowed directories from configuration.

    Returns:
        List of normalized directory paths that can be accessed
    """
    cfg = get_config()
    allowed = getattr(cfg, 'filesystem', {}).get('allowed_directories', [])

    if not allowed:
        logger.warning(
            "No filesystem directories configured for access. All operations will be rejected.",
            emoji_key="security"
        )
        return []

    # os.path functions are generally CPU-bound and fast, keep sync for simplicity here
    normalized = [os.path.normpath(os.path.abspath(os.path.expanduser(d))) for d in allowed]

    logger.info(
        f"Filesystem tools configured with {len(normalized)} allowed directories",
        emoji_key="config",
        allowed_directories=normalized
    )

    return normalized

# Cache allowed directories to avoid recomputing
_ALLOWED_DIRECTORIES: Optional[List[str]] = None
_CACHE_TIMESTAMP: float = 0
_CACHE_TTL = 300  # 5 minutes in seconds

# get_allowed_dirs remains synchronous
def get_allowed_dirs() -> List[str]:
    """Get cached allowed directories or compute them if not cached or expired.

    Returns:
        List of normalized allowed directory paths
    """
    global _ALLOWED_DIRECTORIES, _CACHE_TIMESTAMP

    current_time = time.time()

    if _ALLOWED_DIRECTORIES is None or (current_time - _CACHE_TIMESTAMP) > _CACHE_TTL:
        _ALLOWED_DIRECTORIES = get_allowed_directories()
        _CACHE_TIMESTAMP = current_time

    return _ALLOWED_DIRECTORIES

# validate_path becomes async due to filesystem checks
async def validate_path(path: str, must_exist: bool = True) -> str:
    """Validate a path for security and accessibility using async I/O.

    Args:
        path: The file or directory path to validate
        must_exist: Whether the path must exist already

    Returns:
        Normalized absolute path

    Raises:
        ToolInputError: If the path is invalid, outside allowed directories, or doesn't exist
        ToolError: For underlying filesystem errors
    """
    if not path or not isinstance(path, str):
        raise ToolInputError(
            "Path must be a non-empty string.",
            param_name="path",
            provided_value=path
        )

    # Path normalization remains sync
    try:
        expanded_path = os.path.expanduser(path)
        if not os.path.isabs(expanded_path):
            expanded_path = os.path.abspath(expanded_path)
        normalized_path = os.path.normpath(expanded_path)
    except Exception as e:
        raise ToolInputError(
            f"Invalid path format: {str(e)}",
            param_name="path",
            provided_value=path
        )

    # Check against allowed directories (sync operation)
    allowed_dirs = get_allowed_dirs()
    if not allowed_dirs:
        raise ToolError(
            "Filesystem access is disabled - no allowed directories configured.",
            context={"configured_directories": 0}
        )

    is_allowed = any(normalized_path.startswith(allowed_dir) for allowed_dir in allowed_dirs)
    if not is_allowed:
        raise ToolInputError(
            f"Access denied: path '{path}' is outside allowed directories.",
            param_name="path",
            provided_value=path,
            context={"allowed_dirs": allowed_dirs}
        )

    # Use aiofiles.os for existence checks
    try:
        path_exists = await aiofiles.os.path.exists(normalized_path)

        if must_exist and not path_exists:
            raise ToolInputError(
                f"Path '{path}' does not exist.",
                param_name="path",
                provided_value=path
            )

        # If path is a symlink, validate the real path asynchronously
        if path_exists and await aiofiles.os.path.islink(normalized_path):
            try:
                real_path = await aiofiles.os.path.realpath(normalized_path)
                real_normalized = os.path.normpath(real_path) # normpath is sync

                # Check if the real path is within allowed directories
                is_real_allowed = any(real_normalized.startswith(allowed_dir) for allowed_dir in allowed_dirs)
                if not is_real_allowed:
                    raise ToolInputError(
                        f"Access denied: symlink target '{real_path}' is outside allowed directories.",
                        param_name="path",
                        provided_value=path
                    )
            except OSError as e: # Catch potential OS errors during realpath
                 raise ToolInputError(
                    f"Error resolving symlink: {str(e)}",
                    param_name="path",
                    provided_value=path
                )
            except Exception as e: # Catch other unexpected errors
                raise ToolError(f"Unexpected error resolving symlink: {str(e)}", context={"path": path})


        # If path doesn't exist but must be created, check parent directory asynchronously
        if not path_exists and not must_exist:
            parent_dir = os.path.dirname(normalized_path) # dirname is sync
            if not parent_dir: # Handle edge case like '/'
                 raise ToolInputError(
                    f"Cannot determine parent directory for path '{path}'.",
                    param_name="path",
                    provided_value=path
                )
            if not await aiofiles.os.path.exists(parent_dir):
                raise ToolInputError(
                    f"Parent directory '{parent_dir}' does not exist.",
                    param_name="path",
                    provided_value=path
                )
                
    except OSError as e:
        # Catch filesystem errors during async checks
        raise ToolError(f"Filesystem error validating path '{path}': {str(e)}", context={"path": path, "error": str(e)})
    except ToolInputError: # Re-raise ToolInputErrors
        raise
    except Exception as e:
        # Catch unexpected errors
        logger.error(f"Unexpected error during path validation for {path}: {e}", exc_info=True)
        raise ToolError(f"An unexpected error occurred validating path: {str(e)}", context={"path": path})


    return normalized_path

# --- Helper Functions ---

# format_file_info becomes async due to stat/is_dir/is_file
async def format_file_info(path: str) -> Dict[str, Any]:
    """Get detailed file or directory information asynchronously.

    Args:
        path: Path to file or directory

    Returns:
        Dictionary with file/directory details
    """
    try:
        stat_info = await aiofiles.os.stat(path)
        is_dir = await aiofiles.os.path.isdir(path)
        is_file = await aiofiles.os.path.isfile(path)

        info = {
            "name": os.path.basename(path), # basename is sync
            "path": path,
            "size": stat_info.st_size,
            # Use datetime for consistency and easier parsing if needed later
            "created": time.ctime(stat_info.st_ctime),
            "modified": time.ctime(stat_info.st_mtime),
            "accessed": time.ctime(stat_info.st_atime),
            "is_directory": is_dir,
            "is_file": is_file,
            "permissions": oct(stat_info.st_mode)[-3:],
        }
        return info
    except OSError as e:
        logger.error(f"Error getting file info for {path}: {str(e)}", emoji_key="error")
        return {
            "name": os.path.basename(path),
            "path": path,
            "error": f"Failed to get info: {str(e)}"
        }
    except Exception as e: # Catch unexpected errors
        logger.error(f"Unexpected error getting file info for {path}: {e}", exc_info=True, emoji_key="error")
        return {
            "name": os.path.basename(path),
            "path": path,
            "error": f"An unexpected error occurred: {str(e)}"
        }


# This function doesn't perform I/O itself, but called by async funcs. Make async for consistency.
async def create_unified_diff(original_content: str, new_content: str, filepath: str) -> str:
    """Create a unified diff between original and new content.

    Args:
        original_content: Original file content
        new_content: New file content
        filepath: Path to the file (for display purposes)

    Returns:
        Unified diff as a string
    """
    # Normalize line endings (sync operation)
    original_lines = original_content.splitlines()
    new_lines = new_content.splitlines()

    # Generate unified diff (sync operation)
    diff_generator = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"{filepath} (original)",
        tofile=f"{filepath} (modified)",
        lineterm=""
    )

    return "\n".join(diff_generator)

# read_file_content is already async using aiofiles
async def read_file_content(filepath: str) -> str:
    """Read file content using async I/O. Handles potential errors.

    Args:
        filepath: Path to the file to read

    Returns:
        File content as string

    Raises:
        ToolError: If reading fails.
    """
    try:
        async with aiofiles.open(filepath, mode='r', encoding='utf-8') as f:
            return await f.read()
    except UnicodeDecodeError:
        logger.warning(f"File {filepath} is not UTF-8 encoded. Cannot read as text.", emoji_key="warning")
        raise ToolError(f"File is not UTF-8 encoded: {filepath}", context={"path": filepath, "encoding": "utf-8"})
    except OSError as e:
        logger.error(f"OS error reading file {filepath}: {e}", emoji_key="error")
        raise ToolError(f"Error reading file: {str(e)}", context={"path": filepath})
    except Exception as e:
        logger.error(f"Unexpected error reading file {filepath}: {e}", exc_info=True, emoji_key="error")
        raise ToolError(f"An unexpected error occurred while reading file: {str(e)}", context={"path": filepath})

# read_binary_file_content is already async using aiofiles
async def read_binary_file_content(filepath: str) -> bytes:
    """Read binary file content using async I/O. Handles potential errors.

    Args:
        filepath: Path to the file to read

    Returns:
        File content as bytes

    Raises:
        ToolError: If reading fails.
    """
    try:
        async with aiofiles.open(filepath, mode='rb') as f:
            return await f.read()
    except OSError as e:
        logger.error(f"OS error reading binary file {filepath}: {e}", emoji_key="error")
        raise ToolError(f"Error reading binary file: {str(e)}", context={"path": filepath})
    except Exception as e:
        logger.error(f"Unexpected error reading binary file {filepath}: {e}", exc_info=True, emoji_key="error")
        raise ToolError(f"An unexpected error occurred while reading binary file: {str(e)}", context={"path": filepath})


# write_file_content is already async using aiofiles
async def write_file_content(filepath: str, content: str) -> None:
    """Write file content using async I/O. Handles potential errors.

    Args:
        filepath: Path to the file to write
        content: Content to write

    Raises:
        ToolError: If writing fails.
    """
    try:
        # Ensure parent directory exists (use sync os.path as it's usually fast)
        parent_dir = os.path.dirname(filepath)
        if parent_dir: # Check if parent_dir is not empty (e.g., for root files)
             # Use sync os.makedirs, as async version might not be necessary
             # unless creating dirs is frequent and on slow filesystems.
             # If needed, replace with: await aiofiles.os.makedirs(parent_dir, exist_ok=True)
             os.makedirs(parent_dir, exist_ok=True)

        async with aiofiles.open(filepath, mode='w', encoding='utf-8') as f:
            await f.write(content)
    except OSError as e:
        logger.error(f"OS error writing file {filepath}: {e}", emoji_key="error")
        raise ToolError(f"Error writing file: {str(e)}", context={"path": filepath})
    except Exception as e:
        logger.error(f"Unexpected error writing file {filepath}: {e}", exc_info=True, emoji_key="error")
        raise ToolError(f"An unexpected error occurred while writing file: {str(e)}", context={"path": filepath})

# apply_file_edits is already async, just ensure called helpers are awaited
async def apply_file_edits(
    filepath: str,
    edits: List[Dict[str, str]],
    dry_run: bool = False
) -> Tuple[str, str]:
    """Apply a series of text replacements to a file asynchronously.

    Args:
        filepath: Path to the file to edit
        edits: List of edit operations, each with 'oldText' and 'newText'
        dry_run: If True, don't write changes to disk

    Returns:
        Tuple of (diff, new_content)

    Raises:
        ToolError: If reading/writing fails or edits are invalid.
        ToolInputError: If edits are malformed or text not found.
    """
    # Read original content asynchronously
    content = await read_file_content(filepath) # Handles errors internally

    original_content = content

    # Apply each edit (sync string operations)
    for i, edit in enumerate(edits):
        old_text = edit.get('oldText') # Use .get() for safer access
        new_text = edit.get('newText')

        # Ensure both oldText and newText are provided and are strings
        if old_text is None or not isinstance(old_text, str):
             raise ToolInputError(
                f"Edit #{i+1} is missing 'oldText' or it's not a string.",
                param_name=f"edits[{i}]",
                provided_value=edit
            )
        if new_text is None or not isinstance(new_text, str):
             raise ToolInputError(
                f"Edit #{i+1} is missing 'newText' or it's not a string.",
                param_name=f"edits[{i}]",
                provided_value=edit
            )

        # Try exact replacement first
        if old_text in content:
            content = content.replace(old_text, new_text)
        else:
            # If exact match fails, try line-by-line with trimmed whitespace
            old_lines = old_text.splitlines()
            content_lines = content.splitlines()
            found_match = False

            for line_idx in range(len(content_lines) - len(old_lines) + 1):
                is_match = all(
                    old_lines[j].strip() == content_lines[line_idx + j].strip()
                    for j in range(len(old_lines))
                )

                if is_match:
                    # Found a match with trimmed lines, replace with proper indentation
                    new_lines = new_text.splitlines()

                    # Preserve the original indentation of the first line
                    indent = ""
                    if len(content_lines) > line_idx:
                        first_line = content_lines[line_idx]
                        indent = first_line[:len(first_line) - len(first_line.lstrip())]

                    # Apply indentation to all new lines
                    indented_new_lines = [indent + line.lstrip() for line in new_lines]

                    # Replace the lines
                    content_lines[line_idx : line_idx + len(old_lines)] = indented_new_lines
                    content = '\n'.join(content_lines)
                    found_match = True
                    break # Stop searching once a match is found and replaced for this edit

            if not found_match:
                # No match found even with whitespace trimming
                raise ToolInputError(
                    f"Could not find text to replace in edit #{i+1}: '{old_text[:100]}{'...' if len(old_text)>100 else ''}'",
                    param_name=f"edits[{i}].oldText",
                    provided_value=edit
                )

    # Create diff asynchronously
    diff = await create_unified_diff(original_content, content, filepath)

    # Write the changes if not a dry run asynchronously
    if not dry_run:
        await write_file_content(filepath, content) # Handles errors internally

    return diff, content

# format_mcp_content remains synchronous
def format_mcp_content(text_content: str) -> List[Dict[str, Any]]:
    """Format text content according to MCP protocol schema.

    Args:
        text_content: Text to format

    Returns:
        MCP-formatted content array
    """
    # Basic check for large content to avoid overwhelming downstream systems
    MAX_LEN = 50000 # Example limit, adjust as needed
    if len(text_content) > MAX_LEN:
        logger.warning(f"Formatting potentially large content ({len(text_content)} chars)")
        # Optionally truncate or handle differently
        # text_content = text_content[:MAX_LEN] + "\n... [truncated]"

    return [{"type": "text", "text": text_content}]

# create_tool_response remains synchronous
def create_tool_response(content: Any, is_error: bool = False) -> Dict[str, Any]:
    """Create a properly formatted tool response according to MCP protocol.

    Args:
        content: The content to include in the response
        is_error: Whether this response represents an error

    Returns:
        Properly formatted tool response dictionary
    """
    formatted_content: List[Dict[str, Any]]

    if isinstance(content, str):
        formatted_content = format_mcp_content(content)
    elif isinstance(content, dict) and 'files' in content and isinstance(content['files'], list):
        # Special handling for read_multiple_files structure
        # Convert file contents to text blocks, keep structure
        formatted_files = []
        for file_result in content['files']:
            if file_result.get('success') and 'content' in file_result:
                 formatted_files.append({
                     "type": "text",
                     "text": f"--- File: {file_result['path']} ({file_result.get('size', 'N/A')} bytes) ---\n{file_result['content']}"
                 })
            elif 'error' in file_result:
                 formatted_files.append({
                     "type": "text",
                     "text": f"--- File: {file_result['path']} (Error) ---\n{file_result['error']}"
                 })
        # Add summary
        summary = f"Read {content.get('succeeded', 0)} files successfully, {content.get('failed', 0)} failed."
        formatted_content = [{"type": "text", "text": summary}] + formatted_files
    elif isinstance(content, dict):
        # Attempt to pretty-print other dictionaries as JSON
        try:
            json_content = json.dumps(content, indent=2, ensure_ascii=False)
            formatted_content = format_mcp_content(f"```json\n{json_content}\n```")
        except TypeError:
            # Fallback if dict is not JSON serializable
            formatted_content = format_mcp_content(str(content))
    elif isinstance(content, list) and all(isinstance(item, dict) and "type" in item for item in content):
        # Assume it's already MCP formatted
        formatted_content = content
    else:
        # Convert anything else to string and format
        formatted_content = format_mcp_content(str(content))

    response = {
        "content": formatted_content
    }

    if is_error:
        response["isError"] = True

    return response

# --- Tool Functions (Async Conversions) ---

@with_tool_metrics
@with_error_handling
async def read_file(
    path: str,
) -> Dict[str, Any]:
    """Read the contents of a file from the filesystem asynchronously.

    Ensures the path is within allowed directories and handles text/binary files.

    Args:
        path: Path to the file to read.

    Returns:
        A properly formatted MCP tool response containing the file content or error.
    """
    start_time = time.time()

    try:
        validated_path = await validate_path(path, must_exist=True)

        if not await aiofiles.os.path.isfile(validated_path):
             # Use create_tool_response for consistent error formatting
            return create_tool_response(
                f"Error: Path '{path}' exists but is not a file.",
                is_error=True
            )

        content: str
        is_binary = False
        try:
            # Attempt to read as text first
            async with aiofiles.open(validated_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
        except UnicodeDecodeError:
            # If text fails, read as binary
            logger.warning(f"File {path} is not UTF-8 encoded, reading as binary.", emoji_key="warning")
            try:
                async with aiofiles.open(validated_path, mode='rb') as f:
                    binary_content = await f.read()
                # Provide a safe representation for binary data
                hex_preview = binary_content[:200].hex() # Show first 200 bytes as hex
                content = f"<binary file detected, hex preview (first 200 bytes)>: {hex_preview}"
                is_binary = True
            except OSError as bin_err:
                 logger.error(f"Error reading file {path} as binary after UTF-8 failure: {bin_err}", emoji_key="error")
                 return create_tool_response(f"Error reading file as binary: {str(bin_err)}", is_error=True)
        except OSError as text_err:
             logger.error(f"Error reading file {path} as text: {text_err}", emoji_key="error")
             return create_tool_response(f"Error reading file: {str(text_err)}", is_error=True)

        # Get size asynchronously
        file_size = (await aiofiles.os.stat(validated_path)).st_size
        processing_time = time.time() - start_time

        logger.success(
            f"Successfully read file: {path}",
            emoji_key="file",
            size=file_size,
            time=processing_time,
            is_binary=is_binary
        )

        result_message = (
            f"File: {os.path.basename(validated_path)}\n"
            f"Path: {validated_path}\n"
            f"Size: {file_size} bytes\n"
            f"{'(Binary content preview)' if is_binary else ''}\n"
            f"\n{content}"
        )

        return create_tool_response(result_message)

    # Catch exceptions from validate_path or other unexpected issues
    except (ToolInputError, ToolError) as e:
        logger.error(f"Error in read_file '{path}': {e}", emoji_key="error", details=getattr(e, 'context', None))
        return create_tool_response(f"Error reading file: {str(e)}", is_error=True)
    except Exception as e:
        logger.error(f"Unexpected error in read_file '{path}': {e}", exc_info=True, emoji_key="error")
        return create_tool_response(f"An unexpected error occurred: {str(e)}", is_error=True)


@with_tool_metrics
@with_error_handling
async def read_multiple_files(
    paths: List[str],
) -> Dict[str, Any]:
    """Read the contents of multiple files asynchronously and concurrently.

    Validates each path and handles errors individually.

    Args:
        paths: List of file paths to read.

    Returns:
        Dictionary summarizing results, suitable for create_tool_response.
    """
    start_time = time.time()

    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise ToolInputError(
            "Paths must be a list of strings.",
            param_name="paths",
            provided_value=paths
        )

    if not paths:
        raise ToolInputError(
            "At least one path must be provided.",
            param_name="paths",
            provided_value=paths
        )

    # Define the async task for reading a single file
    async def read_single_file_task(path: str) -> Dict[str, Any]:
        try:
            validated_path = await validate_path(path, must_exist=True)

            if not await aiofiles.os.path.isfile(validated_path):
                return {"path": path, "error": f"Path exists but is not a file", "success": False}

            content: str
            is_binary = False
            try:
                async with aiofiles.open(validated_path, mode='r', encoding='utf-8') as f:
                    content = await f.read()
            except UnicodeDecodeError:
                 logger.warning(f"File {path} in multi-read is not UTF-8, reading as binary.", emoji_key="warning")
                 try:
                     async with aiofiles.open(validated_path, mode='rb') as f:
                         binary_content = await f.read()
                     hex_preview = binary_content[:200].hex()
                     content = f"<binary file detected, hex preview (first 200 bytes)>: {hex_preview}"
                     is_binary = True
                 except OSError as bin_err:
                     return {"path": path, "error": f"Error reading as binary: {str(bin_err)}", "success": False}
            except OSError as text_err:
                 return {"path": path, "error": f"Error reading as text: {str(text_err)}", "success": False}

            file_size = (await aiofiles.os.stat(validated_path)).st_size

            return {
                "path": validated_path,
                "content": content,
                "size": file_size,
                "success": True,
                "is_binary": is_binary
            }
        except (ToolInputError, ToolError) as e:
            # Handle validation or specific tool errors
            return {"path": path, "error": str(e), "success": False}
        except Exception as e:
            # Catch unexpected errors during processing of a single file
            logger.error(f"Unexpected error reading single file {path} in multi-read: {e}", exc_info=True, emoji_key="error")
            return {"path": path, "error": f"Unexpected error: {str(e)}", "success": False}

    # Execute reads concurrently
    results = await asyncio.gather(*(read_single_file_task(p) for p in paths), return_exceptions=False) # Let tasks handle own exceptions

    successful_count = sum(1 for r in results if r.get("success", False))
    failed_count = len(paths) - successful_count
    processing_time = time.time() - start_time

    logger.success(
        f"Read multiple files: {successful_count} succeeded, {failed_count} failed",
        emoji_key="file",
        total_files=len(paths),
        time=processing_time
    )

    # Return a dictionary structure that create_tool_response can format well
    return {
        "files": results,
        "succeeded": successful_count,
        "failed": failed_count,
        "success": True # Overall operation success (individual files might fail)
    }


@with_tool_metrics
@with_error_handling
async def write_file(
    path: str,
    content: str,
) -> Dict[str, Any]:
    """Write content to a file asynchronously, creating/overwriting as needed.

    Ensures the path is within allowed directories.

    Args:
        path: Path to the file to write.
        content: Text content to write.

    Returns:
        Dictionary confirming success and details.
    """
    start_time = time.time()

    if not isinstance(content, str):
         raise ToolInputError("Content must be a string.", param_name="content", provided_value=type(content))

    validated_path = await validate_path(path, must_exist=False) # File doesn't need to exist

    # write_file_content handles directory creation and writing errors
    await write_file_content(validated_path, content)

    # Verify write success and get size
    try:
        stat_info = await aiofiles.os.stat(validated_path)
        file_size = stat_info.st_size
    except OSError as e:
        raise ToolError(f"File written but failed to get status afterwards: {str(e)}", context={"path": validated_path})

    processing_time = time.time() - start_time

    logger.success(
        f"Successfully wrote file: {path}",
        emoji_key="file",
        size=file_size,
        time=processing_time
    )

    return {
        "path": validated_path,
        "size": file_size,
        "success": True
    }

@with_tool_metrics
@with_error_handling
async def edit_file(
    path: str,
    edits: List[Dict[str, str]],
    dry_run: bool = False
) -> Dict[str, Any]:
    """Edit a file asynchronously by applying text replacements.

    Includes dry run option and generates a diff.

    Args:
        path: Path to the file to edit.
        edits: List of edit operations [{'oldText': ..., 'newText': ...}].
        dry_run: If True, preview changes without saving.

    Returns:
        Dictionary with diff, success status, and dry_run flag.
    """
    start_time = time.time()

    validated_path = await validate_path(path, must_exist=True) # File must exist

    if not await aiofiles.os.path.isfile(validated_path):
        raise ToolInputError(f"Path '{path}' is not a file.", param_name="path", provided_value=path)

    # Validate edits structure (basic check)
    if not isinstance(edits, list) or not edits:
        raise ToolInputError(
            "Edits must be a non-empty list of operations.",
            param_name="edits", provided_value=edits
        )
    # Deeper validation happens within apply_file_edits

    # apply_file_edits handles reading, editing logic, diffing, and writing (if not dry_run)
    diff, _ = await apply_file_edits(validated_path, edits, dry_run) # Ignore new_content here

    processing_time = time.time() - start_time

    logger.success(
        f"Successfully {'previewed' if dry_run else 'applied'} edits for file: {path}",
        emoji_key="file",
        num_edits=len(edits),
        dry_run=dry_run,
        time=processing_time
    )

    return {
        "path": validated_path,
        "diff": diff if diff else ("No changes detected." if not dry_run else "No changes to apply."),
        "success": True,
        "dry_run": dry_run
    }


@with_tool_metrics
@with_error_handling
async def create_directory(
    path: str,
) -> Dict[str, Any]:
    """Create a directory asynchronously (like 'mkdir -p').

    Ensures path is within allowed directories. Idempotent.

    Args:
        path: Path to the directory to create.

    Returns:
        Dictionary confirming success and whether it was newly created.
    """
    start_time = time.time()

    validated_path = await validate_path(path, must_exist=False)

    created = False
    try:
        if await aiofiles.os.path.exists(validated_path):
            if not await aiofiles.os.path.isdir(validated_path):
                 raise ToolInputError(
                    f"Path '{path}' exists but is not a directory.",
                    param_name="path", provided_value=path
                )
            # Directory already exists
            logger.info(f"Directory already exists: {path}", emoji_key="directory")
        else:
            # Create directory(ies)
            await aiofiles.os.makedirs(validated_path, exist_ok=True)
            created = True
            logger.success(f"Successfully created directory: {path}", emoji_key="directory")

    except OSError as e:
        raise ToolError(f"Error creating directory '{path}': {str(e)}", context={"path": path})
    except Exception as e:
         logger.error(f"Unexpected error creating directory {path}: {e}", exc_info=True, emoji_key="error")
         raise ToolError(f"An unexpected error occurred creating directory: {str(e)}", context={"path": path})


    processing_time = time.time() - start_time

    return {
        "path": validated_path,
        "created": created,
        "success": True
    }


@with_tool_metrics
@with_error_handling
async def list_directory(
    path: str,
) -> Dict[str, Any]:
    """List files and directories asynchronously within a given directory.

    Args:
        path: Path to the directory to list.

    Returns:
        Dictionary with directory entries (name, type, size for files).
    """
    start_time = time.time()

    validated_path = await validate_path(path, must_exist=True)

    if not await aiofiles.os.path.isdir(validated_path):
        raise ToolInputError(f"Path '{path}' is not a directory.", param_name="path", provided_value=path)

    entries = []
    try:
        # Use async iteration with aiofiles.os.scandir
        async for entry in aiofiles.os.scandir(validated_path):
            try:
                entry_info = {"name": entry.name}
                is_dir = await entry.is_dir() # Use await on async method
                is_file = await entry.is_file() # Use await on async method

                if is_dir:
                    entry_info["type"] = "directory"
                elif is_file:
                    entry_info["type"] = "file"
                    try:
                         # Get size asynchronously
                        stat_res = await entry.stat()
                        entry_info["size"] = stat_res.st_size
                    except OSError as stat_err:
                         logger.warning(f"Could not stat file {entry.path}: {stat_err}", emoji_key="warning")
                         entry_info["error"] = f"Could not get size: {stat_err}"
                else:
                    # Handle other types like symlinks, etc., if necessary
                    entry_info["type"] = "other"

                entries.append(entry_info)
            except OSError as entry_err:
                 logger.warning(f"Could not process entry {entry.path}: {entry_err}", emoji_key="warning")
                 entries.append({"name": entry.name, "type": "error", "error": str(entry_err)})


        # Sort entries - directories first, then files, both alphabetically (sync sort is fine)
        entries.sort(key=lambda e: (0 if e.get("type") == "directory" else (1 if e.get("type") == "file" else 2), e["name"]))

    except OSError as e:
        raise ToolError(f"Error listing directory '{path}': {str(e)}", context={"path": path})
    except Exception as e:
        logger.error(f"Unexpected error listing directory {path}: {e}", exc_info=True, emoji_key="error")
        raise ToolError(f"An unexpected error occurred listing directory: {str(e)}", context={"path": path})

    processing_time = time.time() - start_time

    logger.success(
        f"Listed directory: {path} ({len(entries)} entries)",
        emoji_key="directory",
        time=processing_time
    )

    return {
        "path": validated_path,
        "entries": entries,
        "success": True
    }

@with_tool_metrics
@with_error_handling
async def directory_tree(
    path: str,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """Get a recursive tree view of a directory structure asynchronously.

    Args:
        path: Path to the directory for the tree view.
        max_depth: Maximum recursion depth (-1 for unlimited).

    Returns:
        Dictionary with the hierarchical tree structure.
    """
    start_time = time.time()

    validated_path = await validate_path(path, must_exist=True)

    if not await aiofiles.os.path.isdir(validated_path):
        raise ToolInputError(f"Path '{path}' is not a directory.", param_name="path", provided_value=path)

    # Define the recursive async helper function
    async def build_tree_recursive(current_path: str, current_depth: int) -> List[Dict[str, Any]]:
        if max_depth != -1 and current_depth > max_depth:
            return [] # Reached max depth

        if current_depth > 10 and max_depth == -1: # Add safeguard for unlimited depth
             logger.warning(f"Reached depth {current_depth} in unlimited tree view for {path}. Stopping recursion.", emoji_key="warning")
             return [{"name": "... (depth limit reached)", "type": "info"}]


        result: List[Dict[str, Any]] = []
        try:
            async for entry in aiofiles.os.scandir(current_path):
                entry_data: Dict[str, Any] = {"name": entry.name}
                try:
                    # Check type asynchronously
                    if await entry.is_dir():
                        entry_data["type"] = "directory"
                        # Recurse asynchronously
                        entry_data["children"] = await build_tree_recursive(entry.path, current_depth + 1)
                    elif await entry.is_file():
                         entry_data["type"] = "file"
                         try:
                             # Get size asynchronously
                             stat_res = await entry.stat()
                             entry_data["size"] = stat_res.st_size
                         except OSError as stat_err:
                             logger.warning(f"Could not stat file {entry.path} in tree: {stat_err}", emoji_key="warning")
                             entry_data["error"] = f"Could not get size: {stat_err}"
                    else:
                         entry_data["type"] = "other"

                    result.append(entry_data)
                except OSError as entry_err:
                    logger.warning(f"Could not process entry {entry.path} in tree: {entry_err}", emoji_key="warning")
                    result.append({"name": entry.name, "type": "error", "error": str(entry_err)})

            # Sort entries at the current level (sync sort is fine)
            result.sort(key=lambda e: (0 if e.get("type") == "directory" else (1 if e.get("type") == "file" else 2), e["name"]))
            return result
        except OSError as e:
            logger.error(f"Error scanning directory {current_path} for tree: {str(e)}", emoji_key="error")
            # Return error indicator instead of raising to allow partial trees
            return [{"name": f"... (Error scanning: {e})", "type": "error"}]
        except Exception as e:
            logger.error(f"Unexpected error scanning directory {current_path} for tree: {e}", exc_info=True, emoji_key="error")
            return [{"name": f"... (Unexpected error: {e})", "type": "error"}]


    # Generate tree starting from the validated path
    tree = await build_tree_recursive(validated_path, 0)

    processing_time = time.time() - start_time

    logger.success(
        f"Generated directory tree for: {path}",
        emoji_key="directory",
        max_depth=max_depth,
        time=processing_time
    )

    return {
        "path": validated_path,
        "tree": tree,
        "success": True
    }

@with_tool_metrics
@with_error_handling
async def move_file(
    source: str,
    destination: str,
) -> Dict[str, Any]:
    """Move or rename a file or directory asynchronously.

    Ensures both source and destination are within allowed directories.

    Args:
        source: Path to the file or directory to move.
        destination: New path for the file or directory.

    Returns:
        Dictionary confirming the move operation.
    """
    start_time = time.time()

    # Validate source path (must exist)
    validated_source = await validate_path(source, must_exist=True)

    # Validate destination path (doesn't need to exist, but parent must be allowed and exist)
    validated_dest = await validate_path(destination, must_exist=False)

    try:
        # Check if destination already exists (async)
        if await aiofiles.os.path.exists(validated_dest):
            raise ToolInputError(
                f"Destination '{destination}' already exists.",
                param_name="destination", provided_value=destination
            )

        # Parent directory existence checked during validate_path(must_exist=False)

        # Move file or directory asynchronously
        await aiofiles.os.rename(validated_source, validated_dest)

    except OSError as e:
        raise ToolError(f"Error moving/renaming from '{source}' to '{destination}': {str(e)}", context={"source": source, "destination": destination})
    except Exception as e:
        logger.error(f"Unexpected error moving {source} to {destination}: {e}", exc_info=True, emoji_key="error")
        raise ToolError(f"An unexpected error occurred during move: {str(e)}", context={"source": source, "destination": destination})


    processing_time = time.time() - start_time

    logger.success(
        f"Moved '{source}' to '{destination}'",
        emoji_key="file",
        time=processing_time
    )

    return {
        "source": validated_source,
        "destination": validated_dest,
        "success": True
    }


# Helper for async recursive directory walking
async def async_walk(top: str, topdown=True, onerror=None, followlinks=False, exclude_patterns: Optional[List[str]] = None, base_path: Optional[str] = None):
    """Async version of os.walk using aiofiles.os.scandir. Handles excludes."""
    if base_path is None:
        base_path = top # Keep track of the original root for exclusion matching

    dirs = []
    nondirs = []
    try:
        scandir_it = aiofiles.os.scandir(top)
        async for entry in scandir_it:
            # Calculate relative path for exclusion check
            try:
                 # entry.path should be absolute if 'top' is absolute
                 rel_path = os.path.relpath(entry.path, base_path)
            except ValueError:
                 # Handle cases where paths might be on different drives (Windows)
                 rel_path = entry.name # Fallback for exclusion

            is_excluded = False
            if exclude_patterns:
                for pattern in exclude_patterns:
                    # Match against full relative path or just name
                    if fnmatch(rel_path, pattern) or fnmatch(entry.name, pattern):
                         is_excluded = True
                         logger.debug(f"Excluding '{entry.path}' due to pattern '{pattern}'")
                         break
            if is_excluded:
                continue

            try:
                is_dir = await entry.is_dir()
            except OSError as e:
                 if onerror is not None:
                     onerror(e)
                 is_dir = False # Treat error state as not a directory for walk purposes

            if is_dir:
                dirs.append(entry.name)
            else:
                nondirs.append(entry.name)
    except OSError as err:
        if onerror is not None:
            onerror(err)
        return # Stop iteration for this path on error

    if topdown:
        yield top, dirs, nondirs

    # Recurse into subdirectories
    for name in dirs:
        new_path = os.path.join(top, name)
        if followlinks or not await aiofiles.os.path.islink(new_path):
             # Use 'async for' to delegate iteration
            async for x in async_walk(new_path, topdown, onerror, followlinks, exclude_patterns, base_path):
                yield x

    if not topdown:
        yield top, dirs, nondirs


@with_tool_metrics
@with_error_handling
async def search_files(
    path: str,
    pattern: str,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Search for files/directories matching a pattern asynchronously and recursively.

    Performs case-insensitive substring matching on names. Supports exclusion patterns.

    Args:
        path: Directory to search within.
        pattern: Text pattern to find in names (case-insensitive substring).
        exclude_patterns: Optional list of glob patterns to exclude.

    Returns:
        Dictionary with search results.
    """
    start_time = time.time()

    validated_path = await validate_path(path, must_exist=True)

    if not await aiofiles.os.path.isdir(validated_path):
        raise ToolInputError(f"Path '{path}' is not a directory.", param_name="path", provided_value=path)

    if not isinstance(pattern, str) or not pattern:
         raise ToolInputError("Search pattern must be a non-empty string.", param_name="pattern", provided_value=pattern)

    if exclude_patterns and not isinstance(exclude_patterns, list):
        raise ToolInputError("Exclude patterns must be a list of strings.", param_name="exclude_patterns", provided_value=exclude_patterns)

    matches = []
    pattern_lower = pattern.lower()
    search_errors = []

    def onerror(os_error: OSError):
        logger.warning(f"Permission or access error during search: {os_error}", emoji_key="warning")
        search_errors.append(str(os_error))

    try:
        # Use the custom async_walk
        async for root, dirs, files in async_walk(validated_path, onerror=onerror, exclude_patterns=exclude_patterns, base_path=validated_path):
            # Check matching directories (already filtered by exclude in async_walk)
            for dirname in dirs:
                if pattern_lower in dirname.lower():
                    matches.append(os.path.join(root, dirname))

            # Check matching files (already filtered by exclude in async_walk)
            for filename in files:
                if pattern_lower in filename.lower():
                    matches.append(os.path.join(root, filename))

    except Exception as e:
        # Catch unexpected errors during the async iteration itself
        logger.error(f"Unexpected error during file search in {path}: {e}", exc_info=True, emoji_key="error")
        raise ToolError(f"An unexpected error occurred during search: {str(e)}", context={"path": path, "pattern": pattern})


    processing_time = time.time() - start_time

    logger.success(
        f"Searched for '{pattern}' in {path} ({len(matches)} matches)",
        emoji_key="search",
        errors_encountered=len(search_errors),
        time=processing_time
    )

    result = {
        "path": validated_path,
        "pattern": pattern,
        "matches": sorted(matches), # Sort results for consistency
        "success": True
    }
    if search_errors:
        result["warnings"] = [f"Search encountered access errors: {len(search_errors)}"]
        # Optionally include specific errors if needed, truncated:
        # result["warning_details"] = search_errors[:5]

    return result


@with_tool_metrics
@with_error_handling
async def get_file_info(
    path: str,
) -> Dict[str, Any]:
    """Get detailed metadata about a file or directory asynchronously.

    Args:
        path: Path to the file or directory.

    Returns:
        Dictionary containing detailed file information.
    """
    start_time = time.time()

    # Validate path (must exist)
    validated_path = await validate_path(path, must_exist=True)

    # Get file information asynchronously using the helper
    info = await format_file_info(validated_path) # Handles its own errors

    # Check if the helper returned an error structure
    if "error" in info:
         # Propagate the error appropriately
         # Use ToolError if it was an OS issue, ToolInputError might not be right here
         raise ToolError(f"Failed to get file info: {info['error']}", context={"path": path})


    processing_time = time.time() - start_time

    logger.success(
        f"Got file info for: {path}",
        emoji_key="file",
        time=processing_time
    )

    # Add success flag to the info dictionary
    info["success"] = True
    return info


@with_tool_metrics
@with_error_handling
async def list_allowed_directories() -> Dict[str, Any]:
    """List all directories configured as allowed for filesystem access.

    Returns:
        Dictionary containing the list of allowed directories.
    """
    start_time = time.time()

    # get_allowed_dirs is synchronous and cached
    allowed_dirs = get_allowed_dirs()

    processing_time = time.time() - start_time

    logger.success(
        f"Listed {len(allowed_dirs)} allowed directories",
        emoji_key="config",
        time=processing_time
    )

    return {
        "directories": allowed_dirs,
        "success": True
    }