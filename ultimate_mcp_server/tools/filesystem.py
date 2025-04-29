"""Secure asynchronous filesystem tools for Ultimate MCP Server.

This module provides secure asynchronous filesystem operations, including reading, writing,
deleting, and manipulating files and directories, with robust security controls to limit access
and optional heuristics to prevent accidental mass deletion/modification by LLMs.
"""

import asyncio
import datetime  # Using datetime for standard timestamp representation.
import difflib
import json
import math  # For isnan checks
import os
import statistics  # For calculating standard deviation in protection heuristics
import time
from fnmatch import fnmatch  # Keep sync fnmatch for pattern matching
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Set, Tuple, Union, cast

import aiofiles
import aiofiles.os
from pydantic import BaseModel

from ultimate_mcp_server.config import FilesystemConfig, GatewayConfig, get_config
from ultimate_mcp_server.exceptions import ToolError, ToolInputError
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.filesystem")


class ProtectionTriggeredError(ToolError):
    """Exception raised when a security protection measure is triggered."""

    def __init__(self, message, protection_type=None, context=None, details=None):
        """Initialize the protection triggered error.

        Args:
            message: Error message
            protection_type: Type of protection triggered (e.g., "deletion_protection", "path_protection")
            context: Context information about the protection trigger
            details: Additional error details
        """
        error_details = details or {}
        if protection_type:
            error_details["protection_type"] = protection_type

        self.context = context or {}
        if context:
            error_details["context"] = context

        super().__init__(message, error_code="PROTECTION_TRIGGERED", details=error_details)


# --- Configuration and Security ---


def get_filesystem_config() -> "FilesystemConfig":  # Use type hint from config module
    """Get filesystem configuration object from the main config."""
    cfg: GatewayConfig = get_config()
    # Access the validated FilesystemConfig object directly
    fs_config = cfg.filesystem
    if not fs_config:  # Should not happen with default_factory, but check defensively
        logger.error(
            "Filesystem configuration missing after load. Using defaults.", emoji_key="config"
        )
        from ultimate_mcp_server.config import (
            FilesystemConfig,  # Local import to avoid circularity at top level
        )

        return FilesystemConfig()
    return fs_config


def get_protection_config(operation_type: Literal["deletion", "modification"]) -> Dict[str, Any]:
    """Get protection settings for a specific operation type as a dictionary."""
    fs_config = get_filesystem_config()
    protection_attr_name = f"file_{operation_type}_protection"

    if hasattr(fs_config, protection_attr_name):
        protection_config_obj = getattr(fs_config, protection_attr_name)
        if protection_config_obj and isinstance(
            protection_config_obj, BaseModel
        ):  # Check it's a Pydantic model instance
            # Convert the Pydantic model to a dictionary for consistent access
            return protection_config_obj.model_dump()
        else:
            logger.warning(
                f"Protection config for '{operation_type}' is not a valid model instance. Using defaults.",
                emoji_key="config",
            )
    else:
        logger.warning(
            f"Protection config attribute '{protection_attr_name}' not found. Using defaults.",
            emoji_key="config",
        )

    # Return default dictionary structure if config is missing or invalid
    # Fetch defaults from the Pydantic model definition if possible
    try:
        from ultimate_mcp_server.config import FilesystemProtectionConfig

        return FilesystemProtectionConfig().model_dump()
    except ImportError:  # Fallback if import fails
        return {
            "enabled": False,
            "max_files_threshold": 100,
            "datetime_stddev_threshold_sec": 60 * 60 * 24 * 30,
            "file_type_variance_threshold": 5,
            "max_stat_errors_pct": 10.0,
        }


def get_allowed_directories() -> List[str]:
    """Get allowed directories from configuration.

    Reads the configuration, normalizes paths (absolute, resolves symlinks),
    and returns a list of unique allowed directory paths. Assumes paths were expanded
    during config load.

    Returns:
        List of normalized, absolute directory paths that can be accessed.
    """
    fs_config = get_filesystem_config()
    # Access the already expanded list from the validated config object
    allowed_config: List[str] = fs_config.allowed_directories

    if not allowed_config:
        logger.warning(
            "No filesystem directories configured or loaded for access. All operations may be rejected.",
            emoji_key="security",
        )
        return []

    # Paths should already be expanded and absolute from config loading.
    # We still need to normalize and ensure uniqueness.
    normalized: List[str] = []
    for d in allowed_config:
        try:
            # Basic normalization (separator consistency)
            norm_d = os.path.normpath(d)
            if norm_d not in normalized:  # Avoid duplicates
                normalized.append(norm_d)
        except Exception as e:
            # Log errors during normalization but continue
            logger.error(
                f"Error normalizing configured allowed directory '{d}': {e}. Skipping.",
                emoji_key="config",
            )

    if not normalized and allowed_config:
        logger.error(
            "Filesystem access potentially misconfigured: No valid allowed directories remain after normalization.",
            emoji_key="security",
        )
    elif not normalized:
        # Warning about no configured dirs was logged earlier if allowed_config was empty.
        pass

    # Debug log the final effective list used by tools
    logger.debug(
        f"Filesystem tools operating with {len(normalized)} normalized allowed directories",
        allowed_directories=normalized,
    )
    return normalized


# --- Path Validation ---
async def validate_path(
    path: str,
    check_exists: Optional[bool] = None,
    check_parent_writable: bool = False,
    resolve_symlinks: bool = False,
) -> str:
    """Validate a path for security and accessibility using async I/O.

    Performs several checks:
    1.  Ensures path is a non-empty string.
    2.  Normalizes the path (expands user, makes absolute, resolves '../').
    3.  Checks if the normalized path is within the configured allowed directories.
    4.  If check_exists is True, checks if the path exists. If False, checks it does NOT exist.
    5.  If resolve_symlinks is True, resolves symbolic links and re-validates the real path against allowed dirs.
    6.  If check_exists is False, checks parent directory existence.
    7.  If `check_parent_writable` is True and path likely needs creation, checks parent directory write permissions.

    Args:
        path: The file or directory path input string to validate.
        check_exists: If True, path must exist. If False, path must NOT exist. If None, existence is not checked.
        check_parent_writable: If True and path doesn't exist/creation is implied, check if parent dir is writable.
        resolve_symlinks: If True, follows symlinks and returns their target path. If False, keeps the symlink path.

    Returns:
        The normalized, absolute, validated path string.

    Raises:
        ToolInputError: If the path is invalid (format, permissions, existence violation,
                        outside allowed dirs, symlink issue).
        ToolError: For underlying filesystem errors or configuration issues.
    """
    if not path or not isinstance(path, str):
        raise ToolInputError(
            "Path must be a non-empty string.",
            param_name="path",
            provided_value=repr(path),  # Use repr for clarity on non-string types
        )

    # Path normalization (sync, generally fast)
    try:
        path_expanded = os.path.expanduser(path)
        path_abs = os.path.abspath(path_expanded)
        # Normalize '.','..' and separators
        normalized_path = os.path.normpath(path_abs)
    except Exception as e:
        raise ToolInputError(
            f"Invalid path format or resolution error: {str(e)}",
            param_name="path",
            provided_value=path,
        ) from e

    # --- Use get_allowed_directories which reads from config ---
    allowed_dirs = get_allowed_directories()
    if not allowed_dirs:
        raise ToolError(
            "Filesystem access is disabled: No allowed directories are configured or loadable.",
            context={"configured_directories": 0},  # Provide context
        )

    # Ensure normalized_path is truly *under* an allowed dir.
    is_allowed = False
    original_validated_path = normalized_path  # Store before symlink resolution
    for allowed_dir in allowed_dirs:
        # Ensure allowed_dir is also normalized for comparison
        norm_allowed_dir = os.path.normpath(allowed_dir)
        # Path must be exactly the allowed dir or start with the allowed dir + separator.
        if normalized_path == norm_allowed_dir or normalized_path.startswith(
            norm_allowed_dir + os.sep
        ):
            is_allowed = True
            break

    if not is_allowed:
        logger.warning(
            f"Path '{normalized_path}' denied access. Not within allowed directories: {allowed_dirs}",
            emoji_key="security",
        )
        raise ToolInputError(
            f"Access denied: Path '{path}' resolves to '{normalized_path}', which is outside the allowed directories.",
            param_name="path",
            provided_value=path,
            # Add context about allowed dirs for debugging? Potentially sensitive.
            # context={"allowed": allowed_dirs}
        )

    # Filesystem checks using aiofiles.os
    current_validated_path = normalized_path  # Start with normalized path
    is_symlink = False
    symlink_target_path = None

    try:
        # Use stat with follow_symlinks=False to check the item itself (similar to lstat)
        try:
            lstat_info = await aiofiles.os.stat(current_validated_path, follow_symlinks=False)
            path_exists_locally = True  # stat succeeded, so the path entry itself exists
            is_symlink = os.path.stat.S_ISLNK(lstat_info.st_mode)

            if is_symlink:
                try:
                    # Get the target for information purposes
                    symlink_target = await aiofiles.os.readlink(current_validated_path)
                    symlink_target_path = symlink_target
                    logger.debug(f"Path '{path}' is a symlink pointing to '{symlink_target}'")
                except OSError as link_err:
                    logger.warning(
                        f"Error reading symlink target for '{current_validated_path}': {link_err}"
                    )

        except FileNotFoundError:
            path_exists_locally = False
            is_symlink = False
        except OSError as e:
            # Handle other OS errors during stat check
            logger.error(
                f"OS Error during stat check for '{current_validated_path}': {e}", exc_info=True
            )
            raise ToolError(
                f"Filesystem error checking path status for '{path}': {str(e)}",
                context={"path": path, "resolved_path": current_validated_path},
            ) from e

        # Resolve symlink if it exists and re-validate
        if is_symlink and resolve_symlinks:
            try:
                # Use synchronous os.path.realpath since aiofiles.os.path doesn't have it
                real_path = os.path.realpath(current_validated_path)
                real_normalized = os.path.normpath(real_path)
                symlink_target_path = real_normalized  # noqa: F841

                # Re-check if the *real* resolved path is within allowed directories
                is_real_allowed = False
                for allowed_dir in allowed_dirs:
                    norm_allowed_dir = os.path.normpath(allowed_dir)
                    if real_normalized == norm_allowed_dir or real_normalized.startswith(
                        norm_allowed_dir + os.sep
                    ):
                        is_real_allowed = True
                        break

                if not is_real_allowed:
                    raise ToolInputError(
                        f"Access denied: Path '{path}' is a symbolic link pointing to '{real_normalized}', which is outside allowed directories.",
                        param_name="path",
                        provided_value=path,
                    )

                # If validation passed, use the real path for further checks *about the target*
                current_validated_path = real_normalized
                # Re-check existence *of the target* - use exists instead of lexists
                path_exists = await aiofiles.os.path.exists(current_validated_path)

            except OSError as e:
                # Handle errors during realpath resolution (e.g., broken link, permissions)
                if isinstance(e, FileNotFoundError):
                    # Broken link - the link entry exists, but target doesn't
                    path_exists = False
                    if check_exists is True:
                        raise ToolInputError(
                            f"Required path '{path}' is a symbolic link pointing to a non-existent target ('{original_validated_path}' -> target missing).",
                            param_name="path",
                            provided_value=path,
                        ) from e
                    # If check_exists is False or None, a broken link might be acceptable depending on the operation.
                    # Keep current_validated_path as the *link path itself* if the target doesn't exist.
                    current_validated_path = original_validated_path
                else:
                    raise ToolInputError(
                        f"Error resolving symbolic link '{path}': {str(e)}",
                        param_name="path",
                        provided_value=path,
                    ) from e
            except ToolInputError:  # Re-raise specific input errors
                raise
            except Exception as e:  # Catch other unexpected errors during link resolution
                raise ToolError(
                    f"Unexpected error resolving symbolic link for '{path}': {str(e)}",
                    context={"path": path},
                ) from e
        else:
            # Not a link or not resolving it, so existence check result is based on the initial check
            path_exists = path_exists_locally

        # Check existence requirement *after* potential symlink resolution
        if check_exists is True and not path_exists:
            raise ToolInputError(
                f"Required path '{path}' (resolved to '{current_validated_path}') does not exist.",
                param_name="path",
                provided_value=path,
                details={
                    "path": path,
                    "resolved_path": current_validated_path,
                    "error_type": "PATH_NOT_FOUND",
                },
            )
        elif check_exists is False and path_exists:
            raise ToolInputError(
                f"Path '{path}' (resolved to '{current_validated_path}') already exists, but non-existence was required.",
                param_name="path",
                provided_value=path,
                details={
                    "path": path,
                    "resolved_path": current_validated_path,
                    "error_type": "PATH_ALREADY_EXISTS",
                },
            )
        # else: check_exists is None, or condition met

        # If path doesn't exist and creation is likely (check_exists is False or None), check parent.
        parent_dir = os.path.dirname(current_validated_path)
        if (
            parent_dir and parent_dir != current_validated_path
        ):  # Check parent_dir is not empty and not the root itself
            try:
                parent_exists = await aiofiles.os.path.exists(
                    parent_dir
                )  # Check if parent exists first
                if parent_exists:
                    if not await aiofiles.os.path.isdir(parent_dir):
                        raise ToolInputError(
                            f"Cannot operate on '{path}': Parent path '{parent_dir}' exists but is not a directory.",
                            param_name="path",
                            provided_value=path,
                        )
                    # Parent exists and is a directory, check writability if requested
                    if check_parent_writable:
                        if not os.access(parent_dir, os.W_OK | os.X_OK):
                            raise ToolInputError(
                                f"Cannot operate on '{path}': Parent directory '{parent_dir}' exists but is not writable or accessible.",
                                param_name="path",
                                provided_value=path,
                            )
                # else: Parent does NOT exist.
                # If check_parent_writable was True, it's okay if parent doesn't exist because makedirs will create it.
                # If check_parent_writable was False (or not requested for this scenario), we might still want to error if parent doesn't exist depending on the operation context.
                # For create_directory context, non-existence of parent is fine.
            except OSError as e:
                raise ToolError(
                    f"Filesystem error checking parent directory '{parent_dir}' for '{path}': {str(e)}",
                    context={"path": path, "parent": parent_dir},
                ) from e

    except OSError as e:
        # Catch filesystem errors during async checks like exists, isdir, islink on the primary path
        raise ToolError(
            f"Filesystem error validating path '{path}': {str(e)}",
            context={"path": path, "resolved_path": current_validated_path, "error": str(e)},
        ) from e
    except ToolInputError:  # Re-raise ToolInputErrors from validation logic
        raise
    except Exception as e:
        # Catch unexpected errors during validation logic
        logger.error(f"Unexpected error during path validation for {path}: {e}", exc_info=True)
        raise ToolError(
            f"An unexpected error occurred validating path: {str(e)}", context={"path": path}
        ) from e

    # Always return the validated path string
    return current_validated_path


# --- Helper Functions ---


async def format_file_info(file_path: str, follow_symlinks: bool = False) -> Dict[str, Any]:
    """Get detailed file or directory information asynchronously.

    Uses `aiofiles.os.stat` to retrieve metadata.

    Args:
        file_path: Path to file or directory (assumed validated).
        follow_symlinks: If True, follows symlinks to get info about their targets.
                        If False, gets info about the symlink itself.

    Returns:
        Dictionary with file/directory details (name, path, size, timestamps, type, permissions).
        If an OS error occurs during stat, returns a dict containing 'name', 'path', and 'error'.
    """
    try:
        # Use stat results directly where possible to avoid redundant checks
        # Use stat with follow_symlinks parameter to control whether we stat the link or the target
        stat_info = await aiofiles.os.stat(file_path, follow_symlinks=follow_symlinks)
        mode = stat_info.st_mode
        is_dir = os.path.stat.S_ISDIR(mode)
        is_file = os.path.stat.S_ISREG(mode)  # Check for regular file
        is_link = os.path.stat.S_ISLNK(mode)  # Check if the item stat looked at is a link

        # Use timezone-aware ISO format timestamps for machine readability.
        # Handle potential platform differences in ctime availability (fallback to mtime).
        try:
            # Some systems might not have birthtime (st_birthtime)
            # ctime is platform dependent (creation on Windows, metadata change on Unix)
            # Use mtime as the most reliable "last modified" timestamp.
            # Let's report both ctime and mtime if available.
            ctime_ts = stat_info.st_ctime
        except AttributeError:
            ctime_ts = stat_info.st_mtime  # Fallback

        # Ensure timestamps are valid before conversion
        def safe_isoformat(timestamp):
            try:
                # Handle potential negative timestamps or values outside valid range
                if timestamp < 0:
                    return "Invalid Timestamp (Negative)"
                # Check against a reasonable range (e.g., year 1 to 9999)
                min_ts = datetime.datetime(1, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
                max_ts = datetime.datetime(
                    9999, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc
                ).timestamp()
                if not (min_ts <= timestamp <= max_ts):
                    return f"Invalid Timestamp (Out of Range: {timestamp})"

                return datetime.datetime.fromtimestamp(
                    timestamp, tz=datetime.timezone.utc
                ).isoformat()
            except (OSError, ValueError) as e:  # Handle potential errors like invalid values
                logger.warning(
                    f"Invalid timestamp {timestamp} for {file_path}: {e}", emoji_key="warning"
                )
                return f"Invalid Timestamp ({type(e).__name__})"

        info = {
            "name": os.path.basename(file_path),
            "path": file_path,
            "size": stat_info.st_size,
            "created_os_specific": safe_isoformat(ctime_ts),  # Note platform dependency
            "modified": safe_isoformat(stat_info.st_mtime),
            "accessed": safe_isoformat(stat_info.st_atime),
            "is_directory": is_dir,
            "is_file": is_file,
            "is_symlink": is_link,  # Indicate if the path itself is a symlink
            # Use S_IMODE for standard permission bits (mode & 0o777)
            "permissions": oct(os.path.stat.S_IMODE(mode)),
        }

        if is_link or (not follow_symlinks and await aiofiles.os.path.islink(file_path)):
            try:
                # Attempt to read the link target
                link_target = await aiofiles.os.readlink(file_path)
                info["symlink_target"] = link_target
            except OSError as link_err:
                info["symlink_target"] = f"<Error reading link: {link_err}>"

        return info
    except OSError as e:
        logger.warning(f"Error getting file info for {file_path}: {str(e)}", emoji_key="warning")
        # Return consistent error structure, let caller decide severity.
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "error": f"Failed to get info: {str(e)}",
        }
    except Exception as e:  # Catch unexpected errors
        logger.error(
            f"Unexpected error getting file info for {file_path}: {e}",
            exc_info=True,
            emoji_key="error",
        )
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "error": f"An unexpected error occurred: {str(e)}",
        }


def create_unified_diff(original_content: str, new_content: str, filepath: str) -> str:
    """Create a unified diff string between original and new content.

    Args:
        original_content: Original file content as a single string.
        new_content: New file content as a single string.
        filepath: Path to the file (used in the diff header).

    Returns:
        Unified diff as a multi-line string, or empty string if no differences.
    """
    # Normalize line endings for consistent diffing
    original_lines = original_content.splitlines()
    new_lines = new_content.splitlines()

    # Generate unified diff using difflib (synchronous)
    diff_lines = list(
        difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"{filepath} (original)",  # Label for 'from' file in diff
            tofile=f"{filepath} (modified)",  # Label for 'to' file in diff
            lineterm="",  # Keep lines without added newlines by difflib
        )
    )

    # Return empty string if no changes, otherwise join lines into single string.
    return "\n".join(diff_lines) if diff_lines else ""


async def read_file_content(filepath: str) -> str:
    """Read text file content using async I/O. Assumes UTF-8 encoding.

    Args:
        filepath: Path to the file to read (assumed validated).

    Returns:
        File content as string.

    Raises:
        ToolError: If reading fails or decoding fails. Includes specific context.
    """
    try:
        # Open asynchronously, read with strict UTF-8 decoding.
        async with aiofiles.open(filepath, mode="r", encoding="utf-8", errors="strict") as f:
            return await f.read()
    except UnicodeDecodeError as e:
        logger.warning(f"File {filepath} is not valid UTF-8: {e}", emoji_key="warning")
        # Provide context about the decoding error.
        raise ToolError(
            f"File is not valid UTF-8 encoded text: {filepath}. Cannot read as text. Details: {e}",
            context={"path": filepath, "encoding": "utf-8", "error_details": str(e)},
        ) from e
    except OSError as e:
        logger.error(f"OS error reading file {filepath}: {e}", emoji_key="error")
        raise ToolError(
            f"Error reading file: {str(e)}", context={"path": filepath, "errno": e.errno}
        ) from e
    except Exception as e:
        logger.error(
            f"Unexpected error reading file {filepath}: {e}", exc_info=True, emoji_key="error"
        )
        raise ToolError(
            f"An unexpected error occurred while reading file: {str(e)}", context={"path": filepath}
        ) from e


async def read_binary_file_content(filepath: str) -> bytes:
    """Read binary file content using async I/O.

    Args:
        filepath: Path to the file to read (assumed validated).

    Returns:
        File content as bytes.

    Raises:
        ToolError: If reading fails.
    """
    try:
        # Open asynchronously in binary read mode ('rb').
        async with aiofiles.open(filepath, mode="rb") as f:
            return await f.read()
    except OSError as e:
        logger.error(f"OS error reading binary file {filepath}: {e}", emoji_key="error")
        raise ToolError(
            f"Error reading binary file: {str(e)}", context={"path": filepath, "errno": e.errno}
        ) from e
    except Exception as e:
        logger.error(
            f"Unexpected error reading binary file {filepath}: {e}",
            exc_info=True,
            emoji_key="error",
        )
        raise ToolError(
            f"An unexpected error occurred while reading binary file: {str(e)}",
            context={"path": filepath},
        ) from e


async def write_file_content(filepath: str, content: Union[str, bytes]) -> None:
    """Write text or binary content to a file using async I/O. Creates parent dirs.

    Args:
        filepath: Path to the file to write (assumed validated, including parent writability).
        content: Content to write (string for text UTF-8, bytes for binary).

    Raises:
        ToolError: If writing fails.
        TypeError: If content is not str or bytes.
    """
    # Determine mode and encoding based on content type.
    if isinstance(content, str):
        mode = "w"
        encoding = "utf-8"
        data_to_write = content
    elif isinstance(content, bytes):
        mode = "wb"
        encoding = None  # No encoding for binary mode
        data_to_write = content
    else:
        raise TypeError("Content to write must be str or bytes")

    try:
        # Ensure parent directory exists asynchronously.
        parent_dir = os.path.dirname(filepath)
        if (
            parent_dir and parent_dir != filepath
        ):  # Check parent_dir is not empty and not the root itself
            # Use async makedirs for consistency. exist_ok=True makes it idempotent.
            await aiofiles.os.makedirs(parent_dir, exist_ok=True)

        # Open file asynchronously and write content.
        async with aiofiles.open(filepath, mode=mode, encoding=encoding) as f:
            await f.write(data_to_write)
            # await f.flush() # Often not necessary as context manager handles flush/close, but uncomment for critical writes if needed.

    except OSError as e:
        logger.error(f"OS error writing file {filepath}: {e}", emoji_key="error")
        raise ToolError(
            f"Error writing file: {str(e)}", context={"path": filepath, "errno": e.errno}
        ) from e
    except Exception as e:
        logger.error(
            f"Unexpected error writing file {filepath}: {e}", exc_info=True, emoji_key="error"
        )
        raise ToolError(
            f"An unexpected error occurred while writing file: {str(e)}", context={"path": filepath}
        ) from e


async def apply_file_edits(
    filepath: str, edits: List[Dict[str, str]], dry_run: bool = False
) -> Tuple[str, str]:
    """Apply a series of text replacements to a file asynchronously.

    Reads the file (UTF-8), applies edits sequentially. If an exact match for
    'oldText' isn't found, it attempts a line-by-line match ignoring leading/trailing
    whitespace, preserving the original indentation of the matched block.
    Generates a diff and optionally writes back the changes.

    Args:
        filepath: Path to the file to edit (assumed validated and is a text file).
        edits: List of edit operations. Each dict must have 'oldText' and 'newText' (both strings).
        dry_run: If True, calculate changes and diff but do not write to disk.

    Returns:
        Tuple of (diff_string, new_content_string). The diff string is empty if no changes occurred.

    Raises:
        ToolError: If reading/writing fails.
        ToolInputError: If edits are malformed or text specified in 'oldText' cannot be found.
    """
    # Read original content asynchronously (raises ToolError if fails)
    content = await read_file_content(filepath)
    original_content = content
    current_content = content  # Work on a mutable copy

    # Apply each edit sequentially (string operations are sync)
    for i, edit in enumerate(edits):
        # Validate edit structure
        if not isinstance(edit, dict):
            raise ToolInputError(
                f"Edit #{i + 1} is not a dictionary.",
                param_name=f"edits[{i}]",
                provided_value=type(edit),
            )
        old_text = edit.get("oldText")
        new_text = edit.get("newText")

        if not isinstance(old_text, str):
            raise ToolInputError(
                f"Edit #{i + 1} is missing 'oldText' or it's not a string.",
                param_name=f"edits[{i}].oldText",
                provided_value=edit,
            )
        if not isinstance(new_text, str):
            raise ToolInputError(
                f"Edit #{i + 1} is missing 'newText' or it's not a string.",
                param_name=f"edits[{i}].newText",
                provided_value=edit,
            )
        # Warn about potentially ambiguous empty oldText
        if not old_text:
            logger.warning(
                f"Edit #{i + 1} has empty 'oldText'. Python's string.replace behavior with empty strings might be unexpected.",
                emoji_key="warning",
            )

        # Try exact replacement first
        if old_text in current_content:
            # Replace *all* occurrences of old_text with new_text
            current_content = current_content.replace(old_text, new_text)
        else:
            # Fallback: Try line-by-line matching with stripped whitespace comparison.
            old_lines = old_text.splitlines()
            # Avoid fallback if old_text was only whitespace but wasn't found exactly
            if not any(line.strip() for line in old_lines) and old_text:
                logger.warning(
                    f"Edit #{i + 1}: 'oldText' consists only of whitespace, but was not found exactly. Skipping whitespace-insensitive fallback.",
                    emoji_key="warning",
                )
                raise ToolInputError(
                    f"Could not find exact whitespace text to replace in edit #{i + 1}: '{old_text[:100]}{'...' if len(old_text) > 100 else ''}'",
                    param_name=f"edits[{i}].oldText",
                    provided_value=edit,
                )

            if not old_lines:  # If old_text was empty string and not found, error out.
                raise ToolInputError(
                    f"Could not find empty 'oldText' to replace in edit #{i + 1}.",
                    param_name=f"edits[{i}].oldText",
                    provided_value=edit,
                )

            content_lines = current_content.splitlines()
            found_match = False
            line_idx = 0
            # Iterate through possible starting lines for the block match
            while line_idx <= len(content_lines) - len(old_lines):
                # Check if the slice matches ignoring leading/trailing whitespace on each line
                is_match = all(
                    old_lines[j].strip() == content_lines[line_idx + j].strip()
                    for j in range(len(old_lines))
                )

                if is_match:
                    # Found a match based on content, now replace respecting original indentation.
                    new_lines = new_text.splitlines()

                    # Determine indentation from the *first original line* being replaced.
                    first_original_line = content_lines[line_idx]
                    leading_whitespace = first_original_line[
                        : len(first_original_line) - len(first_original_line.lstrip())
                    ]

                    # Apply this leading whitespace to all *new* lines.
                    indented_new_lines = (
                        [leading_whitespace + line for line in new_lines] if new_lines else []
                    )

                    # Replace the slice in the original lines list
                    content_lines[line_idx : line_idx + len(old_lines)] = indented_new_lines
                    # Reconstruct content string immediately after modification
                    current_content = "\n".join(content_lines)
                    found_match = True
                    # Stop searching for matches for *this specific edit dict* once one is found and replaced.
                    # To replace all occurrences, logic would need modification (e.g., restart search or adjust indices).
                    break

                else:
                    line_idx += 1  # Move to the next line to check

            if not found_match:
                # No match found even with whitespace trimming fallback
                raise ToolInputError(
                    f"Could not find text to replace in edit #{i + 1}. Text searched (approx first 100 chars): '{old_text[:100]}{'...' if len(old_text) > 100 else ''}'. "
                    f"Exact match failed, and whitespace-insensitive line match also failed.",
                    param_name=f"edits[{i}].oldText",
                    provided_value=edit,
                )

    # Create diff (sync function call) using original vs final content
    diff = create_unified_diff(original_content, current_content, filepath)

    # Write the changes if not a dry run asynchronously
    if not dry_run:
        # Ensure content is string before writing
        await write_file_content(filepath, str(current_content))  # Handles errors internally

    return diff, str(current_content)


# --- MCP Formatting Helpers ---


def format_mcp_content(text_content: str) -> List[Dict[str, Any]]:
    """Format text content into a simple MCP text block.

    Applies length truncation to avoid oversized responses.

    Args:
        text_content: Text to format.

    Returns:
        List containing a single MCP text block dictionary.
    """
    # Max length for content block to avoid overwhelming downstream systems.
    MAX_LEN = 100000  # Adjust based on LLM/platform constraints.
    if len(text_content) > MAX_LEN:
        # Note: Truncation happens here, not a placeholder.
        trunc_msg = f"\n... [Content truncated - {len(text_content)} bytes total]"
        text_content = text_content[: MAX_LEN - len(trunc_msg)] + trunc_msg
        logger.warning(f"Content length > {MAX_LEN}, truncated.", emoji_key="warning")

    # Standard MCP text block structure.
    return [{"type": "text", "text": text_content}]


def create_tool_response(content: Any, is_error: bool = False) -> Dict[str, Any]:
    """Create a standard tool response dictionary for the Ultimate MCP Server.

    Formats various content types (str, dict, list) into MCP 'content' blocks,
    attempting to provide useful representations for common tool outputs.

    Args:
        content: The primary result or message from the tool. Can be str, dict,
                 list (assumed pre-formatted MCP), or other types (converted to str).
        is_error: If True, marks the response as an error response using "isError".

    Returns:
        A dictionary structured for the Ultimate MCP Server tool response schema.
    """
    formatted_content: List[Dict[str, Any]]
    response: Dict[str, Any] = {}  # Initialize response dict

    if is_error:
        # --- Error Handling Logic ---
        response["success"] = False  # Explicitly set success to False for errors
        response["isError"] = True  # Mark as an error response

        if isinstance(content, dict) and "message" in content:
            error_message = content.get("message", "Unknown error")
            error_code = content.get("error_code", "TOOL_ERROR")
            error_type = content.get("error_type", "ToolError")

            response["error"] = error_message
            response["error_code"] = error_code
            response["error_type"] = error_type

            # Add details if available
            if "details" in content:
                response["details"] = content["details"]

            # Format the content text for error display, including context if available
            context = content.get("context")  # Context might be added by the calling function
            if context:
                try:
                    # Try to pretty-print context if it's dict/list
                    if isinstance(context, (dict, list)):
                        context_str = json.dumps(context, indent=2, default=str)
                    else:
                        context_str = str(context)
                    error_display_text = f"Error: {error_message}\nContext: {context_str}"
                except Exception:  # Fallback if context serialization fails
                    error_display_text = f"Error: {error_message}\nContext: (Could not serialize context: {type(context).__name__})"
            else:
                error_display_text = f"Error: {error_message}"

            formatted_content = format_mcp_content(error_display_text)

        else:  # Handle cases where error content is not a detailed dict
            error_message = str(content)  # Use string representation of the error content
            formatted_content = format_mcp_content(f"Error: {error_message}")
            response["error"] = error_message
            # Provide default error codes/types if not available
            response["error_code"] = "UNKNOWN_ERROR"
            response["error_type"] = "UnknownError"

        # Add protectionTriggered flag if applicable (check original error dict if passed)
        if isinstance(content, dict) and content.get("protection_triggered"):
            response["protectionTriggered"] = True

        response["content"] = formatted_content

    else:
        # --- Success Case ---
        response["success"] = True  # <<< THIS WAS THE MISSING LINE

        # --- Existing success formatting logic ---
        try:
            if isinstance(content, dict):
                # Handle specific known dictionary structures first
                if (
                    "files" in content
                    and isinstance(content.get("files"), list)
                    and "succeeded" in content
                ):
                    # Format output from read_multiple_files
                    blocks = []
                    summary = f"Read {content.get('succeeded', 0)} files successfully, {content.get('failed', 0)} failed."
                    blocks.append({"type": "text", "text": summary})
                    for file_result in content.get("files", []):
                        if isinstance(file_result, dict):
                            path = file_result.get("path", "Unknown path")
                            if file_result.get("success") and "content" in file_result:
                                size_info = (
                                    f" ({file_result.get('size', 'N/A')} bytes)"
                                    if "size" in file_result
                                    else ""
                                )
                                binary_info = (
                                    " (Binary Preview)" if file_result.get("is_binary") else ""
                                )
                                header = f"--- File: {path}{size_info}{binary_info} ---"
                                file_content_str = str(file_result["content"])
                                blocks.extend(format_mcp_content(f"{header}\n{file_content_str}"))
                            elif "error" in file_result:
                                header = f"--- File: {path} (Error) ---"
                                error_msg = str(file_result["error"])
                                blocks.extend(format_mcp_content(f"{header}\n{error_msg}"))
                            else:
                                blocks.extend(
                                    format_mcp_content(
                                        f"--- File: {path} (Unknown status) ---\n{str(file_result)}"
                                    )
                                )
                        else:
                            blocks.extend(
                                format_mcp_content(
                                    f"--- Invalid entry in results ---\n{str(file_result)}"
                                )
                            )
                    formatted_content = blocks
                elif "tree" in content and "path" in content:
                    # Format output from directory_tree as JSON block
                    tree_json = json.dumps(
                        content["tree"], indent=2, ensure_ascii=False, default=str
                    )
                    MAX_JSON_LEN = 50000
                    if len(tree_json) > MAX_JSON_LEN:
                        trunc_msg = "\n... [Tree JSON truncated]"
                        tree_json = tree_json[: MAX_JSON_LEN - len(trunc_msg)] + trunc_msg
                    formatted_content = format_mcp_content(
                        f"Directory Tree for: {content['path']}\n```json\n{tree_json}\n```"
                    )
                elif "entries" in content and "path" in content:
                    # Format output from list_directory
                    list_strs = [f"Directory Listing for: {content['path']}"]
                    for entry in content.get("entries", []):
                        if isinstance(entry, dict):
                            name = entry.get("name", "?")
                            etype = entry.get("type", "unknown")
                            size_str = (
                                f" ({entry.get('size')} bytes)"
                                if etype == "file" and "size" in entry
                                else ""
                            )
                            error_str = (
                                f" (Error: {entry.get('error')})" if "error" in entry else ""
                            )
                            link_str = (
                                f" -> {entry.get('symlink_target')}"
                                if etype == "symlink" and "symlink_target" in entry
                                else ""
                            )
                            list_strs.append(f"- {name} [{etype}]{size_str}{error_str}{link_str}")
                        else:
                            list_strs.append(f"- Invalid entry: {str(entry)}")
                    formatted_content = format_mcp_content("\n".join(list_strs))
                elif "matches" in content and "path" in content and "pattern" in content:
                    # Format output from search_files
                    search_strs = [
                        f"Search Results for '{content['pattern']}' in '{content['path']}':"
                    ]
                    matches = content.get("matches", [])
                    if matches:
                        for match in matches:
                            search_strs.append(f"- {match}")
                    else:
                        search_strs.append("(No matches found)")
                    if "warnings" in content:
                        search_strs.append("\nWarnings:")
                        search_strs.extend(content["warnings"])
                    formatted_content = format_mcp_content("\n".join(search_strs))
                else:
                    # Attempt to pretty-print other dictionaries as JSON
                    json_content = json.dumps(content, indent=2, ensure_ascii=False, default=str)
                    formatted_content = format_mcp_content(f"```json\n{json_content}\n```")

            elif isinstance(content, str):
                # Simple string content
                formatted_content = format_mcp_content(content)
            elif isinstance(content, list) and all(
                isinstance(item, dict) and "type" in item for item in content
            ):
                # Assume it's already MCP formatted - pass through directly
                formatted_content = content
            else:
                # Convert anything else to string and format
                formatted_content = format_mcp_content(str(content))

            # Ensure formatted_content is valid before adding to response
            if isinstance(formatted_content, list):
                response["content"] = formatted_content
            else:
                # Fallback if formatting somehow failed during success path
                fallback_text = f"Successfully processed, but failed to format response content: {str(formatted_content)}"
                logger.error(fallback_text)  # Log this internal error
                response["content"] = format_mcp_content(fallback_text)
                response["warning"] = "Response content formatting failed."  # Add a warning field

        except (TypeError, ValueError) as json_err:  # Catch JSON serialization errors specifically
            logger.warning(
                f"Could not serialize successful dictionary response to JSON: {json_err}",
                exc_info=True,
                emoji_key="warning",
            )
            formatted_content = format_mcp_content(
                f"(Response dictionary could not be formatted as JSON)\n{str(content)}"
            )
            response["content"] = formatted_content
            response["warning"] = "Could not format successful response as JSON."
        except Exception as format_err:  # Catch any other formatting errors
            logger.error(
                f"Unexpected error formatting successful response content: {format_err}",
                exc_info=True,
            )
            response["content"] = format_mcp_content(f"Error formatting response: {format_err}")
            response["warning"] = "Unexpected error formatting response content."

    return response


# --- Protection Heuristics Implementation ---


async def _get_minimal_stat(path: str) -> Optional[Tuple[Tuple[float, float], str]]:
    """Helper to get minimal stat info (ctime, mtime, extension) for protection checks."""
    try:
        # Use stat with follow_symlinks=False to get info without following links, as we care about the items being listed
        stat_info = await aiofiles.os.stat(path, follow_symlinks=False)
        # Use mtime and ctime (platform-dependent creation/metadata change time)
        mtime = stat_info.st_mtime
        try:
            ctime = stat_info.st_ctime
        except AttributeError:
            ctime = mtime  # Fallback if ctime not available

        extension = (
            os.path.splitext(path)[1].lower()
            if not os.path.stat.S_ISDIR(stat_info.st_mode)
            else ".<dir>"
        )
        return ((ctime, mtime), extension)
    except OSError as e:
        logger.debug(f"Stat failed for protection check on {path}: {e}", emoji_key="warning")
        return None  # Return None on OS error


async def _check_protection_heuristics(
    paths: List[str], operation_type: Literal["deletion", "modification"]
) -> None:
    """
    Check if a bulk operation on multiple files triggers safety protection heuristics.
    (Modified to use get_protection_config which reads from validated config)
    """
    protection_config = get_protection_config()

    if not protection_config.get("enabled", False):
        return  # Protection disabled for this operation type

    num_paths = len(paths)
    # --- Read thresholds from the loaded config dictionary ---
    max_files_threshold = protection_config.get("max_files_threshold", 100)

    # Only run detailed checks if the number of files exceeds the threshold
    if num_paths <= max_files_threshold:
        return

    logger.info(
        f"Performing detailed safety check for {operation_type} of {num_paths} paths (threshold: {max_files_threshold})...",
        emoji_key="security",
    )

    # --- Gather Metadata Asynchronously ---
    stat_tasks = [_get_minimal_stat(p) for p in paths]
    stat_results = await asyncio.gather(*stat_tasks)

    successful_stats: List[Tuple[Tuple[float, float], str]] = []
    failed_stat_count = 0
    for result in stat_results:
        if result is not None:
            successful_stats.append(result)
        else:
            failed_stat_count += 1

    total_attempted = len(paths)
    # --- Read threshold from config dict ---
    max_errors_pct = protection_config.get("max_stat_errors_pct", 10.0)
    if total_attempted > 0 and (failed_stat_count / total_attempted * 100) > max_errors_pct:
        raise ProtectionTriggeredError(
            f"Operation blocked because safety check could not reliably gather file metadata ({failed_stat_count}/{total_attempted} failures).",
            context={"failed_stats": failed_stat_count, "total_paths": total_attempted},
        )

    num_valid_stats = len(successful_stats)
    if num_valid_stats < 2:
        logger.info(
            "Protection check skipped: Not enough valid file metadata points obtained.",
            emoji_key="info",
        )
        return

    # --- Calculate Heuristics ---
    creation_times = [ts[0] for ts, ext in successful_stats]
    modification_times = [ts[1] for ts, ext in successful_stats]
    extensions = {ext for ts, ext in successful_stats if ext and ext != ".<dir>"}

    try:
        ctime_stddev = statistics.pstdev(creation_times) if num_valid_stats > 1 else 0.0
        mtime_stddev = statistics.pstdev(modification_times) if num_valid_stats > 1 else 0.0
        if math.isnan(ctime_stddev):
            ctime_stddev = 0.0
        if math.isnan(mtime_stddev):
            mtime_stddev = 0.0
    except statistics.StatisticsError as e:
        logger.warning(
            f"Could not calculate timestamp standard deviation: {e}", emoji_key="warning"
        )
        ctime_stddev = 0.0
        mtime_stddev = 0.0

    num_unique_extensions = len(extensions)

    # --- Read thresholds from config dict ---
    dt_threshold = protection_config.get("datetime_stddev_threshold_sec", 60 * 60 * 24 * 30)
    type_threshold = protection_config.get("file_type_variance_threshold", 5)

    triggered = False
    reasons = []

    if ctime_stddev > dt_threshold:
        triggered = True
        reasons.append(
            f"High variance in creation times (std dev: {ctime_stddev:.2f}s > threshold: {dt_threshold}s)"
        )
    if mtime_stddev > dt_threshold:
        triggered = True
        reasons.append(
            f"High variance in modification times (std dev: {mtime_stddev:.2f}s > threshold: {dt_threshold}s)"
        )
    if num_unique_extensions > type_threshold:
        triggered = True
        reasons.append(
            f"High variance in file types ({num_unique_extensions} unique types > threshold: {type_threshold})"
        )

    if triggered:
        reason_str = (
            f"Operation involves a large number of files ({num_paths}) with suspicious characteristics: "
            + "; ".join(reasons)
        )
        logger.warning(f"Protection Triggered! Reason: {reason_str}", emoji_key="security")
        raise ProtectionTriggeredError(
            reason_str,
            protection_type=f"{operation_type}_protection",  # Add protection type
            context={  # Use context for structured data
                "num_paths": num_paths,
                "num_valid_stats": num_valid_stats,
                "ctime_stddev_sec": round(ctime_stddev, 2),
                "mtime_stddev_sec": round(mtime_stddev, 2),
                "unique_file_types": num_unique_extensions,
                "threshold_max_files": max_files_threshold,
                "threshold_datetime_stddev_sec": dt_threshold,
                "threshold_file_type_variance": type_threshold,
                "failed_stat_count": failed_stat_count,
            },
        )
    else:
        logger.info(
            f"Safety check passed for {operation_type} of {num_paths} paths.",
            emoji_key="security",
            ctime_stddev=round(ctime_stddev, 2),
            mtime_stddev=round(mtime_stddev, 2),
            unique_types=num_unique_extensions,
        )


# --- Async Walk and Delete Helpers ---


async def async_walk(
    top: str,
    topdown: bool = True,
    onerror: Optional[callable] = None,
    followlinks: bool = False,
    exclude_patterns: Optional[List[str]] = None,
    base_path: Optional[str] = None,
) -> AsyncGenerator[Tuple[str, List[str], List[str]], None]:
    """Async version of os.walk using aiofiles.os.scandir. Handles excludes.

    Yields (current_dir_path, dir_names, file_names) tuples asynchronously.
    Filters entries based on exclude_patterns matched against relative path from base_path.
    Handles errors during scanning via the onerror callback.

    Args:
        top: Root directory path to start walking.
        topdown: Whether to yield the current directory before (True) or after (False) its subdirectories.
        onerror: Optional callable that takes an OSError instance when errors occur during scandir or stat.
        followlinks: If True, recurse into directories pointed to by symlinks.
        exclude_patterns: List of glob-style patterns to exclude files/directories.
        base_path: The original starting path of the walk (used for relative path calculation).

    Yields:
        Tuples of (directory_path, list_of_subdir_names, list_of_file_names)
    """
    if base_path is None:
        base_path = top  # Keep track of the original root for exclusion matching

    dirs: List[str] = []
    nondirs: List[str] = []
    walk_error: Optional[OSError] = None

    try:
        # Use async scandir for efficient directory iteration
        # scandir returns a coroutine that must be awaited to get the entries
        scandir_it = await aiofiles.os.scandir(top)
        # Now iterate through the scandir entries (which should be a list or iterable)
        for entry in scandir_it:
            try:
                # Calculate relative path for exclusion check
                try:
                    # entry.path should be absolute if 'top' is absolute
                    rel_path = os.path.relpath(entry.path, base_path)
                except ValueError:
                    # Fallback if paths are on different drives (Windows) or other issues
                    rel_path = entry.name  # Use name as fallback for matching
                    logger.debug(
                        f"Could not get relative path for {entry.path} from {base_path}, using name '{entry.name}' for exclusion check."
                    )

                # Check exclusion patterns (case-insensitive matching where appropriate)
                is_excluded = False
                if exclude_patterns:
                    norm_rel_path = os.path.normcase(rel_path)
                    norm_entry_name = os.path.normcase(entry.name)
                    for pattern in exclude_patterns:
                        norm_pattern = os.path.normcase(pattern)
                        # Match pattern against full relative path OR just the name
                        if fnmatch(norm_rel_path, norm_pattern) or fnmatch(
                            norm_entry_name, norm_pattern
                        ):
                            is_excluded = True
                            logger.debug(f"Excluding '{entry.path}' due to pattern '{pattern}'")
                            break  # Stop checking patterns for this entry
                if is_excluded:
                    continue  # Skip this excluded entry

                # Check entry type, handling potential errors using lstat to check link itself
                try:
                    # Determine if it's a directory (respecting followlinks for recursion decision later)
                    is_entry_dir = entry.is_dir(follow_symlinks=followlinks)
                    is_entry_link = entry.is_symlink()  # Check if entry *itself* is a link

                    if is_entry_dir and (not is_entry_link or followlinks):
                        # It's a directory, or it's a link to a directory and we follow links
                        dirs.append(entry.name)
                    else:
                        # It's a file, a link we don't follow, or something else
                        nondirs.append(entry.name)

                except OSError as stat_err:
                    # Error determining type (e.g., permissions)
                    if onerror is not None:
                        onerror(stat_err)
                    logger.warning(
                        f"Skipping entry '{entry.name}' in {top} due to stat error: {stat_err}",
                        emoji_key="warning",
                    )
                    continue  # Skip entry if type cannot be determined

            except OSError as entry_proc_err:
                # Error during exclusion check or other processing of the entry itself
                if onerror is not None:
                    onerror(entry_proc_err)
                logger.warning(
                    f"Error processing entry '{entry.name}' in {top}: {entry_proc_err}",
                    emoji_key="warning",
                )
                # Continue processing other entries in the directory

    except OSError as err:
        # Error during the initial scandir call itself (e.g., permissions on 'top')
        walk_error = err
        if onerror is not None:
            onerror(err)
        # Stop iteration for this path if scandir failed

    # --- Yield results and recurse ---
    if walk_error is None:
        if topdown:
            yield top, dirs, nondirs

        # Recurse into subdirectories discovered
        for name in dirs:
            new_path = os.path.join(top, name)
            # Recurse using 'async for' to delegate iteration properly down the recursion
            async for x in async_walk(
                new_path, topdown, onerror, followlinks, exclude_patterns, base_path
            ):
                yield x

        if not topdown:
            yield top, dirs, nondirs


async def _list_paths_recursive(root_path: str) -> List[str]:
    """Recursively list all file paths within a directory using async_walk."""
    paths = []
    try:
        async for dirpath, _dirnames, filenames in async_walk(
            root_path, topdown=True, followlinks=False
        ):
            for filename in filenames:
                paths.append(os.path.join(dirpath, filename))
            # Also include directory paths themselves if needed for certain checks?
            # For deletion check, primarily care about files within.
            # Let's stick to files for heuristic calculation simplicity.
            # for dirname in dirnames:
            #     paths.append(os.path.join(dirpath, dirname))
    except Exception as e:
        logger.error(
            f"Error listing paths recursively under {root_path}: {e}",
            exc_info=True,
            emoji_key="error",
        )
        # Re-raise as ToolError so the caller knows listing failed
        raise ToolError(
            f"Failed to list contents of directory '{root_path}' for safety check: {e}"
        ) from e
    return paths


async def _async_rmtree(path: str):
    """Asynchronously remove a directory and its contents, similar to shutil.rmtree."""
    logger.debug(f"Initiating async rmtree for: {path}", emoji_key="action")
    errors: List[Tuple[str, OSError]] = []

    def onerror(os_error: OSError):
        logger.warning(
            f"Error during rmtree operation on {getattr(os_error, 'filename', 'N/A')}: {os_error}",
            emoji_key="warning",
        )
        errors.append((getattr(os_error, "filename", "N/A"), os_error))

    try:
        # Walk bottom-up to remove files first, then directories
        async for root, dirs, files in async_walk(
            path, topdown=False, onerror=onerror, followlinks=False
        ):
            # Remove files in the current directory
            for name in files:
                filepath = os.path.join(root, name)
                try:
                    logger.debug(f"Removing file: {filepath}", emoji_key="delete")
                    await aiofiles.os.remove(filepath)
                except OSError as e:
                    onerror(e)  # Log error and collect it

            # Remove empty subdirectories (should be empty now if walk is bottom-up)
            for name in dirs:
                dirpath = os.path.join(root, name)
                # We only remove dirs listed in the walk *if* they still exist
                # (e.g., link handling might affect this). Re-check existence and type.
                try:
                    if await aiofiles.os.path.islink(dirpath):
                        # Remove symlink itself if not following links
                        logger.debug(
                            f"Removing symlink (treated as file): {dirpath}", emoji_key="delete"
                        )
                        await aiofiles.os.remove(dirpath)
                    elif await aiofiles.os.path.isdir(dirpath):
                        logger.debug(f"Removing directory: {dirpath}", emoji_key="delete")
                        await aiofiles.os.rmdir(dirpath)
                except OSError as e:
                    onerror(e)  # Log error and collect it

        # Finally, remove the top-level directory itself
        try:
            logger.debug(f"Removing root directory: {path}", emoji_key="delete")
            await aiofiles.os.rmdir(path)
        except OSError as e:
            onerror(e)

        if errors:
            # Raise a consolidated error if any deletions failed
            error_summary = "; ".join([f"{p}: {e.strerror}" for p, e in errors[:5]])
            if len(errors) > 5:
                error_summary += " ... (more errors)"
            raise ToolError(
                f"Errors occurred during recursive deletion of '{path}': {error_summary}",
                context={"path": path, "num_errors": len(errors)},
            )

    except Exception as e:
        logger.error(
            f"Unexpected error during async rmtree of {path}: {e}", exc_info=True, emoji_key="error"
        )
        raise ToolError(
            f"An unexpected error occurred during recursive deletion: {str(e)}",
            context={"path": path},
        ) from e


# --- Tool Functions ---


@with_tool_metrics
@with_error_handling
async def read_file(path: str) -> Dict[str, Any]:
    """Read a file's content asynchronously, handling text/binary detection.

    Validates the path, checks it's a file, attempts to read as UTF-8 text.
    If UTF-8 decoding fails, it reads as binary and provides a hex preview.

    Args:
        path: Path to the file to read.

    Returns:
        A dictionary formatted as an MCP tool response containing file content
        or an error message.
    """
    start_time = time.monotonic()
    response_content: Any
    is_response_error = False

    try:
        # Validate path, ensuring it exists and is accessible. check_exists=True
        validated_path = await validate_path(path, check_exists=True, check_parent_writable=False)

        # Ensure it's a file, not a directory or other type.
        if not await aiofiles.os.path.isfile(validated_path):
            # Check if it's a link first before declaring it's not a regular file
            if await aiofiles.os.path.islink(validated_path):
                # If it's a link, let read proceed, it might link to a file.
                # The isfile check follows links, so if isfile failed, the target isn't a file.
                raise ToolInputError(
                    f"Path '{path}' is a symbolic link that points to something that is not a regular file.",
                    param_name="path",
                    provided_value=path,
                    details={
                        "path": path,
                        "resolved_path": validated_path,
                        "error_type": "INVALID_SYMLINK_TARGET",
                    },
                )
            elif await aiofiles.os.path.isdir(validated_path):
                raise ToolInputError(
                    f"Path '{path}' is a directory, not a file. Use list_directory or directory_tree to view its contents.",
                    param_name="path",
                    provided_value=path,
                    details={
                        "path": path,
                        "resolved_path": validated_path,
                        "error_type": "PATH_IS_DIRECTORY",
                    },
                )
            else:
                raise ToolInputError(
                    f"Path '{path}' exists but is not a regular file. It may be a special file type (socket, device, etc.).",
                    param_name="path",
                    provided_value=path,
                    details={
                        "path": path,
                        "resolved_path": validated_path,
                        "error_type": "PATH_NOT_REGULAR_FILE",
                    },
                )

        content: Union[str, bytes]
        is_binary = False
        read_error = None
        file_size = -1  # Default size if stat fails later

        # Attempt to read as text first
        try:
            content = await read_file_content(validated_path)  # Handles UTF-8 check internally
        except ToolError as text_err:
            # Check if the error was specifically UnicodeDecodeError
            if "not valid UTF-8" in str(text_err):
                logger.warning(
                    f"File {path} is not UTF-8 encoded, reading as binary.", emoji_key="warning"
                )
                is_binary = True
                try:
                    # Fallback to binary read
                    content = await read_binary_file_content(
                        validated_path
                    )  # Keep raw bytes for now
                except ToolError as bin_err:
                    read_error = (
                        f"Error reading file as binary after text decode failed: {str(bin_err)}"
                    )
                except Exception as bin_e:
                    read_error = f"Unexpected error reading file as binary: {str(bin_e)}"
            else:
                # Other OS read error during text read attempt
                read_error = f"Error reading file: {str(text_err)}"
        except Exception as text_e:
            read_error = f"Unexpected error reading file as text: {str(text_e)}"

        if read_error:
            # If reading failed entirely (text and binary fallback)
            raise ToolError(read_error, context={"path": validated_path})

        # Successfully read content (either text or binary bytes)
        try:
            # Use stat with follow_symlinks=False for consistency
            file_size = (await aiofiles.os.stat(validated_path, follow_symlinks=False)).st_size
        except OSError as stat_err:
            logger.warning(
                f"Could not get size for file {validated_path} after reading: {stat_err}",
                emoji_key="warning",
            )
            # Continue without size info

        # Prepare response content string
        basename = os.path.basename(validated_path)
        size_str = f"{file_size} bytes" if file_size >= 0 else "Size unavailable"

        if is_binary:
            # Provide a safe representation for binary data
            binary_data = cast(bytes, content)
            hex_preview_len = 200  # Bytes to show as hex
            hex_preview = binary_data[:hex_preview_len].hex(" ")  # Use spaces for readability
            preview_msg = f"(showing first {min(hex_preview_len, len(binary_data))} bytes as hex)"
            ellipsis = "..." if len(binary_data) > hex_preview_len else ""
            response_text = (
                f"File: {basename}\n"
                f"Path: {validated_path}\n"
                f"Size: {size_str}\n"
                f"Content: <Binary file detected> {preview_msg}\n"
                f"{hex_preview}{ellipsis}"
            )
        else:
            # Content is already string
            response_text = (
                f"File: {basename}\n"
                f"Path: {validated_path}\n"
                f"Size: {size_str}\n"
                f"Content:\n"  # Add newline for separation
                f"{content}"
            )

        response_content = response_text
        processing_time = time.monotonic() - start_time
        logger.success(
            f"Successfully read file: {path}",
            emoji_key="file",
            size=file_size,
            time=processing_time,
            is_binary=is_binary,
        )

    except (ToolInputError, ToolError) as e:
        logger.error(
            f"Error in read_file for '{path}': {e}",
            emoji_key="error",
            details=getattr(e, "context", None),
        )
        # Return a formatted error response with detailed info
        error_type = e.__class__.__name__
        error_details = getattr(e, "details", {}) or {}
        # Get the context from the error if available (used in base ToolError)
        context = getattr(e, "context", None)
        if context and isinstance(context, dict):
            error_details.update(context)
        # Include error type and code for better error display
        response_content = {
            "message": str(e),
            "error_code": getattr(e, "error_code", "TOOL_ERROR"),
            "error_type": error_type,
            "details": error_details,
        }
        is_response_error = True
    except FileNotFoundError as e:
        # Specific error for file not found
        raise ToolInputError(
            f"File not found: {path}",
            param_name="path",
            provided_value=path,
            details={"errno": e.errno, "error_type": "PATH_NOT_FOUND"},
        ) from e
    except IsADirectoryError as e:
        # Specific error for path being a directory
        raise ToolInputError(
            f"Path is a directory, not a file: {path}",
            param_name="path",
            provided_value=path,
            details={"errno": e.errno, "error_type": "PATH_IS_DIRECTORY"},
        ) from e
    except PermissionError as e:
        # Specific error for permissions
        raise ToolInputError(
            f"Permission denied reading file: {path}",
            param_name="path",
            provided_value=path,
            details={"errno": e.errno, "error_type": "PERMISSION_DENIED"},
        ) from e
    except UnicodeDecodeError as e:
        # This is handled in read_file_content now, but keep check here just in case
        raise ToolError(
            f"File is not valid UTF-8 encoded text: {validated_path}. Details: {e}",
            context={"path": validated_path, "encoding": "utf-8"},
        ) from e
    except OSError as e:
        # General OS error during read
        raise ToolError(
            f"OS error reading file: {str(e)}", context={"path": validated_path, "errno": e.errno}
        ) from e
    except Exception as e:
        logger.error(
            f"Unexpected error in read_file for '{path}': {e}", exc_info=True, emoji_key="error"
        )
        response_content = {
            "message": f"An unexpected error occurred while reading '{path}': {str(e)}",
            "error_code": "UNEXPECTED_ERROR",
            "error_type": type(e).__name__,
            "details": {"error_class": type(e).__name__, "path": path},
        }
        is_response_error = True

    # Use create_tool_response for consistent formatting of success/error messages.
    return create_tool_response(response_content, is_error=is_response_error)


@with_tool_metrics
@with_error_handling
async def read_multiple_files(paths: List[str]) -> Dict[str, Any]:
    """Read the contents of multiple files asynchronously and concurrently.

    Validates each path and attempts to read each file (text/binary),
    handling individual errors gracefully.

    Args:
        paths: A list of file paths to read.

    Returns:
        A dictionary summarizing the results (suitable for create_tool_response),
        including success/failure counts and content/errors for each file.
        The operation itself succeeds unless input validation fails.
    """
    start_time = time.monotonic()

    # Input validation
    if not isinstance(paths, list):
        raise ToolInputError(
            "Input must be a list of paths.", param_name="paths", provided_value=type(paths)
        )
    if not paths:  # Handle empty list input explicitly
        logger.info("read_multiple_files called with empty list.", emoji_key="info")
        return {
            "files": [],
            "succeeded": 0,
            "failed": 0,
            "success": True,
            "message": "No paths provided to read.",
        }
    if not all(isinstance(p, str) for p in paths):
        invalid_path = next((p for p in paths if not isinstance(p, str)), None)
        raise ToolInputError(
            "All items in the 'paths' list must be strings.",
            param_name="paths",
            provided_value=f"List contains element of type {type(invalid_path)}",
        )

    # --- Inner Task Definition ---
    async def read_single_file_task(path: str) -> Dict[str, Any]:
        """Task to read and process a single file for read_multiple_files."""
        task_result: Dict[str, Any] = {"path": path, "success": False}  # Initialize result dict
        validated_path: Optional[str] = None
        try:
            validated_path = await validate_path(
                path, check_exists=True, check_parent_writable=False
            )
            task_result["path"] = validated_path  # Update path in result if validation succeeds

            # check isfile (follows links)
            if not await aiofiles.os.path.isfile(validated_path):
                # Check if link before failing
                if await aiofiles.os.path.islink(validated_path):
                    task_result["error"] = "Path is a link, but does not point to a regular file"
                else:
                    task_result["error"] = "Path exists but is not a regular file"
                return task_result

            read_error = None
            file_size = -1

            # Try reading as text (UTF-8)
            try:
                content_str = await read_file_content(validated_path)
                task_result["content"] = content_str
            except ToolError as text_err:
                if "not valid UTF-8" in str(text_err):
                    try:
                        binary_content = await read_binary_file_content(validated_path)
                        # For multi-read, just provide preview directly in result content
                        hex_preview_len = 200
                        hex_preview = binary_content[:hex_preview_len].hex(" ")
                        ellipsis = "..." if len(binary_content) > hex_preview_len else ""
                        preview_msg = f"<binary file detected, hex preview (first {min(hex_preview_len, len(binary_content))} bytes)>: {hex_preview}{ellipsis}"
                        task_result["content"] = preview_msg
                        task_result["is_binary"] = True
                    except ToolError as bin_err:
                        read_error = f"Error reading as binary: {str(bin_err)}"
                    except Exception as bin_e:
                        read_error = f"Unexpected error reading as binary: {str(bin_e)}"
                else:
                    read_error = f"Error reading file: {str(text_err)}"
            except Exception as text_e:
                read_error = f"Unexpected error reading file as text: {str(text_e)}"

            if read_error:
                task_result["error"] = read_error
                return task_result  # Mark as failed

            # Successfully read content (string or binary preview string)
            task_result["success"] = True

            # Try to get size (use stat with follow_symlinks=False for consistency)
            try:
                file_size = (await aiofiles.os.stat(validated_path, follow_symlinks=False)).st_size
                task_result["size"] = file_size
            except OSError as stat_err:
                logger.warning(
                    f"Could not get size for {validated_path} in multi-read: {stat_err}",
                    emoji_key="warning",
                )
                task_result["warning"] = "Could not retrieve file size."

            return task_result

        except (ToolInputError, ToolError) as e:
            # Handle validation or specific tool errors for this path
            task_result["error"] = str(e)
            task_result["path"] = (
                validated_path or path
            )  # Use original path if validation failed early
            return task_result
        except Exception as e:
            # Catch unexpected errors during processing of a single file
            logger.error(
                f"Unexpected error reading single file {path} in multi-read: {e}",
                exc_info=True,
                emoji_key="error",
            )
            task_result["error"] = f"Unexpected error: {str(e)}"
            task_result["path"] = validated_path or path
            return task_result

    # --- End Inner Task Definition ---

    # Execute reads concurrently using asyncio.gather
    results = await asyncio.gather(
        *(read_single_file_task(p) for p in paths), return_exceptions=True
    )

    # Process results (handle potential exceptions returned by gather)
    processed_results: List[Dict[str, Any]] = []
    successful_count = 0
    failed_count = 0

    for i, res in enumerate(results):
        original_path = paths[i]  # Keep track of the requested path
        if isinstance(res, Exception):
            # An unexpected exception occurred *outside* the try/except in the task (unlikely)
            logger.error(
                f"Gather returned exception for path '{original_path}': {res}",
                exc_info=res,
                emoji_key="error",
            )
            processed_results.append(
                {
                    "path": original_path,
                    "error": f"Internal error during task execution: {res}",
                    "success": False,
                }
            )
            failed_count += 1
        elif isinstance(res, dict):
            # Expected dictionary output from our task
            processed_results.append(res)
            if res.get("success"):
                successful_count += 1
            else:
                failed_count += 1
        else:
            # Should not happen if task always returns dict
            logger.error(
                f"Unexpected result type from task for path '{original_path}': {type(res)}",
                emoji_key="error",
            )
            processed_results.append(
                {
                    "path": original_path,
                    "error": f"Internal error: Unexpected task result type {type(res)}",
                    "success": False,
                }
            )
            failed_count += 1

    processing_time = time.monotonic() - start_time
    logger.success(
        f"Finished read_multiple_files: {successful_count} succeeded, {failed_count} failed",
        emoji_key="file",
        total_files=len(paths),
        time=processing_time,
    )

    # Return a dictionary structure that create_tool_response understands
    return {
        "files": processed_results,
        "succeeded": successful_count,
        "failed": failed_count,
        "success": True,  # The overall tool execution was successful (individual files might have failed)
    }


@with_tool_metrics
@with_error_handling
async def get_unique_filepath(path: str) -> Dict[str, Any]:
    """
    Finds an available (non-existent) filepath based on the requested path.

    If the requested path already doesn't exist, it's returned directly.
    If it exists, it appends counters like '_1', '_2', etc., to the filename stem
    until an unused path is found within the same directory.

    Args:
        path: The desired file path.

    Returns:
        Dictionary containing the unique, validated, absolute path found.

    Raises:
        ToolInputError: If the base path is invalid or outside allowed directories.
        ToolError: If the counter limit is exceeded or filesystem errors occur.
    """
    start_time = time.monotonic()
    MAX_FILENAME_ATTEMPTS = 1000  # Safety limit

    try:
        # 1. Validate the *input* path first.
        #    check_exists=None because the *final* path might not exist.
        #    check_parent_writable=False - we only need read/stat access to check existence.
        #    The parent directory's writability should be checked by the *calling* function
        #    (like write_file or smart_download's directory creation) before this.
        validated_input_path = await validate_path(
            path, check_exists=None, check_parent_writable=False
        )
        logger.debug(
            f"get_unique_filepath: Validated input path resolves to {validated_input_path}"
        )

        # 2. Check if the initial validated path is already available.
        if not await aiofiles.os.path.exists(validated_input_path):
            logger.info(f"Initial path '{validated_input_path}' is available.", emoji_key="file")
            return {
                "path": validated_input_path,
                "attempts": 0,
                "success": True,
                "message": f"Path '{validated_input_path}' is already unique.",
            }

        # 3. Path exists, need to find a unique alternative.
        logger.debug(f"Path '{validated_input_path}' exists, finding unique alternative.")
        dirname = os.path.dirname(validated_input_path)
        original_filename = os.path.basename(validated_input_path)
        stem, suffix = os.path.splitext(original_filename)

        counter = 1
        while counter <= MAX_FILENAME_ATTEMPTS:
            # Construct new filename candidate
            candidate_filename = f"{stem}_{counter}{suffix}"
            candidate_path = os.path.join(dirname, candidate_filename)

            # Check existence asynchronously
            if not await aiofiles.os.path.exists(candidate_path):
                processing_time = time.monotonic() - start_time
                logger.success(
                    f"Found unique path '{candidate_path}' after {counter} attempts.",
                    emoji_key="file",
                    time=processing_time,
                )
                return {
                    "path": candidate_path,
                    "attempts": counter,
                    "success": True,
                    "message": f"Found unique path '{candidate_path}'.",
                }

            counter += 1

        # If loop finishes, we exceeded the limit
        raise ToolError(
            f"Could not find a unique filename based on '{path}' after {MAX_FILENAME_ATTEMPTS} attempts.",
            context={"base_path": validated_input_path, "attempts": MAX_FILENAME_ATTEMPTS},
        )

    except OSError as e:
        # Catch errors during exists checks
        logger.error(
            f"Filesystem error finding unique path based on '{path}': {str(e)}",
            exc_info=True,
            emoji_key="error",
        )
        raise ToolError(
            f"Filesystem error checking path existence: {str(e)}",
            context={"path": path, "errno": e.errno},
        ) from e
    except (
        ToolInputError,
        ToolError,
    ):  # Re-raise specific errors from validate_path or the counter limit
        raise
    except Exception as e:
        # Catch unexpected errors
        logger.error(
            f"Unexpected error finding unique path for {path}: {e}",
            exc_info=True,
            emoji_key="error",
        )
        raise ToolError(
            f"An unexpected error occurred finding a unique path: {str(e)}", context={"path": path}
        ) from e


@with_tool_metrics
@with_error_handling
async def write_file(path: str, content: Union[str, bytes]) -> Dict[str, Any]:
    """Write content to a file asynchronously (UTF-8 or binary), creating/overwriting.

    Ensures the path is valid, within allowed directories, and that the parent
    directory exists and is writable. Fails if the target path exists and is a directory.

    Args:
        path: Path to the file to write.
        content: Content to write (string for text UTF-8, bytes for binary).

    Returns:
        A dictionary confirming success and providing file details (path, size).
    """
    start_time = time.monotonic()

    # Validate content type explicitly at the start.
    if not isinstance(content, (str, bytes)):
        raise ToolInputError(
            "Content to write must be a string or bytes.",
            param_name="content",
            provided_value=type(content),
        )

    # Validate path: doesn't need to exist necessarily (check_exists=None), but parent must exist and be writable.
    validated_path = await validate_path(path, check_exists=None, check_parent_writable=True)

    # Check if the path exists and is a directory (we shouldn't overwrite a dir with a file).
    # Use exists instead of lexists for this check
    if await aiofiles.os.path.exists(validated_path) and await aiofiles.os.path.isdir(
        validated_path
    ):
        raise ToolInputError(
            f"Cannot write file: Path '{path}' (resolved to '{validated_path}') exists and is a directory.",
            param_name="path",
            provided_value=path,
        )

    # write_file_content handles actual writing and parent dir creation
    await write_file_content(validated_path, content)  # Can raise ToolError

    # Verify write success by getting status and size afterwards
    file_size = -1
    try:
        file_size = (await aiofiles.os.stat(validated_path, follow_symlinks=False)).st_size
    except OSError as e:
        # If stat fails after write seemed to succeed, something is wrong.
        logger.error(
            f"File write appeared successful for {validated_path}, but failed to get status afterwards: {e}",
            emoji_key="error",
        )
        raise ToolError(
            f"File written but failed to verify status afterwards: {str(e)}",
            context={"path": validated_path},
        ) from e

    processing_time = time.monotonic() - start_time
    logger.success(
        f"Successfully wrote file: {path}", emoji_key="file", size=file_size, time=processing_time
    )

    # Return a structured success response.
    return {
        "message": f"Successfully wrote {file_size} bytes to '{validated_path}'.",
        "path": validated_path,
        "size": file_size,
        "success": True,
    }


@with_tool_metrics
@with_error_handling
async def edit_file(
    path: str, edits: List[Dict[str, str]], dry_run: bool = False
) -> Dict[str, Any]:
    """Edit a text file asynchronously by applying string replacements (UTF-8).

    Validates path and edits, reads the file, applies changes (with fallbacks for
    whitespace differences), generates a diff, and optionally writes back.

    Args:
        path: Path to the file to edit. Must be an existing text file.
        edits: List of edit operations. Each dict needs 'oldText' (str)
               and 'newText' (str).
        dry_run: If True, calculates changes and diff but does not save them.

    Returns:
        Dictionary containing the generated diff, success status, path,
        and whether it was a dry run.
    """
    start_time = time.monotonic()

    # Validate path must exist and be a file (check_exists=True).
    validated_path = await validate_path(path, check_exists=True, check_parent_writable=False)
    # Check if it's a regular file (follows links)
    if not await aiofiles.os.path.isfile(validated_path):
        if await aiofiles.os.path.islink(validated_path):
            raise ToolInputError(
                f"Path '{path}' (resolved to link '{validated_path}') points to something that is not a regular file.",
                param_name="path",
                provided_value=path,
            )
        else:
            raise ToolInputError(
                f"Path '{path}' (resolved to '{validated_path}') is not a regular file.",
                param_name="path",
                provided_value=path,
            )

    # Validate edits structure
    if not isinstance(edits, list):
        raise ToolInputError(
            "Edits parameter must be a list.", param_name="edits", provided_value=type(edits)
        )
    if not edits:
        # Handle empty edits list as a no-op success.
        logger.info(
            f"edit_file called with empty edits list for {path}. No changes will be made.",
            emoji_key="info",
        )
        return {
            "path": validated_path,
            "diff": "No edits provided.",
            "success": True,
            "dry_run": dry_run,
            "message": "No edits were specified.",
        }
    # Deeper validation of each edit dict happens within apply_file_edits

    # NOTE: Modification protection is currently NOT applied here.
    # This operates on a single file. Bulk editing would require a different tool
    # where protection heuristics might be applied.

    # apply_file_edits handles reading, core editing logic, diffing, conditional writing.
    # Raises ToolInputError/ToolError on failure.
    diff, new_content = await apply_file_edits(validated_path, edits, dry_run)

    processing_time = time.monotonic() - start_time

    action = "Previewed edits for" if dry_run else "Applied edits to"
    logger.success(
        f"Successfully {action} file: {path}",
        emoji_key="file",
        num_edits=len(edits),
        dry_run=dry_run,
        time=processing_time,
        changes_made=(diff != ""),  # Log if actual changes resulted
    )

    # Provide clearer messages based on diff content and dry_run status.
    if diff:
        diff_message = diff
        status_message = f"Successfully {'previewed' if dry_run else 'applied'} {len(edits)} edits."
    else:  # Edits provided, but resulted in no change to content
        diff_message = "No changes detected after applying edits."
        status_message = f"{len(edits)} edits provided, but resulted in no content changes."

    return {
        "path": validated_path,
        "diff": diff_message,
        "success": True,
        "dry_run": dry_run,
        "message": status_message,
        # "new_content": new_content # Optional: return new content, especially for dry runs? Can be large.
    }


@with_tool_metrics
@with_error_handling
async def create_directory(path: str) -> Dict[str, Any]:
    """Create a directory asynchronously, including parent directories (like 'mkdir -p').

    Validates the path is allowed and parent is writable. Idempotent: If the directory
    already exists, it succeeds without error. Fails if the path exists but is a file.

    Args:
        path: Path to the directory to create.

    Returns:
        Dictionary confirming success, path, and whether it was newly created.
    """
    start_time = time.monotonic()

    # Validate path, parent must exist/be writable. Path itself should ideally not exist (check_exists=None allows check).
    validated_path = await validate_path(path, check_exists=None, check_parent_writable=True)

    created = False
    message = ""
    try:
        # Check existence and type before creating, using exists instead of lexists
        if await aiofiles.os.path.exists(validated_path):
            if await aiofiles.os.path.isdir(validated_path):
                # Directory already exists - idempotent success.
                logger.info(
                    f"Directory already exists: {path} (resolved: {validated_path})",
                    emoji_key="directory",
                )
                message = f"Directory '{validated_path}' already exists."
            else:
                # Path exists but is not a directory (e.g., a file or symlink)
                raise ToolInputError(
                    f"Cannot create directory: Path '{path}' (resolved to '{validated_path}') already exists but is not a directory.",
                    param_name="path",
                    provided_value=path,
                )
        else:
            # Path does not exist, proceed with creation using async makedirs.
            await aiofiles.os.makedirs(validated_path, exist_ok=True)
            created = True
            logger.success(
                f"Successfully created directory: {path} (resolved: {validated_path})",
                emoji_key="directory",
            )
            message = f"Successfully created directory '{validated_path}'."

    except OSError as e:
        # Catch errors during the exists/isdir checks or makedirs call
        logger.error(
            f"Error creating directory '{path}' (resolved: {validated_path}): {e}",
            exc_info=True,
            emoji_key="error",
        )
        raise ToolError(
            f"Error creating directory '{path}': {str(e)}",
            context={"path": validated_path, "errno": e.errno},
        ) from e
    except Exception as e:
        # Catch unexpected errors
        logger.error(
            f"Unexpected error creating directory {path}: {e}", exc_info=True, emoji_key="error"
        )
        raise ToolError(
            f"An unexpected error occurred creating directory: {str(e)}",
            context={"path": validated_path},
        ) from e

    processing_time = time.monotonic() - start_time
    logger.success(
        f"Successfully created directory: {path} (resolved: {validated_path})",
        emoji_key="directory",
        time=processing_time,
    )

    return {"path": validated_path, "created": created, "success": True, "message": message}


@with_tool_metrics
@with_error_handling
async def list_directory(path: str) -> Dict[str, Any]:
    """List files and subdirectories within a given directory asynchronously.

    Validates the path exists, is a directory, and is allowed.
    Provides basic info (name, type, size for files, link target for symlinks) for each entry.

    Args:
        path: Path to the directory to list.

    Returns:
        Dictionary containing the path listed and a list of entries.
    """
    start_time = time.monotonic()

    # Validate path exists and is a directory (check_exists=True).
    validated_path = await validate_path(path, check_exists=True, check_parent_writable=False)
    if not await aiofiles.os.path.isdir(validated_path):
        # If it's a link, check if it points to a directory
        if await aiofiles.os.path.islink(validated_path):
            try:
                target_is_dir = await aiofiles.os.path.isdir(
                    await aiofiles.os.path.realpath(validated_path)
                )
                if not target_is_dir:
                    raise ToolInputError(
                        f"Path '{path}' (resolved to link '{validated_path}') points to something that is not a directory.",
                        param_name="path",
                        provided_value=path,
                    )
                # If target is dir, proceed using validated_path (which resolves the link)
            except OSError as e:
                raise ToolError(
                    f"Error resolving or checking link target for directory listing: {e}",
                    context={"path": validated_path},
                ) from e
        else:
            raise ToolInputError(
                f"Path '{path}' (resolved to '{validated_path}') is not a directory.",
                param_name="path",
                provided_value=path,
            )

    entries: List[Dict[str, Any]] = []
    scan_errors: List[str] = []
    try:
        # Use await with scandir rather than async iteration
        entry_list = await aiofiles.os.scandir(validated_path)
        for entry in entry_list:
            entry_info: Dict[str, Any] = {"name": entry.name}
            try:
                # Use async methods on the DirEntry object for efficiency, check link status explicitly
                is_link = entry.is_symlink()  # Checks if entry *itself* is a link
                entry_info["is_symlink"] = is_link

                # Let's be explicit using lstat results for type determination
                try:
                    stat_res = entry.stat(follow_symlinks=False)  # Use lstat via entry
                    mode = stat_res.st_mode
                    l_is_dir = os.path.stat.S_ISDIR(mode)
                    l_is_file = os.path.stat.S_ISREG(mode)
                    l_is_link = os.path.stat.S_ISLNK(mode)  # Should match entry.is_symlink() result

                    if l_is_dir:
                        entry_info["type"] = "directory"
                    elif l_is_file:
                        entry_info["type"] = "file"
                        entry_info["size"] = stat_res.st_size  # Size of file itself
                    elif l_is_link:
                        entry_info["type"] = "symlink"
                        entry_info["size"] = stat_res.st_size  # Size of link itself
                        # Optionally try to resolve link target for display
                        try:
                            target = await aiofiles.os.readlink(entry.path)
                            entry_info["symlink_target"] = target
                        except OSError as link_err:
                            entry_info["error"] = f"Could not read link target: {link_err}"
                    else:
                        entry_info["type"] = "other"  # E.g., socket, fifo
                        entry_info["size"] = stat_res.st_size

                except OSError as stat_err:
                    logger.warning(
                        f"Could not lstat entry {entry.path} in list_directory: {stat_err}",
                        emoji_key="warning",
                    )
                    entry_info["type"] = "error"
                    entry_info["error"] = f"Could not get info: {stat_err}"

                entries.append(entry_info)

            except OSError as entry_err:
                # Error processing a specific entry (e.g., permission denied on is_dir/is_file/is_symlink)
                logger.warning(
                    f"Could not process directory entry '{entry.name}' in {validated_path}: {entry_err}",
                    emoji_key="warning",
                )
                error_message = f"Error processing entry '{entry.name}': {entry_err}"
                scan_errors.append(error_message)
                # Add error entry to the list for visibility
                entries.append({"name": entry.name, "type": "error", "error": str(entry_err)})

        # Sort entries: directories, files, symlinks, others, errors; then alphabetically.
        entries.sort(
            key=lambda e: (
                0
                if e.get("type") == "directory"
                else 1
                if e.get("type") == "file"
                else 2
                if e.get("type") == "symlink"
                else 3
                if e.get("type") == "other"
                else 4,  # Errors last
                e.get("name", "").lower(),  # Case-insensitive sort by name
            )
        )

    except OSError as e:
        # Error during the initial scandir call (e.g., permission denied on the directory itself)
        raise ToolError(
            f"Error listing directory '{path}': {str(e)}",
            context={"path": validated_path, "errno": e.errno},
        ) from e
    except Exception as e:
        # Catch unexpected errors during iteration
        logger.error(
            f"Unexpected error listing directory {path}: {e}", exc_info=True, emoji_key="error"
        )
        raise ToolError(
            f"An unexpected error occurred listing directory: {str(e)}",
            context={"path": validated_path},
        ) from e

    processing_time = time.monotonic() - start_time
    logger.success(
        f"Listed directory: {path} ({len(entries)} entries found, {len(scan_errors)} errors)",
        emoji_key="directory",
        time=processing_time,
    )

    # Structure result clearly, including warnings if errors occurred on entries.
    result = {
        "path": validated_path,
        "entries": entries,
        "success": True,
        "message": f"Found {len(entries)} entries in '{validated_path}'.",
    }
    if scan_errors:
        warning_summary = f"Encountered {len(scan_errors)} errors processing directory entries."
        result["warnings"] = [warning_summary]
        # Optionally include first few errors: result["error_details"] = scan_errors[:5]

    return result


@with_tool_metrics
@with_error_handling
async def directory_tree(
    path: str, max_depth: int = 3, include_size: bool = False
) -> Dict[str, Any]:
    """Get a recursive tree view of a directory structure asynchronously.

    Args:
        path: Path to the root directory for the tree view.
        max_depth: Maximum recursion depth (-1 for effectively unlimited, capped internally).
        include_size: If True, include file sizes in the tree (requires extra stat calls).

    Returns:
        Dictionary containing the path and the hierarchical tree structure.
    """
    start_time = time.monotonic()
    # Internal safety cap for 'unlimited' depth to prevent runaway recursion.
    INTERNAL_DEPTH_CAP = 15

    # Validate path exists and is a directory (check_exists=True)
    validated_path = await validate_path(path, check_exists=True)
    if not await aiofiles.os.path.isdir(validated_path):
        # Check if it's a link to a directory
        if await aiofiles.os.path.islink(validated_path):
            try:
                target_is_dir = await aiofiles.os.path.isdir(
                    await aiofiles.os.path.realpath(validated_path)
                )
                if not target_is_dir:
                    raise ToolInputError(
                        f"Path '{path}' (resolved to link '{validated_path}') points to something that is not a directory.",
                        param_name="path",
                        provided_value=path,
                    )
                # proceed with validated_path which resolves link
            except OSError as e:
                raise ToolError(
                    f"Error resolving or checking link target for directory tree: {e}",
                    context={"path": validated_path},
                ) from e
        else:
            raise ToolInputError(
                f"Path '{path}' (resolved to '{validated_path}') is not a directory.",
                param_name="path",
                provided_value=path,
            )

    # Validate max_depth input
    if not isinstance(max_depth, int):
        raise ToolInputError(
            "max_depth must be an integer.", param_name="max_depth", provided_value=max_depth
        )

    # Apply internal depth cap if necessary
    actual_max_depth = max_depth
    if max_depth < 0 or max_depth > INTERNAL_DEPTH_CAP:
        if max_depth >= 0:  # Only warn if user specified a large number, not for -1
            logger.warning(
                f"Requested max_depth {max_depth} exceeds internal cap {INTERNAL_DEPTH_CAP}. Limiting depth.",
                emoji_key="warning",
            )
        actual_max_depth = INTERNAL_DEPTH_CAP

    # --- Recursive Helper ---
    async def build_tree_recursive(current_path: str, current_depth: int) -> List[Dict[str, Any]]:
        """Recursively builds the directory tree structure."""
        if current_depth > actual_max_depth:
            # Return specific marker if depth limit hit, not just empty list.
            return [{"name": f"... (Max depth {actual_max_depth} reached)", "type": "info"}]

        children_nodes: List[Dict[str, Any]] = []
        try:
            # Await scandir and then iterate over the returned entries
            entries = await aiofiles.os.scandir(current_path)
            for entry in entries:
                entry_data: Dict[str, Any] = {"name": entry.name}
                try:
                    # Use lstat via entry to avoid following links unexpectedly
                    stat_res = entry.stat(follow_symlinks=False)
                    mode = stat_res.st_mode
                    l_is_dir = os.path.stat.S_ISDIR(mode)
                    l_is_file = os.path.stat.S_ISREG(mode)
                    l_is_link = os.path.stat.S_ISLNK(mode)

                    if l_is_dir:
                        entry_data["type"] = "directory"
                        # Recurse asynchronously into subdirectory
                        entry_data["children"] = await build_tree_recursive(
                            entry.path, current_depth + 1
                        )
                    elif l_is_file:
                        entry_data["type"] = "file"
                        if include_size:
                            entry_data["size"] = stat_res.st_size
                    elif l_is_link:
                        entry_data["type"] = "symlink"
                        if include_size:
                            entry_data["size"] = stat_res.st_size  # Size of link itself
                        # Optionally resolve link target? Can be noisy. Let's skip for tree view simplicity.
                        # try: entry_data["target"] = await aiofiles.os.readlink(entry.path)
                        # except OSError: entry_data["target"] = "<Error reading link>"
                    else:
                        entry_data["type"] = "other"
                        if include_size:
                            entry_data["size"] = stat_res.st_size

                    children_nodes.append(entry_data)

                except OSError as entry_err:
                    # Error processing one entry (e.g., permissions on stat)
                    logger.warning(
                        f"Could not process entry {entry.path} in tree: {entry_err}",
                        emoji_key="warning",
                    )
                    children_nodes.append(
                        {"name": entry.name, "type": "error", "error": str(entry_err)}
                    )

            # Sort entries at the current level alphabetically by name, type secondary
            children_nodes.sort(
                key=lambda e: (
                    0
                    if e.get("type") == "directory"
                    else 1
                    if e.get("type") == "file"
                    else 2
                    if e.get("type") == "symlink"
                    else 3,
                    e.get("name", "").lower(),  # Case-insensitive sort
                )
            )
            return children_nodes

        except OSError as e:
            # Error scanning the directory itself (e.g., permissions)
            logger.error(
                f"Error scanning directory {current_path} for tree: {str(e)}", emoji_key="error"
            )
            # Return error indicator instead of raising, allows partial trees if root is accessible
            return [{"name": f"... (Error scanning this directory: {e})", "type": "error"}]
        except Exception as e:
            # Unexpected error during scan
            logger.error(
                f"Unexpected error scanning directory {current_path} for tree: {e}",
                exc_info=True,
                emoji_key="error",
            )
            return [{"name": f"... (Unexpected error scanning: {e})", "type": "error"}]

    # --- End Recursive Helper ---

    # Generate tree starting from the validated path
    tree_structure = await build_tree_recursive(validated_path, 0)

    processing_time = time.monotonic() - start_time
    logger.success(
        f"Generated directory tree for: {path}",
        emoji_key="directory",
        max_depth=actual_max_depth,  # Log the effective depth
        requested_depth=max_depth,
        include_size=include_size,
        time=processing_time,
    )

    # Structure result clearly
    return {
        "path": validated_path,
        "max_depth_reached": actual_max_depth,
        "tree": tree_structure,
        "success": True,
        "message": f"Generated directory tree for '{validated_path}' up to depth {actual_max_depth}.",
    }


@with_tool_metrics
@with_error_handling
async def move_file(
    source: str,
    destination: str,
    overwrite: bool = False,  # Default to NOT overwrite for safety
) -> Dict[str, Any]:
    """Move or rename a file or directory asynchronously.

    Ensures both source and destination paths are within allowed directories.
    Checks for existence and potential conflicts before moving.

    Args:
        source: Path to the file or directory to move. Must exist.
        destination: New path for the file or directory. Parent must exist and be writable.
        overwrite: If True, allows overwriting an existing file or *empty* directory
                   at the destination. USE WITH CAUTION. Defaults False.

    Returns:
        Dictionary confirming the move operation, including source and destination paths.
    """
    start_time = time.monotonic()

    # Validate source path (must exist, check_exists=True)
    validated_source = await validate_path(source, check_exists=True)

    # Validate destination path (parent must exist and be writable, path itself may or may not exist check_exists=None)
    # Check parent writability *before* checking overwrite logic.
    validated_dest = await validate_path(destination, check_exists=None, check_parent_writable=True)

    # Validate overwrite flag type
    if not isinstance(overwrite, bool):
        raise ToolInputError(
            "overwrite parameter must be a boolean (true/false).",
            param_name="overwrite",
            provided_value=type(overwrite),
        )

    # NOTE: Modification protection is currently NOT applied here.
    # If overwriting a directory, current logic only allows overwriting an *empty* one via rmdir.
    # Overwriting a file is a single-file modification.
    # A future enhancement could add protection heuristics here if overwriting non-empty dirs was allowed.

    try:
        # Check if destination already exists using exists instead of lexists
        dest_exists = await aiofiles.os.path.exists(validated_dest)
        dest_is_dir = False
        if dest_exists:
            dest_is_dir = await aiofiles.os.path.isdir(
                validated_dest
            )  # Check type (follows links if dest is link)

        if dest_exists:
            if overwrite:
                # Overwrite requested. Log prominently.
                logger.warning(
                    f"Overwrite flag is True. Attempting to replace existing path '{validated_dest}' with '{validated_source}'.",
                    emoji_key="warning",
                )

                # Check if source and destination types are compatible for overwrite (e.g., cannot replace dir with file easily)
                # Use stat with follow_symlinks=False for source type check as well.
                source_stat = await aiofiles.os.stat(validated_source, follow_symlinks=False)
                is_source_dir = os.path.stat.S_ISDIR(source_stat.st_mode)

                # Simple check: Prevent overwriting dir with file or vice-versa.
                # Note: aiofiles.os.rename might handle some cases, but explicit check is safer.
                # This logic assumes we'd remove the destination first.
                if is_source_dir != dest_is_dir:
                    # Allow replacing link with dir/file, or dir/file with link? Be cautious.
                    # Let's prevent dir/file mismatch for simplicity. Overwriting links is tricky.
                    if not await aiofiles.os.path.islink(
                        validated_dest
                    ):  # Only enforce if dest is not a link
                        raise ToolInputError(
                            f"Cannot overwrite: Source is a {'directory' if is_source_dir else 'file/link'} but destination ('{validated_dest}') exists and is a {'directory' if dest_is_dir else 'file/link'}.",
                            param_name="destination",
                            provided_value=destination,
                        )

                # Attempt to remove the existing destination. This is the dangerous part.
                try:
                    if dest_is_dir:
                        # Use async rmdir - fails if directory is not empty!
                        # This prevents accidental recursive deletion via move+overwrite.
                        await aiofiles.os.rmdir(validated_dest)
                        logger.info(
                            f"Removed existing empty directory destination '{validated_dest}' for overwrite.",
                            emoji_key="action",
                        )
                    else:
                        # Removes a file or symlink
                        await aiofiles.os.remove(validated_dest)
                        logger.info(
                            f"Removed existing file/link destination '{validated_dest}' for overwrite.",
                            emoji_key="action",
                        )
                except OSError as remove_err:
                    # Handle cases like directory not empty, permission error during removal
                    raise ToolError(
                        f"Failed to remove existing destination '{validated_dest}' for overwrite: {remove_err}. Check permissions or if directory is empty.",
                        context={
                            "source": validated_source,
                            "destination": validated_dest,
                            "errno": remove_err.errno,
                        },
                    ) from remove_err

            else:  # Destination exists, and overwrite is False (default)
                raise ToolInputError(
                    f"Cannot move: Destination path '{destination}' (resolved to '{validated_dest}') already exists. Use overwrite=True to replace.",
                    param_name="destination",
                    provided_value=destination,
                )

        # Ensure source and destination are not the same path after normalization/resolution.
        # Note: aiofiles.os.rename handles this check internally too, but explicit check is clearer.
        # Use os.path.samfile to check if they refer to the same actual file/inode (more robust than string comparison)
        # samfile is sync, but should be fast. Requires paths to exist.
        try:
            # Source exists. Does dest exist now (after potential removal)?
            dest_exists_after_remove = await aiofiles.os.path.exists(validated_dest)  # noqa: F841
            # Only call samefile if both paths point to existing things after the removal step.
            # This check is mainly useful if overwrite was False and dest didn't exist initially but resolved same as source.
            # If dest existed and overwrite=True, it should have been removed.
            # Let's simplify: compare final validated paths. If rename fails later, it likely handles identity internally.
            if validated_source == validated_dest:
                logger.info(
                    f"Source and destination paths resolve to the same location ('{validated_source}'). No move needed.",
                    emoji_key="info",
                )
                return {
                    "source": validated_source,
                    "destination": validated_dest,
                    "success": True,
                    "message": "Source and destination are the same path. No operation performed.",
                }
        except OSError:
            # Ignore errors from samefile check, rely on rename's internal checks.
            pass

        # Attempt the move/rename operation asynchronously
        await aiofiles.os.rename(validated_source, validated_dest)

    except OSError as e:
        # Catch errors from exists, rename, remove, rmdir etc.
        logger.error(
            f"Error moving/renaming from '{source}' to '{destination}': {str(e)}",
            exc_info=True,
            emoji_key="error",
        )
        raise ToolError(
            f"Error moving '{source}' to '{destination}': {str(e)}",
            context={"source": validated_source, "destination": validated_dest, "errno": e.errno},
        ) from e
    except (ToolInputError, ToolError, ProtectionTriggeredError):  # Re-raise our specific errors
        raise
    except Exception as e:
        # Catch unexpected errors
        logger.error(
            f"Unexpected error moving {source} to {destination}: {e}",
            exc_info=True,
            emoji_key="error",
        )
        raise ToolError(
            f"An unexpected error occurred during move: {str(e)}",
            context={"source": validated_source, "destination": validated_dest},
        ) from e

    processing_time = time.monotonic() - start_time
    logger.success(
        f"Moved '{source}' to '{destination}'",
        emoji_key="file",
        time=processing_time,
        overwrite_used=overwrite,
    )

    return {
        "source": validated_source,  # Report original source for clarity
        "destination": validated_dest,
        "success": True,
        "message": f"Successfully moved '{validated_source}' to '{validated_dest}'.",
    }


@with_tool_metrics
@with_error_handling
async def delete_path(path: str) -> Dict[str, Any]:
    """Delete a file or an entire directory tree asynchronously.

    Validates the path exists and is allowed. If the path is a directory,
    applies deletion protection heuristics (if enabled) based on the directory's
    contents before proceeding with recursive deletion.

    Args:
        path: Path to the file or directory to delete.

    Returns:
        Dictionary confirming the deletion.
    """
    start_time = time.monotonic()

    # Validate path exists (check_exists=True), but don't resolve symlinks yet
    # We need to know if the original path is a symlink to handle it properly
    validation_result = await validate_path(path, check_exists=True, resolve_symlinks=False)

    # Check if the path is a symlink
    is_symlink = await aiofiles.os.path.islink(path)
    logger.info(f"Note: Deleting the symlink itself (not its target) at path: {path}")
    validated_path = validation_result

    try:
        deleted_type = "unknown"

        # First check if it's a symlink - we want to handle this separately
        # to avoid accidentally deleting the target
        if is_symlink:
            deleted_type = "symlink"
            logger.info(f"Deleting symlink: {validated_path}", emoji_key="delete")
            await aiofiles.os.remove(validated_path)

        # If not a symlink, proceed with regular directory or file deletion
        else:
            # We need to check if it's a directory or file - use follow_symlinks=True because
            # we already handled the symlink case above
            stat_info = await aiofiles.os.stat(validated_path, follow_symlinks=True)
            is_dir = os.path.stat.S_ISDIR(stat_info.st_mode)
            is_file = os.path.stat.S_ISREG(stat_info.st_mode)

            if is_dir:
                deleted_type = "directory"
                logger.info(f"Attempting to delete directory: {validated_path}", emoji_key="delete")
                # --- Deletion Protection Check ---
                try:
                    # List all file paths within the directory for heuristic checks
                    contained_file_paths = await _list_paths_recursive(validated_path)
                    if contained_file_paths:  # Only run check if directory is not empty
                        await _check_protection_heuristics(contained_file_paths, "deletion")
                    else:
                        logger.info(
                            f"Directory '{validated_path}' is empty, skipping detailed protection check.",
                            emoji_key="info",
                        )
                except ProtectionTriggeredError:
                    raise  # Re-raise the specific error if protection blocked the operation
                except ToolError as list_err:
                    # If listing contents failed, block deletion for safety?
                    raise ToolError(
                        f"Could not list directory contents for safety check before deleting '{validated_path}'. Deletion aborted. Reason: {list_err}",
                        context={"path": validated_path},
                    ) from list_err
                # --- End Protection Check ---

                # Protection passed (or disabled/not triggered), proceed with recursive delete
                await _async_rmtree(validated_path)

            elif is_file:
                deleted_type = "file"
                logger.info(f"Attempting to delete file: {validated_path}", emoji_key="delete")
                await aiofiles.os.remove(validated_path)
            else:
                # Should not happen if lexists passed validation, but handle defensively
                raise ToolError(
                    f"Cannot delete path '{validated_path}': It is neither a file, directory, nor a symbolic link.",
                    context={"path": validated_path},
                )

    except OSError as e:
        # Catch errors from remove, rmdir, or during rmtree
        logger.error(
            f"Error deleting path '{path}' (resolved: {validated_path}): {str(e)}",
            exc_info=True,
            emoji_key="error",
        )
        raise ToolError(
            f"Error deleting '{path}': {str(e)}", context={"path": validated_path, "errno": e.errno}
        ) from e
    except (ToolInputError, ToolError, ProtectionTriggeredError):  # Re-raise our specific errors
        raise
    except Exception as e:
        # Catch unexpected errors
        logger.error(f"Unexpected error deleting {path}: {e}", exc_info=True, emoji_key="error")
        raise ToolError(
            f"An unexpected error occurred during deletion: {str(e)}",
            context={"path": validated_path},
        ) from e

    processing_time = time.monotonic() - start_time
    logger.success(
        f"Successfully deleted {deleted_type}: '{path}' (resolved: {validated_path})",
        emoji_key="delete",
        time=processing_time,
    )

    return {
        "path": validated_path,
        "type_deleted": deleted_type,
        "success": True,
        "message": f"Successfully deleted {deleted_type} '{validated_path}'.",
    }


@with_tool_metrics
@with_error_handling
async def search_files(
    path: str,
    pattern: str,
    case_sensitive: bool = False,
    exclude_patterns: Optional[List[str]] = None,
    search_content: bool = False,
    max_content_bytes: int = 1024 * 1024,  # Limit content search size per file (1MB)
) -> Dict[str, Any]:
    """Search for files/directories matching a pattern asynchronously and recursively.

    Supports filename pattern matching (case-sensitive/insensitive substring)
    and optional searching within file content (UTF-8 text only, limited size).
    Allows exclusion patterns (glob format).

    Args:
        path: Directory path to start the search from.
        pattern: Text pattern to find. Used for substring match in names,
                 and exact string find in content if search_content=True.
        case_sensitive: If True, matching is case-sensitive. Defaults False.
        exclude_patterns: Optional list of glob-style patterns for paths/names
                          to exclude (e.g., ["*.log", ".git/"]). Matched case-insensitively
                          on appropriate systems.
        search_content: If True, also search *inside* text files for the pattern.
                        This can be significantly slower and memory intensive.
        max_content_bytes: Max bytes to read from each file when search_content=True.

    Returns:
        Dictionary with search parameters and a list of matching file/directory paths.
        May include warnings about errors encountered during search.
    """
    start_time = time.monotonic()

    # Validate path exists and is a directory (check_exists=True)
    validated_path = await validate_path(path, check_exists=True)
    if not await aiofiles.os.path.isdir(validated_path):
        # Check links to dirs
        if await aiofiles.os.path.islink(validated_path):
            try:
                if not await aiofiles.os.path.isdir(
                    await aiofiles.os.path.realpath(validated_path)
                ):
                    raise ToolInputError(
                        f"Path '{path}' (resolved to link '{validated_path}') points to something that is not a directory.",
                        param_name="path",
                        provided_value=path,
                    )
                # proceed with validated_path
            except OSError as e:
                raise ToolError(
                    f"Error resolving or checking link target for search: {e}",
                    context={"path": validated_path},
                ) from e
        else:
            raise ToolInputError(
                f"Path '{path}' (resolved to '{validated_path}') is not a directory.",
                param_name="path",
                provided_value=path,
            )

    # Validate other inputs
    if not isinstance(pattern, str) or not pattern:
        raise ToolInputError(
            "Search pattern must be a non-empty string.",
            param_name="pattern",
            provided_value=pattern,
        )
    if exclude_patterns:  # Ensure it's a list of strings if provided
        if not isinstance(exclude_patterns, list):
            raise ToolInputError(
                "Exclude patterns must be a list of strings.",
                param_name="exclude_patterns",
                provided_value=exclude_patterns,
            )
        if not all(isinstance(p, str) for p in exclude_patterns):
            raise ToolInputError(
                "All items in exclude_patterns must be strings.", param_name="exclude_patterns"
            )
    if not isinstance(case_sensitive, bool):
        raise ToolInputError(
            "case_sensitive must be a boolean.",
            param_name="case_sensitive",
            provided_value=case_sensitive,
        )
    if not isinstance(search_content, bool):
        raise ToolInputError(
            "search_content must be a boolean.",
            param_name="search_content",
            provided_value=search_content,
        )
    if not isinstance(max_content_bytes, int) or max_content_bytes < 0:
        raise ToolInputError(
            "max_content_bytes must be a non-negative integer.",
            param_name="max_content_bytes",
            provided_value=max_content_bytes,
        )

    search_errors: List[str] = []
    # Using a set to store matched paths avoids duplicates efficiently.
    matched_paths: Set[str] = set()

    # Prepare pattern based on case sensitivity
    search_pattern = pattern if case_sensitive else pattern.lower()

    # Error handler callback for async_walk
    MAX_REPORTED_ERRORS = 50

    def onerror(os_error: OSError):
        """Callback to handle and log errors during file tree walking."""
        err_msg = f"Permission or access error during search near '{getattr(os_error, 'filename', 'N/A')}': {getattr(os_error, 'strerror', str(os_error))}"
        # Limit number of reported errors to avoid flooding logs/results
        if len(search_errors) < MAX_REPORTED_ERRORS:
            logger.warning(err_msg, emoji_key="warning")
            search_errors.append(err_msg)
        elif len(search_errors) == MAX_REPORTED_ERRORS:
            suppress_msg = f"... (Further {type(os_error).__name__} errors suppressed)"
            logger.warning(suppress_msg, emoji_key="warning")
            search_errors.append(suppress_msg)

    # --- File Content Search Task (if enabled) ---
    async def check_file_content(filepath: str) -> bool:
        """Reads limited file content and checks for the pattern."""
        try:
            # Read limited chunk, ignore decoding errors for content search robustness
            async with aiofiles.open(filepath, mode="r", encoding="utf-8", errors="ignore") as f:
                content_chunk = await f.read(max_content_bytes)
            # Perform search (case sensitive or insensitive)
            if case_sensitive:
                return pattern in content_chunk
            else:
                return search_pattern in content_chunk.lower()  # Compare lower case
        except OSError as read_err:
            # Log content read errors but don't necessarily fail the whole search
            logger.warning(
                f"Could not read content of {filepath} for search: {read_err}", emoji_key="warning"
            )
            onerror(read_err)  # Report as a search error
        except Exception as read_unexpected_err:
            logger.error(
                f"Unexpected error reading content of {filepath}: {read_unexpected_err}",
                exc_info=True,
                emoji_key="error",
            )
            # Do not report unexpected errors via onerror, log them fully.
        return False

    # --- End File Content Search Task ---

    try:
        # Use the async_walk helper for efficient traversal and exclusion handling
        # followlinks=True: Search should probably follow links to find matches within linked dirs too.
        # Exclude patterns will apply to paths within the linked directories relative to the base path.
        # Since we've fixed async_walk to use await and iterate properly, this should work as expected
        async for root, dirs, files in async_walk(
            validated_path,
            onerror=onerror,
            exclude_patterns=exclude_patterns,
            base_path=validated_path,  # Use original validated path as base for excludes
            followlinks=True,
        ):
            # Check matching directory names
            for dirname in dirs:
                name_to_check = dirname if case_sensitive else dirname.lower()
                if search_pattern in name_to_check:
                    match_path = os.path.join(root, dirname)
                    matched_paths.add(match_path)  # Add to set (handles duplicates)

            # Check matching file names OR content
            content_check_tasks = []
            files_to_check_content = []

            for filename in files:
                name_to_check = filename if case_sensitive else filename.lower()
                match_path = os.path.join(root, filename)
                name_match = search_pattern in name_to_check

                if name_match:
                    # Check if already added via content search in a previous iteration (unlikely but possible)
                    if match_path not in matched_paths:
                        matched_paths.add(match_path)  # Add name match

                # If searching content AND name didn't already match, schedule content check
                # Also check if path isn't already matched from a directory name match higher up.
                if search_content and match_path not in matched_paths:
                    # Avoid queueing files already matched by name (implicitly handled by set)
                    files_to_check_content.append(match_path)
                    # Create task but don't await yet, gather below
                    content_check_tasks.append(check_file_content(match_path))

            # Run content checks concurrently for this directory level
            if content_check_tasks:
                content_results = await asyncio.gather(*content_check_tasks, return_exceptions=True)
                for idx, result in enumerate(content_results):
                    file_path_checked = files_to_check_content[idx]
                    if isinstance(result, Exception):
                        logger.warning(
                            f"Error during content check task for {file_path_checked}: {result}",
                            emoji_key="error",
                        )
                        # Errors during check_file_content are logged there, potentially reported via onerror too.
                    elif result is True:  # Content matched
                        matched_paths.add(file_path_checked)  # Add content match

    except Exception as e:
        # Catch unexpected errors during the async iteration/walk setup itself
        logger.error(
            f"Unexpected error during file search setup or walk in {path}: {e}",
            exc_info=True,
            emoji_key="error",
        )
        raise ToolError(
            f"An unexpected error occurred during search execution: {str(e)}",
            context={"path": path, "pattern": pattern},
        ) from e

    processing_time = time.monotonic() - start_time
    # Convert the set of unique matched paths to a sorted list for consistent output.
    unique_matches = sorted(list(matched_paths))

    logger.success(
        f"Search for '{pattern}' in {path} completed ({len(unique_matches)} unique matches found)",
        emoji_key="search",
        errors_encountered=len(search_errors),
        time=processing_time,
        case_sensitive=case_sensitive,
        search_content=search_content,
    )

    result = {
        "path": validated_path,
        "pattern": pattern,
        "case_sensitive": case_sensitive,
        "search_content": search_content,
        "matches": unique_matches,
        "success": True,
        "message": f"Found {len(unique_matches)} unique matches for '{pattern}' in '{validated_path}'.",
    }
    if search_errors:
        # Add consolidated warning if errors occurred during walk/stat.
        result["warnings"] = search_errors  # Include actual errors reported by onerror
        result["message"] += (
            f" Encountered {len(search_errors)} access or read errors during search."
        )

    return result


@with_tool_metrics
@with_error_handling
async def get_file_info(path: str) -> Dict[str, Any]:
    """Get detailed metadata about a specific file or directory asynchronously.

    Validates the path exists and is allowed. Returns information like size,
    timestamps, type, and permissions. Uses lstat to report info about the path
    itself (including if it's a symlink).

    Args:
        path: Path to the file or directory.

    Returns:
        Dictionary containing detailed file information, marked with success=True.
        If info retrieval fails after validation, raises ToolError.
    """
    start_time = time.monotonic()

    # Validate path (must exist, check_exists=True), but don't resolve symlinks
    # We want to get info about the path as specified by the user
    validation_result = await validate_path(path, check_exists=True, resolve_symlinks=False)

    # Check if this is a symlink
    is_symlink = isinstance(validation_result, dict) and validation_result.get("is_symlink")
    if is_symlink:
        validated_path = validation_result.get("symlink_path")
    else:
        validated_path = validation_result

    # Get file information asynchronously using the helper
    # Don't follow symlinks - we want info about the path itself
    info = await format_file_info(validated_path, follow_symlinks=False)

    # Check if the helper returned an error structure
    if "error" in info:
        # Propagate the error. Since path validation passed, this is likely
        # a transient issue or permission problem reading metadata. Use ToolError.
        raise ToolError(
            f"Failed to get file info for '{validated_path}': {info['error']}",
            context={"path": validated_path},
        )

    # Info retrieval successful
    processing_time = time.monotonic() - start_time
    logger.success(
        f"Got file info for: {path} (resolved: {validated_path})",
        emoji_key="file",
        time=processing_time,
    )

    # Add success flag and descriptive message to the info dictionary
    info["success"] = True
    info["message"] = f"Successfully retrieved info for '{validated_path}'."
    return info


@with_tool_metrics
@with_error_handling
async def list_allowed_directories() -> Dict[str, Any]:
    """List all directories configured as allowed for filesystem access.

    This is primarily an administrative/debugging tool. Reads from the loaded config.

    Returns:
        Dictionary containing the list of allowed base directory paths.
    """
    start_time = time.monotonic()

    # --- Use get_allowed_directories which reads from config ---
    try:
        allowed_dirs = get_allowed_directories()
    except Exception as e:
        # Handle rare errors during config retrieval itself
        raise ToolError(f"Failed to retrieve allowed directories configuration: {e}") from e

    processing_time = time.monotonic() - start_time
    logger.success(
        f"Listed {len(allowed_dirs)} allowed directories", emoji_key="config", time=processing_time
    )

    return {
        "directories": allowed_dirs,
        "count": len(allowed_dirs),
        "success": True,
        "message": f"Retrieved {len(allowed_dirs)} configured allowed directories.",
    }
