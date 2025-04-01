"""Utility functions for LLM Gateway."""
from llm_gateway.utils.logging.console import console
from llm_gateway.utils.logging.logger import (
    critical,
    debug,
    error,
    get_logger,
    info,
    logger,
    section,
    success,
    warning,
)
from llm_gateway.utils.parsing import parse_result, process_mcp_result

__all__ = [
    # Logging utilities
    "logger",
    "console",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "critical",
    "section",
    "get_logger",
    
    # Parsing utilities
    "parse_result",
    "process_mcp_result",
]
