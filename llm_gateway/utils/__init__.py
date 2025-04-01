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

__all__ = [
    "logger",
    "console",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "critical",
    "section"
]
