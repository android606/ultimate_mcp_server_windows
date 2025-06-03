"""Ultimate MCP Server: Multi-provider LLM Gateway with Tools."""
import sys
import warnings

# Check Python version when the package is imported
REQUIRED_VERSION = (3, 13)
if sys.version_info[:2] < REQUIRED_VERSION:
    warnings.warn(
        f"Ultimate MCP Server requires Python {REQUIRED_VERSION[0]}.{REQUIRED_VERSION[1]}+. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}, "
        f"which may cause compatibility issues.",
        RuntimeWarning, stacklevel=2
    )

# Package metadata and version
__version__ = "1.0.0"  # Update as needed