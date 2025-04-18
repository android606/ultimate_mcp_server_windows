"""Command-line interface for Ultimate MCP Server."""
# Instead of importing main directly, we'll let it be imported as needed
# This avoids the circular import issue

__all__ = ["main"]

# Delayed import to avoid circular reference
def main(args=None):
    from ultimate_mcp_server.cli.main import main as _main
    return _main(args)