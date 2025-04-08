"""Command-line interface for LLM Gateway."""
# Instead of importing main directly, we'll let it be imported as needed
# This avoids the circular import issue

__all__ = ["main"]

# Delayed import to avoid circular reference
def main(args=None):
    from llm_gateway.cli.main import main as _main
    return _main(args)