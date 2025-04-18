#!/usr/bin/env python3
"""Main server entry point for Ultimate MCP Server."""

import argparse

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.core.server import start_server
from ultimate_mcp_server.utils import get_logger

# Get logger using our fixed get_logger function
logger = get_logger("ultimate.server_runner")

def main():
    """Run the Ultimate MCP Server with CLI flag for transport mode."""
    parser = argparse.ArgumentParser(
        description="Start the Ultimate MCP Server (stdio or SSE mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--transport-mode",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: 'stdio' for CLI/pipe, 'sse' for HTTP SSE server"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (overrides config)"
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level (overrides config)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (overrides config)"
    )
    args = parser.parse_args()

    config = get_config()
    version = config.server.version

    # Log server info with emoji keys for better visual formatting
    logger.info(f"Starting Ultimate MCP Server server v{version}", emoji_key="start")
    logger.info(f"Server name: {config.server.name}", emoji_key="id")
    logger.info(f"Host: {args.host or config.server.host}")
    logger.info(f"Port: {args.port or config.server.port}")
    logger.info(f"Workers: {args.workers or config.server.workers}")
    logger.info(f"Transport mode: {args.transport_mode}")

    # Start the server using the harmonized entrypoint
    start_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        transport_mode=args.transport_mode
    )

if __name__ == "__main__":
    main() 