#!/usr/bin/env python3
"""Main server entry point for Ultimate MCP Server."""

import argparse

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.core.server import start_server
from ultimate_mcp_server.utils import get_logger

# Get logger using our fixed get_logger function
logger = get_logger("ultimate_mcp_server.server_runner")

def main():
    """
    Run the Ultimate MCP Server with configurable transport mode and server settings.
    
    This function is the main entry point for starting the Ultimate MCP Server when
    running it directly as a standalone application. It provides a command-line interface
    for configuring key server parameters, including:
    
    - Transport mode: Choose between stdio communication for CLI/pipe usage (default) or SSE 
      over HTTP for browser/UI integration
    - Server host/port: Configure the network binding for SSE mode
    - Worker count: Control parallelism for request handling
    - Logging level: Adjust verbosity for debugging or quieter operation
    
    The function performs the following steps:
    1. Parse command-line arguments with sensible defaults
    2. Load and apply configuration from files and environment variables
    3. Log server startup information with key settings
    4. Start the server using the appropriate transport mode
    
    Command-line arguments take precedence over configuration file settings,
    allowing for quick overrides without modifying configuration files.
    
    When called without arguments, the server starts in stdio mode (the default)
    with settings from the configuration file or environment variables.
    
    Examples:
        # Start in stdio mode (default)
        $ python -m ultimate_mcp_server.server
        
        # Start in SSE mode on specific host/port
        $ python -m ultimate_mcp_server.server --transport-mode sse --host 0.0.0.0 --port 8080
        
        # Start with verbose logging
        $ python -m ultimate_mcp_server.server --log-level debug
    """
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