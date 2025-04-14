#!/usr/bin/env python3
"""Main server entry point for LLM Gateway MCP Server."""

from llm_gateway.config import get_config
from llm_gateway.core.server import Gateway
from llm_gateway.utils import get_logger

# Get logger using our fixed get_logger function
logger = get_logger("llm_gateway.server_runner")

def main():
    """Run the LLM Gateway MCP server."""
    config = get_config()
    version = config.server.version
    
    # Log server info with emoji keys for better visual formatting
    logger.info(f"Starting LLM Gateway server v{version}", emoji_key="start")
    logger.info(f"Server name: {config.server.name}", emoji_key="id")
    logger.info(f"Host: {config.server.host}")
    logger.info(f"Port: {config.server.port}")
    logger.info(f"Workers: {config.server.workers}")
    
    # Create and run the server
    server = Gateway()
    
    # Run the MCP server
    server.run()

if __name__ == "__main__":
    main() 