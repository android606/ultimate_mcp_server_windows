#!/usr/bin/env python3
"""
Standalone CLI for running and interacting with the Ultimate MCP Server.

This module provides a comprehensive command-line interface for the Ultimate MCP
(Model Control Protocol) Server, enabling users to perform various operations including:

- Starting and configuring the server with custom settings
- Testing and benchmarking different LLM providers (OpenAI, Anthropic, etc.)
- Generating text completions directly from the command line
- Managing the completion cache for performance optimization
- Listing available providers and models with their current status

The CLI is designed to be user-friendly with comprehensive help text and follows
a hierarchical command structure similar to tools like git. It can be used both 
for server administration and as a quick interface for testing LLM capabilities
without writing any code.

Usage examples:
    # Start the server on default host/port
    $ python main.py run
    
    # Generate a completion with a specific provider
    $ python main.py complete --provider openai --prompt "Write a haiku about coding"
    
    # List all available providers and check their API key status
    $ python main.py providers --check
    
    # Benchmark multiple providers and models
    $ python main.py benchmark --providers openai anthropic --runs 5

For full documentation on available commands and options, use:
    $ python main.py --help
    $ python main.py <command> --help
"""
import argparse
import asyncio
import os
import sys
from typing import List, Optional

from ultimate_mcp_server import __version__
from ultimate_mcp_server.cli.commands import (
    benchmark_providers,
    check_cache,
    generate_completion,
    list_providers,
    run_server,
    test_provider,
)
from ultimate_mcp_server.utils import get_logger

# Import tool registration functions/modules
# Import Marqo health check


# Use consistent namespace
logger = get_logger("ultimate_mcp_server.main")


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser for the MCP Server CLI.
    
    This function defines the complete command-line interface structure, including
    all available commands, their arguments, and help text. The parser provides
    a hierarchical command structure with the following top-level commands:
    
    - run: Launch the MCP server with configurable host/port/workers
    - providers: List and check status of available LLM providers
    - test: Test a specific provider with a sample prompt
    - complete: Generate text completions using a chosen provider/model
    - cache: Manage the completion cache (status, clearing)
    - benchmark: Run performance benchmarks across multiple providers
    
    Each command has its own set of arguments tailored to its functionality.
    The parser also supports global flags like --version and --debug that
    apply across all commands.
    
    The parser uses argparse's subparser functionality to organize commands
    hierarchically, enabling a git-like command structure (e.g., 
    'ultimate-mcp-server run --host 0.0.0.0').
    
    Returns:
        ArgumentParser: Fully configured argument parser ready for parsing
    
    Example Usage:
        parser = create_parser()
        args = parser.parse_args(['run', '--port', '9000'])
        # args.command would contain 'run'
        # args.port would contain 9000
    """
    parser = argparse.ArgumentParser(
        prog="ultimate-mcp-server",
        description="Ultimate MCP Server - Multi-provider LLM management server",
        epilog="For more information, visit: https://github.com/ultimate-mcp-server/ultimate-mcp-server"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"Ultimate MCP Server {__version__}"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run server command
    server_parser = subparsers.add_parser("run", help="Run the Ultimate MCP Server server")
    server_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from config)"
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on (default: from config)"
    )
    server_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: from config)"
    )
    
    # List providers command
    provider_parser = subparsers.add_parser("providers", help="List available providers")
    provider_parser.add_argument(
        "--check",
        action="store_true",
        help="Check API keys for all providers"
    )
    provider_parser.add_argument(
        "--models",
        action="store_true",
        help="List available models for each provider"
    )
    
    # Test provider command
    test_parser = subparsers.add_parser("test", help="Test a specific provider")
    test_parser.add_argument(
        "provider",
        type=str,
        help="Provider to test (openai, anthropic, deepseek, gemini)"
    )
    test_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to test (default: provider's default model)"
    )
    test_parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, world!",
        help="Test prompt to send to the model"
    )
    
    # Generate completion command
    completion_parser = subparsers.add_parser("complete", help="Generate a completion")
    completion_parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Provider to use (default: openai)"
    )
    completion_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: provider's default model)"
    )
    completion_parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt text (if not provided, read from stdin)"
    )
    completion_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature (default: 0.7)"
    )
    completion_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: provider's default)"
    )
    completion_parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt (for providers that support it)"
    )
    completion_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response"
    )
    
    # Cache command
    cache_parser = subparsers.add_parser("cache", help="Manage the cache")
    cache_parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status"
    )
    cache_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the cache"
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark providers")
    benchmark_parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["openai", "anthropic", "deepseek", "gemini", "openrouter"],
        help="Providers to benchmark (default: all)"
    )
    benchmark_parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to benchmark (default: default models for each provider)"
    )
    benchmark_parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to use for benchmarking (if not provided, built-in prompts will be used)"
    )
    benchmark_parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs for each benchmark (default: 3)"
    )
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the Ultimate MCP Server CLI.
    
    This function parses command-line arguments and executes the appropriate command
    based on user input. The CLI supports multiple commands including:
    
    - run: Start the MCP server
    - providers: List available LLM providers and their status
    - test: Test a specific provider with a sample prompt
    - complete: Generate a completion using a specific provider
    - cache: Manage the completion cache
    - benchmark: Run performance benchmarks across providers
    
    The function handles command execution, including proper error handling for
    user interruptions and unexpected exceptions.
    
    Args:
        args: Command-line arguments to parse. If None, sys.argv[1:] is used.
        
    Returns:
        Exit code (0 for success, 1 for general error, 130 for keyboard interrupt)
        
    Example:
        # Run the server on a custom host and port
        main(["run", "--host", "0.0.0.0", "--port", "8000"])
        
        # Test the OpenAI provider
        main(["test", "openai", "--prompt", "Hello, world!"])
    """
    # Parse arguments
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Set debug mode if requested
    if parsed_args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    # Handle command
    try:
        if parsed_args.command == "run":
            # Run server
            run_server(
                host=parsed_args.host,
                port=parsed_args.port,
                workers=parsed_args.workers
            )
            return 0
            
        elif parsed_args.command == "providers":
            # List providers
            asyncio.run(list_providers(
                check_keys=parsed_args.check,
                list_models=parsed_args.models
            ))
            return 0
            
        elif parsed_args.command == "test":
            # Test provider
            asyncio.run(test_provider(
                provider=parsed_args.provider,
                model=parsed_args.model,
                prompt=parsed_args.prompt
            ))
            return 0
            
        elif parsed_args.command == "complete":
            # Generate completion
            # Get prompt from stdin if not provided
            prompt = parsed_args.prompt
            if prompt is None:
                if sys.stdin.isatty():
                    print("Enter prompt (Ctrl+D to finish):")
                prompt = sys.stdin.read().strip()
            
            asyncio.run(generate_completion(
                provider=parsed_args.provider,
                model=parsed_args.model,
                prompt=prompt,
                temperature=parsed_args.temperature,
                max_tokens=parsed_args.max_tokens,
                system=parsed_args.system,
                stream=parsed_args.stream
            ))
            return 0
            
        elif parsed_args.command == "cache":
            # Cache management
            asyncio.run(check_cache(
                show_status=parsed_args.status,
                clear=parsed_args.clear
            ))
            return 0
            
        elif parsed_args.command == "benchmark":
            # Benchmark providers
            asyncio.run(benchmark_providers(
                providers=parsed_args.providers,
                models=parsed_args.models,
                prompt=parsed_args.prompt,
                runs=parsed_args.runs
            ))
            return 0
            
        else:
            # No command or unrecognized command
            parser.print_help()
            return 0
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user", emoji_key="info")
        return 130
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", emoji_key="error", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())