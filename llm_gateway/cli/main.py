"""Command-line interface for LLM Gateway."""
import argparse
import asyncio
import sys
from typing import List, Optional

from llm_gateway import __version__
from llm_gateway.cli.commands import (
    benchmark_providers,
    check_cache,
    generate_completion,
    list_providers,
    list_tools,
    run_server,
    test_provider,
)
from llm_gateway.utils import get_logger

# Use the consistent namespace approach
logger = get_logger("llm_gateway.cli")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser.
    
    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="llm-gateway",
        description="LLM Gateway - Multi-provider LLM management server",
        epilog="For more information, visit: https://github.com/llm-gateway/llm-gateway"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"LLM Gateway {__version__}"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run server command
    server_parser = subparsers.add_parser("run", help="Run the LLM Gateway server")
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
    server_parser.add_argument(
        "--transport-mode",
        type=str,
        choices=["sse", "stdio"],
        default="stdio",
        help="Transport mode to use: 'sse' for HTTP server or 'stdio' for Claude Desktop (default: stdio)"
    )
    # Add new tool filtering options
    server_parser.add_argument(
        "--include-tools",
        type=str,
        nargs="+",
        help="Only register the specified tools (default: register all tools)"
    )
    server_parser.add_argument(
        "--exclude-tools",
        type=str,
        nargs="+",
        help="Exclude specified tools from registration (takes precedence over --include-tools)"
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
    
    # Add a new command to list available tools
    tools_parser = subparsers.add_parser("tools", help="List available tools")
    tools_parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter tools by category (e.g., 'filesystem', 'completion')"
    )
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Command-line arguments (if None, sys.argv[1:] is used)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Set debug mode if requested - but don't try to set config.logging.level
    debug_mode = parsed_args.debug
    
    # Handle command
    try:
        if parsed_args.command == "run":
            # Run server with log level from debug flag
            log_level = "DEBUG" if debug_mode else "INFO"
            
            run_server(
                host=parsed_args.host,
                port=parsed_args.port,
                workers=parsed_args.workers,
                log_level=log_level,
                transport_mode=parsed_args.transport_mode,
                include_tools=parsed_args.include_tools,
                exclude_tools=parsed_args.exclude_tools
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
                    sys.stderr.write("Enter prompt (Ctrl+D to finish):\n")
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
            
        elif parsed_args.command == "tools":
            # List available tools
            asyncio.run(list_tools(
                category=parsed_args.category
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