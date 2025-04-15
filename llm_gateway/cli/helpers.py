"""Helper functions for the LLM Gateway CLI."""
import json
import sys
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from llm_gateway.config import get_env
from llm_gateway.constants import COST_PER_MILLION_TOKENS, Provider
from llm_gateway.utils import get_logger

logger = get_logger(__name__)
console = Console(file=sys.stderr)


def print_cost_table() -> None:
    """Print a table of model costs."""
    # Create table
    table = Table(title="Model Cost Per Million Tokens")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="blue")
    table.add_column("Input ($/M)", style="green")
    table.add_column("Output ($/M)", style="yellow")
    
    # Group models by provider
    models_by_provider = {}
    for model, costs in COST_PER_MILLION_TOKENS.items():
        # Determine provider
        provider = None
        if "gpt" in model:
            provider = Provider.OPENAI.value
        elif "claude" in model:
            provider = Provider.ANTHROPIC.value
        elif "deepseek" in model:
            provider = Provider.DEEPSEEK.value
        elif "gemini" in model:
            provider = Provider.GEMINI.value
        else:
            provider = "other"
        
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        
        models_by_provider[provider].append((model, costs))
    
    # Add rows for each provider's models
    for provider in sorted(models_by_provider.keys()):
        models = sorted(models_by_provider[provider], key=lambda x: x[0])
        
        for model, costs in models:
            table.add_row(
                provider,
                model,
                f"${costs['input']:.3f}",
                f"${costs['output']:.3f}"
            )
    
    # Print table
    console.print(table)


def format_tokens(tokens: int) -> str:
    """Format token count with thousands separator.
    
    Args:
        tokens: Token count
        
    Returns:
        Formatted token count
    """
    return f"{tokens:,}"


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration
    """
    if seconds < 0.1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def save_output_to_file(text: str, file_path: str, mode: str = "w") -> bool:
    """Save output text to a file.
    
    Args:
        text: Text to save
        file_path: Path to save to
        mode: File mode (w for write, a for append)
        
    Returns:
        True if successful
    """
    try:
        with open(file_path, mode, encoding="utf-8") as f:
            f.write(text)
        
        console.print(f"[green]Output saved to {file_path}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error saving output: {str(e)}[/red]")
        return False


def load_file_content(file_path: str) -> Optional[str]:
    """Load content from a file.
    
    Args:
        file_path: Path to load from
        
    Returns:
        File content or None if error
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        console.print(f"[red]Error loading file: {str(e)}[/red]")
        return None


def print_markdown(markdown_text: str) -> None:
    """Print formatted Markdown.
    
    Args:
        markdown_text: Markdown text to print
    """
    md = Markdown(markdown_text)
    console.print(md)


def print_json(json_data: Union[Dict, List]) -> None:
    """Print formatted JSON.
    
    Args:
        json_data: JSON data to print
    """
    json_str = json.dumps(json_data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", word_wrap=True)
    console.print(syntax)


def print_code(code: str, language: str = "python") -> None:
    """Print formatted code.
    
    Args:
        code: Code to print
        language: Programming language
    """
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)


def print_model_comparison(
    provider: str,
    models: List[str],
    metrics: List[Dict[str, Any]]
) -> None:
    """Print a comparison of models for a provider.
    
    Args:
        provider: Provider name
        models: List of model names
        metrics: List of metrics dictionaries
    """
    # Create table
    table = Table(title=f"{provider.capitalize()} Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Response Time", style="green")
    table.add_column("Tokens/Sec", style="yellow")
    table.add_column("Cost", style="magenta")
    table.add_column("Total Tokens", style="dim")
    
    # Add rows for each model
    for model, metric in zip(models, metrics, strict=False):
        table.add_row(
            model,
            format_duration(metric.get("time", 0)),
            f"{metric.get('tokens_per_second', 0):.1f}",
            f"${metric.get('cost', 0):.6f}",
            format_tokens(metric.get("total_tokens", 0))
        )
    
    # Print table
    console.print(table)


def print_environment_info() -> None:
    """Print information about the current environment."""
    # Create table
    table = Table(title="Environment Information")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # Add API key info
    for provider in [p.value for p in Provider]:
        env_var = f"{provider.upper()}_API_KEY"
        has_key = bool(get_env(env_var))
        table.add_row(env_var, "✅ Set" if has_key else "❌ Not set")
    
    # Add other environment variables
    for var in ["LOG_LEVEL", "CACHE_ENABLED", "CACHE_DIR"]:
        value = get_env(var, "Not set")
        table.add_row(var, value)
    
    # Print table
    console.print(table)


def print_examples() -> None:
    """Print examples of using the CLI."""
    examples = """
# Run the server
llm-gateway run --host 0.0.0.0 --port 8013

# List available providers
llm-gateway providers --check

# Test a provider
llm-gateway test openai --model gpt-4.1-mini --prompt "Hello, world!"

# Generate a completion
llm-gateway complete --provider anthropic --model claude-3-5-haiku-20241022 --prompt "Explain quantum computing"

# Stream a completion
llm-gateway complete --provider openai --stream --prompt "Write a poem about AI"

# Run benchmarks
llm-gateway benchmark --providers openai anthropic --runs 3

# Check cache status
llm-gateway cache --status

# Clear cache
llm-gateway cache --clear
"""
    
    syntax = Syntax(examples, "bash", theme="monokai", word_wrap=True)
    console.print(Panel(syntax, title="CLI Examples", border_style="cyan"))


def confirm_action(message: str, default: bool = False) -> bool:
    """Prompt the user to confirm an action.
    
    Args:
        message: Confirmation message
        default: Default response if user just presses Enter
        
    Returns:
        True if confirmed, False if canceled
    """
    default_str = "Y/n" if default else "y/N"
    response = input(f"{message} [{default_str}]: ")
    
    if not response:
        return default
    
    return response.lower() in ["y", "yes"]