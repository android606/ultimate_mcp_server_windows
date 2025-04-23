#!/usr/bin/env python
"""Comprehensive diagnostic tool for Ollama integration with Ultimate MCP Server."""
import asyncio
import os
import sys
import traceback
from pathlib import Path
import inspect
import json

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Rich for pretty printing
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)

console = Console()

# Try to import key components
console.print("[bold blue]Importing key components...[/bold blue]")
try:
    from ultimate_mcp_server.constants import Provider
    from ultimate_mcp_server.core.server import Gateway
    from ultimate_mcp_server.config import get_config, load_config
    from ultimate_mcp_server.utils import get_logger
    from ultimate_mcp_server.core.providers.ollama import OllamaProvider, OllamaConfig
    
    console.print("[green]Imports successful[/green]")
except Exception as e:
    console.print("[bold red]Error during imports:[/bold red]")
    console.print_exception()
    sys.exit(1)

# Set up logger
logger = get_logger("diagnostic.ollama")


def print_dict_as_table(data, title):
    """Print a dictionary as a rich table."""
    if data is None:
        console.print(f"[yellow]{title}: None[/yellow]")
        return
        
    table = Table(title=title)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="yellow")
    
    if isinstance(data, dict):
        for key, value in data.items():
            table.add_row(str(key), str(value))
    else:
        # Try to convert to dict if possible
        try:
            if hasattr(data, '__dict__'):
                for key, value in data.__dict__.items():
                    table.add_row(str(key), str(value))
            elif hasattr(data, 'dict') and callable(data.dict):
                for key, value in data.dict().items():
                    table.add_row(str(key), str(value))
            else:
                # Just show string representation
                table.add_row("object", str(data))
        except Exception as e:
            table.add_row("Error", f"Couldn't convert to table: {str(e)}")
    
    console.print(table)


async def check_direct_connection():
    """Test direct connection to Ollama API."""
    console.print("\n[bold blue]STAGE 1: Testing direct connection to Ollama API[/bold blue]")
    
    try:
        import aiohttp
        urls = ["http://localhost:11434", "http://127.0.0.1:11434"]
        
        for url in urls:
            console.print(f"\nTesting URL: [cyan]{url}[/cyan]")
            try:
                async with aiohttp.ClientSession() as session:
                    console.print(f"  Connecting to {url}/api/tags...")
                    async with session.get(f"{url}/api/tags", timeout=5.0) as response:
                        status = response.status
                        console.print(f"  Status code: [cyan]{status}[/cyan]")
                        
                        if status == 200:
                            data = await response.json()
                            models_count = len(data.get("models", []))
                            console.print(f"  [green]Success! Found {models_count} models.[/green]")
                            if models_count > 0:
                                console.print("  Models:")
                                for model in data.get("models", []):
                                    console.print(f"    - {model.get('name')}")
                        else:
                            text = await response.text()
                            console.print(f"  [red]Error: {text[:200]}[/red]")
            except Exception as e:
                console.print(f"  [red]Connection error: {type(e).__name__} - {str(e)}[/red]")
    except Exception as e:
        console.print(f"[bold red]Error testing direct connection:[/bold red]")
        console.print_exception()


async def test_ollama_provider_directly():
    """Test OllamaProvider class directly."""
    console.print("\n[bold blue]STAGE 2: Testing OllamaProvider class directly[/bold blue]")
    
    try:
        # Print module info
        console.print("\nModule info:")
        console.print(f"  Module: {inspect.getmodule(OllamaProvider)}")
        console.print(f"  File: {inspect.getfile(OllamaProvider)}")
        
        # Create provider instance
        console.print("\nCreating OllamaProvider instance...")
        provider = OllamaProvider()
        
        # Print config
        console.print("\nProvider configuration:")
        print_dict_as_table(provider.config, "OllamaConfig")
        
        # Attempt initialization
        console.print("\nAttempting to initialize provider...")
        initialized = await provider.initialize()
        console.print(f"Initialization result: [{'green' if initialized else 'red'}]{initialized}[/{'green' if initialized else 'red'}]")
        
        if initialized:
            # Try to list models
            console.print("\nListing models...")
            try:
                models = await provider.list_models()
                console.print(f"[green]Found {len(models)} models[/green]")
                for model in models:
                    console.print(f"  - {model['id']}")
            except Exception as e:
                console.print("[red]Error listing models:[/red]")
                console.print_exception()
                
            # Make a simple completion
            console.print("\nTesting a simple completion...")
            try:
                result = await provider.generate_completion(
                    prompt="Hello, world!",
                    model="llama3.2-vision" if "llama3.2-vision" in [m["id"] for m in models] else models[0]["id"],
                    max_tokens=20
                )
                console.print(f"Completion result: {result.text}")
                console.print(f"Tokens: input={result.input_tokens}, output={result.output_tokens}")
            except Exception as e:
                console.print("[red]Error generating completion:[/red]")
                console.print_exception()
        
        # Clean up
        await provider.shutdown()
            
    except Exception as e:
        console.print(f"[bold red]Error testing OllamaProvider:[/bold red]")
        console.print_exception()


async def inspect_config():
    """Inspect configuration values."""
    console.print("\n[bold blue]STAGE 3: Inspecting configuration[/bold blue]")
    
    try:
        # Ensure config is loaded
        load_config()
        config = get_config()
        
        console.print("\nDumping providers configuration:")
        if hasattr(config, 'providers'):
            providers_config = config.providers
            console.print(f"providers attribute exists: {providers_config}")
            
            # Check if Ollama is in providers
            if hasattr(providers_config, Provider.OLLAMA.value):
                ollama_config = getattr(providers_config, Provider.OLLAMA.value)
                console.print(f"[green]Ollama configuration found[/green]")
                print_dict_as_table(ollama_config, "Ollama Config")
            else:
                console.print(f"[yellow]No Ollama provider configuration found in config.providers[/yellow]")
                
            # List all providers
            console.print("\nAll configured providers:")
            for provider in dir(providers_config):
                if not provider.startswith('_'):
                    console.print(f"  - {provider}")
        else:
            console.print("[yellow]No 'providers' attribute in config[/yellow]")
            
    except Exception as e:
        console.print(f"[bold red]Error inspecting configuration:[/bold red]")
        console.print_exception()


async def test_gateway_initialization():
    """Test Gateway initialization with Ollama provider."""
    console.print("\n[bold blue]STAGE 4: Testing Gateway initialization[/bold blue]")
    
    try:
        # Create Gateway instance
        console.print("\nCreating Gateway instance...")
        gateway = Gateway("diagnostic-gateway", register_tools=False)
        
        # Check provider configuration before initialization
        console.print("\nProvider configuration before initialization:")
        if hasattr(gateway, 'provider_exclusions'):
            console.print(f"provider_exclusions: {gateway.provider_exclusions}")
        
        # Initialize providers
        console.print("\nInitializing providers...")
        await gateway._initialize_providers()
        
        # Check provider status after initialization
        console.print("\nProvider status after initialization:")
        
        if Provider.OLLAMA.value in gateway.provider_status:
            status = gateway.provider_status[Provider.OLLAMA.value]
            console.print(f"[green]Ollama provider status found[/green]")
            try:
                console.print(f"  enabled: {status.enabled}")
                console.print(f"  available: {status.available}")
                console.print(f"  api_key_configured: {status.api_key_configured}")
                console.print(f"  models: {len(status.models)} models")
                if hasattr(status, 'error') and status.error:
                    console.print(f"  [red]error: {status.error}[/red]")
            except Exception as e:
                console.print(f"[red]Error accessing status properties: {str(e)}[/red]")
                console.print_exception()
        else:
            console.print(f"[yellow]No Ollama provider status found in gateway.provider_status[/yellow]")
            
        # Check provider instance after initialization
        console.print("\nProvider instance after initialization:")
        if Provider.OLLAMA.value in gateway.providers:
            provider = gateway.providers[Provider.OLLAMA.value]
            console.print(f"[green]Ollama provider instance found[/green]")
            console.print(f"  Type: {type(provider)}")
            console.print(f"  Initialized: {provider._initialized if hasattr(provider, '_initialized') else 'Unknown'}")
        else:
            console.print(f"[yellow]No Ollama provider instance found in gateway.providers[/yellow]")
            
        # List all provider statuses
        console.print("\nAll provider statuses:")
        for name, status in gateway.provider_status.items():
            console.print(f"  - {name}: enabled={status.enabled}, available={status.available}")
            
    except Exception as e:
        console.print(f"[bold red]Error testing Gateway initialization:[/bold red]")
        console.print_exception()


async def inspect_gateway_code():
    """Inspect Gateway code to understand initialization flow."""
    console.print("\n[bold blue]STAGE 5: Inspecting Gateway initialization flow[/bold blue]")
    
    try:
        # Get the source file and line number for key methods
        gateway_file = inspect.getfile(Gateway)
        console.print(f"Gateway class defined in: {gateway_file}")
        
        # Find the _initialize_providers method
        init_providers_source = inspect.getsource(Gateway._initialize_providers)
        console.print("\nKey parts of Gateway._initialize_providers method:")
        
        # Extract and print the Ollama-specific handling
        for line in init_providers_source.splitlines():
            if "Ollama" in line or "Provider.OLLAMA" in line:
                console.print(f"  [cyan]{line.strip()}[/cyan]")
            
        # Find the _initialize_provider method
        init_provider_source = inspect.getsource(Gateway._initialize_provider)
        console.print("\nKey parts of Gateway._initialize_provider method:")
        
        # Extract and print the Ollama-specific handling
        for line in init_provider_source.splitlines():
            if "Ollama" in line or "Provider.OLLAMA" in line:
                console.print(f"  [cyan]{line.strip()}[/cyan]")
            
    except Exception as e:
        console.print(f"[bold red]Error inspecting Gateway code:[/bold red]")
        console.print_exception()


async def check_environment():
    """Check environment variables and other factors."""
    console.print("\n[bold blue]STAGE 6: Checking environment[/bold blue]")
    
    try:
        # Check config file locations
        console.print("\nChecking config file locations:")
        home_config = Path.home() / ".config" / "umcp" / "config.yaml"
        console.print(f"Home config path: {home_config}")
        console.print(f"  Exists: {home_config.exists()}")
        
        local_config = Path.cwd() / "config.yaml"
        console.print(f"Local config path: {local_config}")
        console.print(f"  Exists: {local_config.exists()}")
        
        # Check environment variables
        console.print("\nEnvironment variables:")
        for key, value in os.environ.items():
            if "OLLAMA" in key.upper() or "MCP" in key.upper() or "UMCP" in key.upper():
                console.print(f"  {key}: {value}")
                
    except Exception as e:
        console.print(f"[bold red]Error checking environment:[/bold red]")
        console.print_exception()


async def show_example_config():
    """Show example configuration that should work."""
    console.print("\n[bold blue]STAGE 7: Example working configuration[/bold blue]")
    
    example_config = {
        "providers": {
            "ollama": {
                "enabled": True,
                "api_url": "http://127.0.0.1:11434",
                "default_model": "llama3.2-vision",
                "request_timeout": 300
            }
        }
    }
    
    console.print("\nExample configuration to put in ~/.config/umcp/config.yaml or config.yaml in project root:")
    console.print(Panel(json.dumps(example_config, indent=2), title="Example Config", border_style="green"))
    
    console.print("\nTo create this config file:")
    console.print("```bash")
    console.print("mkdir -p ~/.config/umcp/")
    console.print("cat > ~/.config/umcp/config.yaml << EOF")
    console.print("providers:")
    console.print("  ollama:")
    console.print("    enabled: true")
    console.print("    api_url: \"http://127.0.0.1:11434\"")
    console.print("    default_model: \"llama3.2-vision\"")
    console.print("    request_timeout: 300")
    console.print("EOF")
    console.print("```")


async def main():
    """Run all diagnostic tests."""
    console.print(Panel.fit(
        "[bold]Ultimate MCP Server Ollama Integration Diagnostics[/bold]\n\n"
        "This tool will help diagnose issues with Ollama integration in Ultimate MCP Server.",
        title="Ollama Diagnostics",
        border_style="blue"
    ))
    
    # Run all diagnostic tests
    await check_direct_connection()
    await test_ollama_provider_directly()
    await inspect_config()
    await test_gateway_initialization()
    await inspect_gateway_code()
    await check_environment()
    await show_example_config()
    
    console.print("\n[bold green]Diagnostics complete![/bold green]")
    

if __name__ == "__main__":
    asyncio.run(main()) 