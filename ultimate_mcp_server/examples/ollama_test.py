#!/usr/bin/env python
"""
Test script for the Ollama provider integration in Ultimate MCP Server.

This script demonstrates how to use the Ollama provider for:
1. Getting provider status and available models
2. Generating text completions
3. Streaming text completions
4. Creating embeddings

Prerequisites:
- Ollama must be installed and running locally (https://ollama.com/download)
- At least one model must be pulled (e.g., 'ollama pull llama3.2')
"""

import asyncio
import sys

from rich.console import Console
from rich.table import Table

sys.path.insert(0, ".")  # Add current directory to path

# Import provider-related modules
from ultimate_mcp_server.core.providers.ollama import OllamaProvider
from ultimate_mcp_server.utils import get_logger

# Set up logger and console
logger = get_logger(__name__)
console = Console()


async def test_provider_status():
    """Test getting provider status."""
    provider = OllamaProvider()
    await provider.initialize()
    
    status = provider.get_status()
    
    # Print status in a table
    table = Table(title="Ollama Provider Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Name", status.name)
    table.add_row("Enabled", str(status.enabled))
    table.add_row("Available", str(status.available))
    table.add_row("API Key Configured", str(status.api_key_configured))
    table.add_row("Default Model", status.default_model)
    table.add_row("Supports Streaming", str(status.features.supports_streaming))
    table.add_row("Supports Chat", str(status.features.supports_chat_completions))
    table.add_row("Supports Embeddings", str(status.features.supports_embeddings))
    
    console.print(table)
    
    # Check if Ollama is accessible
    api_key_valid = await provider.check_api_key()
    console.print(f"Ollama service accessible: [{'green' if api_key_valid else 'red'}]{api_key_valid}[/]")
    
    await provider.shutdown()


async def test_list_models():
    """Test listing available models."""
    provider = OllamaProvider()
    await provider.initialize()
    
    try:
        models = await provider.list_models()
        
        # Print models in a table
        table = Table(title="Available Ollama Models")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description", style="yellow")
        
        for model in models:
            table.add_row(model.id, model.name, model.description)
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing models: {str(e)}[/red]")
    
    await provider.shutdown()


async def test_generate_completion():
    """Test generating a completion."""
    provider = OllamaProvider()
    await provider.initialize()
    
    prompt = "Write a haiku about artificial intelligence."
    model = "llama3.2"  # Change to a model you have downloaded
    
    console.print(f"[cyan]Generating completion using model:[/cyan] {model}")
    console.print(f"[cyan]Prompt:[/cyan] {prompt}")
    
    try:
        with console.status("[bold green]Generating...[/bold green]"):
            result = await provider.generate_completion(
                prompt=prompt,
                model=model,
                temperature=0.7
            )
        
        console.print("\n[bold green]Completion:[/bold green]")
        console.print(result.text)
        
        # Print metrics
        console.print("\n[bold]Metrics:[/bold]")
        console.print(f"Model: {result.model}")
        console.print(f"Input tokens: {result.input_tokens}")
        console.print(f"Output tokens: {result.output_tokens}")
        console.print(f"Total tokens: {result.total_tokens}")
        console.print(f"Cost: ${result.cost:.8f}")
        console.print(f"Processing time: {result.processing_time:.2f}s")
    except Exception as e:
        console.print(f"[red]Error generating completion: {str(e)}[/red]")
    
    await provider.shutdown()


async def test_stream_completion():
    """Test streaming a completion."""
    provider = OllamaProvider()
    await provider.initialize()
    
    prompt = "List 5 benefits of using local LLMs like Ollama."
    model = "llama3.2"  # Change to a model you have downloaded
    
    console.print(f"[cyan]Streaming completion using model:[/cyan] {model}")
    console.print(f"[cyan]Prompt:[/cyan] {prompt}")
    console.print("\n[bold green]Streaming Response:[/bold green]")
    
    try:
        metadata = None
        async for chunk, meta in provider.generate_completion_stream(
            prompt=prompt,
            model=model,
            temperature=0.7
        ):
            console.print(chunk, end="", highlight=False)
            metadata = meta
        
        console.print("\n\n[bold]Final Metrics:[/bold]")
        if metadata:
            console.print(f"Model: {metadata.model}")
            console.print(f"Input tokens: {metadata.input_tokens}")
            console.print(f"Output tokens: {metadata.output_tokens}")
            console.print(f"Total tokens: {metadata.total_tokens}")
            console.print(f"Cost: ${metadata.cost:.8f}")
            console.print(f"Processing time: {metadata.processing_time:.2f}s")
    except Exception as e:
        console.print(f"\n[red]Error streaming completion: {str(e)}[/red]")
    
    await provider.shutdown()


async def test_create_embeddings():
    """Test creating embeddings."""
    provider = OllamaProvider()
    await provider.initialize()
    
    texts = [
        "Artificial intelligence is revolutionizing computing.",
        "Language models can process and generate human language.",
        "Running models locally gives you privacy and control."
    ]
    model = "llama3.2"  # Change to a model you have downloaded
    
    console.print(f"[cyan]Creating embeddings using model:[/cyan] {model}")
    console.print(f"[cyan]Number of texts:[/cyan] {len(texts)}")
    
    try:
        with console.status("[bold green]Creating embeddings...[/bold green]"):
            result = await provider.create_embeddings(
                texts=texts,
                model=model
            )
        
        console.print("\n[bold green]Embeddings Created:[/bold green]")
        console.print(f"Number of embeddings: {len(result.embeddings)}")
        console.print(f"Dimensions: {result.dimensions}")
        
        # Print first few values of first embedding
        if result.embeddings:
            first_embedding = result.embeddings[0]
            preview = ", ".join(f"{v:.4f}" for v in first_embedding[:5])
            console.print(f"First embedding (first 5 values): [{preview}, ...]")
        
        # Print metrics
        console.print("\n[bold]Metrics:[/bold]")
        console.print(f"Model: {result.model}")
        console.print(f"Total tokens: {result.total_tokens}")
        console.print(f"Cost: ${result.cost:.8f}")
        console.print(f"Processing time: {result.processing_time:.2f}s")
        
        if result.errors:
            console.print("[yellow]Errors encountered:[/yellow]")
            for error in result.errors:
                console.print(f"- {error}")
    except Exception as e:
        console.print(f"[red]Error creating embeddings: {str(e)}[/red]")
    
    await provider.shutdown()


async def main():
    """Run all tests."""
    console.print("[bold blue]Testing Ollama Provider Integration[/bold blue]")
    console.print("Make sure Ollama is installed and running locally!")
    console.print()
    
    # Test provider status
    console.print("[bold]1. Testing Provider Status[/bold]")
    await test_provider_status()
    console.print()
    
    # Test listing models
    console.print("[bold]2. Testing List Models[/bold]")
    await test_list_models()
    console.print()
    
    # Test generate completion
    console.print("[bold]3. Testing Generate Completion[/bold]")
    await test_generate_completion()
    console.print()
    
    # Test stream completion
    console.print("[bold]4. Testing Stream Completion[/bold]")
    await test_stream_completion()
    console.print()
    
    # Test create embeddings
    console.print("[bold]5. Testing Create Embeddings[/bold]")
    await test_create_embeddings()
    console.print()
    
    console.print("[bold green]All tests completed![/bold green]")


if __name__ == "__main__":
    asyncio.run(main()) 