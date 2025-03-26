#!/usr/bin/env python
"""Simple completion demo using LLM Gateway's direct provider functionality."""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from decouple import config as decouple_config

from llm_gateway.constants import Provider
from llm_gateway.core.server import Gateway
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.simple_completion")

async def run_model_demo():
    """Run a simple demo using direct provider access."""
    logger.info("Starting simple completion demo", emoji_key="start")
    
    # Create Gateway instance
    gateway = Gateway("simple-demo")
    
    # Initialize providers
    logger.info("Initializing providers", emoji_key="provider")
    await gateway._initialize_providers()
    
    # Get provider (OpenAI)
    provider_name = Provider.OPENAI.value
    provider = gateway.providers.get(provider_name)
    
    if not provider:
        logger.error(f"Provider {provider_name} not available", emoji_key="error")
        return 1
        
    logger.success(f"Provider {provider_name} initialized", emoji_key="success")
    
    # List available models
    models = await provider.list_models()
    logger.info(f"Available models: {len(models)}", emoji_key="model")
    
    # Pick a valid model from the provider
    model = "gpt-4o-mini"  # A valid model from constants.py
    
    # Generate a completion
    prompt = "Explain quantum computing in simple terms."
    
    logger.info(f"Generating completion with {model}", emoji_key="processing")
    result = await provider.generate_completion(
        prompt=prompt,
        model=model,
        temperature=0.7,
        max_tokens=150
    )
    
    # Print the result
    logger.success("Completion generated successfully!", emoji_key="success")
    print("\n" + "-" * 80)
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print("\nCompletion:")
    print(result.text)
    print("-" * 80 + "\n")
    
    # Print stats
    logger.info(
        f"Stats: {result.input_tokens} input tokens, " +
        f"{result.output_tokens} output tokens, " +
        f"${result.cost:.6f} cost, " +
        f"{result.processing_time:.2f}s",
        emoji_key="token"
    )
    
    return 0

async def main():
    """Run the demo."""
    try:
        return await run_model_demo()
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical")
        return 1

if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 