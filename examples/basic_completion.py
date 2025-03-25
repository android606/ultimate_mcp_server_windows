#!/usr/bin/env python
"""Basic completion example using LLM Gateway."""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.basic_completion")


async def run_basic_completion():
    """Run a basic completion example."""
    logger.info("Starting basic completion example", emoji_key="start")

    # Prompt to complete
    prompt = "Explain the concept of federated learning in simple terms."
    
    # Get OpenAI provider (defaulting to environment variable)
    provider_name = Provider.OPENAI.value
    try:
        provider = get_provider(provider_name)
        await provider.initialize()
        
        logger.info(
            f"Using provider: {provider_name}",
            emoji_key="provider",
        )
        
        # Get default model
        model = provider.get_default_model()
        logger.info(f"Using model: {model}", emoji_key="model")
        
        # Generate completion
        logger.info("Generating completion...", emoji_key="processing")
        result = await provider.generate_completion(
            prompt=prompt,
            model=model,
            temperature=0.7,
            max_tokens=200
        )
        
        # Log success with stats
        logger.success(
            "Completion generated successfully!",
            emoji_key="success",
            tokens={
                "input": result.input_tokens,
                "output": result.output_tokens
            },
            cost=result.cost,
            time=result.processing_time
        )
        
        # Print the completion
        logger.info("Generated text:", emoji_key="info")
        print("\n" + "-" * 80)
        print(result.text.strip())
        print("-" * 80 + "\n")
        
        # Print stats
        logger.info(
            f"Stats: {result.input_tokens} input tokens, " +
            f"{result.output_tokens} output tokens, " +
            f"${result.cost:.6f} cost",
            emoji_key="token"
        )
        
    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}", emoji_key="error")
        raise


async def run_streaming_completion():
    """Run a streaming completion example."""
    logger.info("Starting streaming completion example", emoji_key="start")

    # Prompt to complete
    prompt = "Write a short poem about artificial intelligence."
    
    # Get OpenAI provider (defaulting to environment variable)
    provider_name = Provider.OPENAI.value
    try:
        provider = get_provider(provider_name)
        await provider.initialize()
        
        logger.info(
            f"Using provider: {provider_name} in streaming mode",
            emoji_key="provider",
        )
        
        # Get default model
        model = provider.get_default_model()
        
        # Generate streaming completion
        logger.info("Generating streaming completion...", emoji_key="processing")
        
        print("\n" + "-" * 80)
        
        # Start timer
        import time
        start_time = time.time()
        
        # Process the stream
        stream = provider.generate_completion_stream(
            prompt=prompt,
            model=model,
            temperature=0.7,
            max_tokens=200
        )
        
        full_text = ""
        token_count = 0
        
        async for chunk, metadata in stream:
            print(chunk, end="", flush=True)
            full_text += chunk
            token_count += 1
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        print("\n" + "-" * 80 + "\n")
        
        # Log success with stats
        logger.success(
            "Streaming completion generated successfully!",
            emoji_key="success",
            chunks=token_count,
            time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error generating streaming completion: {str(e)}", emoji_key="error")
        raise


async def main():
    """Run completion examples."""
    try:
        # Run basic completion
        await run_basic_completion()
        
        print("\n")
        
        # Run streaming completion
        await run_streaming_completion()
        
    except Exception as e:
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)