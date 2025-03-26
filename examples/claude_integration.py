#!/usr/bin/env python
"""Claude integration example for LLM Gateway."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.claude_integration")


async def compare_claude_models():
    """Compare different Claude models."""
    logger.info("Starting Claude models comparison", emoji_key="start")
    
    # Get API key directly from decouple
    from decouple import config as decouple_config
    api_key = decouple_config('ANTHROPIC_API_KEY', default=None)
    
    # Get Claude provider with API key
    provider_name = Provider.ANTHROPIC.value
    try:
        provider = get_provider(
            provider_name,
            api_key=api_key
        )
        await provider.initialize()
        
        logger.info(
            f"Using provider: {provider_name}",
            emoji_key="provider",
        )
        
        # Get available Claude models
        models = await provider.list_models()
        logger.info(f"Found {len(models)} Claude models", emoji_key="model")
        
        # Select models to compare
        # In real usage, you might want to check if these models are available
        claude_models = [
            "claude-3-5-haiku-latest",
            "claude-3-7-sonnet-latest", 
            "claude-3-5-sonnet-latest"
        ]
        
        # Define a consistent prompt for comparison
        prompt = """
        Explain the concept of quantum entanglement in a way that a high school student would understand.
        Keep your response brief and accessible.
        """
        
        # Compare models
        results = []
        
        for model_name in claude_models:
            try:
                logger.info(f"Testing model: {model_name}", emoji_key="model")
                
                # Generate completion
                start_time = time.time()
                result = await provider.generate_completion(
                    prompt=prompt,
                    model=model_name,
                    temperature=0.3,
                    max_tokens=300
                )
                processing_time = time.time() - start_time
                
                # Store result
                results.append({
                    "model": model_name,
                    "text": result.text,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens
                    },
                    "cost": result.cost,
                    "time": processing_time
                })
                
                # Log completion
                logger.success(
                    f"Completion for {model_name} successful",
                    emoji_key="success",
                    tokens={
                        "input": result.input_tokens,
                        "output": result.output_tokens
                    },
                    cost=result.cost,
                    time=processing_time
                )
                
            except Exception as e:
                logger.error(
                    f"Error testing model {model_name}: {str(e)}",
                    emoji_key="error"
                )
        
        # Display comparison results
        if results:
            logger.info("Model comparison results:", emoji_key="info")
            print("\n" + "=" * 80)
            
            for result in results:
                model = result["model"]
                time_ms = result["time"] * 1000
                tokens = result["tokens"]["total"]
                tokens_per_second = tokens / result["time"] if result["time"] > 0 else 0
                
                print(f"\n## {model}")
                print(f"Time: {time_ms:.0f}ms | Tokens: {tokens} | " +
                      f"Speed: {tokens_per_second:.1f} tokens/sec | " +
                      f"Cost: ${result['cost']:.6f}")
                print("-" * 80)
                print(result["text"].strip())
                print("-" * 80)
                
            print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}", emoji_key="error")
        raise


async def demonstrate_system_prompt():
    """Demonstrate Claude with system prompts."""
    logger.info("Demonstrating Claude with system prompts", emoji_key="start")
    
    # Get API key directly from decouple
    from decouple import config as decouple_config
    api_key = decouple_config('ANTHROPIC_API_KEY', default=None)
    
    # Get Claude provider with API key
    provider_name = Provider.ANTHROPIC.value
    try:
        provider = get_provider(
            provider_name,
            api_key=api_key
        )
        await provider.initialize()
        
        # Use a fast Claude model
        model = "claude-3-haiku-20240307"
        
        # Define system prompt and user prompt
        system_prompt = """
        You are a helpful assistant with expertise in physics.
        Keep all explanations accurate but very concise.
        Always provide real-world examples to illustrate concepts.
        """
        
        user_prompt = "Explain the concept of gravity."
        
        logger.info(
            "Generating completion with system prompt",
            emoji_key="processing"
        )
        
        # Generate completion with system prompt
        result = await provider.generate_completion(
            prompt=user_prompt,
            model=model,
            temperature=0.7,
            system=system_prompt,
            max_tokens=1000
        )
        
        # Log success
        logger.success(
            "Completion with system prompt successful",
            emoji_key="success",
            tokens={
                "input": result.input_tokens,
                "output": result.output_tokens
            },
            cost=result.cost,
            time=result.processing_time
        )
        
        # Display result
        print("\n" + "=" * 80)
        print("SYSTEM PROMPT:")
        print(system_prompt.strip())
        print("\nUSER PROMPT:")
        print(user_prompt.strip())
        print("\nRESPONSE:")
        print(result.text.strip())
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Error in system prompt demonstration: {str(e)}", emoji_key="error")
        raise


async def main():
    """Run Claude integration examples."""
    try:
        # Get API key directly from decouple
        from decouple import config as decouple_config
        api_key = decouple_config('ANTHROPIC_API_KEY', default=None)
        
        if not api_key:
            logger.warning(
                "No Anthropic API key found in .env file. " +
                "Set ANTHROPIC_API_KEY in your .env file to run this example.",
                emoji_key="warning"
            )
            return 1
            
        # Run model comparison
        await compare_claude_models()
        
        print("\n")
        
        # Run system prompt demonstration
        await demonstrate_system_prompt()
        
    except Exception as e:
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)