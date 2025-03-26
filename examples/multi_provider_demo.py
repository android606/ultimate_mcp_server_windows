#!/usr/bin/env python
"""Multi-provider completion demo using LLM Gateway."""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))


from llm_gateway.constants import Provider
from llm_gateway.core.server import Gateway
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.multi_provider")

async def run_provider_comparison():
    """Run a comparison of completions across multiple providers."""
    logger.info("Starting multi-provider comparison demo", emoji_key="start")
    
    # Create Gateway instance
    gateway = Gateway("multi-provider-demo")
    
    # Initialize providers
    logger.info("Initializing providers", emoji_key="provider")
    await gateway._initialize_providers()
    
    # Common prompt for all providers
    prompt = "Explain the advantages of quantum computing in 3-4 sentences."
    
    # Provider and model configurations to test
    configs = [
        {"provider": Provider.OPENAI.value, "model": "gpt-4o-mini"},
        {"provider": Provider.ANTHROPIC.value, "model": "claude-3-5-haiku-latest"},
        {"provider": Provider.GEMINI.value, "model": "gemini-2.0-flash-lite"},
        {"provider": Provider.DEEPSEEK.value, "model": "deepseek-chat"}
    ]
    
    results = []
    
    # Run completions for each provider
    for config in configs:
        provider_name = config["provider"]
        model_name = config["model"]
        
        provider = gateway.providers.get(provider_name)
        if not provider:
            logger.warning(f"Provider {provider_name} not available, skipping", emoji_key="warning")
            continue
            
        try:
            logger.info(f"Generating completion with {provider_name}/{model_name}", emoji_key="processing")
            
            # Generate completion
            result = await provider.generate_completion(
                prompt=prompt,
                model=model_name,
                temperature=0.7,
                max_tokens=150
            )
            
            # Store result
            results.append({
                "provider": provider_name,
                "model": model_name,
                "text": result.text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost": result.cost,
                "processing_time": result.processing_time
            })
            
            logger.success(
                f"Completion from {provider_name}/{model_name} generated successfully!",
                emoji_key="success"
            )
            
        except Exception as e:
            logger.error(f"Error with {provider_name}/{model_name}: {str(e)}", emoji_key="error")
    
    # Print comparison results
    print("\n" + "=" * 80)
    print("PROVIDER COMPARISON RESULTS")
    print("=" * 80)
    
    for result in results:
        print("\n" + "-" * 80)
        print(f"Provider: {result['provider']}")
        print(f"Model: {result['model']}")
        print(f"Cost: ${result['cost']:.6f}")
        print(f"Time: {result['processing_time']:.2f}s")
        print(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
        print("\nResponse:")
        print(result['text'].strip())
        print("-" * 80)
    
    # Comparison summary
    if results:
        fastest = min(results, key=lambda r: r['processing_time'])
        cheapest = min(results, key=lambda r: r['cost'])
        most_tokens = max(results, key=lambda r: r['output_tokens'])
        
        print("\nSummary:")
        print(f"Fastest: {fastest['provider']}/{fastest['model']} ({fastest['processing_time']:.2f}s)")
        print(f"Cheapest: {cheapest['provider']}/{cheapest['model']} (${cheapest['cost']:.6f})")
        print(f"Most detailed: {most_tokens['provider']}/{most_tokens['model']} ({most_tokens['output_tokens']} tokens)")
    
    return 0

async def main():
    """Run the demo."""
    try:
        return await run_provider_comparison()
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical")
        return 1

if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 