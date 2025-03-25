#!/usr/bin/env python
"""Cost optimization examples for LLM Gateway."""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_gateway.constants import Provider, COST_PER_MILLION_TOKENS
from llm_gateway.core.providers.base import get_provider
from llm_gateway.tools.optimization import OptimizationTools
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.cost_optimization")


async def demonstrate_cost_optimization():
    """Demonstrate cost optimization features."""
    logger.info("Starting cost optimization demonstration", emoji_key="start")
    
    # Create a sample prompt
    prompt = """
    Write a comprehensive analysis of how machine learning is being applied in the healthcare industry,
    focusing on diagnostic tools, treatment optimization, and administrative efficiency.
    Include specific examples and potential future developments.
    """
    
    # Estimate costs for different models
    logger.info("Estimating costs for different models", emoji_key="cost")
    
    # Create optimization tools instance
    optimization_tools = OptimizationTools(None)
    
    # Models to compare
    models_to_compare = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "gemini-2.0-pro",
        "gemini-2.0-flash"
    ]
    
    # Estimate costs
    print("\n" + "=" * 80)
    print("COST ESTIMATION FOR DIFFERENT MODELS")
    print("=" * 80)
    
    for model in models_to_compare:
        # Check if we have cost data for this model
        if model not in COST_PER_MILLION_TOKENS:
            print(f"No cost data available for {model}")
            continue
            
        # Estimate cost
        cost_estimate = await optimization_tools.mcp.execute("estimate_cost", {
            "prompt": prompt,
            "model": model,
            "max_tokens": 1000,  # Assume a lengthy response
            "include_output": True
        })
        
        # Print estimate
        print(f"\nModel: {model}")
        print(f"  Input tokens: {cost_estimate['tokens']['input']}")
        print(f"  Output tokens: {cost_estimate['tokens']['output']}")
        print(f"  Total tokens: {cost_estimate['tokens']['total']}")
        print(f"  Estimated cost: ${cost_estimate['cost']:.6f}")
        
        # Get rates
        input_rate = COST_PER_MILLION_TOKENS[model]["input"]
        output_rate = COST_PER_MILLION_TOKENS[model]["output"]
        print(f"  Rates: ${input_rate}/M input, ${output_rate}/M output")
    
    print("=" * 80)
    
    # Demonstrate model recommendation
    logger.info("Demonstrating model recommendation", emoji_key="model")
    
    # Get model recommendation
    recommendation = await optimization_tools.mcp.execute("recommend_model", {
        "task_type": "summarization",
        "expected_input_length": len(prompt) * 4,  # Rough token count estimation
        "expected_output_length": 1000 * 4,  # Assume 1000 tokens output
        "required_capabilities": ["reasoning", "knowledge"],
        "max_cost": 0.10,  # Maximum cost of $0.10
        "priority": "balanced"  # Balance cost and quality
    })
    
    # Display recommendations
    print("\n" + "=" * 80)
    print("MODEL RECOMMENDATIONS")
    print("=" * 80)
    
    if recommendation["recommendations"]:
        print(f"\nTop recommended models (priority: {recommendation['priority']}):")
        for i, model_rec in enumerate(recommendation["recommendations"]):
            print(f"\n{i+1}. {model_rec['model']}")
            print(f"   Quality score: {model_rec['quality']}/10")
            print(f"   Speed: {10-model_rec['speed']}/10 (lower is faster)")
            print(f"   Estimated cost: ${model_rec['cost']:.6f}")
            if model_rec.get("specialized"):
                print(f"   ✓ Specialized for {recommendation['task_type']} tasks")
    else:
        print("No models match the criteria.")
        
    print("=" * 80)
    
    # Demonstrate cost comparison with real API calls
    logger.info("Comparing actual costs with real API calls", emoji_key="cost")
    
    # Select 2 models for actual comparison
    compare_models = ["gpt-3.5-turbo", "gemini-2.0-flash"]
    
    print("\n" + "=" * 80)
    print("ACTUAL COST COMPARISON")
    print("=" * 80)
    
    # Simplified prompt for actual calls
    short_prompt = "Briefly explain the concept of neural networks."
    
    for model_name in compare_models:
        try:
            provider_name = _get_provider_for_model(model_name)
            if not provider_name:
                logger.warning(f"No provider found for model {model_name}", emoji_key="warning")
                continue
                
            provider = get_provider(provider_name)
            await provider.initialize()
            
            logger.info(f"Testing {model_name} with actual API call", emoji_key="processing")
            
            # Generate completion
            result = await provider.generate_completion(
                prompt=short_prompt,
                model=model_name,
                temperature=0.7,
                max_tokens=200  # Limit output for example
            )
            
            # Print results
            print(f"\nModel: {model_name}")
            print(f"  Provider: {provider_name}")
            print(f"  Input tokens: {result.input_tokens}")
            print(f"  Output tokens: {result.output_tokens}")
            print(f"  Total tokens: {result.total_tokens}")
            print(f"  Actual cost: ${result.cost:.6f}")
            print(f"  Completion time: {result.processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error testing {model_name}: {str(e)}", emoji_key="error")
    
    print("=" * 80)


def _get_provider_for_model(model_name: str) -> str:
    """Get provider name for a model.
    
    Args:
        model_name: Model name
        
    Returns:
        Provider name or None if not found
    """
    if model_name.startswith("gpt-"):
        return Provider.OPENAI.value
    elif model_name.startswith("claude-"):
        return Provider.ANTHROPIC.value
    elif model_name.startswith("deepseek-"):
        return Provider.DEEPSEEK.value
    elif model_name.startswith("gemini-"):
        return Provider.GEMINI.value
    return None


async def main():
    """Run cost optimization examples."""
    try:
        await demonstrate_cost_optimization()
        
    except Exception as e:
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)