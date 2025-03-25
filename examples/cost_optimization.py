#!/usr/bin/env python
"""Cost optimization examples for LLM Gateway."""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from decouple import config as decouple_config
from mcp.server.fastmcp import Context, FastMCP

from llm_gateway.constants import COST_PER_MILLION_TOKENS, Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.tools.optimization import OptimizationTools
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.cost_optimization")

# Initialize FastMCP server
mcp = FastMCP("Cost Optimization Demo")

@mcp.tool()
async def estimate_cost(
    prompt: str,
    model: str,
    max_tokens: int = None,
    include_output: bool = True,
    ctx: Context = None
) -> dict:
    """Estimate the cost of a request without executing it."""
    # Estimate input tokens
    # Simple approximation: 1 token ≈ 4 characters
    input_tokens = len(prompt) // 4
    
    # Estimate output tokens if not provided
    if max_tokens is None:
        if include_output:
            # Default estimate: output is about 50% of input size
            output_tokens = input_tokens // 2
        else:
            output_tokens = 0
    else:
        output_tokens = max_tokens if include_output else 0
    
    # Get cost rates for model
    cost_data = COST_PER_MILLION_TOKENS.get(model)
    if not cost_data:
        return {
            "error": f"Unknown model: {model}",
            "cost": 0.0,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            }
        }
    
    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * cost_data["input"]
    output_cost = (output_tokens / 1_000_000) * cost_data["output"]
    total_cost = input_cost + output_cost
    
    return {
        "cost": total_cost,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        },
        "rate": {
            "input": cost_data["input"],
            "output": cost_data["output"]
        },
        "model": model
    }

@mcp.tool()
async def recommend_model(
    task_type: str,
    expected_input_length: int,
    expected_output_length: int = None,
    required_capabilities: list = None,
    max_cost: float = None,
    priority: str = "balanced",
    ctx: Context = None
) -> dict:
    """Recommend the most suitable model for a given task."""
    # Convert input length to tokens
    input_tokens = expected_input_length // 4
    
    # Estimate output tokens if not provided
    if expected_output_length is None:
        if task_type == "summarization":
            output_tokens = input_tokens // 3  # Summaries are typically shorter
        elif task_type == "extraction":
            output_tokens = input_tokens // 4  # Extraction is typically concise
        elif task_type == "generation":
            output_tokens = input_tokens * 2  # Generation often creates more content
        else:
            output_tokens = input_tokens  # Default 1:1 ratio
    else:
        output_tokens = expected_output_length // 4
    
    # Define capability requirements
    required_capabilities = required_capabilities or []
    
    # Model capability mapping (simplified)
    model_capabilities = {
        # OpenAI models
        "gpt-4o": ["reasoning", "coding", "knowledge", "instruction-following", "math"],
        "gpt-4o-mini": ["reasoning", "coding", "knowledge", "instruction-following"],
        "gpt-3.5-turbo": ["coding", "knowledge", "instruction-following"],
        
        # Claude models
        "claude-3-opus-20240229": ["reasoning", "coding", "knowledge", "instruction-following", "math"],
        "claude-3-sonnet-20240229": ["reasoning", "coding", "knowledge", "instruction-following"],
        "claude-3-haiku-20240307": ["knowledge", "instruction-following"],
        
        # Other models
        "gemini-2.0-pro": ["reasoning", "knowledge", "instruction-following", "math"],
        "gemini-2.0-flash": ["knowledge", "instruction-following"],
    }
    
    # Model latency characteristics (lower is faster)
    model_speed = {
        # OpenAI models
        "gpt-4o": 3,
        "gpt-4o-mini": 2,
        "gpt-3.5-turbo": 1,
        
        # Claude models
        "claude-3-opus-20240229": 5,
        "claude-3-sonnet-20240229": 3,
        "claude-3-haiku-20240307": 2,
        
        # Other models
        "gemini-2.0-pro": 3,
        "gemini-2.0-flash": 2,
    }
    
    # Model quality characteristics (higher is better)
    model_quality = {
        # OpenAI models
        "gpt-4o": 9,
        "gpt-4o-mini": 7,
        "gpt-3.5-turbo": 5,
        
        # Claude models
        "claude-3-opus-20240229": 9,
        "claude-3-sonnet-20240229": 8,
        "claude-3-haiku-20240307": 6,
        
        # Other models
        "gemini-2.0-pro": 7,
        "gemini-2.0-flash": 5,
    }
    
    # Calculate scores for each model
    recommendations = []
    for model, capabilities in model_capabilities.items():
        # Check if model meets capability requirements
        if not all(cap in capabilities for cap in required_capabilities):
            continue
            
        # Calculate cost estimate
        cost_data = COST_PER_MILLION_TOKENS.get(model)
        if not cost_data:
            continue
            
        input_cost = (input_tokens / 1_000_000) * cost_data["input"]
        output_cost = (output_tokens / 1_000_000) * cost_data["output"]
        total_cost = input_cost + output_cost
        
        # Skip if cost exceeds maximum
        if max_cost and total_cost > max_cost:
            continue
            
        # Calculate score based on priority
        if priority == "cost":
            score = 1 / (total_cost + 1e-6)  # Avoid division by zero
        elif priority == "quality":
            score = model_quality[model]
        elif priority == "speed":
            score = 10 - model_speed[model]  # Invert speed (lower is better)
        else:  # balanced
            score = (model_quality[model] * 0.5 + 
                    (10 - model_speed[model]) * 0.3 + 
                    (1 / (total_cost + 1e-6)) * 0.2)
        
        recommendations.append({
            "model": model,
            "quality": model_quality[model],
            "speed": model_speed[model],
            "cost": total_cost,
            "score": score,
            "specialized": task_type in capabilities
        })
    
    # Sort by score
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "recommendations": recommendations[:3],  # Top 3 recommendations
        "priority": priority,
        "task_type": task_type
    }

async def demonstrate_cost_optimization():
    """Demonstrate cost optimization features."""
    logger.info("Starting cost optimization demonstration", emoji_key="start")
    
    # Create optimization tools instance with MCP server
    OptimizationTools(mcp)
    
    # Create a sample prompt
    prompt = """
    Write a comprehensive analysis of how machine learning is being applied in the healthcare industry,
    focusing on diagnostic tools, treatment optimization, and administrative efficiency.
    Include specific examples and potential future developments.
    """
    
    # Estimate costs for different models
    logger.info("Estimating costs for different models", emoji_key="cost")
    
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
    print("COST ESTIMATES")
    print("=" * 80)
    
    for model in models_to_compare:
        # Check if we have cost data for this model
        if model not in COST_PER_MILLION_TOKENS:
            print(f"No cost data available for {model}")
            continue
            
        # Estimate cost
        cost_estimate = await estimate_cost(
            prompt=prompt,
            model=model,
            max_tokens=1000,  # Assume a lengthy response
            include_output=True
        )
        
        # Print estimate
        print(f"\nModel: {model}")
        print(f"  Input tokens: {cost_estimate['tokens']['input']}")
        print(f"  Output tokens: {cost_estimate['tokens']['output']}")
        print(f"  Total tokens: {cost_estimate['tokens']['total']}")
        print(f"  Estimated cost: ${cost_estimate['cost']:.6f}")
        
        # Get rates
        input_rate = cost_estimate['rate']['input']
        output_rate = cost_estimate['rate']['output']
        print(f"  Rates: ${input_rate}/M input, ${output_rate}/M output")
    
    print("=" * 80)
    
    # Demonstrate model recommendation
    logger.info("Demonstrating model recommendation", emoji_key="model")
    
    # Get model recommendation
    recommendation = await recommend_model(
        task_type="summarization",
        expected_input_length=len(prompt) * 4,  # Rough token count estimation
        expected_output_length=1000 * 4,  # Assume 1000 tokens output
        required_capabilities=["reasoning", "knowledge"],
        max_cost=0.10,  # Maximum cost of $0.10
        priority="balanced"  # Balance cost and quality
    )
    
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
            
            # Get API key from config
            api_key = None
            if provider_name == Provider.OPENAI.value:
                api_key = decouple_config("OPENAI_API_KEY", default=None)
            elif provider_name == Provider.ANTHROPIC.value:
                api_key = decouple_config("ANTHROPIC_API_KEY", default=None)
            elif provider_name == Provider.GEMINI.value:
                api_key = decouple_config("GEMINI_API_KEY", default=None)
                
            provider = get_provider(provider_name, api_key=api_key)
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