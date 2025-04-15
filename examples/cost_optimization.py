#!/usr/bin/env python
"""Cost optimization examples for LLM Gateway."""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from decouple import config as decouple_config
from mcp.server.fastmcp import FastMCP
from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.tools.optimization import estimate_cost, recommend_model
from llm_gateway.utils import get_logger

# --- Import display utilities ---
from llm_gateway.utils.display import parse_and_display_result

# --- Add Rich Imports ---
from llm_gateway.utils.logging.console import console
from llm_gateway.utils.text import count_tokens  # Import proper token counting function

# ----------------------

# Initialize logger
logger = get_logger("example.cost_optimization")

# Initialize FastMCP server
mcp = FastMCP("Cost Optimization Demo")

# Create optimization tools instance with MCP server - this registers all the tools
# OptimizationTools(mcp)

# Manually register the tools needed for this demo on the local MCP instance
mcp.tool()(estimate_cost)
mcp.tool()(recommend_model)
logger.info("Manually registered optimization tools (estimate_cost, recommend_model).")

# Helper function to unpack tool results that might be returned as a list
def unpack_tool_result(result):
    """
    Handles the case where tool results are returned as a list instead of a dictionary.
    
    Args:
        result: Result from an MCP tool call
        
    Returns:
        Unpacked result as a dictionary if possible, or the original result
    """
    # Check if result is a list with content
    if isinstance(result, list) and result:
        # Try to get the first item if it's a dictionary
        first_item = result[0]
        if isinstance(first_item, dict):
            return first_item
            
        # Handle case where item might be an object with a text attribute
        if hasattr(first_item, 'text'):
            try:
                import json
                # Try to parse as JSON
                return json.loads(first_item.text)
            except (json.JSONDecodeError, AttributeError):
                pass
    
    # Return the original result if we can't unpack
    return result

async def demonstrate_cost_optimization():
    """Demonstrate cost optimization features using Rich."""
    console.print(Rule("[bold blue]Cost Optimization Demonstration[/bold blue]"))
    logger.info("Starting cost optimization demonstration", emoji_key="start")
    
    prompt = """
    Write a comprehensive analysis of how machine learning is being applied in the healthcare industry,
    focusing on diagnostic tools, treatment optimization, and administrative efficiency.
    Include specific examples and potential future developments.
    """
    
    # Note for the demo: Use proper token counting, not character estimation
    logger.info("Calculating tokens for the prompt with tiktoken", emoji_key="info")
    models_to_show = ["gpt-4o", "claude-3-5-sonnet-latest"]
    for model_name in models_to_show:
        token_count = count_tokens(prompt, model_name)
        logger.info(f"Model {model_name}: {token_count} input tokens", emoji_key="token")
    
    estimated_output_tokens = 500 # Estimate output length for the prompt
    
    # --- Cost Estimation --- 
    console.print(Rule("[cyan]Cost Estimation[/cyan]"))
    logger.info("Estimating costs for different models", emoji_key="cost")
    
    models_to_compare = [
        "gpt-4o",
        "gpt-4.1-mini",
        "claude-3-opus-20240229",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
        "gemini-2.5-pro-latest",
        "gemini-2.5-flash-latest",
        "deepseek-chat"
    ]
    
    cost_table = Table(title="Estimated Costs", box=box.ROUNDED, show_header=True)
    cost_table.add_column("Model", style="magenta")
    cost_table.add_column("Input Tokens", style="white", justify="right")
    cost_table.add_column("Output Tokens", style="white", justify="right")
    cost_table.add_column("Input Rate ($/M)", style="dim blue", justify="right")
    cost_table.add_column("Output Rate ($/M)", style="dim blue", justify="right")
    cost_table.add_column("Estimated Cost", style="green", justify="right")

    cost_estimates = []
    for model in models_to_compare:
        try:
            # Call the estimate_cost tool
            raw_result = await mcp.call_tool("estimate_cost", {
                "prompt": prompt,
                "model": model,
                "max_tokens": estimated_output_tokens
            })
            
            # Unpack the result
            estimate_result = unpack_tool_result(raw_result)
            
            if "error" in estimate_result:
                logger.warning(f"Could not estimate cost for {model}: {estimate_result['error']}", emoji_key="warning")
                cost_table.add_row(escape(model), "-", "-", "-", "-", f"[dim red]{estimate_result['error']}[/dim red]")
            else:
                cost_estimates.append(estimate_result) # Store for later use if needed
                cost_table.add_row(
                    escape(model),
                    str(estimate_result["tokens"]["input"]),
                    str(estimate_result["tokens"]["output"]),
                    f"${estimate_result['rate']['input']:.2f}",
                    f"${estimate_result['rate']['output']:.2f}",
                    f"${estimate_result['cost']:.6f}"
                )
        except Exception as e:
            logger.error(f"Error calling estimate_cost for {model}: {e}", emoji_key="error", exc_info=True)
            cost_table.add_row(escape(model), "-", "-", "-", "-", "[red]Error[/red]")
            
    console.print(cost_table)
    console.print()

    # --- Model Recommendation --- 
    console.print(Rule("[cyan]Model Recommendation[/cyan]"))
    logger.info("Getting model recommendations based on different priorities", emoji_key="recommend")
    
    # Define task parameters for recommendation
    task_info = {
        "task_type": "analysis_generation",
        "expected_input_length": len(prompt),
        "expected_output_length": estimated_output_tokens * 4, # Convert tokens back to chars approx
        "required_capabilities": ["reasoning", "knowledge"], 
        "max_cost": 0.005 # Example max cost constraint
    }
    
    priorities = ["balanced", "cost", "quality", "speed"]
    
    recommendation_table = Table(title="Model Recommendations", box=box.ROUNDED, show_header=True)
    recommendation_table.add_column("Priority", style="yellow")
    recommendation_table.add_column("1st Rec", style="magenta")
    recommendation_table.add_column("Cost", style="green", justify="right")
    recommendation_table.add_column("Quality", style="blue", justify="right")
    recommendation_table.add_column("Speed", style="cyan", justify="right")
    recommendation_table.add_column("Score", style="white", justify="right")
    recommendation_table.add_column("Other Recs", style="dim")

    recommendation_results = {}
    
    for priority in priorities:
        try:
            # Call the recommend_model tool
            raw_result = await mcp.call_tool("recommend_model", {
                **task_info,
                "priority": priority
            })
            
            # Unpack the result
            rec_result = unpack_tool_result(raw_result)
            
            if "error" in rec_result:
                logger.warning(f"Could not get recommendations for priority '{priority}': {rec_result['error']}", emoji_key="warning")
                recommendation_table.add_row(priority, "-", "-", "-", "-", "-", f"[dim red]{rec_result['error']}[/dim red]")
            elif not rec_result.get("recommendations"):
                 logger.info(f"No models met criteria for priority '{priority}'", emoji_key="info")
                 recommendation_table.add_row(priority, "[dim]None[/dim]", "-", "-", "-", "-", "No models fit criteria")
            else:
                recs = rec_result["recommendations"]
                top_rec = recs[0]
                other_recs_str = ", ".join([escape(r["model"]) for r in recs[1:]]) if len(recs) > 1 else "None"
                
                # Calculate score if missing (library versions may differ)
                if 'score' not in top_rec:
                    if priority == "cost":
                        # Lower cost is better
                        score = 10.0 / (float(top_rec['cost']) + 0.001)
                    elif priority == "quality":
                        # Higher quality is better
                        score = float(top_rec['quality'])
                    elif priority == "speed":
                        # Lower speed value is better
                        score = 10.0 - float(top_rec['speed'])
                    else:  # balanced
                        # Balanced score
                        score = (float(top_rec['quality']) * 0.5 - 
                                float(top_rec['cost']) * 100.0 - 
                                float(top_rec['speed']) * 0.3)
                else:
                    score = top_rec['score']
                
                recommendation_table.add_row(
                    priority,
                    escape(top_rec["model"]),
                    f"${top_rec['cost']:.6f}",
                    str(top_rec["quality"]),
                    str(top_rec["speed"]),
                    f"{score:.2f}",
                    other_recs_str
                )
                
                # Store for later use
                recommendation_results[priority] = rec_result
                
        except Exception as e:
             logger.error(f"Error calling recommend_model for priority {priority}: {e}", emoji_key="error", exc_info=True)
             recommendation_table.add_row(priority, "-", "-", "-", "-", "-", "[red]Error[/red]")

    console.print(recommendation_table)
    console.print()

    # --- Run with Recommended Model (Example) ---
    # Find the balanced recommendation
    balanced_rec = None
    try:
        # Use stored result if available
        if "balanced" in recommendation_results:
            rec_result = recommendation_results["balanced"]
            if rec_result.get("recommendations"):
                balanced_rec = rec_result["recommendations"][0]["model"]
        else:
            # Otherwise, try to get a fresh recommendation
            raw_result = await mcp.call_tool("recommend_model", {
                **task_info,
                "priority": "balanced"
            })
            
            # Unpack the result
            rec_result = unpack_tool_result(raw_result)
            
            if rec_result.get("recommendations"):
                balanced_rec = rec_result["recommendations"][0]["model"]
    except Exception as e:
        logger.error(f"Error getting balanced recommendation: {e}", emoji_key="error")
        pass # Ignore errors here, just trying to get a model

    if balanced_rec:
        console.print(Rule(f"[cyan]Executing with Recommended Model ({escape(balanced_rec)})[/cyan]"))
        logger.info(f"Running completion with balanced recommendation: {balanced_rec}", emoji_key="processing")
        
        provider_name = _get_provider_for_model(balanced_rec)
        if not provider_name:
            logger.error(f"Could not determine provider for recommended model {balanced_rec}", emoji_key="error")
            return
        
        api_key = decouple_config(f"{provider_name.upper()}_API_KEY", default=None)
        if not api_key:
            logger.warning(f"API key for {provider_name} not found. Cannot run completion.", emoji_key="warning")
            return
        
        try:
            provider = get_provider(provider_name, api_key=api_key)
            await provider.initialize()
            
            # Generate the completion and get accurate token counts from the response
            result = await provider.generate_completion(
                prompt=prompt,
                model=balanced_rec,
                temperature=0.7,
                max_tokens=estimated_output_tokens
            )
            
            # Display result information using actual token counts from the API response
            logger.success(f"Completion with {balanced_rec} successful", emoji_key="success")
            logger.info(f"Actual input tokens: {result.input_tokens}, output tokens: {result.output_tokens}", emoji_key="token")
            logger.info(f"Cost based on actual usage: ${result.cost:.6f}", emoji_key="cost")
            
            console.print(Panel(
                escape(result.text.strip()),
                title=f"[bold green]Response from {escape(balanced_rec)}[/bold green]",
                border_style="green"
            ))
            
            # Display Stats using display utility
            parse_and_display_result(
                f"Execution Stats for {balanced_rec}",
                None,
                {
                    "model": balanced_rec,
                    "provider": provider_name,
                    "cost": result.cost,
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.input_tokens + result.output_tokens
                    },
                    "processing_time": result.processing_time
                }
            )
            
        except Exception as e:
             logger.error(f"Error running completion with {balanced_rec}: {e}", emoji_key="error", exc_info=True)
    else:
        logger.info("Could not get a balanced recommendation to execute.", emoji_key="info")


def _get_provider_for_model(model_name: str) -> str:
    """Helper to determine provider from model name."""
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