#!/usr/bin/env python
"""Enhanced demo of the Advanced Response Comparator & Synthesizer Tool."""
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_gateway.constants import Provider
from llm_gateway.core.server import Gateway  # Use Gateway to get MCP
from llm_gateway.tools.meta import MetaTools  # Import MetaTools
from llm_gateway.utils import get_logger

# Initialize logger
logger = get_logger("example.compare_synthesize_v2")

# Global MCP instance (will be populated from Gateway)
mcp = None

async def setup_gateway_and_tools():
    """Set up the gateway and register tools."""
    global mcp
    logger.info("Initializing Gateway and MetaTools for enhanced demo...", emoji_key="start")
    gateway = Gateway("compare-synthesize-demo-v2")

    # Initialize providers (needed for the tool to function)
    try:
        await gateway._initialize_providers()
    except Exception as e:
        logger.critical(f"Failed to initialize providers: {e}. Check API keys.", emoji_key="critical")
        sys.exit(1) # Exit if providers can't be initialized

    # Create MetaTools instance, registering tools on the gateway's MCP
    meta_tools = MetaTools(gateway) # Pass the gateway instance  # noqa: F841
    mcp = gateway.mcp # Store the MCP server instance

    # Verify tool registration
    tool_list = await mcp.list_tools()
    tool_names = [t.name for t in tool_list] # Access name attribute directly
    print(f"Registered tools: {tool_names}") # Print the list of discovered tool names
    if "compare_and_synthesize" in tool_names:
        logger.success("compare_and_synthesize tool registered successfully.", emoji_key="success")
    else:
        logger.error("compare_and_synthesize tool FAILED to register.", emoji_key="error")
        sys.exit(1) # Exit if the required tool isn't available

    logger.success("Setup complete.", emoji_key="success")

def print_result(title: str, result: dict):
    """Helper function to print results clearly."""
    print(f"\n--- {title} ---")
    
    # Handle list result format - convert to dict
    if isinstance(result, list) and len(result) > 0:
        if hasattr(result[0], 'text'):
            try:
                result_dict = json.loads(result[0].text)
                result = result_dict
            except Exception:
                # If parsing fails, create a basic result dict
                result = {
                    "synthesis": {"error": "Could not parse result"},
                    "error": "Failed to parse result from list format"
                }
        else:
            # If no text attribute, use the first item directly
            result = result[0]
    
    if result.get("error"):
        print(f"Error: {result['error']}")
        if "partial_results" in result and result["partial_results"]:
             print(f"Partial Results:\n{json.dumps(result['partial_results'], indent=2)}")
    else:
        # Print relevant sections based on response format
        if "synthesis" in result:
            synthesis_data = result["synthesis"]
            if isinstance(synthesis_data, dict):
                if "best_response_text" in synthesis_data:
                    print("\n*Best Response Text*:")
                    print(synthesis_data["best_response_text"])
                if "synthesized_response" in synthesis_data:
                    print("\n*Synthesized Response*:")
                    print(synthesis_data["synthesized_response"])
                if "ranking" in synthesis_data:
                    print("\n*Ranking*:")
                    print(json.dumps(synthesis_data["ranking"], indent=2))
                if "comparative_analysis" in synthesis_data:
                    print("\n*Comparative Analysis*:")
                    print(json.dumps(synthesis_data["comparative_analysis"], indent=2))
                if synthesis_data.get("best_response", {}).get("reasoning"):
                    print("\n*Best Response Reasoning*:")
                    print(synthesis_data["best_response"]["reasoning"])
                if synthesis_data.get("synthesis_strategy"):
                    print("\n*Synthesis Strategy Explanation*:")
                    print(synthesis_data["synthesis_strategy"])

                # Optionally print evaluations for context
                # print("\n*Evaluations*:")
                # print(json.dumps(synthesis_data.get("evaluations", []), indent=2))

            else: # If synthesis failed parsing, show raw text
                print(f"Synthesis Output (raw):\n{synthesis_data}")

        print(f"\n*Synthesis/Evaluation Model*: {result.get('synthesis_provider')}/{result.get('synthesis_model')}")
        print(f"*Total Cost*: ${result.get('cost', {}).get('total_cost', 0.0):.6f}")
        print(f"*Processing Time*: {result.get('processing_time', 0.0):.2f}s")
    print("-" * (len(title) + 4) + "\n")


async def run_comparison_demo():
    """Demonstrate different modes of compare_and_synthesize."""
    if not mcp:
        logger.error("MCP server not initialized. Run setup first.", emoji_key="error")
        return

    prompt = "Explain the main benefits of using asynchronous programming in Python for a moderately technical audience. Provide 2-3 key advantages."

    # --- Configuration for initial responses ---
    # Define which models to query initially. Use a mix for better comparison.
    initial_configs = [
        {"provider": Provider.OPENAI.value, "model": "gpt-4o-mini", "parameters": {"temperature": 0.6, "max_tokens": 150}},
        {"provider": Provider.ANTHROPIC.value, "model": "claude-3-5-haiku-latest", "parameters": {"temperature": 0.5, "max_tokens": 150}},
        {"provider": Provider.GEMINI.value, "model": "gemini-2.0-flash", "parameters": {"temperature": 0.7, "max_tokens": 150}},
        # Example of adding a higher-capability model
        {"provider": Provider.ANTHROPIC.value, "model": "claude-3-7-sonnet-latest", "parameters": {"temperature": 0.6, "max_tokens": 150}},
    ]

    # --- Evaluation Criteria ---
    # More specific criteria can lead to better evaluations
    criteria = [
        "Clarity: Is the explanation clear and easy to understand for the target audience?",
        "Accuracy: Are the stated benefits of async programming technically correct?",
        "Relevance: Does the response directly address the prompt and focus on key advantages?",
        "Conciseness: Is the explanation brief and to the point?",
        "Completeness: Does it mention 2-3 distinct and significant benefits?",
    ]

    # --- Criteria Weights (Optional) ---
    # Emphasize clarity and accuracy more
    criteria_weights = {
        "Clarity: Is the explanation clear and easy to understand for the target audience?": 0.3,
        "Accuracy: Are the stated benefits of async programming technically correct?": 0.3,
        "Relevance: Does the response directly address the prompt and focus on key advantages?": 0.15,
        "Conciseness: Is the explanation brief and to the point?": 0.1,
        "Completeness: Does it mention 2-3 distinct and significant benefits?": 0.15,
    }

    # --- Synthesis/Evaluation Model (Optional) ---
    # Let the tool pick automatically first, then specify one
    synthesis_model_config = {"provider": Provider.OPENAI.value, "model": "gpt-4o"} # Changed from Claude to gpt-4o

    common_args = {
        "prompt": prompt,
        "configs": initial_configs,
        "criteria": criteria,
        "criteria_weights": criteria_weights,
    }

    # --- Demo 1: Select Best Response ---
    logger.info("Running format 'best'...", emoji_key="processing")
    try:
        result = await mcp.call_tool("compare_and_synthesize", {
            **common_args,
            "response_format": "best",
            "include_reasoning": True, # Show why it was selected
            # "synthesis_model": synthesis_model_config # Optionally specify
        })
        print_result("Response Format: 'best' (with reasoning)", result)
    except Exception as e:
        logger.error(f"Error during 'best' format demo: {e}", emoji_key="error", exc_info=True)

    # --- Demo 2: Synthesize Responses (Comprehensive Strategy) ---
    logger.info("Running format 'synthesis' (comprehensive)...", emoji_key="processing")
    try:
        result = await mcp.call_tool("compare_and_synthesize", {
            **common_args,
            "response_format": "synthesis",
            "synthesis_strategy": "comprehensive",
            "synthesis_model": synthesis_model_config, # Specify model for consistency
            "include_reasoning": True,
        })
        print_result("Response Format: 'synthesis' (Comprehensive Strategy)", result)
    except Exception as e:
        logger.error(f"Error during 'synthesis comprehensive' demo: {e}", emoji_key="error", exc_info=True)

    # --- Demo 3: Synthesize Responses (Conservative Strategy, No Reasoning) ---
    logger.info("Running format 'synthesis' (conservative, no reasoning)...", emoji_key="processing")
    try:
        result = await mcp.call_tool("compare_and_synthesize", {
            **common_args,
            "response_format": "synthesis",
            "synthesis_strategy": "conservative",
            "synthesis_model": synthesis_model_config,
            "include_reasoning": False, # Hide the synthesis strategy explanation
        })
        print_result("Response Format: 'synthesis' (Conservative, No Reasoning)", result)
    except Exception as e:
        logger.error(f"Error during 'synthesis conservative' demo: {e}", emoji_key="error", exc_info=True)

    # --- Demo 4: Rank Responses ---
    logger.info("Running format 'ranked'...", emoji_key="processing")
    try:
        result = await mcp.call_tool("compare_and_synthesize", {
            **common_args,
            "response_format": "ranked",
            "include_reasoning": True, # Show reasoning for ranks
            "synthesis_model": synthesis_model_config,
        })
        print_result("Response Format: 'ranked' (with reasoning)", result)
    except Exception as e:
        logger.error(f"Error during 'ranked' format demo: {e}", emoji_key="error", exc_info=True)

    # --- Demo 5: Analyze Responses ---
    logger.info("Running format 'analysis'...", emoji_key="processing")
    try:
        result = await mcp.call_tool("compare_and_synthesize", {
            **common_args,
            "response_format": "analysis",
            # No reasoning needed for analysis format, it's inherent
            "synthesis_model": synthesis_model_config,
        })
        print_result("Response Format: 'analysis'", result)
    except Exception as e:
        logger.error(f"Error during 'analysis' format demo: {e}", emoji_key="error", exc_info=True)


async def main():
    """Run the enhanced compare_and_synthesize demo."""
    await setup_gateway_and_tools()
    await run_comparison_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Demo stopped by user.")
    except Exception as main_err:
         logger.critical(f"Demo failed with unexpected error: {main_err}", emoji_key="critical", exc_info=True)