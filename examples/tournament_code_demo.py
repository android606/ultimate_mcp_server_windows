#!/usr/bin/env python3
"""
Tournament Code Demo - Demonstrates running a code improvement tournament

This script shows how to:
1. Create a tournament with multiple models
2. Track progress across multiple rounds
3. Retrieve and analyze the improved code

The tournament task is to write and iteratively improve a Python function for
parsing messy CSV data, handling various edge cases.

Usage:
  python examples/tournament_code_demo.py

Options:
  --task TASK       Specify a different coding task (default: parse_csv)
"""

import argparse
import asyncio
import json
import re
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from llm_gateway.core.server import Gateway
from llm_gateway.services.prompts import PromptTemplate
from llm_gateway.tools import extract_code_from_response

# Import tournament tools
from llm_gateway.tools.tournament import (
    create_tournament,
    get_tournament_results,
    get_tournament_status,
)
from llm_gateway.utils import get_logger, process_mcp_result
from llm_gateway.utils.display import (
    CostTracker,
    display_tournament_results,
    display_tournament_status,
)
from llm_gateway.utils.logging.console import console


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run a code improvement tournament demo")
    parser.add_argument(
        "--task", 
        type=str, 
        default="parse_csv",
        help="Coding task (parse_csv, calculator, string_util, or custom)"
    )
    parser.add_argument(
        "--custom-task", 
        type=str, 
        help="Custom coding task description (used when --task=custom)"
    )
    return parser.parse_args()


# Initialize logger using get_logger
logger = get_logger("example.tournament_code")

# Create a simple structure for cost tracking from dict (tokens might be missing)
TrackableResult = namedtuple("TrackableResult", ["cost", "input_tokens", "output_tokens", "provider", "model", "processing_time"])

# Initialize global gateway
gateway = None

# --- Configuration ---
# Adjust model IDs based on your configured providers
MODEL_IDS = [
    "openai:gpt-4.1-mini",
    "deepseek:deepseek-chat",
    "gemini:gemini-2.5-pro-exp-03-25"
]
NUM_ROUNDS = 2  # Changed from 3 to 2 for faster execution and debugging
TOURNAMENT_NAME = "Code Improvement Tournament Demo"

# The generic code prompt template
TEMPLATE_CODE = """
# GENERIC CODE TOURNAMENT PROMPT TEMPLATE

Write a {{code_type}} that {{task_description}}.

{{context}}

Your solution should:

{% for requirement in requirements %}
{{ loop.index }}. {{requirement}}
{% endfor %}

{% if example_inputs %}
Example inputs:
```
{{example_inputs}}
```
{% endif %}

{% if example_outputs %}
Expected outputs:
```
{{example_outputs}}
```
{% endif %}

Provide ONLY the Python code for your solution, enclosed in triple backticks (```python ... ```).
"""

# Define predefined tasks
TASKS = {
    "parse_csv": {
        "code_type": "Python function",
        "task_description": "parses a CSV string that may use different delimiters and contains various edge cases",
        "context": "Your function should be robust enough to handle real-world messy CSV data.",
        "requirements": [
            "Implement `parse_csv_string(csv_data: str) -> list[dict]`",
            "Accept a string `csv_data` which might contain CSV data",
            "Automatically detect the delimiter (comma, semicolon, or tab)",
            "Handle quoted fields correctly, including escaped quotes within fields",
            "Treat the first row as the header",
            "Return a list of dictionaries, where each dictionary represents a row",
            "Handle errors gracefully by logging warnings and skipping problematic rows",
            "Return an empty list if the input is empty or only contains a header",
            "Include necessary imports",
            "Be efficient for moderately large inputs"
        ],
        "example_inputs": """name,age,city
"Smith, John",42,New York
"Doe, Jane",39,"Los Angeles, CA"
"\"Williams\", Bob",65,"Chicago"
""",
        "example_outputs": """[
    {"name": "Smith, John", "age": "42", "city": "New York"},
    {"name": "Doe, Jane", "age": "39", "city": "Los Angeles, CA"},
    {"name": "\"Williams\", Bob", "age": "65", "city": "Chicago"}
]"""
    },
    "calculator": {
        "code_type": "Python class",
        "task_description": "implements a scientific calculator with basic and advanced operations",
        "context": "Implement a Calculator class that supports both basic arithmetic and scientific operations.",
        "requirements": [
            "Create a `Calculator` class with appropriate methods",
            "Support basic operations: add, subtract, multiply, divide",
            "Support scientific operations: power, square root, logarithm, sine, cosine, tangent",
            "Handle edge cases (division by zero, negative square roots, etc.)",
            "Include proper error handling with descriptive error messages",
            "Maintain calculation history",
            "Implement a method to clear the history",
            "Allow chaining operations (e.g., calc.add(5).multiply(2))",
            "Include proper docstrings and type hints"
        ],
        "example_inputs": """# Create calculator and perform operations
calc = Calculator()
calc.add(5).multiply(2).subtract(3)
result = calc.value()
""",
        "example_outputs": """7.0  # (5 * 2 - 3 = 7)"""
    },
    "string_util": {
        "code_type": "Python utility module",
        "task_description": "provides advanced string processing functions",
        "context": "Create a comprehensive string utility module that goes beyond Python's built-in string methods.",
        "requirements": [
            "Create a module named `string_utils.py`",
            "Implement `remove_duplicates(text: str) -> str` to remove duplicate characters while preserving order",
            "Implement `is_balanced(text: str) -> bool` to check if brackets/parentheses are balanced",
            "Implement `find_longest_palindrome(text: str) -> str` to find the longest palindrome in a string",
            "Implement `count_words(text: str) -> dict` that returns word frequencies (case-insensitive)",
            "Implement `generate_ngrams(text: str, n: int) -> list` that returns all n-grams",
            "Properly handle edge cases (empty strings, invalid inputs)",
            "Include appropriate error handling",
            "Add comprehensive docstrings and type hints",
            "Include a simple example usage section"
        ],
        "example_inputs": """text = "Hello world, the world is amazing!"
count_words(text)""",
        "example_outputs": """{'hello': 1, 'world': 2, 'the': 1, 'is': 1, 'amazing': 1}"""
    }
}

# Create custom task template
def create_custom_task_variables(task_description):
    """Create a simple custom task with standard requirements"""
    return {
        "code_type": "Python function",
        "task_description": task_description,
        "context": "",
        "requirements": [
            "Implement the solution as specified in the task description",
            "Include proper error handling",
            "Handle edge cases appropriately",
            "Write clean, readable code",
            "Include helpful comments",
            "Use appropriate data structures and algorithms",
            "Follow Python best practices",
            "Include necessary imports"
        ],
        "example_inputs": "",
        "example_outputs": ""
    }

# Create the prompt template object
code_template = PromptTemplate(
    template=TEMPLATE_CODE,
    template_id="code_tournament_template",
    description="A template for code tournament prompts",
    required_vars=["code_type", "task_description", "context", "requirements"]
)

async def setup_gateway():
    """Set up the gateway for demonstration."""
    global gateway
    
    # Create gateway instance
    logger.info("Initializing gateway for demonstration", emoji_key="start")
    gateway = Gateway("code-tournament-demo", register_tools=False)
    
    # Initialize the server with all providers and built-in tools
    await gateway._initialize_providers()
    
    # Manually register tournament tools
    mcp = gateway.mcp
    mcp.tool()(create_tournament)
    mcp.tool()(get_tournament_status)
    mcp.tool()(get_tournament_results)
    logger.info("Manually registered tournament tools.")

    # Verify tools are registered
    tools = await gateway.mcp.list_tools()
    tournament_tools = [t.name for t in tools if t.name.startswith('tournament') or 'tournament' in t.name]
    logger.info(f"Registered tournament tools: {tournament_tools}", emoji_key="info")
    
    if not any('tournament' in t.lower() for t in [t.name for t in tools]):
        logger.warning("No tournament tools found. Make sure tournament plugins are registered.", emoji_key="warning")
    
    logger.success("Gateway initialized", emoji_key="success")


async def poll_tournament_status(tournament_id: str, storage_path: Optional[str] = None, interval: int = 5) -> Optional[str]:
    """Poll the tournament status until it reaches a final state.
    
    Args:
        tournament_id: ID of the tournament to poll
        storage_path: Optional storage path to avoid tournament not found issues
        interval: Time between status checks in seconds
    """
    logger.info(f"Polling status for tournament {tournament_id}...", emoji_key="poll")
    final_states = ["COMPLETED", "FAILED", "CANCELLED"]
    
    # Add direct file polling capability to handle case where tournament manager can't find the tournament
    if storage_path:
        storage_dir = Path(storage_path)
        state_file = storage_dir / "tournament_state.json"
        logger.debug(f"Will check tournament state file directly at: {state_file}")
    
    while True:
        status_input = {"tournament_id": tournament_id}
        status_result = await gateway.mcp.call_tool("get_tournament_status", status_input)
        status_data = await process_mcp_result(status_result)
        
        if "error" in status_data:
            # If tournament manager couldn't find the tournament but we have the storage path,
            # try to read the state file directly (this is a fallback mechanism)
            if storage_path and "not found" in status_data.get("error", "").lower():
                try:
                    logger.debug(f"Attempting to read tournament state directly from: {state_file}")
                    if state_file.exists():
                        with open(state_file, 'r', encoding='utf-8') as f:
                            direct_status_data = json.load(f)
                            status = direct_status_data.get("status")
                            current_round = direct_status_data.get("current_round", 0)
                            total_rounds = direct_status_data.get("config", {}).get("rounds", 0)
                            
                            # Create a status object compatible with our display function
                            status_data = {
                                "tournament_id": tournament_id,
                                "status": status,
                                "current_round": current_round,
                                "total_rounds": total_rounds,
                                "storage_path": storage_path
                            }
                            logger.debug(f"Successfully read direct state: {status}")
                    else:
                        logger.warning(f"State file not found at: {state_file}")
                except Exception as e:
                    logger.error(f"Error reading state file directly: {e}")
                    logger.error(f"Error fetching status: {status_data['error']}", emoji_key="error")
                    return None # Indicate error during polling
            else:
                # Standard error case
                logger.error(f"Error fetching status: {status_data['error']}", emoji_key="error")
                return None # Indicate error during polling
            
        # Display improved status using the imported function
        display_tournament_status(status_data)
        
        status = status_data.get("status")
        if status in final_states:
            logger.success(f"Tournament reached final state: {status}", emoji_key="success")
            return status
            
        await asyncio.sleep(interval)

def try_code_in_sandbox(code_str: str, task_name: str) -> Dict[str, Any]:
    """Test the generated code in a sandbox to validate it.
    
    Args:
        code_str: The code to test
        task_name: The name of the task, used to determine appropriate test cases
        
    Returns:
        Dictionary with test results
    """
    logger.info("Testing code in sandbox", emoji_key="sandbox")
    
    # Strip markdown code block markers if present
    if code_str.startswith("```python"):
        code_str = code_str[len("```python"):].strip()
    if code_str.endswith("```"):
        code_str = code_str[:-3].strip()
    
    # Results to return
    results = {
        "success": False,
        "error": None,
        "output": None,
        "function_name": None
    }
    
    # Extract the function definition and attempt to run it
    try:
        # Create a local environment to execute the code
        local_env = {}
        exec(code_str, local_env)
        
        # Look for defined functions in the local environment
        function_names = [name for name, obj in local_env.items() 
                         if callable(obj) and not name.startswith('__')]
        
        if not function_names:
            results["error"] = "No functions found in the code"
            logger.error(results["error"], emoji_key="error")
            return results
        
        # Use the first function found 
        main_function_name = function_names[0]
        main_function = local_env[main_function_name]
        
        logger.info(f"Found function: {main_function_name}", emoji_key="function")
        results["function_name"] = main_function_name
        
        # Custom test cases based on task
        if task_name == "parse_csv":
            # Test for CSV parser
            test_input = """name,age,city
"Smith, John",42,New York
"Doe, Jane",39,"Los Angeles, CA"
"""
            try:
                parsed_data = main_function(test_input)
                if isinstance(parsed_data, list) and len(parsed_data) == 2:
                    results["success"] = True
                    results["output"] = parsed_data
                else:
                    results["error"] = f"Expected a list of 2 items, got: {type(parsed_data)} with {len(parsed_data) if isinstance(parsed_data, list) else 'unknown'} items"
            except Exception as e:
                results["error"] = f"Error running function: {str(e)}"
        elif task_name == "calculator":
            # Test for calculator
            try:
                # Check if Calculator class exists
                if "Calculator" in local_env:
                    calc = local_env["Calculator"]()
                    # Run simple operations
                    if hasattr(calc, "add") and hasattr(calc, "subtract"):
                        result_val = calc.add(5).subtract(3).value() if hasattr(calc, "value") else None
                        results["success"] = result_val == 2
                        results["output"] = result_val
                    else:
                        results["error"] = "Calculator is missing add/subtract methods"
                else:
                    results["error"] = "Calculator class not found"
            except Exception as e:
                results["error"] = f"Error testing calculator: {str(e)}"
        else:
            # For other tasks, just confirm the function can be called
            results["success"] = True
            results["output"] = "Function passes basic inspection"
            
    except Exception as e:
        results["error"] = f"Error in code execution: {str(e)}"
        
    return results


def analyze_code_quality(code_str: str) -> Dict[str, Any]:
    """Analyze the quality of the code.
    
    Args:
        code_str: The code to analyze
        
    Returns:
        Dictionary with code quality metrics
    """
    metrics = {
        "line_count": 0,
        "complexity_score": 0
    }
    
    if not code_str:
        return metrics
    
    # Count lines (excluding empty lines)
    lines = [line for line in code_str.split("\n") if line.strip()]
    metrics["line_count"] = len(lines)
    
    # Simple complexity metric based on control structures and functions
    complexity = 0
    control_patterns = [
        r"\bif\b", r"\belse\b", r"\belif\b", r"\bfor\b", r"\bwhile\b",
        r"\btry\b", r"\bexcept\b", r"\bwith\b", r"\bdef\b", r"\bclass\b",
        r"\blambda\b", r"\breturn\b", r"\braise\b"
    ]
    
    for pattern in control_patterns:
        for line in lines:
            if re.search(pattern, line):
                complexity += 1
    
    metrics["complexity_score"] = complexity
    
    return metrics


# --- Main Script Logic ---
async def run_tournament_demo(tracker: CostTracker):
    """Run the code tournament demo."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine which task to use
    if args.task == "custom" and args.custom_task:
        # Custom task provided via command line
        task_name = "custom"
        task_variables = create_custom_task_variables(args.custom_task)
        task_description = args.custom_task
        log_task_info = f"Using custom task: [yellow]{escape(task_description)}[/yellow]"
    elif args.task in TASKS:
        # Use one of the predefined tasks
        task_name = args.task
        task_variables = TASKS[args.task]
        task_description = task_variables["task_description"]
        log_task_info = f"Using predefined task: [yellow]{escape(task_description)}[/yellow]"
    else:
        # Default to parse_csv if task not recognized
        task_name = "parse_csv"
        task_variables = TASKS[task_name]
        task_description = task_variables['task_description']
        log_task_info = f"Using default task: [yellow]{escape(task_description)}[/yellow]"
    
    # Use Rich Rule for title
    console.print(Rule(f"[bold blue]{TOURNAMENT_NAME} - {task_name.replace('_', ' ').title()}[/bold blue]"))
    console.print(log_task_info)
    console.print(f"Models: [cyan]{', '.join(MODEL_IDS)}[/cyan]")
    console.print(f"Rounds: [cyan]{NUM_ROUNDS}[/cyan]")
    
    # Render the template
    try:
        rendered_prompt = code_template.render(task_variables)
        logger.info(f"Template rendered for task: {task_name}", emoji_key="template")
        
        # Show prompt preview in a Panel
        prompt_preview = rendered_prompt.split("\n")[:10]  # Show first 10 lines
        preview_text = "\n".join(prompt_preview) + "\n..."
        console.print(Panel(escape(preview_text), title="[bold]Rendered Prompt Preview[/bold]", border_style="dim blue", expand=False))
        
    except Exception as e:
        logger.error(f"Template rendering failed: {str(e)}", emoji_key="error", exc_info=True)
        # Log template and variables for debugging
        logger.debug(f"Template: {TEMPLATE_CODE}")
        logger.debug(f"Variables: {escape(str(task_variables))}")
        return 1
    
    # 1. Create the tournament
    create_input = {
        "name": f"{TOURNAMENT_NAME} - {task_name.replace('_', ' ').title()}",
        "prompt": rendered_prompt,
        "model_ids": MODEL_IDS,
        "rounds": NUM_ROUNDS,
        "tournament_type": "code"
    }
    
    try:
        logger.info("Creating tournament...", emoji_key="processing")
        create_result = await gateway.mcp.call_tool("create_tournament", create_input)
        create_data = await process_mcp_result(create_result)
        
        if "error" in create_data:
            error_msg = create_data.get("error", "Unknown error")
            logger.error(f"Failed to create tournament: {error_msg}. Exiting.", emoji_key="error")
            return 1
            
        tournament_id = create_data.get("tournament_id")
        if not tournament_id:
            logger.error("No tournament ID returned. Exiting.", emoji_key="error")
            return 1
            
        # Extract storage path for reference
        storage_path = create_data.get("storage_path")
        logger.info(f"Tournament created with ID: {tournament_id}", emoji_key="tournament")
        if storage_path:
            logger.info(f"Tournament storage path: {storage_path}", emoji_key="path")
            
        # Add a small delay to ensure the tournament state is saved before proceeding
        await asyncio.sleep(2)
        
        # 2. Poll for status
        final_status = await poll_tournament_status(tournament_id, storage_path)

        # 3. Fetch and display final results
        if final_status == "COMPLETED":
            logger.info("Fetching final results...", emoji_key="results")
            results_input = {"tournament_id": tournament_id}
            final_results = await gateway.mcp.call_tool("get_tournament_results", results_input)
            results_data = await process_mcp_result(final_results)

            if "error" not in results_data:
                # Use the imported display function for tournament results
                display_tournament_results(results_data)
                
                # Track aggregated cost
                if isinstance(results_data, dict) and "cost" in results_data:
                    try:
                        total_cost = results_data.get("cost", {}).get("total_cost", 0.0)
                        processing_time = results_data.get("total_processing_time", 0.0)
                        # Provider/Model is ambiguous here, use a placeholder
                        trackable = TrackableResult(
                            cost=total_cost,
                            input_tokens=0, # Not aggregated
                            output_tokens=0, # Not aggregated
                            provider="tournament",
                            model="code_tournament",
                            processing_time=processing_time
                        )
                        tracker.add_call(trackable)
                    except Exception as track_err:
                        logger.warning(f"Could not track tournament cost: {track_err}", exc_info=False)

                # Analyze round progression if available
                rounds_results = results_data.get('rounds_results', [])
                if rounds_results:
                    console.print(Rule("[bold blue]Code Evolution Analysis[/bold blue]"))

                    for round_idx, round_data in enumerate(rounds_results):
                        console.print(f"[bold]Round {round_idx} Analysis:[/bold]")
                        responses = round_data.get('responses', {})
                        
                        round_table = Table(box=box.MINIMAL, show_header=True, expand=False)
                        round_table.add_column("Model", style="magenta")
                        round_table.add_column("Lines", style="green", justify="right")
                        round_table.add_column("Complexity", style="yellow", justify="right")

                        has_responses = False
                        for model_id, response in responses.items():
                            display_model = escape(model_id.split(':')[-1])
                            extracted_code = response.get('extracted_code', '')
                            
                            # If no extracted code but response_text exists, try to extract code from it
                            if not extracted_code and response.get('response_text'):
                                extracted_code = await extract_code_from_response(response.get('response_text', ''), tracker=tracker)
                            
                            if extracted_code:
                                has_responses = True
                                metrics = analyze_code_quality(extracted_code)
                                round_table.add_row(
                                    display_model, 
                                    str(metrics['line_count']),
                                    str(metrics['complexity_score'])
                                )
                        
                        if has_responses:
                            console.print(round_table)
                        else:
                             console.print("[dim]No valid code responses recorded for this round.[/dim]")
                
                # Allow optional testing of final code
                winner_model = results_data.get('winner_model')
                if winner_model:
                    final_code = results_data.get('model_results', {}).get(winner_model, {}).get('extracted_code')
                    if final_code:
                        console.print(Rule("[bold blue]Testing Winner's Code[/bold blue]"))
                        console.print(f"Testing code from winner model: [cyan]{winner_model}[/cyan]")
                        
                        sandbox_results = try_code_in_sandbox(final_code, task_name)
                        
                        if sandbox_results["success"]:
                            console.print(Panel(
                                f"[green]Code test successful![/green]\nFunction: {sandbox_results['function_name']}\nOutput: {sandbox_results['output']}",
                                title="[bold green]Test Results[/bold green]",
                                border_style="green"
                            ))
                        else:
                            console.print(Panel(
                                f"[red]Code test failed:[/red] {sandbox_results['error']}",
                                title="[bold red]Test Results[/bold red]",
                                border_style="red"
                            ))
            else:
                logger.error(f"Error getting results: {results_data.get('error', 'Unknown error')}", emoji_key="error")
        else:
            logger.error(f"Tournament did not complete successfully (status: {final_status})", emoji_key="error")
            
        # Display cost summary at the end
        tracker.display_summary(console)

        return 0
        
    except Exception as e:
        logger.error(f"Error running tournament: {str(e)}", emoji_key="error", exc_info=True)
        return 1


async def main():
    """Run the code tournament demo."""
    tracker = CostTracker() # Instantiate tracker
    try:
        await setup_gateway()
        return await run_tournament_demo(tracker) # Pass tracker
    except Exception as e:
        logger.critical(f"Tournament demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 