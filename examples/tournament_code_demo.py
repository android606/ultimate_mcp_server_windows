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
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_gateway.core.server import Gateway
from llm_gateway.core.models.requests import CompletionRequest
from llm_gateway.core.providers.base import get_provider
from llm_gateway.services.prompts import PromptTemplate
from llm_gateway.utils import get_logger
from llm_gateway.utils.logging.console import console
from llm_gateway.utils.display import display_tournament_status, display_tournament_results
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box
from rich.markup import escape


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

# Initialize global gateway
gateway = None

# --- Configuration ---
# Adjust model IDs based on your configured providers
MODEL_IDS = [
    "openai:gpt-4o-mini",
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

# --- Helper Functions ---
def parse_result(result):
    """Parse the result from a tool call into a usable dictionary.
    
    Handles various return types from MCP tools.
    """
    try:
        # Handle TextContent object (which has a .text attribute)
        if hasattr(result, 'text'):
            try:
                # Try to parse the text as JSON
                return json.loads(result.text)
            except json.JSONDecodeError:
                # Return the raw text if not JSON
                return {"text": result.text}
                
        # Handle list result
        if isinstance(result, list):
            if result:
                first_item = result[0]
                if hasattr(first_item, 'text'):
                    try:
                        return json.loads(first_item.text)
                    except json.JSONDecodeError:
                        return {"text": first_item.text}
                else:
                    return first_item
            return {}
            
        # Handle dictionary directly
        if isinstance(result, dict):
            return result
            
        # Handle other potential types or return error
        else:
            return {"error": f"Unexpected result type: {type(result)}"}
        
    except Exception as e:
        return {"error": f"Error parsing result: {str(e)}"}


async def setup_gateway():
    """Set up the gateway for demonstration."""
    global gateway
    
    # Create gateway instance
    logger.info("Initializing gateway for demonstration", emoji_key="start")
    gateway = Gateway("code-tournament-demo")
    
    # Initialize the server with all providers and built-in tools
    await gateway._initialize_providers()
    
    # Verify tools are registered
    tools = await gateway.mcp.list_tools()
    tournament_tools = [t.name for t in tools if t.name.startswith('tournament') or 'tournament' in t.name]
    logger.info(f"Registered tournament tools: {tournament_tools}", emoji_key="info")
    
    if not any('tournament' in t.lower() for t in [t.name for t in tools]):
        logger.warning("No tournament tools found. Make sure tournament plugins are registered.", emoji_key="warning")
    
    logger.success("Gateway initialized", emoji_key="success")


# Helper function to process results from MCP tool calls
def process_mcp_result(result):
    """Process result from MCP tool call, handling both list and dictionary formats."""
    # If result is a list, use the first item
    if isinstance(result, list) and result:
        result = result[0]
    
    # Handle TextContent objects (access their text attribute)
    if hasattr(result, 'text'):
        try:
            # Try to parse the text as JSON
            parsed = json.loads(result.text)
            return parsed
        except json.JSONDecodeError:
            # If it's not valid JSON, return a dictionary with the text
            return {"text": result.text}
            
    # Return as is if already a dictionary or other type
    return result


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
        status_data = process_mcp_result(status_result)
        
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
            # Execute the function with the test input
            test_output = main_function(test_input)
            results["output"] = test_output
            results["success"] = True
            logger.info(f"Test result: {json.dumps(test_output, indent=2)}", emoji_key="result")
            
            # Basic validation - should return a list of dictionaries
            if not isinstance(test_output, list):
                results["error"] = f"Expected list output, got {type(test_output).__name__}"
                results["success"] = False
            elif test_output and not all(isinstance(item, dict) for item in test_output):
                results["error"] = "Not all items in output list are dictionaries"
                results["success"] = False
            
        elif task_name == "calculator":
            # Test for calculator implementation
            try:
                Calculator = local_env.get("Calculator")
                if Calculator:
                    # Create calculator and test operations
                    calc = Calculator()
                    
                    # Test basic functions
                    calc.add(5)
                    calc.multiply(2)
                    calc.subtract(3)
                    result = calc.value() if hasattr(calc, "value") else None
                    
                    results["output"] = result
                    results["success"] = result == 7.0
                    logger.info(f"Calculator test result: {result}", emoji_key="result")
                    
                    if result != 7.0:
                        results["error"] = f"Expected 7.0, got {result}"
                else:
                    results["error"] = "Calculator class not found"
                    results["success"] = False
            except Exception as e:
                results["error"] = f"Calculator test error: {str(e)}"
                results["success"] = False
                
        elif task_name == "string_util":
            # Test for string utility functions
            test_functions = {
                "count_words": "Hello world, the world is amazing!",
                "is_balanced": "([{}])",
                "remove_duplicates": "hello",
                "find_longest_palindrome": "racecar is my favorite",
            }
            
            # Find which functions exist in the code
            test_results = {}
            for func_name, test_input in test_functions.items():
                if func_name in local_env and callable(local_env[func_name]):
                    try:
                        test_output = local_env[func_name](test_input)
                        test_results[func_name] = test_output
                    except Exception as e:
                        test_results[func_name] = f"Error: {str(e)}"
            
            results["output"] = test_results
            results["success"] = bool(test_results)  # Success if any function was tested
            logger.info(f"String utility test results: {json.dumps(test_results, indent=2)}", emoji_key="result")
        
        else:
            # Generic approach for other functions
            try:
                # Try with a simple string argument as default
                test_output = main_function("test input")
                results["output"] = test_output
                results["success"] = True
                logger.info(f"Generic test result: {test_output}", emoji_key="result")
            except TypeError:
                try:
                    # Try with no arguments
                    test_output = main_function()
                    results["output"] = test_output
                    results["success"] = True
                    logger.info(f"Generic test result (no args): {test_output}", emoji_key="result")
                except Exception as e:
                    results["error"] = f"Could not test function: {str(e)}"
                    results["success"] = False
        
        if results["success"]:
            logger.success("Code executed successfully!", emoji_key="success")
        elif not results["error"]:
            results["error"] = "Tests ran but no specific success criteria met"
            
    except Exception as e:
        results["error"] = f"Error executing code: {str(e)}"
        logger.error(results["error"], emoji_key="error")
    
    return results


def analyze_code_quality(code: str) -> Dict[str, Any]:
    """Basic code quality analysis."""
    line_count = len(code.split('\n'))
    char_count = len(code)
    
    # Simple complexity measure based on control structures
    complexity_indicators = [
        'if ', 'else:', 'elif ', 'for ', 'while ', 'try:', 'except:', 
        'with ', 'def ', 'class ', 'return', 'yield'
    ]
    complexity_score = sum(code.count(indicator) for indicator in complexity_indicators)
    
    return {
        "line_count": line_count,
        "char_count": char_count,
        "complexity_score": complexity_score
    }


async def evaluate_code(code_by_model: Dict[str, str]) -> Dict[str, Any]:
    """Use LLM to evaluate which code solution is the best.
    
    Args:
        code_by_model: Dictionary mapping model IDs to their code solutions
        
    Returns:
        Dictionary with evaluation results
    """
    if not code_by_model or len(code_by_model) < 2:
        return {"error": "Not enough code samples to compare"}
    
    try:
        # Format the code for evaluation
        evaluation_prompt = "# Code Evaluation\n\nPlease analyze the following code solutions to the same problem and determine which one is the best. "
        evaluation_prompt += "Consider factors such as correctness, efficiency, readability, error handling, and overall quality.\n\n"
        
        # Add each code sample
        for i, (model_id, code) in enumerate(code_by_model.items(), 1):
            display_model = model_id.split(':')[-1] if ':' in model_id else model_id
            # Limit each code sample to maintain reasonable context window
            truncated_code = code[:5000]  # More generous limit for code
            if len(code) > 5000:
                truncated_code += "\n# ... (code truncated for length) ..."
            evaluation_prompt += f"## Code Sample {i} (by {display_model})\n\n```python\n{truncated_code}\n```\n\n"
        
        evaluation_prompt += "\n# Your Evaluation Task\n\n"
        evaluation_prompt += "1. Rank the code solutions from best to worst\n"
        evaluation_prompt += "2. Explain your reasoning for the ranking\n"
        evaluation_prompt += "3. Highlight specific strengths of the best solution\n"
        evaluation_prompt += "4. Suggest one improvement for each solution\n"
        evaluation_prompt += "5. Comment on correctness, readability, and efficiency\n"
        
        # Use a more capable model for evaluation
        evaluation_model = "gemini:gemini-2.5-pro-exp-03-25"
        
        logger.info(f"Evaluating code using {evaluation_model}...", emoji_key="evaluate")
        
        # Get the provider
        provider_id = evaluation_model.split(':')[0]
        provider = get_provider(provider_id)
        
        if not provider:
            return {
                "error": f"Provider {provider_id} not available for evaluation",
                "model_used": evaluation_model,
                "eval_prompt": evaluation_prompt,
                "cost": 0.0
            }
        
        # Generate completion for evaluation with timeout
        try:
            request = CompletionRequest(prompt=evaluation_prompt, model=evaluation_model)
            
            # Set a timeout for the completion request
            completion_task = provider.generate_completion(
                prompt=request.prompt,
                model=request.model
            )
            
            # 45 second timeout for evaluation
            completion_result = await asyncio.wait_for(completion_task, timeout=45)
            
            # Extract cost information if available
            cost = 0.0
            if hasattr(completion_result, 'cost'):
                cost = completion_result.cost
            elif hasattr(completion_result, 'metrics') and isinstance(completion_result.metrics, dict):
                cost = completion_result.metrics.get('cost', 0.0)
            
            return {
                "evaluation": completion_result.text,
                "model_used": evaluation_model,
                "eval_prompt": evaluation_prompt,
                "cost": cost,
                "input_tokens": getattr(completion_result, 'input_tokens', 0),
                "output_tokens": getattr(completion_result, 'output_tokens', 0)
            }
        except asyncio.TimeoutError:
            logger.warning(f"Evaluation with {evaluation_model} timed out after 45 seconds", emoji_key="warning")
            return {
                "error": f"Evaluation timed out after 45 seconds",
                "model_used": evaluation_model,
                "eval_prompt": evaluation_prompt,
                "cost": 0.0
            }
        except Exception as request_error:
            logger.error(f"Error during model request: {str(request_error)}", emoji_key="error")
            return {
                "error": f"Error during model request: {str(request_error)}",
                "model_used": evaluation_model,
                "eval_prompt": evaluation_prompt,
                "cost": 0.0
            }
    
    except Exception as e:
        logger.error(f"Code evaluation failed: {str(e)}", emoji_key="error", exc_info=True)
        return {
            "error": str(e),
            "model_used": evaluation_model if 'evaluation_model' in locals() else "unknown",
            "eval_prompt": evaluation_prompt if 'evaluation_prompt' in locals() else "Error generating prompt",
            "cost": 0.0
        }


async def calculate_tournament_costs(rounds_results, evaluation_cost=None):
    """Calculate total costs of the tournament by model and grand total.
    
    Args:
        rounds_results: List of round results data from tournament results
        evaluation_cost: Optional cost of the final evaluation step
        
    Returns:
        Dictionary with cost information
    """
    model_costs = {}
    total_cost = 0.0
    
    # Process costs for each round
    for round_idx, round_data in enumerate(rounds_results):
        responses = round_data.get('responses', {})
        for model_id, response in responses.items():
            metrics = response.get('metrics', {})
            cost = metrics.get('cost', 0.0)
            
            # Convert to float if it's a string
            if isinstance(cost, str):
                try:
                    cost = float(cost.replace('$', ''))
                except (ValueError, TypeError):
                    cost = 0.0
            
            # Initialize model if not present
            if model_id not in model_costs:
                model_costs[model_id] = 0.0
                
            # Add to model total and grand total
            model_costs[model_id] += cost
            total_cost += cost
    
    # Add evaluation cost if provided
    if evaluation_cost:
        total_cost += evaluation_cost
        model_costs['evaluation'] = evaluation_cost
    
    return {
        'model_costs': model_costs,
        'total_cost': total_cost
    }

# Replace the regex extraction with LLM-based extraction
async def extract_code_from_response(response_text: str) -> str:
    """Extract code from response text using an LLM.
    
    Args:
        response_text: The raw response text from the model
        
    Returns:
        Extracted code or empty string if no code found
    """
    if not response_text:
        return ""
        
    # Use a lightweight model for extraction
    extraction_model = "openai:gpt-4o-mini"
    
    extraction_prompt = f"""
Extract the complete, executable Python code from the following text. 
Return ONLY the code, with no additional text, explanations, or markdown formatting.
If there are multiple code snippets, combine them into a single coherent program.
If there is no valid Python code, return an empty string.

Text to extract from:
```
{response_text}
```

Python code (no markdown, no explanations, just the complete code):
"""
    
    try:
        # Get the provider
        provider_id = extraction_model.split(':')[0]
        provider = get_provider(provider_id)
        
        if not provider:
            logger.warning(f"Provider {provider_id} not available for code extraction", emoji_key="warning")
            return ""
        
        # Generate completion for extraction
        request = CompletionRequest(prompt=extraction_prompt, model=extraction_model)
        
        # Set a timeout for the completion request
        completion_task = provider.generate_completion(
            prompt=request.prompt,
            model=request.model
        )
        
        # 15 second timeout for extraction
        completion_result = await asyncio.wait_for(completion_task, timeout=15)
        
        extracted_code = completion_result.text.strip()
        
        # If the result starts with ```python or ```, strip it
        if extracted_code.startswith("```python"):
            extracted_code = extracted_code[len("```python"):].strip()
        elif extracted_code.startswith("```"):
            extracted_code = extracted_code[len("```"):].strip()
            
        # If the result ends with ```, strip it
        if extracted_code.endswith("```"):
            extracted_code = extracted_code[:-3].strip()
        
        return extracted_code
        
    except Exception as e:
        logger.warning(f"Error extracting code using LLM: {str(e)}", emoji_key="warning")
        return ""

# --- Main Script Logic ---
async def run_tournament_demo():
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
        create_data = process_mcp_result(create_result)
        
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
            results_data = process_mcp_result(final_results)

            if "error" not in results_data:
                # Use the imported display function for tournament results
                display_tournament_results(results_data)
                
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
                                extracted_code = await extract_code_from_response(response.get('response_text', ''))
                            
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
                        console.print()  # Add space between rounds

                    # Evaluate final code using LLM
                    final_round = rounds_results[-1]
                    final_responses = final_round.get('responses', {})
                    
                    # Track evaluation cost
                    evaluation_cost = 0.0
                    
                    if final_responses:
                        console.print(Rule("[bold blue]AI Evaluation of Code Solutions[/bold blue]"))
                        console.print("[bold]Evaluating final code solutions...[/bold]")
                        
                        code_by_model = {}
                        for model_id, response in final_responses.items():
                            code = response.get('extracted_code', '')
                            
                            # If no extracted code but response_text exists, try to extract code from it
                            if not code and response.get('response_text'):
                                code = await extract_code_from_response(response.get('response_text', ''))
                                
                            if code:
                                code_by_model[model_id] = code
                        
                        evaluation_result = await evaluate_code(code_by_model)
                        
                        if "error" not in evaluation_result:
                            console.print(Panel(
                                escape(evaluation_result["evaluation"]),
                                title=f"[bold]Code Evaluation (by {evaluation_result['model_used'].split(':')[-1]})[/bold]",
                                border_style="green",
                                expand=False
                            ))
                            
                            # Save evaluation result to a file in the tournament directory
                            if storage_path:
                                try:
                                    evaluation_file = os.path.join(storage_path, "code_evaluation.md")
                                    with open(evaluation_file, "w", encoding="utf-8") as f:
                                        f.write(f"# Code Evaluation by {evaluation_result['model_used']}\n\n")
                                        f.write(evaluation_result["evaluation"])
                                    
                                    logger.info(f"Evaluation saved to {evaluation_file}", emoji_key="save")
                                except Exception as e:
                                    logger.warning(f"Could not save evaluation to file: {str(e)}", emoji_key="warning")
                            
                            # Track evaluation cost if available
                            evaluation_cost = evaluation_result.get('cost', 0.0)
                            logger.info(f"Evaluation cost: ${evaluation_cost:.6f}", emoji_key="cost")
                        else:
                            console.print(f"[yellow]Could not evaluate code: {evaluation_result.get('error')}[/yellow]")
                            # Try with fallback model if primary fails
                            if "gemini" in evaluation_result.get("model_used", ""):
                                console.print("[bold]Trying evaluation with fallback model (GPT-4o-mini)...[/bold]")
                                # Similar fallback approach as in text demo
                                # Implementation can be added here
                                pass

                    # Test the code (specific to code tournaments)
                    console.print(Rule("[bold blue]Code Testing[/bold blue]"))
                    console.print("[bold]Testing the best code solution in sandbox...[/bold]")
                    
                    # Find a model with code to test (preferably the winner)
                    final_code = None
                    final_model = None
                    
                    for model_id, response in final_responses.items():
                        code = response.get('extracted_code', '')
                        
                        # If no extracted code but response_text exists, try to extract code from it
                        if not code and response.get('response_text'):
                            code = await extract_code_from_response(response.get('response_text', ''))
                            
                        if code:
                            final_code = code
                            final_model = model_id
                            break
                    
                    if final_code:
                        logger.info(f"Testing code from model: {final_model}", emoji_key="test")
                        test_results = await try_code_in_sandbox(final_code, task_name)
                        
                        # Display test results
                        test_table = Table(box=box.MINIMAL, show_header=True, expand=False)
                        test_table.add_column("Test", style="cyan")
                        test_table.add_column("Result", style="green")
                        
                        test_table.add_row("Status", "[green]PASSED[/green]" if test_results["success"] else "[red]FAILED[/red]")
                        
                        if test_results["function_name"]:
                            test_table.add_row("Function", test_results["function_name"])
                            
                        if test_results["error"]:
                            test_table.add_row("Error", f"[red]{escape(test_results['error'])}[/red]")
                            
                        if test_results["output"] is not None:
                            output_str = str(test_results["output"])
                            if len(output_str) > 80:  # Truncate long outputs
                                output_str = output_str[:77] + "..."
                            test_table.add_row("Output", escape(output_str))
                            
                        console.print(test_table)
                        
                        # Save test results to file
                        if storage_path:
                            try:
                                test_file = os.path.join(storage_path, "code_test_results.md")
                                with open(test_file, "w", encoding="utf-8") as f:
                                    f.write(f"# Code Test Results\n\n")
                                    f.write(f"## Function: {test_results.get('function_name', 'Unknown')}\n\n")
                                    f.write(f"**Status**: {'PASSED' if test_results['success'] else 'FAILED'}\n\n")
                                    
                                    if test_results["error"]:
                                        f.write(f"**Error**: {test_results['error']}\n\n")
                                        
                                    if test_results["output"] is not None:
                                        f.write(f"**Output**:\n```\n{test_results['output']}\n```\n")
                                
                                logger.info(f"Test results saved to {test_file}", emoji_key="save")
                            except Exception as e:
                                logger.warning(f"Could not save test results: {str(e)}", emoji_key="warning")
                    else:
                        logger.warning("No final code found to test", emoji_key="warning")

                    # Find and highlight comparison file for final round
                    comparison_file = final_round.get('comparison_file_path')
                    if comparison_file:
                        console.print(Panel(
                            f"Check the final comparison file for the full code solutions and detailed round comparisons:\n[bold yellow]{escape(comparison_file)}[/bold yellow]",
                            title="[bold]Final Comparison File[/bold]",
                            border_style="yellow",
                            expand=False
                        ))
                    else:
                        logger.warning("Could not find path to final comparison file in results", emoji_key="warning")
                    
                    # Display cost summary
                    costs = await calculate_tournament_costs(rounds_results, evaluation_cost)
                    model_costs = costs.get('model_costs', {})
                    total_cost = costs.get('total_cost', 0.0)
                    
                    console.print(Rule("[bold blue]Tournament Cost Summary[/bold blue]"))
                    
                    cost_table = Table(box=box.MINIMAL, show_header=True, expand=False)
                    cost_table.add_column("Model", style="magenta")
                    cost_table.add_column("Total Cost", style="green", justify="right")
                    
                    # Add model costs to table
                    for model_id, cost in sorted(model_costs.items()):
                        if model_id == 'evaluation':
                            display_model = "Evaluation"
                        else:
                            display_model = model_id.split(':')[-1] if ':' in model_id else model_id
                        
                        cost_table.add_row(
                            display_model,
                            f"${cost:.6f}"
                        )
                    
                    # Add grand total
                    cost_table.add_row(
                        "[bold]GRAND TOTAL[/bold]",
                        f"[bold]${total_cost:.6f}[/bold]"
                    )
                    
                    console.print(cost_table)
                    
                    # Save cost summary to file
                    if storage_path:
                        try:
                            cost_file = os.path.join(storage_path, "cost_summary.md")
                            with open(cost_file, "w", encoding="utf-8") as f:
                                f.write(f"# Tournament Cost Summary\n\n")
                                f.write(f"## Per-Model Costs\n\n")
                                
                                for model_id, cost in sorted(model_costs.items()):
                                    if model_id == 'evaluation':
                                        display_model = "Evaluation"
                                    else:
                                        display_model = model_id.split(':')[-1] if ':' in model_id else model_id
                                    
                                    f.write(f"- **{display_model}**: ${cost:.6f}\n")
                                
                                f.write(f"\n## Grand Total\n\n")
                                f.write(f"**TOTAL COST**: ${total_cost:.6f}\n")
                            
                            logger.info(f"Cost summary saved to {cost_file}", emoji_key="save")
                        except Exception as e:
                            logger.warning(f"Could not save cost summary: {str(e)}", emoji_key="warning")
            else:
                logger.error(f"Could not fetch final results: {results_data.get('error', 'Unknown error')}", emoji_key="error")
        elif final_status:
            logger.warning(f"Tournament ended with status {final_status}. Check logs or status details for more info.", emoji_key="warning")
        
    except Exception as e:
        logger.error(f"Error in tournament demo: {str(e)}", emoji_key="error", exc_info=True)
        return 1

    logger.success("Code Tournament Demo Finished", emoji_key="complete")
    console.print(Panel(
        "To view full code solutions and detailed comparisons, check the storage directory indicated in the results summary.",
        title="[bold]Next Steps[/bold]",
        border_style="dim green",
        expand=False
    ))
    return 0


async def main():
    """Run the tournament demo."""
    try:
        # Set up gateway
        await setup_gateway()
        
        # Run the demo
        return await run_tournament_demo()
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    finally:
        # Clean up
        if gateway:
            pass  # No cleanup needed for Gateway instance


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 