#!/usr/bin/env python
"""Workflow delegation example using LLM Gateway MCP server."""
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import Context, FastMCP

from llm_gateway.constants import Provider
from llm_gateway.core.providers.base import get_provider
from llm_gateway.core.server import Gateway
from llm_gateway.utils import get_logger, process_mcp_result
# --- Add Rich Imports ---
from llm_gateway.utils.logging.console import console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.syntax import Syntax
from rich.markup import escape
from rich import box
# --- Add Display Utils Import ---
from llm_gateway.utils.display import (
    display_text_content_result,
    parse_and_display_result,
    extract_and_parse_content,
    _display_stats,
    _display_json_data
)
# ----------------------

# Initialize logger
logger = get_logger("example.workflow_delegation")

# Initialize FastMCP server
mcp = FastMCP("Workflow Delegation Demo")

# Mock provider initialization function (replace with actual if needed)
async def initialize_providers():
    logger.info("Initializing required providers...", emoji_key="provider")
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    from decouple import config as decouple_config
    all_keys_present = True
    for key in required_keys:
        if not decouple_config(key, default=None):
            logger.warning(f"API key {key} not found. Some demos might fail.", emoji_key="warning")
            console.print(f"[yellow]Warning:[/yellow] API key {key} not found.")
            all_keys_present = False
    # Actual initialization would happen inside the tools or a setup function
    if all_keys_present:
        logger.info("All required API keys seem to be present.", emoji_key="success")
    else:
         logger.warning("Some API keys missing, functionality may be limited.", emoji_key="warning")

# Register meta tools directly
@mcp.tool()
async def analyze_task(
    task_description: str,
    available_providers: Optional[List[str]] = None,
    analyze_features: bool = True,
    analyze_cost: bool = True,
    ctx = None
) -> Dict[str, Any]:
    """Analyze a task and recommend suitable models."""
    start_time = time.time()
    
    # Mock implementation for demonstration
    if not available_providers:
        available_providers = [Provider.OPENAI.value, Provider.GEMINI.value, Provider.ANTHROPIC.value]
    
    # Analyze task type based on description
    task_type = "extraction" if "extract" in task_description.lower() else \
                "summarization" if "summarize" in task_description.lower() else \
                "generation"
    
    # Mock required features
    if "entities" in task_description.lower():
        required_features = ["entity_recognition", "classification"]
        features_explanation = "This task requires entity recognition capabilities to identify key concepts."
    elif "technical" in task_description.lower():
        required_features = ["domain_knowledge", "technical_understanding"]
        features_explanation = "This task requires technical domain knowledge to properly analyze content."
    else:
        required_features = ["text_processing"]
        features_explanation = "This is a general text processing task."
    
    # Generate recommendations
    recommendations = [
        {
            "provider": Provider.OPENAI.value,
            "model": "gpt-4o",
            "explanation": "Best overall quality for complex tasks"
        },
        {
            "provider": Provider.ANTHROPIC.value,
            "model": "claude-3-opus-20240229",
            "explanation": "Excellent for technical content analysis"
        },
        {
            "provider": Provider.GEMINI.value,
            "model": "gemini-2.0-pro",
            "explanation": "Good balance of performance and cost"
        }
    ]
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return {
        "task_type": task_type,
        "required_features": required_features,
        "features_explanation": features_explanation,
        "recommendations": recommendations,
        "processing_time": processing_time
    }

@mcp.tool()
async def delegate_task(
    task_description: str,
    prompt: str,
    optimization_criteria: str = "balanced",
    available_providers: Optional[List[str]] = None,
    max_cost: Optional[float] = None,
    ctx = None
) -> Dict[str, Any]:
    """Delegate a task to the most appropriate provider."""
    start_time = time.time()
    
    # Mock implementation for demonstration
    if not available_providers:
        available_providers = [Provider.OPENAI.value, Provider.GEMINI.value]
    
    # Select a provider based on criteria
    if optimization_criteria == "cost":
        provider = Provider.GEMINI.value
        model = "gemini-2.0-flash-lite"
    elif optimization_criteria == "quality":
        provider = Provider.OPENAI.value
        model = "gpt-4o"
    else:  # balanced
        provider = Provider.OPENAI.value
        model = "gpt-4o-mini"
    
    # Get provider instance
    provider_instance = get_provider(provider)
    await provider_instance.initialize()
    
    # Generate completion
    result = await provider_instance.generate_completion(
        prompt=prompt,
        model=model,
        temperature=0.7,
        max_tokens=300
    )
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return {
        "text": result.text,
        "provider": provider,
        "model": model,
        "processing_time": processing_time,
        "cost": result.cost,
        "tokens": {
            "input": result.input_tokens,
            "output": result.output_tokens,
            "total": result.total_tokens
        }
    }

@mcp.tool()
async def execute_workflow(
    workflow_steps: List[Dict[str, Any]],
    initial_input: str,
    max_concurrency: int = 1,
    ctx = None
) -> Dict[str, Any]:
    """Execute a multi-step workflow."""
    start_time = time.time()
    total_cost = 0.0
    
    # Initialize output collection
    outputs = {}
    
    # Initialize input for first step
    current_input = initial_input
    
    # Process each step sequentially (for demo)
    for step in workflow_steps:
        step_id = step.get("id", "unknown")
        operation = step.get("operation", "")
        provider_name = step.get("provider", Provider.OPENAI.value)
        model_name = step.get("model", "")
        parameters = step.get("parameters", {})
        output_as = step.get("output_as", step_id)
        
        # Check if we should use output from previous step
        if "input_from" in step and step["input_from"] in outputs:
            current_input = outputs[step["input_from"]]
        
        # Get provider instance
        provider = get_provider(provider_name)
        await provider.initialize()
        
        # Create prompts based on operation
        if operation == "summarize":
            prompt = f"Summarize the following text. {parameters.get('format', 'Keep it concise')}:\n\n{current_input}"
        elif operation == "extract_entities":
            entity_types = parameters.get("entity_types", ["organization", "person", "concept"])
            prompt = f"Extract the following entity types from the text: {', '.join(entity_types)}.\n\n{current_input}"
        elif operation == "generate_questions":
            prompt = f"Generate {parameters.get('question_count', 3)} {parameters.get('question_type', 'analytical')} questions about the following text:\n\n{current_input}"
        else:
            prompt = current_input
        
        # Generate completion
        result = await provider.generate_completion(
            prompt=prompt,
            model=model_name,
            temperature=0.7,
            max_tokens=500
        )
        
        # Store output
        outputs[output_as] = result.text
        
        # Add to total cost
        total_cost += result.cost
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return {
        "outputs": outputs,
        "processing_time": processing_time,
        "total_cost": total_cost
    }

@mcp.tool()
async def optimize_prompt(
    prompt: str,
    target_model: str,
    optimization_type: str = "general",
    provider: str = Provider.OPENAI.value,
    ctx = None
) -> Dict[str, Any]:
    """Optimize a prompt for a specific model."""
    # Get provider instance
    provider_instance = get_provider(provider)
    await provider_instance.initialize()
    
    # Create optimization prompt
    optimization_prompt = f"""
    I need to optimize this prompt for the {target_model} model:
    
    "{prompt}"
    
    Please rewrite this prompt to work optimally with {target_model}, 
    focusing on {optimization_type} optimization.
    
    Return ONLY the optimized prompt with no explanations.
    """
    
    # Generate optimized prompt
    result = await provider_instance.generate_completion(
        prompt=optimization_prompt,
        model=provider_instance.get_default_model(),
        temperature=0.7,
        max_tokens=300
    )
    
    # Return optimized prompt
    return {
        "original_prompt": prompt,
        "optimized_prompt": result.text.strip(),
        "target_model": target_model,
        "optimization_type": optimization_type,
        "cost": result.cost
    }

# Enhanced display function for workflow demos
def display_workflow_result(title: str, result: Any):
    """Display workflow result with consistent formatting."""
    console.print(Rule(f"[bold blue]{escape(title)}[/bold blue]"))
    
    # Process result to handle list or dict format
    result = process_mcp_result(result)
    
    # Display outputs if present
    if "outputs" in result and result["outputs"]:
        for output_name, output_text in result["outputs"].items():
            console.print(Panel(
                escape(str(output_text).strip()),
                title=f"[bold magenta]Output: {escape(output_name)}[/bold magenta]",
                border_style="magenta",
                expand=False
            ))
    elif "text" in result:
        # Display single text output if there's no outputs dictionary
        console.print(Panel(
            escape(result["text"].strip()),
            title="[bold magenta]Result[/bold magenta]",
            border_style="magenta",
            expand=False
        ))
    
    # Display execution stats
    _display_stats(result, console)

# Enhanced display function for task analysis
def display_task_analysis(title: str, result: Any):
    """Display task analysis result with consistent formatting."""
    console.print(Rule(f"[bold blue]{escape(title)}[/bold blue]"))
    
    # Process result to handle list or dict format
    result = process_mcp_result(result)
    
    # Display task type and features
    analysis_table = Table(box=box.SIMPLE, show_header=False)
    analysis_table.add_column("Metric", style="cyan")
    analysis_table.add_column("Value", style="white")
    analysis_table.add_row("Task Type", escape(result.get("task_type", "N/A")))
    analysis_table.add_row("Required Features", escape(str(result.get("required_features", []))))
    console.print(analysis_table)
    
    # Display features explanation
    if "features_explanation" in result:
        console.print(Panel(
            escape(result["features_explanation"]),
            title="[bold]Features Explanation[/bold]",
            border_style="dim blue",
            expand=False
        ))
    
    # Display recommendations
    if "recommendations" in result and result["recommendations"]:
        rec_table = Table(title="[bold]Model Recommendations[/bold]", box=box.ROUNDED, show_header=True)
        rec_table.add_column("Provider", style="magenta")
        rec_table.add_column("Model", style="blue")
        rec_table.add_column("Explanation", style="white")
        for rec in result["recommendations"]:
            rec_table.add_row(
                escape(rec.get("provider", "N/A")),
                escape(rec.get("model", "N/A")),
                escape(rec.get("explanation", "N/A"))
            )
        console.print(rec_table)
    
    # Display execution stats
    _display_stats(result, console)

# --- Demo Functions ---

async def run_analyze_task_demo():
    """Demonstrate the analyze_task tool."""
    console.print(Rule("[bold blue]Analyze Task Demo[/bold blue]"))
    logger.info("Running analyze_task demo...", emoji_key="start")
    
    task_description = "Summarize the provided technical document about AI advancements and extract key entities."
    console.print(f"[cyan]Task Description:[/cyan] {escape(task_description)}")
    
    try:
        result = await mcp.call_tool("analyze_task", {
            "task_description": task_description,
            "analyze_features": True
        })
        
        # Use enhanced display function
        display_task_analysis("Analysis Results", result)
        
    except Exception as e:
        logger.error(f"Error in analyze_task demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")
    console.print()


async def run_delegate_task_demo():
    """Demonstrate the delegate_task tool."""
    console.print(Rule("[bold blue]Delegate Task Demo[/bold blue]"))
    logger.info("Running delegate_task demo...", emoji_key="start")
    
    task_description = "Generate a short marketing blurb for a new AI-powered writing assistant."
    prompt = "Write a catchy, 2-sentence marketing blurb for \'AI Writer Pro\', a tool that helps users write faster and better."
    console.print(f"[cyan]Task Description:[/cyan] {escape(task_description)}")
    console.print(f"[cyan]Prompt:[/cyan] {escape(prompt)}")

    priorities = ["balanced", "cost", "quality"]
    
    for priority in priorities:
        console.print(Rule(f"[yellow]Delegating with Priority: {priority}[/yellow]"))
        logger.info(f"Delegating task with priority: {priority}", emoji_key="processing")
        try:
            result = await mcp.call_tool("delegate_task", {
                "task_description": task_description,
                "prompt": prompt,
                "optimization_criteria": priority,
                # "available_providers": [Provider.OPENAI.value] # Example constraint
            })
            
            # Use display_text_content_result for consistent formatting
            result_obj = process_mcp_result(result)
            
            # Display the text result
            text_content = result_obj.get("text", "")
            if not text_content and hasattr(result, 'text'):
                text_content = result.text
                
            console.print(Panel(
                escape(text_content.strip() if text_content else "[red]No text returned[/red]"),
                title=f"[bold green]Delegated Result ({escape(priority)})[/bold green]",
                border_style="green",
                expand=False
            ))
            
            # Display execution stats
            _display_stats(result_obj, console)

        except Exception as e:
            logger.error(f"Error delegating task with priority {priority}: {e}", emoji_key="error", exc_info=True)
            console.print(f"[bold red]Error ({escape(priority)}):[/bold red] {escape(str(e))}")
        console.print()


async def run_workflow_demo():
    """Demonstrate the execute_workflow tool."""
    console.print(Rule("[bold blue]Execute Workflow Demo[/bold blue]"))
    logger.info("Running execute_workflow demo...", emoji_key="start")

    initial_text = """
    Artificial intelligence (AI) is rapidly transforming various sectors. 
    In healthcare, AI algorithms analyze medical images with remarkable accuracy, 
    aiding radiologists like Dr. Evelyn Reed. Pharmaceutical companies, such as InnovatePharma, 
    use AI to accelerate drug discovery. Meanwhile, financial institutions leverage AI 
    for fraud detection and algorithmic trading. The field continues to evolve, 
    driven by researchers like Kenji Tanaka and advancements in machine learning.
    """
    
    workflow = [
        {
            "id": "step1_summarize",
            "operation": "summarize",
            "provider": Provider.ANTHROPIC.value,
            "model": "claude-3-5-haiku-latest",
            "parameters": {"format": "Provide a 2-sentence summary"},
            "output_as": "summary"
        },
        {
            "id": "step2_extract",
            "operation": "extract_entities",
            "provider": Provider.OPENAI.value,
            "model": "gpt-4o-mini",
            "parameters": {"entity_types": ["person", "organization", "field"]},
            "input_from": None, # Use initial_input
            "output_as": "entities"
        },
        {
            "id": "step3_questions",
            "operation": "generate_questions",
            "provider": Provider.GEMINI.value,
            "model": "gemini-2.0-flash-lite",
            "parameters": {"question_count": 2, "question_type": "insightful"},
            "input_from": "summary", # Use output from step 1
            "output_as": "questions"
        }
    ]
    
    console.print("[cyan]Initial Input Text:[/cyan]")
    console.print(Panel(escape(initial_text.strip()), border_style="dim blue", expand=False))
    console.print("[cyan]Workflow Definition:[/cyan]")
    try:
        workflow_json = json.dumps(workflow, indent=2, default=lambda o: o.value if isinstance(o, Provider) else str(o)) # Handle enum serialization
        console.print(Panel(
            Syntax(workflow_json, "json", theme="default", line_numbers=True, word_wrap=True),
            title="[bold]Workflow Steps[/bold]",
            border_style="blue",
            expand=False
        ))
    except Exception as json_err:
         console.print(f"[red]Could not display workflow definition: {escape(str(json_err))}[/red]")
    
    logger.info(f"Executing workflow with {len(workflow)} steps...", emoji_key="processing")
    try:
        result = await mcp.call_tool("execute_workflow", {
            "workflow_steps": workflow, 
            "initial_input": initial_text
        })
        
        # Use enhanced display function
        display_workflow_result("Workflow Results", result)

    except Exception as e:
        logger.error(f"Error executing workflow: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Workflow Execution Error:[/bold red] {escape(str(e))}")
    console.print()


async def run_prompt_optimization_demo():
    """Demonstrate the optimize_prompt tool."""
    console.print(Rule("[bold blue]Prompt Optimization Demo[/bold blue]"))
    logger.info("Running optimize_prompt demo...", emoji_key="start")

    original_prompt = "Tell me about Large Language Models."
    target_model = "claude-3-opus-20240229"
    optimization_type = "detailed_response" # e.g., conciseness, detailed_response, specific_format
    
    console.print(f"[cyan]Original Prompt:[/cyan] {escape(original_prompt)}")
    console.print(f"[cyan]Target Model:[/cyan] {escape(target_model)}")
    console.print(f"[cyan]Optimization Type:[/cyan] {escape(optimization_type)}")
    
    logger.info(f"Optimizing prompt for {target_model}...", emoji_key="processing")
    try:
        result = await mcp.call_tool("optimize_prompt", {
            "prompt": original_prompt,
            "target_model": target_model,
            "optimization_type": optimization_type,
            "provider": Provider.OPENAI.value # Using OpenAI to optimize for Claude
        })
        
        # Process result to handle list or dict format
        result = process_mcp_result(result)
        
        # Get optimized prompt text
        optimized_prompt = result.get("optimized_prompt", "")
        if not optimized_prompt and hasattr(result, 'text'):
            optimized_prompt = result.text
        
        console.print(Panel(
            escape(optimized_prompt.strip() if optimized_prompt else "[red]Optimization failed[/red]"),
            title="[bold green]Optimized Prompt[/bold green]",
            border_style="green",
            expand=False
        ))
        
        # Display execution stats
        _display_stats(result, console)
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Prompt Optimization Error:[/bold red] {escape(str(e))}")
    console.print()


async def main():
    """Run all workflow delegation demonstrations."""
    await initialize_providers() # Ensure keys are checked/providers ready
    console.print(Rule("[bold magenta]Workflow & Delegation Demos Starting[/bold magenta]"))
    
    try:
        await run_analyze_task_demo()
        await run_delegate_task_demo()
        await run_workflow_demo()
        await run_prompt_optimization_demo()
        
    except Exception as e:
        logger.critical(f"Workflow demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"[bold red]Critical Demo Error:[/bold red] {escape(str(e))}")
        return 1
    
    logger.success("Workflow & Delegation Demos Finished Successfully!", emoji_key="complete")
    console.print(Rule("[bold magenta]Workflow & Delegation Demos Complete[/bold magenta]"))
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 