"""Tools for LLM cost estimation, model comparison, recommendation, and workflow execution.

Provides utilities to help manage LLM usage costs and select appropriate models.
"""
import asyncio
import time
import traceback
from typing import Any, Dict, List, Optional

# MCP ToolError might be needed if raised directly, keep if necessary
from mcp.server.fastmcp.exceptions import ToolError

from llm_gateway.constants import COST_PER_MILLION_TOKENS

# Import specific exceptions
from llm_gateway.exceptions import ToolInputError
from llm_gateway.tools.base import with_error_handling, with_tool_metrics

# Import other tools potentially needed by execute_optimized_workflow
# Ensure all tools used in workflows are imported here or handled dynamically.
from llm_gateway.tools.completion import chat_completion
from llm_gateway.tools.document import chunk_document, summarize_document
from llm_gateway.tools.extraction import extract_json
from llm_gateway.tools.rag import (
    add_documents,
    create_knowledge_base,
    generate_with_rag,
    retrieve_context,
)
from llm_gateway.utils import get_logger
from llm_gateway.utils.text import count_tokens

logger = get_logger("llm_gateway.tools.optimization")

# --- Standalone Tool Functions ---

@with_tool_metrics
@with_error_handling
async def estimate_cost(
    prompt: str,
    model: str, # Expects full model ID like "openai/gpt-4o-mini"
    max_tokens: Optional[int] = None,
    include_output: bool = True
) -> Dict[str, Any]:
    """Estimates the monetary cost of an LLM request without executing it.

    Calculates cost based on input prompt tokens and estimated/specified output tokens
    using predefined cost rates for the specified model.

    Args:
        prompt: The text prompt that would be sent to the model.
        model: The full model identifier (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-haiku-20240307").
               Cost data must be available for this model in `COST_PER_MILLION_TOKENS`.
        max_tokens: (Optional) The maximum number of tokens expected in the output. If None,
                      output tokens are estimated as roughly half the input prompt tokens.
        include_output: (Optional) If False, calculates cost based only on input tokens, ignoring
                        `max_tokens` or output estimation. Defaults to True.

    Returns:
        A dictionary containing the cost estimate and token breakdown:
        {
            "cost": 0.000150, # Total estimated cost in USD
            "breakdown": {
                "input_cost": 0.000100,
                "output_cost": 0.000050
            },
            "tokens": {
                "input": 200,   # Tokens counted from the prompt
                "output": 100,  # Estimated or provided max_tokens
                "total": 300
            },
            "rate": {         # Cost per million tokens for this model
                "input": 0.50,
                "output": 1.50
            },
            "model": "openai/gpt-4o-mini",
            "is_estimate": true
        }

    Raises:
        ToolError: If the specified `model` is unknown or cost data is missing.
        ValueError: If token counting fails for the given model and prompt.
    """
    # Add input validation
    if not prompt or not isinstance(prompt, str):
        raise ToolInputError("Prompt must be a non-empty string.")
    if not model or '/' not in model: # Basic check for format provider/model
        raise ToolInputError(f"Invalid model format: '{model}'. Expected format 'provider/model_name'.")

    try:
        # Use the model name directly if it includes provider prefix
        input_tokens = count_tokens(prompt, model_name=model)
    except ValueError as e:
        logger.warning(f"Could not count tokens for model '{model}' using tiktoken: {e}. Using rough estimate.")
        # Fallback: Estimate based on character count (adjust ratio as needed)
        input_tokens = len(prompt) // 4

    # Estimate output tokens if needed
    estimated_output_tokens = 0
    if include_output:
        if max_tokens is not None:
            estimated_output_tokens = max_tokens
        else:
            # Simple estimation logic (e.g., output is ~50% of input)
            estimated_output_tokens = input_tokens // 2
            logger.debug(f"max_tokens not provided, estimating output tokens as {estimated_output_tokens}")
    else:
         estimated_output_tokens = 0 # Explicitly zero if output excluded

    # Use the full model ID to get cost data
    cost_data = COST_PER_MILLION_TOKENS.get(model)
    if not cost_data:
        # Use ToolError for clearer API error handling
        raise ToolError(status_code=400, detail=f"Unknown model or cost data unavailable: {model}")

    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * cost_data["input"]
    output_cost = (estimated_output_tokens / 1_000_000) * cost_data["output"]
    total_cost = input_cost + output_cost

    logger.info(f"Estimated cost for model '{model}': ${total_cost:.6f} (In: {input_tokens} tokens, Out: {estimated_output_tokens} tokens)")
    return {
        "cost": total_cost,
        "breakdown": {
            "input_cost": input_cost,
            "output_cost": output_cost
        },
        "tokens": {
            "input": input_tokens,
            "output": estimated_output_tokens,
            "total": input_tokens + estimated_output_tokens
        },
        "rate": {
            "input": cost_data["input"],
            "output": cost_data["output"]
        },
        "model": model,
        "is_estimate": True
    }

@with_tool_metrics
@with_error_handling
async def compare_models(
    prompt: str,
    models: List[str], # List of full model IDs like "openai/gpt-4o-mini"
    max_tokens: Optional[int] = None,
    include_output: bool = True
) -> Dict[str, Any]:
    """Compares the estimated cost of running a prompt across multiple specified models.

    Uses the `estimate_cost` tool for each model in the list concurrently.

    Args:
        prompt: The text prompt to use for cost comparison.
        models: A list of full model identifiers (e.g., ["openai/gpt-4o-mini", "anthropic/claude-3-haiku-20240307"]).
        max_tokens: (Optional) Maximum output tokens to assume for cost estimation across all models.
                      If None, output is estimated individually per model based on input.
        include_output: (Optional) Whether to include estimated output costs in the comparison. Defaults to True.

    Returns:
        A dictionary containing the cost comparison results:
        {
            "models": {
                "openai/gpt-4o-mini": {
                    "cost": 0.000150,
                    "tokens": { "input": 200, "output": 100, "total": 300 }
                },
                "anthropic/claude-3-haiku-20240307": {
                    "cost": 0.000087,
                    "tokens": { "input": 200, "output": 100, "total": 300 }
                },
                "openai/gpt-4o": { # Example of an error during estimation
                    "error": "Unknown model or cost data unavailable: openai/gpt-4o"
                }
            },
            "ranking": [ # List of models ordered by cost (cheapest first), errors excluded
                "anthropic/claude-3-haiku-20240307",
                "openai/gpt-4o-mini"
            ],
            "cheapest": "anthropic/claude-3-haiku-20240307", # Model with the lowest cost
            "most_expensive": "openai/gpt-4o-mini",        # Model with the highest cost (among successful estimates)
            "prompt_length_chars": 512, # Character length of the input prompt
            "max_tokens_assumed": 100 # Assumed output tokens (estimated or provided)
        }

    Raises:
        ToolInputError: If the `models` list is empty or contains invalid formats.
    """
    if not models or not isinstance(models, list):
        raise ToolInputError("'models' must be a non-empty list of model identifiers.")
    # Add basic format validation for models in the list
    if any('/' not in m for m in models):
         raise ToolInputError("One or more model names in the list are invalid. Expected format 'provider/model_name'.")

    results = {}
    estimated_output_for_summary = None # To store one estimate for the summary

    async def get_estimate(model_name):
        nonlocal estimated_output_for_summary
        try:
            estimate = await estimate_cost(
                prompt=prompt,
                model=model_name,
                max_tokens=max_tokens,
                include_output=include_output
            )
            results[model_name] = {
                "cost": estimate["cost"],
                "tokens": estimate["tokens"],
            }
            # Store the output token estimate from the first successful call for summary
            if estimated_output_for_summary is None:
                estimated_output_for_summary = estimate["tokens"]["output"]
        except ToolError as e:
            logger.warning(f"Could not estimate cost for model {model_name}: {e.detail}")
            results[model_name] = {"error": e.detail}
        except Exception as e:
            logger.error(f"Unexpected error estimating cost for model {model_name}: {e}", exc_info=True)
            results[model_name] = {"error": f"Unexpected error: {str(e)}"}

    await asyncio.gather(*(get_estimate(model) for model in models))

    # Filter out errors before sorting
    successful_estimates = {m: r for m, r in results.items() if "error" not in r}
    sorted_models = sorted(successful_estimates.items(), key=lambda item: item[1]["cost"])

    # Use the stored estimated output tokens or the provided max_tokens for summary
    output_tokens_summary = estimated_output_for_summary if max_tokens is None else max_tokens
    if not include_output:
         output_tokens_summary = 0

    logger.info(f"Compared models: {list(results.keys())}. Cheapest: {sorted_models[0][0] if sorted_models else 'N/A'}")
    return {
        "models": results,
        "ranking": [m for m, _ in sorted_models],
        "cheapest": sorted_models[0][0] if sorted_models else None,
        "most_expensive": sorted_models[-1][0] if sorted_models else None,
        "prompt_length_chars": len(prompt),
        "max_tokens_assumed": output_tokens_summary,
    }

@with_tool_metrics
@with_error_handling
async def recommend_model(
    task_type: str,
    expected_input_length: int, # In characters
    expected_output_length: Optional[int] = None, # In characters
    required_capabilities: Optional[List[str]] = None,
    max_cost: Optional[float] = None,
    priority: str = "balanced" # Options: "cost", "quality", "speed", "balanced"
) -> Dict[str, Any]:
    """Recommends suitable LLM models based on task requirements and optimization priority.

    Evaluates known models against criteria like task type suitability (inferred),
    estimated cost (based on expected lengths), required capabilities, speed, and quality metrics.

    Args:
        task_type: A description of the task (e.g., "summarization", "code generation", "entity extraction",
                   "customer support chat", "complex reasoning question"). Used loosely for capability checks.
        expected_input_length: Estimated length of the input text in characters.
        expected_output_length: (Optional) Estimated length of the output text in characters.
                                If None, it's roughly estimated based on input length.
        required_capabilities: (Optional) A list of specific capabilities the model MUST possess.
                               Current known capabilities include: "reasoning", "coding", "knowledge",
                               "instruction-following", "math". Check model metadata for supported values.
                               Example: ["coding", "instruction-following"]
        max_cost: (Optional) The maximum acceptable estimated cost (in USD) for a single run
                  with the expected input/output lengths. Models exceeding this are excluded.
        priority: (Optional) The primary factor for ranking suitable models.
                  Options:
                  - "cost": Prioritize the cheapest models.
                  - "quality": Prioritize models with the highest quality score.
                  - "speed": Prioritize models with the lowest latency score.
                  - "balanced": (Default) Attempt to find a good mix of cost, quality, and speed.

    Returns:
        A dictionary containing model recommendations:
        {
            "recommendations": [
                {
                    "model": "anthropic/claude-3-haiku-20240307",
                    "estimated_cost": 0.000087,
                    "quality_score": 6,
                    "speed_score": 2,
                    "capabilities": ["knowledge", "instruction-following"],
                    "reason": "Good balance of cost and speed, meets requirements."
                },
                {
                    "model": "openai/gpt-4o-mini",
                    "estimated_cost": 0.000150,
                    "quality_score": 7,
                    "speed_score": 2,
                    "capabilities": ["reasoning", "coding", ...],
                    "reason": "Slightly higher cost, but better quality/capabilities."
                }
                # ... other suitable models
            ],
            "parameters": { # Input parameters for context
                "task_type": "summarization",
                "expected_input_length": 2000,
                "expected_output_length": 500,
                "required_capabilities": [],
                "max_cost": 0.001,
                "priority": "balanced"
            },
            "excluded_models": { # Models evaluated but excluded, with reasons
                 "anthropic/claude-3-opus-20240229": "Exceeds max cost ($0.0015 > $0.001)",
                 "some-other-model": "Missing required capabilities: ['coding']"
            }
        }

    Raises:
        ToolInputError: If priority is invalid or lengths are non-positive.
    """
    if expected_input_length <= 0:
        raise ToolInputError("expected_input_length must be positive.")
    if expected_output_length is not None and expected_output_length <= 0:
        raise ToolInputError("expected_output_length must be positive if provided.")
    if priority not in ["cost", "quality", "speed", "balanced"]:
        raise ToolInputError(f"Invalid priority: '{priority}'. Must be cost, quality, speed, or balanced.")

    # Use a simple placeholder text based on length for cost estimation
    sample_text = "a" * expected_input_length
    required_capabilities = required_capabilities or []

    # Rough estimate for output length if not provided
    if expected_output_length is None:
        # Adjust this heuristic as needed (e.g., summarization shortens, generation might lengthen)
        estimated_output_length_chars = expected_input_length // 4
    else:
         estimated_output_length_chars = expected_output_length
    # Estimate max_tokens based on character length (very rough)
    estimated_max_tokens = estimated_output_length_chars // 3

    # --- Model Metadata (Keep as defined before, ensure it's up-to-date) --- 
    model_capabilities = {
        # OpenAI models
        "openai/gpt-4o": ["reasoning", "coding", "knowledge", "instruction-following", "math"],
        "openai/gpt-4o-mini": ["reasoning", "coding", "knowledge", "instruction-following"],
        
        # Anthropic models (corrected names)
        "anthropic/claude-3-opus-20240229": ["reasoning", "coding", "knowledge", "instruction-following", "math"],
        "anthropic/claude-3-sonnet-20240229": ["reasoning", "coding", "knowledge", "instruction-following"],
        "anthropic/claude-3-haiku-20240307": ["knowledge", "instruction-following"],
        "anthropic/claude-3-5-sonnet-20240620": ["reasoning", "coding", "knowledge", "instruction-following", "math"],
        
        # Other potential models (adjust based on actual availability/support)
        # "deepseek/deepseek-chat": ["coding", "knowledge", "instruction-following"],
        # "deepseek/deepseek-reasoner": ["reasoning", "math", "instruction-following"],
        # "google/gemini-1.5-flash-latest": ["knowledge", "instruction-following"],
        # "google/gemini-1.5-pro-latest": ["reasoning", "knowledge", "instruction-following", "math"],
    }
    
    model_speed = { # Lower is faster (latency score 1-5)
        "openai/gpt-4o": 3,
        "openai/gpt-4o-mini": 2,
        "anthropic/claude-3-opus-20240229": 5,
        "anthropic/claude-3-sonnet-20240229": 3,
        "anthropic/claude-3-haiku-20240307": 2,
        "anthropic/claude-3-5-sonnet-20240620": 3,
        # "deepseek/deepseek-chat": 2,
        # "deepseek/deepseek-reasoner": 3,
        # "google/gemini-1.5-flash-latest": 1,
        # "google/gemini-1.5-pro-latest": 3,
    }
    
    model_quality = { # Higher is better (quality score 1-10)
        "openai/gpt-4o": 9,
        "openai/gpt-4o-mini": 7,
        "anthropic/claude-3-opus-20240229": 10,
        "anthropic/claude-3-sonnet-20240229": 8,
        "anthropic/claude-3-haiku-20240307": 6,
        "anthropic/claude-3-5-sonnet-20240620": 9,
        # "deepseek/deepseek-chat": 7,
        # "deepseek/deepseek-reasoner": 8,
        # "google/gemini-1.5-flash-latest": 5,
        # "google/gemini-1.5-pro-latest": 8,
    }
    # --- End Model Metadata --- 

    candidate_models_data = []
    excluded_models_reasons = {}
    # Use models available in COST_PER_MILLION_TOKENS as the source of truth for available models
    all_known_models = list(COST_PER_MILLION_TOKENS.keys())

    async def evaluate_model(model_name):
        # 1. Check capabilities
        capabilities = model_capabilities.get(model_name, [])
        missing_caps = [cap for cap in required_capabilities if cap not in capabilities]
        if missing_caps:
            excluded_models_reasons[model_name] = f"Missing required capabilities: {missing_caps}"
            return

        # 2. Estimate cost
        try:
            cost_estimate = await estimate_cost(
                prompt=sample_text,
                model=model_name,
                max_tokens=estimated_max_tokens,
                include_output=True
            )
            estimated_cost_value = cost_estimate["cost"]
        except ToolError as e:
            # Model exists in cost list but maybe not elsewhere? Log and exclude.
            excluded_models_reasons[model_name] = f"Cost estimation failed: {e.detail}"
            return
        except Exception as e:
            logger.error(f"Unexpected error estimating cost for {model_name} in recommendation: {e}", exc_info=True)
            excluded_models_reasons[model_name] = f"Cost estimation failed unexpectedly: {str(e)}"
            return

        # 3. Check max cost constraint
        if max_cost is not None and estimated_cost_value > max_cost:
            excluded_models_reasons[model_name] = f"Exceeds max cost (${estimated_cost_value:.6f} > ${max_cost:.6f})"
            return

        # 4. Gather data for scoring
        candidate_models_data.append({
            "model": model_name,
            "cost": estimated_cost_value,
            "quality": model_quality.get(model_name, 5), # Default quality if missing
            "speed": model_speed.get(model_name, 3),     # Default speed if missing
            "capabilities": capabilities
        })

    # Evaluate all known models concurrently
    await asyncio.gather(*(evaluate_model(model) for model in all_known_models))

    # --- Scoring Logic (Keep as defined before, adjust weights if needed) ---
    def calculate_score(model_data):
        cost = model_data['cost']
        quality = model_data['quality']
        speed = model_data['speed'] # Lower is better for speed score

        # Normalize cost (needs context of min/max cost among candidates)
        all_costs = [m['cost'] for m in candidate_models_data if m['cost'] > 0] # Avoid division by zero
        min_cost = min(all_costs) if all_costs else 0.000001
        max_cost = max(all_costs) if all_costs else 0.000001
        cost_range = max_cost - min_cost
        # Normalized score (1 is cheapest, 0 is most expensive) - handle zero range
        norm_cost_score = 1.0 - ((cost - min_cost) / cost_range) if cost_range > 0 else 1.0

        # Normalize quality (scale 1-10)
        norm_quality_score = quality / 10.0

        # Normalize speed (scale 1-5, lower is better, so invert)
        norm_speed_score = (5 - speed + 1) / 5.0 # Maps 1->1, 5->0.2

        # Calculate final score based on priority
        if priority == "cost":
            score = norm_cost_score * 0.7 + norm_quality_score * 0.15 + norm_speed_score * 0.15
        elif priority == "quality":
            score = norm_cost_score * 0.15 + norm_quality_score * 0.7 + norm_speed_score * 0.15
        elif priority == "speed":
            score = norm_cost_score * 0.15 + norm_quality_score * 0.15 + norm_speed_score * 0.7
        else: # balanced
            score = norm_cost_score * 0.34 + norm_quality_score * 0.33 + norm_speed_score * 0.33

        return score
    # --- End Scoring Logic --- 

    # Calculate scores for all candidates
    for model_data in candidate_models_data:
        model_data['score'] = calculate_score(model_data)

    # Sort candidates by score (highest first)
    sorted_candidates = sorted(candidate_models_data, key=lambda x: x['score'], reverse=True)

    # Format recommendations
    recommendations_list = []
    for cand in sorted_candidates:
        # Add a simple reason based on priority
        reason = f"High overall score ({cand['score']:.2f}) according to '{priority}' priority."
        if priority == 'cost' and cand['cost'] <= min(m['cost'] for m in candidate_models_data):
            reason = f"Lowest estimated cost (${cand['cost']:.6f}) and meets requirements."
        elif priority == 'quality' and cand['quality'] >= max(m['quality'] for m in candidate_models_data):
             reason = f"Highest quality score ({cand['quality']}/10) and meets requirements."
        elif priority == 'speed' and cand['speed'] <= min(m['speed'] for m in candidate_models_data):
             reason = f"Fastest speed score ({cand['speed']}/5) and meets requirements."

        recommendations_list.append({
            "model": cand['model'],
            "estimated_cost": cand['cost'],
            "quality_score": cand['quality'],
            "speed_score": cand['speed'],
            "capabilities": cand['capabilities'],
            "reason": reason
        })

    logger.info(f"Recommended models (priority: {priority}): {[r['model'] for r in recommendations_list]}")
    return {
        "recommendations": recommendations_list,
        "parameters": {
            "task_type": task_type,
            "expected_input_length": expected_input_length,
            "expected_output_length": estimated_output_length_chars,
            "required_capabilities": required_capabilities,
            "max_cost": max_cost,
            "priority": priority
        },
        "excluded_models": excluded_models_reasons
    }

@with_tool_metrics
@with_error_handling
async def execute_optimized_workflow(
    documents: Optional[List[str]] = None, # Make documents optional, workflow might not need them
    workflow: List[Dict[str, Any]] = None, # Require workflow definition
    max_concurrency: int = 5
) -> Dict[str, Any]:
    """Executes a predefined workflow consisting of multiple tool calls.

    Processes a list of documents (optional) through a sequence of stages defined in the workflow.
    Handles dependencies between stages (output of one stage as input to another) and allows
    for concurrent execution of independent stages or document processing within stages.

    Args:
        documents: (Optional) A list of input document strings. Required if the workflow references
                   'documents' as input for any stage.
        workflow: A list of dictionaries, where each dictionary defines a stage (a tool call).
                  Required keys per stage:
                  - `stage_id`: A unique identifier for this stage (e.g., "summarize_chunks").
                  - `tool_name`: The name of the tool function to call (e.g., "summarize_document").
                  - `params`: A dictionary of parameters to pass to the tool function.
                     Parameter values can be literal values (strings, numbers, lists) or references
                     to outputs from previous stages using the format `"${stage_id}.output_key"`
                     (e.g., `{"text": "${chunk_stage}.chunks"}`).
                     Special inputs: `"${documents}"` refers to the input `documents` list.
                  Optional keys per stage:
                  - `depends_on`: A list of `stage_id`s that must complete before this stage starts.
                  - `iterate_on`: The key from a previous stage's output list over which this stage
                                  should iterate (e.g., `"${chunk_stage}.chunks"`). The tool will be
                                  called once for each item in the list.
                  - `optimization_hints`: (Future use) Hints for model selection or cost saving for this stage.
        max_concurrency: (Optional) The maximum number of concurrent tasks (tool calls) to run.
                         Defaults to 5.

    Returns:
        A dictionary containing the results of all successful workflow stages:
        {
            "success": true,
            "results": {
                "chunk_stage": { "output": { "chunks": ["chunk1...", "chunk2..."] } },
                "summarize_chunks": { # Example of an iterated stage
                     "output": [
                         { "summary": "Summary of chunk 1..." },
                         { "summary": "Summary of chunk 2..." }
                     ]
                },
                "final_summary": { "output": { "summary": "Overall summary..." } }
            },
            "status": "Workflow completed successfully.",
            "total_processing_time": 15.8
        }
        or an error dictionary if the workflow fails:
        {
            "success": false,
            "results": { ... }, # Results up to the point of failure
            "status": "Workflow failed at stage 'stage_id'.",
            "error": "Error details from the failed stage...",
            "total_processing_time": 8.2
        }

    Raises:
        ToolInputError: If the workflow definition is invalid (missing keys, bad references,
                        circular dependencies - basic checks).
        ToolError: If a tool call within the workflow fails.
        Exception: For unexpected errors during workflow execution.
    """
    start_time = time.time()
    if not workflow or not isinstance(workflow, list):
        raise ToolInputError("'workflow' must be a non-empty list of stage dictionaries.")

    # --- Basic Workflow Validation --- (Could be expanded significantly)
    stage_ids = set()
    for i, stage in enumerate(workflow):
        if not all(k in stage for k in ["stage_id", "tool_name", "params"]):
            raise ToolInputError(f"Workflow stage {i} missing required keys (stage_id, tool_name, params).")
        if not isinstance(stage["params"], dict):
             raise ToolInputError(f"Stage '{stage['stage_id']}' params must be a dictionary.")
        if stage["stage_id"] in stage_ids:
            raise ToolInputError(f"Duplicate stage_id found: '{stage['stage_id']}'.")
        stage_ids.add(stage["stage_id"])
        depends_on = stage.get("depends_on", [])
        if not isinstance(depends_on, list):
            raise ToolInputError(f"Stage '{stage['stage_id']}' depends_on must be a list.")
        # Basic check for dependencies existing (won't catch circular deps here)
        # for dep_id in depends_on:
        #     if dep_id not in stage_ids: # This check is flawed as stages are processed linearly here
        #         pass # A full graph check is needed for proper validation
    # --- End Validation --- 

    # Dictionary to store results of each stage
    stage_results: Dict[str, Any] = {}
    # Set to keep track of completed stages
    completed_stages = set()
    # Dictionary to hold active asyncio tasks
    active_tasks: Dict[str, asyncio.Task] = {}

    # --- Tool Mapping --- (Dynamically import or map tool names to functions)
    # Ensure all tools listed in workflows are mapped here correctly.
    tool_functions = {
        "estimate_cost": estimate_cost,
        "compare_models": compare_models,
        "recommend_model": recommend_model,
        "chat_completion": chat_completion,
        "chunk_document": chunk_document,
        "summarize_document": summarize_document,
        "extract_json": extract_json,
        # RAG Tools
        "create_knowledge_base": create_knowledge_base,
        "add_documents": add_documents,
        "retrieve_context": retrieve_context,
        "generate_with_rag": generate_with_rag,
        # Add other tools as needed...
    }

    # --- Workflow Execution Logic --- 
    # This is a simplified execution loop. A robust implementation would use
    # a proper task graph/scheduler (e.g., using libraries like Prefect, Dagster,
    # or custom asyncio scheduling) to handle complex dependencies, retries, 
    # concurrency limits, and iteration correctly.

    stages_to_process = list(workflow) # Copy the list
    
    while stages_to_process or active_tasks:
        # Launch new tasks that are ready
        runnable_stages = []
        remaining_stages = []
        for stage in stages_to_process:
            stage_id = stage["stage_id"]
            dependencies = stage.get("depends_on", [])
            if all(dep in completed_stages for dep in dependencies):
                runnable_stages.append(stage)
            else:
                remaining_stages.append(stage)
        stages_to_process = remaining_stages

        for stage in runnable_stages:
            if len(active_tasks) >= max_concurrency:
                 stages_to_process.append(stage) # Re-queue if concurrency limit hit
                 continue # Check again later
                 
            stage_id = stage["stage_id"]
            tool_name = stage["tool_name"]
            params = stage["params"]
            iterate_on_ref = stage.get("iterate_on")

            if tool_name not in tool_functions:
                 # Handle error: Tool not found
                 error_msg = f"Workflow failed at stage '{stage_id}': Tool '{tool_name}' not found."
                 logger.error(error_msg)
                 # Need to cancel running tasks properly here
                 return { "success": False, "results": stage_results, "status": f"Workflow failed at stage '{stage_id}'.", "error": error_msg, "total_processing_time": time.time() - start_time }

            tool_func = tool_functions[tool_name]
            
            # Resolve parameters, handle iteration
            try:
                # Parameter resolution logic needs to be robust
                resolved_params, is_iteration, iteration_list = _resolve_params(stage_id, params, iterate_on_ref, stage_results, documents)
            except ValueError as e:
                 error_msg = f"Workflow failed at stage '{stage_id}': Parameter resolution error: {e}"
                 logger.error(error_msg)
                 return { "success": False, "results": stage_results, "status": f"Workflow failed at stage '{stage_id}'.", "error": error_msg, "total_processing_time": time.time() - start_time }

            # Create task(s)
            if is_iteration:
                # Create multiple tasks for iteration
                sub_tasks = []
                for i, item in enumerate(iteration_list):
                     # Inject the iterated item into parameters correctly
                     iter_params = _inject_iteration_item(resolved_params, item)
                     task_id = f"{stage_id}_iter_{i}" 
                     # Note: Concurrency limit needs careful handling with iteration
                     task = asyncio.create_task(tool_func(**iter_params), name=task_id)
                     sub_tasks.append(task)
                # Wrap sub-tasks in a single task representing the stage completion
                stage_task = asyncio.create_task(_gather_iteration_results(stage_id, sub_tasks), name=stage_id)
            else:
                # Single task for the stage
                stage_task = asyncio.create_task(tool_func(**resolved_params), name=stage_id)
            
            active_tasks[stage_id] = stage_task
            logger.info(f"Launched workflow stage '{stage_id}' (Tool: {tool_name}). Active tasks: {len(active_tasks)}")

        # Await completion of any task
        if not active_tasks:
             if stages_to_process: # Should not happen with correct dependency logic, but safety check
                 logger.warning("No active tasks, but stages remain. Potential deadlock or dependency issue.")
                 await asyncio.sleep(0.1) # Avoid busy-waiting
             continue

        done, pending = await asyncio.wait(active_tasks.values(), return_when=asyncio.FIRST_COMPLETED)
        
        for task in done:
             task_name = task.get_name()
             try:
                 result = await task # Get result or raise exception
                 stage_results[task_name] = {"output": result} # Store successful result
                 completed_stages.add(task_name)
                 logger.info(f"Workflow stage '{task_name}' completed successfully.")
             except Exception as e:
                 error_msg = f"Workflow failed at stage '{task_name}'. Error: {type(e).__name__}: {str(e)}"
                 logger.error(error_msg, exc_info=True)
                 # Optionally log traceback: logger.error(traceback.format_exc())
                 stage_results[task_name] = {"error": error_msg, "traceback": traceback.format_exc()} # Store error
                 # Terminate workflow on first error (or implement other strategies)
                 # Cancel pending tasks
                 for p_task in pending:
                     p_task.cancel()
                 for _a_task_id, a_task in active_tasks.items():
                      if not a_task.done(): 
                          a_task.cancel()
                 return { "success": False, "results": stage_results, "status": f"Workflow failed at stage '{task_name}'.", "error": error_msg, "total_processing_time": time.time() - start_time }
             finally:
                 # Remove task from active list regardless of outcome
                 if task_name in active_tasks:
                      del active_tasks[task_name]
        
        # Small sleep to prevent overly tight loop if no tasks complete immediately
        if not done:
             await asyncio.sleep(0.01)

    # If loop finishes, workflow completed successfully
    total_time = time.time() - start_time
    logger.success(f"Workflow completed successfully in {total_time:.2f}s")
    return {
        "success": True,
        "results": stage_results,
        "status": "Workflow completed successfully.",
        "total_processing_time": total_time
    }

# --- Helper functions for workflow execution --- 
# These need careful implementation for robustness

def _resolve_params(stage_id: str, params: Dict, iterate_on_ref: Optional[str], stage_results: Dict, documents: Optional[List[str]]) -> tuple[Dict, bool, Optional[List]]:
    """Resolves parameter values, handling references and iteration.
    Returns resolved_params, is_iteration, iteration_list.
    Raises ValueError on resolution errors.
    """
    resolved = {}
    is_iteration = False
    iteration_list = None
    iteration_param_name = None

    # Check for iteration first
    if iterate_on_ref:
         if not iterate_on_ref.startswith("${") or not iterate_on_ref.endswith("}"):
              raise ValueError(f"Invalid iterate_on reference format: '{iterate_on_ref}'")
         ref_key = iterate_on_ref[2:-1]
         
         if ref_key == "documents":
              if documents is None:
                   raise ValueError(f"Stage '{stage_id}' iterates on documents, but no documents were provided.")
              iteration_list = documents
         else:
              dep_stage_id, output_key = _parse_ref(ref_key)
              if dep_stage_id not in stage_results or "output" not in stage_results[dep_stage_id]:
                   raise ValueError(f"Dependency '{dep_stage_id}' for iteration not found or failed.")
              dep_output = stage_results[dep_stage_id]["output"]
              if not isinstance(dep_output, dict) or output_key not in dep_output:
                   raise ValueError(f"Output key '{output_key}' not found in dependency '{dep_stage_id}' for iteration.")
              iteration_list = dep_output[output_key]
              if not isinstance(iteration_list, list):
                  raise ValueError(f"Iteration target '{ref_key}' is not a list.")
         
         is_iteration = True
         # We still resolve other params, the iteration item is injected later
         logger.debug(f"Stage '{stage_id}' will iterate over {len(iteration_list)} items from '{iterate_on_ref}'")

    # Resolve individual parameters
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            ref_key = value[2:-1]
            if ref_key == "documents":
                 if documents is None:
                      raise ValueError(f"Parameter '{key}' references documents, but no documents provided.")
                 resolved[key] = documents
            else:
                dep_stage_id, output_key = _parse_ref(ref_key)
                if dep_stage_id not in stage_results or "output" not in stage_results[dep_stage_id]:
                    raise ValueError(f"Dependency '{dep_stage_id}' for parameter '{key}' not found or failed.")
                dep_output = stage_results[dep_stage_id]["output"]
                # Handle potential nested keys in output_key later if needed
                if not isinstance(dep_output, dict) or output_key not in dep_output:
                    raise ValueError(f"Output key '{output_key}' not found in dependency '{dep_stage_id}' for parameter '{key}'. Available keys: {list(dep_output.keys()) if isinstance(dep_output, dict) else 'N/A'}")
                resolved[key] = dep_output[output_key]
                # If this resolved param is the one we iterate on, store its name
                if is_iteration and iterate_on_ref == value:
                     iteration_param_name = key
        else:
            resolved[key] = value # Literal value
            
    # Validation: If iterating, one parameter must match the iterate_on reference
    if is_iteration and iteration_param_name is None:
         # This means iterate_on pointed to something not used directly as a param value
         # We need a convention here, e.g., assume the tool takes a list or find the param name
         # For now, let's assume the tool expects the *list* if iterate_on isn't directly a param value.
         # This might need refinement based on tool behavior. A clearer workflow definition could help.
         # Alternative: Raise error if iterate_on target isn't explicitly mapped to a param. 
         # logger.warning(f"Iteration target '{iterate_on_ref}' not directly mapped to a parameter in stage '{stage_id}'. Tool must handle list input.")
         # Let's require the iteration target to be mapped for clarity:
          raise ValueError(f"Iteration target '{iterate_on_ref}' must correspond to a parameter value in stage '{stage_id}'.")

    # Remove the iteration parameter itself from the base resolved params if iterating
    # It will be injected per-item later
    if is_iteration and iteration_param_name in resolved:
        del resolved[iteration_param_name] 
        resolved["_iteration_param_name"] = iteration_param_name # Store the name for injection

    return resolved, is_iteration, iteration_list

def _parse_ref(ref_key: str) -> tuple[str, str]:
    """Parses a reference like 'stage_id.output_key'"""
    parts = ref_key.split('.', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid reference format: '{ref_key}'. Expected 'stage_id.output_key'.")
    return parts[0], parts[1]

def _inject_iteration_item(base_params: Dict, item: Any) -> Dict:
     """Injects the current iteration item into the parameter dict."""
     injected_params = base_params.copy()
     iter_param_name = injected_params.pop("_iteration_param_name", None)
     if iter_param_name:
          injected_params[iter_param_name] = item
     else:
          # This case should be prevented by validation in _resolve_params
          logger.error("Cannot inject iteration item: Iteration parameter name not found in resolved params.")
          # Handle error appropriately, maybe raise
     return injected_params

async def _gather_iteration_results(stage_id: str, tasks: List[asyncio.Task]) -> List[Any]:
     """Gathers results from iteration sub-tasks. Raises exception if any sub-task failed."""
     results = []
     try:
          raw_results = await asyncio.gather(*tasks)
          # Assume each task returns the direct output dictionary
          results = list(raw_results) # gather preserves order
          logger.debug(f"Iteration stage '{stage_id}' completed with {len(results)} results.")
          return results
     except Exception:
          # If any sub-task failed, gather will raise the first exception
          logger.error(f"Iteration stage '{stage_id}' failed: One or more sub-tasks raised an error.", exc_info=True)
          # Cancel any remaining tasks in this iteration group if needed (gather might do this)
          for task in tasks:
               if not task.done(): 
                   task.cancel()
          raise # Re-raise the exception to fail the main workflow stage