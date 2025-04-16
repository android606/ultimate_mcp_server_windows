"""Meta tools for LLM Gateway including LLM instructions on tool usage."""
import json
import asyncio # Add asyncio
import time # Add time
import re # Add re
from typing import Any, Dict, Optional, List, Union, Tuple # Add List, Union, Tuple

# Remove BaseTool import if no longer needed
# from llm_gateway.tools.base import BaseTool 
from llm_gateway.tools.base import with_error_handling, with_tool_metrics
from llm_gateway.utils import get_logger
from llm_gateway.constants import Provider, COST_PER_MILLION_TOKENS # Add COST_PER_MILLION_TOKENS
from llm_gateway.core.providers.base import get_provider # Add get_provider
from llm_gateway.exceptions import ProviderError, ToolInputError, ToolExecutionError # Add ToolInputError
from llm_gateway.tools.completion import generate_completion # Add generate_completion import

logger = get_logger("llm_gateway.tools.meta")

# --- Standalone Tool Functions --- 

# Removed MetaTools class and _register_tools method

# Un-indented get_tool_info and removed @self.mcp.tool()
@with_tool_metrics
@with_error_handling
async def get_tool_info(
    tool_name: Optional[str] = None,
    ctx=None
) -> Dict[str, Any]:
    """
    Get information about available tools and their usage.
    
    This tool provides information about the tools available in the LLM Gateway.
    If a specific tool name is provided, detailed information about that tool will
    be returned. Otherwise, a list of all available tools will be returned.
    
    Args:
        tool_name: Name of the tool to get information about. If None, returns a list of all tools.
        ctx: Context object passed by the MCP server. Required to access the tool registry.
        
    Returns:
        Information about the specified tool or a list of all available tools.
    """
    # Robust context checking is essential here
    if ctx is None or not hasattr(ctx, 'request_context') or ctx.request_context is None or not hasattr(ctx.request_context, 'lifespan_context') or ctx.request_context.lifespan_context is None:
        logger.error("Context or lifespan_context is None or invalid in get_tool_info")
        return {
            "error": "Server context not available. Tool information cannot be retrieved.",
            "tools": []
        }

    # Get tools from registry via context
    lifespan_ctx = ctx.request_context.lifespan_context
    tools = lifespan_ctx.get("tools", {})
    
    if not tools:
         logger.warning("No tools found in lifespan context for get_tool_info.")
         return {
             "message": "No tools seem to be registered or available in the server context.",
             "tools": []
         }

    # Return list of all tools if no specific tool is requested
    if not tool_name:
        return {
            "tools": [
                {
                    "name": name,
                    # Attempt to get description safely
                    "description": tool_info.get("description", "") if isinstance(tool_info, dict) else "(No description available)"
                }
                for name, tool_info in tools.items()
            ]
        }
    
    # Check if requested tool exists
    if tool_name not in tools:
        return {
            "error": f"Tool '{tool_name}' not found",
            "available_tools": list(tools.keys())
        }
    
    # Get detailed information about requested tool
    tool_info = tools[tool_name]
    
    # Create result with basic information
    # Ensure tool_info is a dict before accessing .get()
    result = {
        "name": tool_name,
        "description": tool_info.get("description", "") if isinstance(tool_info, dict) else "(No description available)"
    }
    
    # Add more details if available (e.g., parameters)
    if isinstance(tool_info, dict) and "parameters" in tool_info:
         result["parameters"] = tool_info["parameters"]
            
    return result

# Un-indented get_llm_instructions and removed @self.mcp.tool()
@with_tool_metrics
@with_error_handling
async def get_llm_instructions(
    tool_name: Optional[str] = None,
    task_type: Optional[str] = None,
    ctx=None # Keep ctx for decorator compatibility, even if unused directly
) -> Dict[str, Any]:
    """
    Get LLM-specific instructions on how to use tools effectively.
    
    This tool provides guidance for LLMs on how to effectively use the tools
    provided by the LLM Gateway. It can provide general instructions or
    tool-specific instructions.
    
    Args:
        tool_name: Name of the tool to get instructions for. If None, returns general instructions.
        task_type: Type of task to get instructions for (e.g., "summarization", "extraction").
        ctx: Context object passed by the MCP server.
        
    Returns:
        Dictionary containing instructions for the requested tool or task.
    """
    # General instructions for all LLMs
    general_instructions = """
    # LLM Gateway Tool Usage Guidelines
    
    ## General Principles
    
    1. **Cost-Aware Delegation**: Always consider the cost implications of your tool calls. 
        Delegate to cheaper models when the task doesn't require your full capabilities.
        
    2. **Progressive Refinement**: Start with cheaper/faster models for initial processing, 
        then use more expensive models only if needed for refinement.
        
    3. **Chunked Processing**: When dealing with large documents, use the chunking tools 
        to break them into manageable pieces before processing.
        
    4. **Error Handling**: All tools return standardized error responses with error_code 
        and details fields. Check the "success" field to determine if the call succeeded.
        
    5. **Resource Management**: Use resource-related tools to create and manage persistent 
        resources like documents and embeddings.
    
    ## Tool Selection Guidelines
    
    - For **text generation**, use:
      - `generate_completion` for single responses
      - `chat_completion` for conversational responses
      - `stream_completion` for streaming responses
      
    - For **document processing**, use:
      - `chunk_document` to break documents into manageable pieces
      - `summarize_document` for document summarization
      - `extract_entities` for entity extraction
      
    - For **structured data**, use:
      - `extract_json` to extract structured JSON
      - `extract_table` to extract tabular data
      
    - For **semantic search**, use:
      - `create_embeddings` to generate embeddings
      - `semantic_search` to find similar content
      
    ## Provider Selection
    
    - **OpenAI**: Best for general-purpose tasks, strong JSON capabilities
    - **Anthropic**: Good for long-form content, nuanced reasoning
    - **Gemini**: Cost-effective for summarization and extraction
    - **DeepSeek**: Good performance for code-related tasks
    
    ## Parameter Tips
    
    - Use appropriate `temperature` values:
      - 0.0-0.3: Deterministic, factual responses
      - 0.4-0.7: Balanced creativity and coherence
      - 0.8-1.0: More creative, diverse outputs
      
    - Set appropriate `max_tokens` to control response length
    - Use `additional_params` for provider-specific parameters
    """
    
    # Define tool-specific instructions
    tool_instructions = {
        "generate_completion": """
        # Generate Completion Tool
        
        The `generate_completion` tool generates text based on a prompt using a specified provider.
        
        ## When to Use
        
        - Single, non-conversational completions
        - Tasks like summarization, translation, or text generation
        - When you need just one response to a prompt
        
        ## Best Practices
        
        - Be specific in your prompts for better results
        - Use lower temperatures (0.0-0.3) for factual tasks
        - Use higher temperatures (0.7-1.0) for creative tasks
        - Set `max_tokens` to control response length
        
        ## Provider Selection
        
        - OpenAI (default): Good general performance
        - Anthropic: Better for nuanced, careful responses
        - Gemini: Cost-effective, good performance
        - DeepSeek: Good for technical content
        
        ## Example Usage
        
        ```python
        # Basic usage
        result = await client.tools.generate_completion(
            prompt="Explain quantum computing in simple terms"
        )
        
        # With specific provider and parameters
        result = await client.tools.generate_completion(
            prompt="Translate to French: 'Hello, how are you?'",
            provider="gemini",
            model="gemini-2.0-flash-lite",
            temperature=0.3
        )
        ```
        
        ## Common Errors
        
        - Invalid provider name
        - Model not available for the provider
        - Token limit exceeded
        """
    }
    
    # Task-specific instructions
    task_instructions = {
        "summarization": """
        # Document Summarization Best Practices
        
        ## Recommended Approach
        
        1. **Chunk the document** first if it's large:
           ```python
           chunks = await client.tools.chunk_document(
               document=long_text,
               chunk_size=1000,
               method="semantic"
           )
           ```
        
        2. **Summarize each chunk** with a cost-effective model:
           ```python
           chunk_summaries = []
           for chunk in chunks["chunks"]:
               summary = await client.tools.generate_completion(
                   prompt=f"Summarize this text: {chunk}",
                   provider="gemini",
                   model="gemini-2.0-flash-lite"
               )
               chunk_summaries.append(summary["text"])
           ```
        
        3. **Combine chunk summaries** if needed:
           ```python
           final_summary = await client.tools.generate_completion(
               prompt=f"Combine these summaries into a coherent overall summary: {' '.join(chunk_summaries)}",
               provider="anthropic",
               model="claude-3-5-haiku-20241022"
           )
           ```
        
        ## Provider Recommendations
        
        - For initial chunk summaries: Gemini or gpt-4.1-mini
        - For final summary combination: Claude or GPT-4o
        
        ## Parameters
        
        - Use temperature 0.0-0.3 for factual summaries
        - Use temperature 0.4-0.7 for more engaging summaries
        """
    }
    
    # Return appropriate instructions
    if tool_name is not None:
        # Tool-specific instructions
        if tool_name in tool_instructions:
            return {"instructions": tool_instructions[tool_name]}
        else:
            # Get tool info
            tool_info = await get_tool_info(tool_name=tool_name)
            
            if "error" in tool_info:
                return {
                    "error": f"No specific instructions available for tool: {tool_name}",
                    "general_instructions": general_instructions
                }
            
            # Generate basic instructions from tool info
            basic_instructions = f"""
            # {tool_name} Tool
            
            ## Description
            
            {tool_info.get('description', 'No description available.')}
            
            ## Input Parameters
            
            """
            
            # Add parameters from schema if available
            if "input_schema" in tool_info and "properties" in tool_info["input_schema"]:
                for param_name, param_info in tool_info["input_schema"]["properties"].items():
                    param_desc = param_info.get("description", "No description")
                    param_type = param_info.get("type", "unknown")
                    basic_instructions += f"- **{param_name}** ({param_type}): {param_desc}\n"
            
            # Add examples if available
            if "examples" in tool_info and tool_info["examples"]:
                basic_instructions += "\n## Example Usage\n\n"
                for example in tool_info["examples"]:
                    basic_instructions += f"### {example.get('name', 'Example')}\n\n"
                    basic_instructions += f"{example.get('description', '')}\n\n"
                    basic_instructions += f"Input: {json.dumps(example.get('input', {}), indent=2)}\n\n"
                    basic_instructions += f"Output: {json.dumps(example.get('output', {}), indent=2)}\n\n"
            
            return {"instructions": basic_instructions}
    
    elif task_type is not None:
        # Task-specific instructions
        if task_type in task_instructions:
            return {"instructions": task_instructions[task_type]}
        else:
            return {
                "error": f"No specific instructions available for task type: {task_type}",
                "general_instructions": general_instructions
            }
    
    else:
        # General instructions
        return {"instructions": general_instructions}

# Un-indented get_tool_recommendations and removed @self.mcp.tool()
@with_tool_metrics
@with_error_handling
async def get_tool_recommendations(
    task: str,
    constraints: Optional[Dict[str, Any]] = None,
    ctx=None # Pass ctx along for get_tool_info
) -> Dict[str, Any]:
    """
    Get recommendations for tool and provider selection based on a specific task.
    
    This tool analyzes the described task and provides recommendations on which
    tools and providers to use, along with a suggested workflow.
    
    Args:
        task: Description of the task to be performed (e.g., "summarize a document", 
             "extract entities from text").
        constraints: Optional constraints (e.g., {"max_cost": 0.01, "priority": "speed"}).
        ctx: Context object passed by the MCP server.
        
    Returns:
        Dictionary containing tool and provider recommendations, along with a workflow.
    """
    constraints = constraints or {}
    
    # Get information about available tools using the refactored function
    # Pass the context received by this function
    tools_info = await get_tool_info(ctx=ctx) 
    
    # Handle potential error from get_tool_info if context was bad
    if "error" in tools_info:
        logger.warning(f"Could not get tool info for recommendations: {tools_info['error']}")
        return {
             "error": "Could not retrieve tool information needed for recommendations. Server context might be unavailable.",
             "message": "Cannot provide recommendations without knowing available tools."
         }
         
    available_tools = [t["name"] for t in tools_info.get("tools", [])]
    if not available_tools:
        logger.warning("No available tools found by get_tool_info for recommendations.")
        return {
            "error": "No tools appear to be available.",
            "message": "Cannot provide recommendations as no tools were found."
        }

    # Task-specific recommendations
    task_lower = task.lower()
    
    # Dictionary of task patterns and recommendations
    task_patterns = {
        "summar": {
            "task_type": "summarization",
            "tools": [
                {"tool": "chunk_document", "reason": "Break the document into manageable chunks"},
                {"tool": "summarize_document", "reason": "Summarize document content efficiently"}
            ],
            "providers": [
                {"provider": "gemini", "model": "gemini-2.0-flash-lite", "reason": "Most cost-effective for summarization"},
                {"provider": "openai", "model": "gpt-4.1-mini", "reason": "Good balance of quality and cost"}
            ],
            "workflow": """
            1. First chunk the document with `chunk_document`
            2. Summarize each chunk with `summarize_document` using Gemini
            3. For the final combined summary, use `generate_completion` with a more capable model if needed
            """
        },
        "extract": {
            "task_type": "extraction",
            "tools": [
                {"tool": "extract_entities", "reason": "Extract named entities from text"},
                {"tool": "extract_json", "reason": "Extract structured data in JSON format"}
            ],
            "providers": [
                {"provider": "openai", "model": "gpt-4.1-mini", "reason": "Excellent at structured extraction"},
                {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "reason": "Good balance of accuracy and cost"}
            ],
            "workflow": """
            1. First determine the schema for extraction
            2. Use `extract_json` with OpenAI models for structured extraction
            3. For specific entity types, use `extract_entities` as an alternative
            """
        },
        "translate": {
            "task_type": "translation",
            "tools": [
                {"tool": "generate_completion", "reason": "Simple translation tasks"},
                {"tool": "batch_process", "reason": "For translating multiple texts"}
            ],
            "providers": [
                {"provider": "gemini", "model": "gemini-2.0-flash-lite", "reason": "Cost-effective for translations"},
                {"provider": "deepseek", "model": "deepseek-chat", "reason": "Good performance for technical content"}
            ],
            "workflow": """
            1. For simple translations, use `generate_completion` with Gemini
            2. For batch translations, use `batch_process` to handle multiple texts efficiently
            """
        },
        "search": {
            "task_type": "semantic_search",
            "tools": [
                {"tool": "create_embeddings", "reason": "Generate embeddings for search"},
                {"tool": "semantic_search", "reason": "Search using semantic similarity"}
            ],
            "providers": [
                {"provider": "openai", "model": "text-embedding-ada-002", "reason": "High-quality embeddings"},
                {"provider": "openai", "model": "gpt-4.1-mini", "reason": "For processing search results"}
            ],
            "workflow": """
            1. First create embeddings with `create_embeddings`
            2. Perform semantic search with `semantic_search`
            3. Process and enhance results with a completion if needed
            """
        }
    }
    
    # Find matching task pattern
    matching_pattern = None
    for pattern, recommendations in task_patterns.items():
        if pattern in task_lower:
            matching_pattern = recommendations
            break
    
    # If no specific pattern matches, provide general recommendations
    if matching_pattern is None:
        return {
            "message": "No specific recommendations available for this task.",
            "general_recommendations": {
                "tools": [
                    {"tool": "generate_completion", "reason": "General text generation"},
                    {"tool": "chat_completion", "reason": "Conversational interactions"}
                ],
                "providers": [
                    {"provider": "openai", "model": "gpt-4o", "reason": "High-quality general purpose"},
                    {"provider": "gemini", "model": "gemini-2.0-flash-lite", "reason": "Cost-effective alternative"}
                ],
                "note": "For more specific recommendations, try describing your task in more detail."
            }
        }
    
    # Apply constraints if provided
    if "max_cost" in constraints:
        max_cost = constraints["max_cost"]
        # Adjust provider recommendations based on cost
        if max_cost < 0.005:  # Very low cost
            matching_pattern["providers"] = [
                {"provider": "gemini", "model": "gemini-2.0-flash-lite", "reason": "Lowest cost option"},
                {"provider": "deepseek", "model": "deepseek-chat", "reason": "Low cost alternative"}
            ]
    
    # Get instructions for this task type
    task_type_str = matching_pattern.get("task_type")
    task_instructions = await get_llm_instructions(task_type=task_type_str, ctx=ctx) # Pass ctx
    
    # Return recommendations
    result = {
        "task_type": matching_pattern.get("task_type", "general"),
        "recommended_tools": [
            tool for tool in matching_pattern.get("tools", [])
            if tool["tool"] in available_tools
        ],
        "recommended_providers": matching_pattern.get("providers", []),
        "workflow": matching_pattern.get("workflow", "No specific workflow available.")
    }
    
    # Add instructions if available
    if "instructions" in task_instructions:
        result["detailed_instructions"] = task_instructions["instructions"]
    
    return result

# --- Standalone multi_completion Tool --- 
@with_tool_metrics
@with_error_handling
async def multi_completion(
    prompt: str,
    configs: List[Dict[str, Any]],
    timeout: Optional[float] = 60.0
) -> Dict[str, Any]:
    """Generates completions from multiple models/providers in parallel.

    Args:
        prompt: The prompt to send to all models.
        configs: List of configurations for each model/provider.
                 Each config dict should have at least 'provider'. 
                 Optional keys: 'model', 'parameters' (dict for temp, max_tokens, etc.).
                 Example: `[{"provider": "openai", "model": "gpt-4.1-mini", "parameters": {"temperature": 0.5}}]`
        timeout: Timeout for each individual completion operation in seconds. Defaults to 60.0.

    Returns:
        Dictionary containing completions from each model.
        {
            "completions": [
                { "provider": "openai", "model": "openai/gpt-4.1-mini", "text": "...", ... },
                { "provider": "anthropic", "model": "anthropic/claude-3-5-haiku-20241022", "error": "Timeout", ... }
            ],
            "total_cost": 0.00123,
            "total_tokens": 1500,
            "success_count": 1,
            "processing_time": 5.2
        }
    """
    start_time = time.time()
    
    if not configs:
        raise ToolInputError("No model configurations provided.")
    
    logger.info(
        f"Generating multi-completions with {len(configs)} configurations",
        emoji_key="meta",
        configs_count=len(configs)
    )
    
    async def execute_single_completion(config: Dict[str, Any]) -> Dict[str, Any]:
        provider_name = config.get("provider")
        model_name = config.get("model") # Can be None
        params = config.get("parameters", {}) # Gets temp, max_tokens, etc.
        
        if not provider_name:
             return { "provider": "unknown", "model": model_name, "error": "Provider name missing in config." }

        result_data = {
            "provider": provider_name,
            "model": f"{provider_name}/{model_name}" if model_name else f"{provider_name}/default",
            "text": None,
            "tokens": None,
            "cost": 0.0,
            "error": None,
            "processing_time": 0.0
        }
        single_start_time = time.time()
        try:
            # Call the imported generate_completion directly
            completion_result = await generate_completion(
                prompt=prompt,
                provider=provider_name,
                model=model_name,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens"),
                additional_params=params.get("additional_params")
            )
            result_data["processing_time"] = time.time() - single_start_time
            
            # Check the nested success flag from generate_completion's potential error wrapper
            if completion_result.get("success", False):
                 result_data.update({
                     "text": completion_result.get("text"),
                     "model": completion_result.get("model"), # Use actual model returned
                     "tokens": completion_result.get("tokens"),
                     "cost": completion_result.get("cost", 0.0),
                 })
            else:
                 # If generate_completion failed, its error is in the result dict
                 result_data["error"] = completion_result.get("error", "Unknown completion error")
                 result_data["model"] = completion_result.get("model") # Capture model even on failure
                 
        except asyncio.TimeoutError:
             result_data["processing_time"] = time.time() - single_start_time
             result_data["error"] = f"Operation timed out after {timeout}s"
             logger.warning(f"Timeout for {result_data['model']} in multi_completion")
        except Exception as e:
            result_data["processing_time"] = time.time() - single_start_time
            error_msg = f"Error in multi_completion sub-task ({result_data['model']}): {type(e).__name__}: {str(e)}"
            logger.error(error_msg, emoji_key="error")
            result_data["error"] = error_msg
            # Try to capture actual model from error if available (e.g., ProviderError)
            if hasattr(e, 'model') and e.model:
                 result_data["model"] = e.model
                 
        return result_data
        
    tasks = [execute_single_completion(config) for config in configs]
    completions = []
    try:
        # Use the timeout for the gather operation itself
        # Note: timeout in execute_single_completion is not needed if gather has timeout
        # Keeping the inner timeout provides per-task timeout, gather provides overall
        # Let's rely on the overall timeout for simplicity here.
        # We pass None to wait_for to disable the outer timeout if timeout arg is None
        if timeout is not None and timeout > 0:
             completions = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        else:
             completions = await asyncio.gather(*tasks) # Run without overall timeout

    except asyncio.TimeoutError:
         logger.error(f"Multi-completion gather operation timed out after {timeout}s", emoji_key="error")
         # Need to handle partial results if some tasks completed before timeout
         # For now, return an error message, potentially losing partial results
         raise ToolExecutionError(f"Multi-completion operation timed out after {timeout}s")

    total_cost = sum(c.get("cost", 0.0) for c in completions if c and c.get("error") is None)
    # Safely sum total tokens
    total_tokens = 0
    for c in completions:
         if c and c.get("error") is None:
              tokens_dict = c.get("tokens")
              if isinstance(tokens_dict, dict):
                   total_tokens += tokens_dict.get("total", 0)
                   
    processing_time = time.time() - start_time
    success_count = sum(1 for c in completions if c and c.get("error") is None)
    
    logger.success(
        f"Generated {success_count}/{len(configs)} multi-completions successfully",
        time=processing_time,
        cost=total_cost
    )
    
    return {
        "completions": completions,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "success_count": success_count,
        "processing_time": processing_time,
        "success": success_count > 0 # Consider success if at least one completed
    }

# --- analyze_task Tool and Helpers (Adapted from old code) ---

def _analyze_task_type(task_description: str) -> str:
    """Analyze task description to determine task type."""
    task_lower = task_description.lower()
    if any(kw in task_lower for kw in ["summarize", "summary", "summarization", "tldr"]):
        return "summarization"
    if any(kw in task_lower for kw in ["extract", "identify", "find all", "list the", "parse"]):
        return "extraction"
    if any(kw in task_lower for kw in ["classify", "categorize", "what type of", "which category"]):
        return "classification"
    if any(kw in task_lower for kw in ["translate", "translation", "convert to language"]):
        return "translation"
    if any(kw in task_lower for kw in ["write story", "create article", "generate content", "creative"]):
        return "creative_writing"
    if any(kw in task_lower for kw in ["write code", "implement function", "create program", "coding"]):
        return "coding"
    if any(kw in task_lower for kw in ["reason", "analyze", "evaluate", "assess", "interpret"]):
        return "reasoning"
    if any(kw in task_lower for kw in ["chat", "conversation", "discuss", "talk", "respond"]):
        return "conversation"
    return "general"

def _analyze_required_features(task_description: str) -> Tuple[List[str], str]:
    """Analyze task description to determine required features."""
    task_lower = task_description.lower()
    required_features = []
    explanation = ""
    feature_map = {
        "reasoning": ["reason", "reasoning", "analyze", "complex", "nuanced"],
        "coding": ["code", "function", "programming", "implement", "algorithm"],
        "math": ["math", "calculate", "computation", "formula", "equation"],
        "knowledge": ["knowledge", "facts", "information", "domain"],
        "instruction-following": ["instruction", "specific format", "follow", "adhere", "precise"],
        "creativity": ["creative", "imagination", "original", "innovative", "novel"],
        "complex-reasoning": ["multi-step", "complex", "sophisticated", "difficult"],
    }
    explanations = {
        "reasoning": "Task requires reasoning capabilities for analysis and complex logic. ",
        "coding": "Task involves code generation or programming knowledge. ",
        "math": "Task requires mathematical computation or understanding. ",
        "knowledge": "Task requires factual knowledge or domain expertise. ",
        "instruction-following": "Task requires precise instruction following or specific output formatting. ",
        "creativity": "Task requires creative or original thinking. ",
        "complex-reasoning": "Task involves multi-step or complex reasoning. "
    }
    for feature, keywords in feature_map.items():
        if any(kw in task_lower for kw in keywords):
            required_features.append(feature)
            explanation += explanations[feature]
    if not required_features:
        required_features.append("instruction-following")
        explanation = "Task requires basic instruction following."
    return required_features, explanation.strip()

async def _get_provider_options(
    required_features: List[str],
    available_providers: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Get provider options based on required features."""
    provider_options = {}
    # TODO: Centralize this model capability mapping
    model_capabilities = {
        "openai/gpt-4o": ["reasoning", "coding", "knowledge", "instruction-following", "math", "creativity", "complex-reasoning"],
        "openai/gpt-4.1-mini": ["reasoning", "coding", "knowledge", "instruction-following", "creativity"],
        "anthropic/claude-3-opus-20240229": ["reasoning", "coding", "knowledge", "instruction-following", "math", "creativity", "complex-reasoning"],
        "anthropic/claude-3-sonnet-20240229": ["reasoning", "coding", "knowledge", "instruction-following", "creativity"],
        "anthropic/claude-3-5-haiku-20241022": ["knowledge", "instruction-following"],
        "anthropic/claude-3-7-sonnet-20250219": ["reasoning", "coding", "knowledge", "instruction-following", "math", "creativity", "complex-reasoning"],
        "deepseek/deepseek-chat": ["coding", "knowledge", "instruction-following"],
        "deepseek/deepseek-coder": ["coding", "instruction-following"],
        "google/gemini-1.5-flash-latest": ["knowledge", "instruction-following"],
        "google/gemini-1.5-pro-latest": ["reasoning", "knowledge", "instruction-following", "math", "creativity"],
    }
    
    for provider_name in available_providers:
        try:
            provider_instance = await get_provider(provider_name)
            # Ensure model listing works even if provider not fully initialized elsewhere
            # await provider_instance.initialize() # get_provider should handle init now
            models = await provider_instance.list_models()
            suitable_models = []
            for model_info in models:
                model_id = model_info.get("id") # Includes provider prefix from list_models
                if not model_id: continue
                
                caps = model_capabilities.get(model_id, [])
                if all(feat in caps for feat in required_features):
                    suitable_models.append({
                        "id": model_id,
                        "provider": provider_name,
                        "description": model_info.get("description", ""),
                        "capabilities": caps,
                    })
            if suitable_models:
                provider_options[provider_name] = suitable_models
        except Exception as e:
            logger.error(f"Failed to get models for provider {provider_name}: {str(e)}", emoji_key="error")
            
    return provider_options

def _analyze_cost(
    provider_options: Dict[str, List[Dict[str, Any]]],
    input_tokens: int,
    output_tokens: int
) -> Dict[str, Any]:
    """Analyze cost for task execution based on token estimates."""
    cost_estimates = {}
    for provider_name, models in provider_options.items():
        provider_costs = []
        for model_info in models:
            model_id = model_info["id"]
            cost_data = COST_PER_MILLION_TOKENS.get(model_id)
            if cost_data:
                input_cost = (input_tokens / 1_000_000) * cost_data["input"]
                output_cost = (output_tokens / 1_000_000) * cost_data["output"]
                provider_costs.append({
                    "model": model_id,
                    "cost": input_cost + output_cost,
                })
        if provider_costs:
            cost_estimates[provider_name] = sorted(provider_costs, key=lambda x: x["cost"])
            
    all_costs = [m_cost for p_costs in cost_estimates.values() for m_cost in p_costs]
    lowest = min(all_costs, key=lambda x: x["cost"]) if all_costs else None
    highest = max(all_costs, key=lambda x: x["cost"]) if all_costs else None
    
    return {
        "lowest_cost_model": lowest["model"] if lowest else None,
        "lowest_cost_value": lowest["cost"] if lowest else None,
        "highest_cost_model": highest["model"] if highest else None,
        "highest_cost_value": highest["cost"] if highest else None,
        "details": cost_estimates
    }

def _generate_recommendations(
    provider_options: Dict[str, List[Dict[str, Any]]],
    cost_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate recommendations based on provider options and cost analysis."""
    recommendations = {
        "lowest_cost": None,
        "best_quality": None, # Simplified, just pick one high-quality one
        "balanced": None,
    }
    # TODO: Centralize quality/speed scores
    model_quality = {
        "openai/gpt-4o": 9.5, "openai/gpt-4.1-mini": 8.5,
        "anthropic/claude-3-opus-20240229": 9.5, "anthropic/claude-3-sonnet-20240229": 9.0,
        "anthropic/claude-3-5-haiku-20241022": 8.0, "anthropic/claude-3-7-sonnet-20250219": 9.5,
        "deepseek/deepseek-chat": 7.0, "deepseek/deepseek-coder": 8.0,
        "google/gemini-1.5-pro-latest": 9.0, "google/gemini-1.5-flash-latest": 7.5
    }
    all_models = []
    for provider, models in provider_options.items():
        for model in models:
            model_id = model["id"]
            cost_data = next((m for m in cost_analysis.get("details", {}).get(provider, []) if m["model"] == model_id), None)
            all_models.append({
                "provider": provider,
                "model": model_id,
                "quality": model_quality.get(model_id, 7.0),
                "cost": cost_data["cost"] if cost_data else float('inf'),
                "capabilities": model["capabilities"]
            })
            
    if not all_models: return recommendations

    all_models.sort(key=lambda x: x["cost"]) # Sort by cost primarily
    recommendations["lowest_cost"] = all_models[0]
    
    quality_sorted = sorted(all_models, key=lambda x: x["quality"], reverse=True)
    recommendations["best_quality"] = quality_sorted[0]

    # Simple balanced: lowest cost model with quality >= 8.5, or fallback to lowest cost
    balanced = next((m for m in all_models if m["quality"] >= 8.5), None)
    recommendations["balanced"] = balanced if balanced else all_models[0]
    
    return recommendations

@with_tool_metrics
@with_error_handling
async def analyze_task(
    task_description: str,
    available_providers: Optional[List[str]] = None,
    analyze_features: bool = True,
    analyze_cost: bool = True,
    # Add token estimation params if cost analysis needs them
    estimated_input_tokens: Optional[int] = None,
    estimated_output_tokens: Optional[int] = None,
    ctx=None # Add ctx for potential future use or consistency
) -> Dict[str, Any]:
    """Analyze a task and recommend the best provider and model based on requirements."""
    start_time = time.time()
    
    if available_providers is None:
        available_providers = [p.value for p in Provider] # Use enum values
    
    task_type = _analyze_task_type(task_description)
    required_features, features_explanation = [], ""
    if analyze_features:
        required_features, features_explanation = _analyze_required_features(task_description)
    
    # Get provider options based on capabilities
    provider_options = await _get_provider_options(required_features, available_providers)
    
    cost_analysis = {}
    if analyze_cost:
        # Estimate tokens if not provided (very rough estimate based on description length)
        input_tokens = estimated_input_tokens if estimated_input_tokens else len(task_description) // 3
        output_tokens = estimated_output_tokens if estimated_output_tokens else input_tokens // 2
        cost_analysis = _analyze_cost(provider_options, input_tokens, output_tokens)
    
    recommendations = _generate_recommendations(provider_options, cost_analysis)
    
    processing_time = time.time() - start_time
    logger.success(f"Task analysis completed: {task_type}", time=processing_time)
    
    return {
        "task_type": task_type,
        "required_features": required_features,
        "features_explanation": features_explanation,
        "providers": provider_options, # List suitable providers/models
        "cost_analysis": cost_analysis,
        "recommendations": recommendations,
        "processing_time": processing_time,
    }

# --- Helper Functions for compare_and_synthesize ---

def _get_criteria_definitions(criteria: List[str]) -> str:
    """
    Generate detailed definitions for each evaluation criterion.
    """
    criteria_descriptions = {
        "factual_accuracy": "Assess whether the information provided is correct, verifiable, and free from errors or misleading statements. Check if claims align with established knowledge and whether appropriate qualifiers are used for uncertain information.",
        
        "completeness": "Evaluate if the response addresses all aspects of the prompt and provides sufficient depth. A complete response leaves no important questions unanswered and provides context where needed.",
        
        "relevance": "Determine how well the response addresses the specific query or task in the prompt. A relevant response stays focused on the user's needs without unnecessary tangents.",
        
        "coherence": "Assess the logical flow and structure of the response. Look for clear organization, smooth transitions between ideas, and a consistent narrative that's easy to follow.",
        
        "depth_of_reasoning": "Evaluate the sophistication of analysis and logical reasoning. Higher scores indicate nuanced thinking, consideration of multiple perspectives, and well-supported conclusions.",
        
        "clarity": "Assess how easy the response is to understand. Clear responses use precise language, explain complex concepts appropriately, and avoid jargon unless necessary.",
        
        "creativity": "Evaluate originality, innovative thinking, and novel approaches in the response. Consider whether the response introduces fresh perspectives or solutions.",
        
        "practical_utility": "Assess how useful and actionable the information is for the user's likely purpose. Consider whether the response provides practical guidance that can be implemented.",
        
        "conciseness": "Evaluate whether the response is appropriately brief while still being complete. Lower scores indicate unnecessary verbosity or repetition.",
        
        "tone_appropriateness": "Assess whether the style, formality level, and emotional tone match what would be appropriate for the context of the request.",
        
        "safety": "Evaluate whether the response adheres to ethical guidelines, avoids harmful content, and maintains appropriate boundaries.",
    }
    
    definitions = []
    for criterion in criteria:
        if criterion in criteria_descriptions:
            definitions.append(f"- **{criterion}**: {criteria_descriptions[criterion]}")
        else:
            # Generic definition for custom criteria
            definitions.append(f"- **{criterion}**: Evaluate the response based on {criterion}.")
    
    return "\n".join(definitions)

def _get_synthesis_strategy_description(strategy: str) -> str:
    """
    Generate description for the requested synthesis strategy.
    """
    if strategy == "comprehensive":
        return "Create a thorough synthesis that combines the strongest elements from all responses, integrating different perspectives and insights to produce a more complete and nuanced answer than any individual response."
    
    elif strategy == "conservative":
        return "Prioritize accuracy and reliability. Only include information that appears consistently across multiple responses or is provided by the most reliable source. Explicitly acknowledge uncertainties and avoid speculative content."
    
    elif strategy == "creative":
        return "Build upon the insights from all responses to generate novel connections and ideas that weren't present in any individual response. The synthesis should extend beyond the original responses while maintaining accuracy."
    
    else:
        return "Create a balanced synthesis that accurately represents the information from all responses."

def _get_format_specific_instructions(
    response_format: str,
    synthesis_strategy: str,
    include_reasoning: bool,
    criteria: List[str]
) -> str:
    """
    Generate detailed instructions based on the requested response format.
    """
    # Base scores structure that's common across formats
    scores_json = ", ".join([f'"{criterion}": <score 1-10>' for criterion in criteria])
    
    if response_format == "best":
        instructions = f"""Carefully evaluate each response based on the provided criteria. Then select the BEST overall response.

Your analysis should follow these steps:
1. Evaluate each response individually against all criteria, assigning scores from 1-10
2. Consider the weights of different criteria when calculating overall scores
3. Identify key strengths and weaknesses of each response
4. Determine which response performs best overall, considering both average scores and crucial criteria
5. Provide clear reasoning for your selection

Your response MUST be in valid JSON format with the following structure:
{{
"evaluations": [
    {{
    "response_index": 1,
    "provider": "provider_name",
    "model": "model_name",
    "scores": {{ {scores_json} }},
    "weighted_average": <calculated_weighted_average>,
    "strengths": ["specific strength 1", "specific strength 2"...],
    "weaknesses": ["specific weakness 1", "specific weakness 2"...]
    }},
    ...
],
"best_response": {{
    "response_index": <best_index>,
    "provider": "provider_name",
    "model": "model_name",
    "reasoning": "<detailed explanation of why this response was selected as best>"
}},
"best_response_text": "<the full text of the best response>"
}}"""

        if not include_reasoning:
            instructions = instructions.replace('"reasoning": "<detailed explanation of why this response was selected as best>"', '"reasoning": null')
            
    elif response_format == "synthesis":
        instructions = f"""Carefully evaluate each response against the provided criteria. Then create a new synthesized response that combines the best elements of all responses using the {synthesis_strategy} strategy.

Your analysis should follow these steps:
1. Evaluate each response individually against all criteria, assigning scores from 1-10
2. Identify the unique strengths, insights, and valuable content from each response
3. Create a cohesive new response that integrates the best elements according to the {synthesis_strategy} strategy
4. Ensure your synthesized response is coherent, well-structured, and addresses the original prompt effectively

Your response MUST be in valid JSON format with the following structure:
{{
"evaluations": [
    {{
    "response_index": 1,
    "provider": "provider_name",
    "model": "model_name",
    "scores": {{ {scores_json} }},
    "weighted_average": <calculated_weighted_average>,
    "key_contributions": ["specific insight or strength that was incorporated into synthesis", ...]
    }},
    ...
],
"synthesis_strategy": "<explanation of how you combined the responses>",
"synthesized_response": "<the full synthesized response that combines the best elements according to the strategy>",
"metadata": {{
    "agreement_level": "<high/medium/low> - how consistent the responses were",
    "key_disagreements": ["specific point of disagreement 1", ...],
    "confidence": <1-10> - confidence in the quality of the synthesis
}}
}}"""

        if not include_reasoning:
            instructions = instructions.replace('"synthesis_strategy": "<explanation of how you combined the responses>"', '"synthesis_strategy": null')
            
    elif response_format == "ranked":
        instructions = f"""Carefully evaluate each response against the provided criteria. Then rank all responses from best to worst.

Your analysis should follow these steps:
1. Evaluate each response individually against all criteria, assigning scores from 1-10
2. Calculate weighted average scores based on the criteria weights
3. Rank the responses from highest to lowest overall quality
4. For each response, identify key strengths and weaknesses
5. Provide brief reasoning for each ranking position

Your response MUST be in valid JSON format with the following structure:
{{
"evaluations": [
    {{
    "response_index": 1,
    "provider": "provider_name",
    "model": "model_name",
    "scores": {{ {scores_json} }},
    "weighted_average": <calculated_weighted_average>,
    "strengths": ["specific strength 1", "specific strength 2"...],
    "weaknesses": ["specific weakness 1", "specific weakness 2"...]
    }},
    ...
],
"ranking": [
    {{
    "rank": 1,
    "response_index": <best_index>,
    "provider": "provider_name",
    "model": "model_name",
    "reasoning": "<explanation of why this response received this rank>"
    }},
    ...
]
}}"""

        if not include_reasoning:
            instructions = instructions.replace('"reasoning": "<explanation of why this response received this rank>"', '"reasoning": null')
            
    else:  # analysis
        instructions = f"""Carefully evaluate each response against the provided criteria, but do NOT select a winner or create a synthesis. Instead, provide a detailed comparative analysis.

Your analysis should follow these steps:
1. Evaluate each response individually against all criteria, assigning scores from 1-10
2. Identify patterns, similarities, and differences across the responses
3. Analyze the unique approaches, strengths, and limitations of each response
4. Assess what this comparison reveals about different approaches to the prompt

Your response MUST be in valid JSON format with the following structure:
{{
"evaluations": [
    {{
    "response_index": 1,
    "provider": "provider_name",
    "model": "model_name",
    "scores": {{ {scores_json} }},
    "weighted_average": <calculated_weighted_average>,
    "key_characteristics": ["notable characteristic 1", "notable characteristic 2"...]
    }},
    ...
],
"comparative_analysis": {{
    "patterns": ["observed pattern 1", "observed pattern 2"...],
    "differences": ["key difference 1", "key difference 2"...],
    "strengths_distribution": {{ "<criterion>": "description of how models performed on this criterion" }},
    "insights": ["analytical insight 1", "analytical insight 2"...]
}},
"metadata": {{
    "agreement_level": "<high/medium/low> - how consistent the responses were",
    "key_disagreements": ["specific point of disagreement 1", ...],
    "most_challenging_criteria": ["criterion that showed greatest variance", ...]
}}
}}"""
    
    return instructions

def _parse_synthesis_response(text: str) -> Union[Dict[str, Any], str]:
    """
    Parse the synthesis response, handling various edge cases.
    """
    # Try direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON with regex
    json_match = re.search(r'(\{[\s\S]*\})', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # If we can't parse JSON, return the raw text with error indication
    return {
        "error": "Failed to parse synthesis JSON",
        "raw_text": text
    }

async def _execute_comparison_synthesis(
    prompt: str,
    configs: List[Dict[str, Any]],
    criteria: List[str],
    criteria_weights: Dict[str, float],
    synthesis_model: Optional[Dict[str, Any]],
    response_format: str,
    synthesis_strategy: str,
    include_reasoning: bool,
    ctx: Any # Add ctx to access MCP
) -> Dict[str, Any]:
    """
    Internal method to execute the comparison and synthesis.
    """
    try:
        # Remove context/MCP checks as we call local functions directly
        # if ctx is None or not hasattr(ctx, 'mcp') or ctx.mcp is None:
        #      raise ValueError("MCP context (ctx.mcp) is required for calling sub-tools")
             
        # Generate completions from multiple models by calling local multi_completion
        multi_result = await multi_completion(
            prompt=prompt,
            configs=configs
            # Pass timeout if multi_completion supports it, or handle overall timeout
        )
        
        # Check the structure returned by our new multi_completion
        if not isinstance(multi_result, dict) or not multi_result.get("success", False):
             error_detail = multi_result.get("error", "multi_completion tool failed without specific error") if isinstance(multi_result, dict) else "multi_completion returned invalid format"
             logger.error(f"multi_completion call failed: {error_detail}")
             return { "error": f"Failed to get initial completions: {error_detail}" }

        completions = multi_result.get("completions", [])
        
        if not completions:
            return {
                "error": "No completions generated by multi_completion tool",
            }
            
        # Select synthesis model if not specified
        if not synthesis_model:
            try:
                # Call local analyze_task directly
                analysis_result = await analyze_task(
                    task_description=f"Compare and synthesize multiple responses based on criteria: {', '.join(criteria)} with strategy: {synthesis_strategy}",
                    analyze_features=True,
                    analyze_cost=False # Cost not needed here, just need model recommendation
                    # Pass available_providers if needed? Assumes defaults for now.
                )
                
                # Check analysis result structure
                if not isinstance(analysis_result, dict):
                     raise ValueError("analyze_task returned invalid format")
                     
                analysis = analysis_result # Use the dict directly

            except Exception as e:
                logger.error(f"Failed to call local analyze_task: {e}")
                analysis = {"recommendations": {}} # Default on error
            
            recommendations = analysis.get("recommendations", {})
            # Use balanced recommendation as default synthesis model
            if recommendations and recommendations.get("balanced"): # Changed from best_quality to balanced
                # Recommendation already provides full model ID (provider/model)
                synth_model_id = recommendations["balanced"]["model"] 
                synth_provider = recommendations["balanced"]["provider"]
                synth_model = synth_model_id # Use the full ID directly
            else:
                # Fallback if recommendation fails
                logger.warning("analyze_task failed to recommend a balanced model, falling back to default.")
                synth_provider = Provider.OPENAI.value # Default provider
                synth_model = "openai/gpt-4o" # Default high-capability model (full ID)
        else:
            # If synthesis_model config is provided
            synth_provider = synthesis_model.get("provider", Provider.OPENAI.value)
            synth_model_name = synthesis_model.get("model", "gpt-4o") # Assume model name only if provider given
            # Ensure model has provider prefix 
            if synth_provider not in synth_model_name:
                 synth_model = f"{synth_provider}/{synth_model_name}"
            else:
                 synth_model = synth_model_name # Already has prefix

        # Log the selected synthesis model
        logger.info(f"Using synthesis model: {synth_model}")

        # Generate criteria definitions and guidance
        criteria_definitions = _get_criteria_definitions(criteria)
        criteria_weights_text = "\n".join([f"- {criterion}: {weight:.2f}" for criterion, weight in criteria_weights.items()])
        
        responses_text = ""
        for i, completion in enumerate(completions):
            provider = completion.get("provider", "unknown")
            model = completion.get("model", "unknown")
            text = completion.get("text", "")
            responses_text += f"\n\n=== RESPONSE {i+1} (From {provider}/{model}) ===\n{text}\n=== END OF RESPONSE {i+1} ==="
        
        meta_prompt = f"""# TASK: EVALUATE AND {"SYNTHESIZE" if response_format == "synthesis" else "COMPARE"} MULTIPLE AI RESPONSES

## ORIGINAL PROMPT
{prompt}

## EVALUATION CRITERIA
{criteria_definitions}

## CRITERIA WEIGHTS
{criteria_weights_text}

## RESPONSES TO EVALUATE
{responses_text}

## SYNTHESIS STRATEGY: {synthesis_strategy.upper()}
{_get_synthesis_strategy_description(synthesis_strategy)}

## INSTRUCTIONS
"""

        meta_prompt += _get_format_specific_instructions(
            response_format=response_format,
            synthesis_strategy=synthesis_strategy,
            include_reasoning=include_reasoning,
            criteria=criteria
        )
        
        try:
            # Extract provider from model string if it contains a provider prefix
            # This ensures we're using the correct provider (e.g., openai) and not defaulting to openrouter
            if '/' in synth_model:
                extracted_provider, extracted_model = synth_model.split('/', 1)
                # Only use the extracted provider if it's a known provider
                if extracted_provider in [Provider.OPENAI.value, Provider.ANTHROPIC.value, 
                                        Provider.GEMINI.value, Provider.DEEPSEEK.value, 
                                        Provider.GROK.value, Provider.OPENROUTER.value]:
                    synth_provider = extracted_provider
                    synth_model = extracted_model
                    logger.info(f"Extracted provider '{synth_provider}' and model '{synth_model}' from '{synth_provider}/{synth_model}'")
            
            # Use standardized generate_completion instead of provider_instance.generate_completion
            completion_result = await generate_completion(
                prompt=meta_prompt,
                provider=synth_provider,
                model=synth_model,
                temperature=0.2,
                max_tokens=4000
            )
            
            # Check for errors in completion
            if not completion_result.success:
                raise completion_result.error or ValueError("Synthesis completion failed without specific error")
                
            result = completion_result
        except Exception as e:
            logger.error(f"Primary synthesis model {synth_provider}/{synth_model} failed: {str(e)}", emoji_key="error")
            # Use a different fallback model
            fallback_provider = Provider.ANTHROPIC.value
            fallback_model = "claude-3-7-sonnet-20250219"
            # Avoid retrying with the same model if it was the fallback
            if synth_model == fallback_model and synth_provider == fallback_provider:
                 logger.error("Fallback model also failed or was the primary model. Cannot proceed.")
                 raise ProviderError(f"Synthesis failed with both primary and fallback models: {e}", provider=synth_provider, model=synth_model, cause=e) from e

            logger.info(f"Attempting fallback to {fallback_provider}/{fallback_model}", emoji_key="warning")
            
            # Use standardized generate_completion for fallback
            fallback_result = await generate_completion(
                prompt=meta_prompt,
                provider=fallback_provider,
                model=fallback_model,
                temperature=0.2,
                max_tokens=4000
            )
            
            # Check for errors in fallback completion
            if not fallback_result.success:
                raise fallback_result.error or ValueError("Fallback synthesis completion failed without specific error")
                
            result = fallback_result
            synth_provider = fallback_provider # Update provider if fallback used
        
        synthesis = _parse_synthesis_response(result.text)
        
        total_cost = result.cost
        total_tokens = result.total_tokens
        
        for completion in completions:
            if completion:
                 total_cost += completion.get("cost", 0.0)
                 # Safely access nested dictionary
                 tokens_dict = completion.get("tokens")
                 if isinstance(tokens_dict, dict):
                      total_tokens += tokens_dict.get("total", 0)
        
        final_result = {
            "synthesis": synthesis,
            "completions": completions,
            "synthesis_provider": synth_provider,
            "synthesis_model": result.model,
            "criteria": criteria,
            "criteria_weights": criteria_weights,
            "response_format": response_format,
            "synthesis_strategy": synthesis_strategy,
            "tokens": {
                "synthesis_input": result.input_tokens,
                "synthesis_output": result.output_tokens,
                "synthesis_total": result.total_tokens,
                "total": total_tokens,
            },
            "cost": {
                "synthesis_cost": result.cost,
                "total_cost": total_cost,
            },
        }
        
        if isinstance(synthesis, dict) and "metadata" in synthesis:
            final_result["metadata"] = synthesis["metadata"]
        
        return final_result
        
    except Exception as e:
        logger.error(f"Error during comparison/synthesis execution: {e}", exc_info=True)
        raise e # Re-raise for the decorator to catch

# --- compare_and_synthesize Tool Function ---
@with_tool_metrics
@with_error_handling
async def compare_and_synthesize(
    prompt: str,
    configs: List[Dict[str, Any]],
    criteria: Optional[List[str]] = None,
    criteria_weights: Optional[Dict[str, float]] = None,
    synthesis_model: Optional[Dict[str, Any]] = None,
    response_format: str = "best",
    synthesis_strategy: str = "comprehensive",
    include_reasoning: bool = True,
    # max_retries: int = 2, # Removed max_retries, error handling decorator handles this
    timeout: Optional[float] = 120.0,
    ctx=None # Added ctx
) -> Dict[str, Any]:
    """
    Generate responses from multiple models/providers and synthesize or select the best one.
    
    Args:
        prompt: The prompt to send to all models
        configs: List of configurations for each model/provider
                Each config should have: {"provider": "...", "model": "...", "parameters": {...}}
        criteria: Criteria for evaluation (accuracy, completeness, etc.)
        criteria_weights: Optional weights for each criterion (e.g., {"accuracy": 0.6, "creativity": 0.4})
        synthesis_model: Config for the model to use for synthesis
                        If None, will select a high-capability model automatically
        response_format: Format of the response ("best", "synthesis", "ranked", or "analysis")
        synthesis_strategy: Strategy for synthesis ("comprehensive", "conservative", "creative")
        include_reasoning: Whether to include detailed reasoning in the output
        timeout: Timeout for the entire operation in seconds
        ctx: Context object passed by the MCP server. Required for calling sub-tools.
        
    Returns:
        Dictionary containing the synthesized results
    """
    start_time = time.time()
    log_extra = {"emoji_key": "meta", "configs_count": len(configs)}
    
    logger.info(
        f"Starting response comparison and synthesis with {len(configs)} configurations",
        **log_extra
    )
    
    if not configs:
        return {
            "error": "No model configurations provided",
            "processing_time": time.time() - start_time,
        }
    
    if len(configs) < 2 and response_format != "analysis": # Allow analysis with 1 model
        logger.warning(
            "Only one model configuration provided. Comparison/synthesis requires at least two models.",
            **log_extra
        )
        # Potentially short-circuit if only one model and not analysis? For now, proceed.

    if not criteria:
        criteria = [
            "factual_accuracy", "completeness", "relevance", "coherence", 
            "depth_of_reasoning", "clarity", "safety",
        ]
    
    if not criteria_weights:
        criteria_weights = {criterion: 1.0 / len(criteria) for criterion in criteria}
    else:
        weight_sum = sum(criteria_weights.values())
        if weight_sum > 0:
            criteria_weights = {k: v / weight_sum for k, v in criteria_weights.items()}
            # Ensure all criteria have weights
            for criterion in criteria:
                if criterion not in criteria_weights:
                     criteria_weights[criterion] = 0.0 # Assign 0 if missing after normalization
        else:
             criteria_weights = {criterion: 1.0 / len(criteria) for criterion in criteria} # Reset if sum is 0

    try:
        completion_task = asyncio.create_task(_execute_comparison_synthesis(
            prompt=prompt,
            configs=configs,
            criteria=criteria,
            criteria_weights=criteria_weights,
            synthesis_model=synthesis_model,
            response_format=response_format,
            synthesis_strategy=synthesis_strategy,
            include_reasoning=include_reasoning,
            ctx=ctx # Pass context
        ))
        
        if timeout:
            result = await asyncio.wait_for(completion_task, timeout=timeout)
        else:
            result = await completion_task
            
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        logger.success(
            f"Response comparison and synthesis completed ({len(configs)} models, {response_format} format)",
            time=processing_time, cost=result.get("cost",{}).get("total_cost"),
            **log_extra
        )
        
        # Ensure we return a dict, not a list (already handled inside _execute...)
        return result
        
    except asyncio.TimeoutError:
        completion_task.cancel()
        processing_time = time.time() - start_time
        logger.error(f"Response comparison and synthesis timed out after {timeout}s", **log_extra)
        return {
            "error": f"Operation timed out after {timeout} seconds",
            "processing_time": processing_time,
            "partial_results": None, # Indicate potential partial results if needed later
        }
        
    except Exception as e:
        # Error should be caught by the decorator, but include a failsafe
        processing_time = time.time() - start_time
        logger.error(f"Unhandled error in compare_and_synthesize: {str(e)}", time=processing_time, exc_info=True)
        return {
            "error": f"Response comparison failed: {str(e)}",
            "processing_time": processing_time,
        }