"""Meta tools for LLM Gateway including LLM instructions on tool usage."""
import json
from typing import Any, Dict, Optional

# Remove BaseTool import if no longer needed
# from llm_gateway.tools.base import BaseTool 
from llm_gateway.tools.base import with_error_handling, with_tool_metrics
from llm_gateway.utils import get_logger

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
               model="claude-3-haiku-20240307"
           )
           ```
        
        ## Provider Recommendations
        
        - For initial chunk summaries: Gemini or GPT-4o-mini
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
                {"provider": "openai", "model": "gpt-4o-mini", "reason": "Good balance of quality and cost"}
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
                {"provider": "openai", "model": "gpt-4o-mini", "reason": "Excellent at structured extraction"},
                {"provider": "anthropic", "model": "claude-3-haiku-20240307", "reason": "Good balance of accuracy and cost"}
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
                {"provider": "openai", "model": "gpt-4o-mini", "reason": "For processing search results"}
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