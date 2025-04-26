"""MCP Tools for Ultimate MCP Server."""

import inspect
from typing import Any, Dict

from ultimate_mcp_server.tools.base import (
    BaseTool,  # Keep BaseTool in case other modules use it
    register_tool,
    with_error_handling,
    with_retry,
    with_tool_metrics,
)
from ultimate_mcp_server.utils import get_logger

from .audio_transcription import (
    chat_with_transcript,
    extract_audio_transcript_key_points,
    transcribe_audio,
)

# Import cognitive and agent memory tools
from .cognitive_and_agent_memory import (
    add_action_dependency,
    auto_update_focus,
    compute_memory_statistics,
    consolidate_memories,
    create_memory_link,
    create_thought_chain,
    create_workflow,
    delete_expired_memories,
    focus_memory,
    generate_reflection,
    generate_workflow_report,
    get_action_dependencies,
    get_action_details,
    get_artifact_by_id,
    get_artifacts,
    get_linked_memories,
    get_memory_by_id,
    get_recent_actions,
    get_thought_chain,
    get_workflow_context,
    get_workflow_details,
    get_working_memory,
    hybrid_search_memories,
    initialize_memory_system,
    list_workflows,
    load_cognitive_state,
    optimize_working_memory,
    promote_memory_level,
    query_memories,
    record_action_completion,
    record_action_start,
    record_artifact,
    record_thought,
    save_cognitive_state,
    search_semantic_memories,
    store_memory,
    summarize_context_block,
    summarize_text,
    update_memory,
    update_workflow_status,
    visualize_memory_network,
    visualize_reasoning_chain,
)

# Import base decorators/classes that might be used by other tool modules
from .completion import chat_completion, generate_completion, multi_completion, stream_completion
from .docstring_refiner import refine_tool_documentation
from .document import (
    chunk_document,
    extract_entities,
    generate_qa_pairs,
    process_document_batch,
    summarize_document,
)
from .document_conversion_tool import convert_document
from .entity_relation_graph import extract_entity_graph

# Import new standalone functions from extraction.py
from .extraction import (
    extract_code_from_response,
    extract_json,
    extract_key_value_pairs,
    extract_semantic_schema,
    extract_table,
)
from .filesystem import (
    create_directory,
    directory_tree,
    edit_file,
    get_file_info,
    list_allowed_directories,
    list_directory,
    move_file,
    read_file,
    read_multiple_files,
    search_files,
    write_file,
)
from .html_to_markdown import (
    batch_format_texts,
    clean_and_format_text_as_markdown,
    detect_content_type,
    optimize_markdown_formatting,
)
from .marqo_fused_search import marqo_fused_search
from .meta import (
    get_llm_instructions,
    get_tool_info,
    get_tool_recommendations,
)
from .meta_api_tool import register_api_meta_tools

# Import OCR tools from ocr_tools.py
from .ocr_tools import (
    analyze_pdf_structure,
    batch_process_documents,
    enhance_ocr_text,
    extract_text_from_pdf,
    process_image_ocr,
)

# Import standalone functions from optimization.py
from .optimization import (
    compare_models,
    estimate_cost,
    execute_optimized_workflow,
    recommend_model,
)
from .provider import get_provider_status, list_models

# Import python js sandbox tools
from .python_js_sandbox import (
    execute_python_code,
)
from .rag import (
    add_documents,
    create_knowledge_base,
    delete_knowledge_base,
    generate_with_rag,
    list_knowledge_bases,
    retrieve_context,
)
from .sentiment_analysis import analyze_business_sentiment, analyze_business_text_batch

# Import smart browser tools
from .smart_browser import (
    search_web,
)
from .text_classification import text_classification
from .text_redline_tools import (
    compare_documents_redline,
    create_html_redline,
)
from .tournament import (
    cancel_tournament,
    create_tournament,
    get_tournament_results,
    get_tournament_status,
    list_tournaments,
)

__all__ = [
    # Base decorators/classes
    "BaseTool",
    "with_tool_metrics",
    "with_retry",
    "with_error_handling",
    "register_tool", 
    
    # Standalone tool functions (explicitly list them)
    "generate_completion",
    "stream_completion",
    "chat_completion",
    "multi_completion",
    "get_provider_status",
    "list_models",
    "get_tool_info",
    "get_llm_instructions",
    "get_tool_recommendations",
    "chunk_document",
    "summarize_document",
    "extract_entities",
    "generate_qa_pairs",
    "process_document_batch",
    "extract_json",
    "extract_table",
    "extract_key_value_pairs",
    "extract_semantic_schema",
    "extract_entity_graph",
    "create_knowledge_base",
    "list_knowledge_bases",
    "delete_knowledge_base",
    "add_documents",
    "retrieve_context",
    "generate_with_rag",
    "text_classification",
    "create_tournament",
    "get_tournament_status",
    "list_tournaments",
    "get_tournament_results",
    "cancel_tournament",
    "estimate_cost",
    "compare_models",
    "recommend_model",
    "execute_optimized_workflow",
    "refine_tool_documentation",
    
    # Filesystem tools
    "read_file",
    "read_multiple_files",
    "write_file",
    "edit_file",
    "create_directory",
    "list_directory",
    "directory_tree",
    "move_file",
    "search_files",
    "get_file_info",
    "list_allowed_directories",
    
    # OCR tools
    "extract_text_from_pdf",
    "process_image_ocr",
    "enhance_ocr_text",
    "analyze_pdf_structure",
    "batch_process_documents",
    
    # HTML to Markdown tools
    "clean_and_format_text_as_markdown",
    "detect_content_type",
    "batch_format_texts",
    "optimize_markdown_formatting",

    # Text Redline tools
    "compare_documents_redline",
    "create_html_redline",

    # Utility functions
    "extract_code_from_response",
    
    # Meta API tools
    "register_api_meta_tools",

    # Marqo tool
    "marqo_fused_search",

    # SQL tools


    # Audio tools
    "transcribe_audio",
    "extract_audio_transcript_key_points",
    "chat_with_transcript",
    
    # Browser automation tools


    # Document conversion tool
    "convert_document",

    # Sentiment analysis tool
    "analyze_business_sentiment",
    "analyze_business_text_batch",
    
    # Cognitive and agent memory tools
    "initialize_memory_system",
    "create_workflow",
    "update_workflow_status",
    "record_action_start",
    "record_action_completion",
    "get_action_details",
    "summarize_context_block",
    "add_action_dependency",
    "get_action_dependencies",
    "record_artifact",
    "record_thought",
    "store_memory",
    "get_memory_by_id",
    "search_semantic_memories",
    "hybrid_search_memories",
    "create_memory_link",
    "query_memories",
    "list_workflows",
    "get_workflow_details",
    "get_recent_actions",
    "get_artifacts",
    "get_artifact_by_id",
    "create_thought_chain",
    "get_thought_chain",
    "get_working_memory",
    "focus_memory",
    "optimize_working_memory",
    "save_cognitive_state",
    "load_cognitive_state",
    "get_workflow_context",
    "auto_update_focus",
    "promote_memory_level",
    "update_memory",
    "get_linked_memories",
    "consolidate_memories",
    "generate_reflection",
    "summarize_text",
    "delete_expired_memories",
    "compute_memory_statistics",
    "generate_workflow_report",
    "visualize_reasoning_chain",
    "visualize_memory_network",
]

logger = get_logger("ultimate_mcp_server.tools")


# --- Tool Registration --- 

# List of standalone functions to register
STANDALONE_TOOL_FUNCTIONS = [
    generate_completion,
    stream_completion,
    chat_completion,
    multi_completion,
    get_provider_status,
    list_models,
    get_tool_info,
    get_llm_instructions,
    get_tool_recommendations,
    chunk_document,
    summarize_document,
    extract_entities,
    generate_qa_pairs,
    process_document_batch,
    extract_json,
    extract_table,
    extract_key_value_pairs,
    extract_semantic_schema,
    extract_entity_graph,
    create_knowledge_base,
    list_knowledge_bases,
    delete_knowledge_base,
    add_documents,
    retrieve_context,
    generate_with_rag,
    text_classification,
    create_tournament,
    get_tournament_status,
    list_tournaments,
    get_tournament_results,
    cancel_tournament,
    estimate_cost,
    compare_models,
    recommend_model,
    execute_optimized_workflow,
    refine_tool_documentation,
    
    # Filesystem tools
    read_file,
    read_multiple_files,
    write_file,
    edit_file,
    create_directory,
    list_directory,
    directory_tree,
    move_file,
    search_files,
    get_file_info,
    list_allowed_directories,
    
    # OCR tools
    extract_text_from_pdf,
    process_image_ocr,
    enhance_ocr_text,
    analyze_pdf_structure,
    batch_process_documents,

    # HTML to Markdown tools
    clean_and_format_text_as_markdown,
    detect_content_type,
    batch_format_texts,
    optimize_markdown_formatting,

    # Text Redline tools
    compare_documents_redline,
    create_html_redline,

    # Marqo tool
    marqo_fused_search,

    # Added SQL tools

    # Added Audio tools
    transcribe_audio,
    extract_audio_transcript_key_points,
    chat_with_transcript,

    # Browser automation tools

    
    # Cognitive and agent memory tools
    initialize_memory_system,
    create_workflow,
    update_workflow_status,
    record_action_start,
    record_action_completion,
    get_action_details,
    summarize_context_block,
    add_action_dependency,
    get_action_dependencies,
    record_artifact,
    record_thought,
    store_memory,
    get_memory_by_id,
    search_semantic_memories,
    hybrid_search_memories,
    create_memory_link,
    query_memories,
    list_workflows,
    get_workflow_details,
    get_recent_actions,
    get_artifacts,
    get_artifact_by_id,
    create_thought_chain,
    get_thought_chain,
    get_working_memory,
    focus_memory,
    optimize_working_memory,
    save_cognitive_state,
    load_cognitive_state,
    get_workflow_context,
    auto_update_focus,
    promote_memory_level,
    update_memory,
    get_linked_memories,
    consolidate_memories,
    generate_reflection,
    summarize_text,
    delete_expired_memories,
    compute_memory_statistics,
    generate_workflow_report,
    visualize_reasoning_chain,
    visualize_memory_network,
]


def register_all_tools(mcp_server) -> Dict[str, Any]:
    """Registers all tools (standalone and class-based) with the MCP server.

    Args:
        mcp_server: The MCP server instance.

    Returns:
        Dictionary containing information about registered tools.
    """
    from ultimate_mcp_server.config import get_config
    cfg = get_config()
    filter_enabled = cfg.tool_registration.filter_enabled
    included_tools = cfg.tool_registration.included_tools
    excluded_tools = cfg.tool_registration.excluded_tools
    
    logger.info("Registering tools based on configuration...")
    if filter_enabled:
        if included_tools:
            logger.info(f"Tool filtering enabled: including only {len(included_tools)} specified tools")
        if excluded_tools:
            logger.info(f"Tool filtering enabled: excluding {len(excluded_tools)} specified tools")
    
    registered_tools: Dict[str, Any] = {}
    
    # --- Register Standalone Functions ---
    standalone_count = 0
    for tool_func in STANDALONE_TOOL_FUNCTIONS:
        if not callable(tool_func) or not inspect.iscoroutinefunction(tool_func):
            logger.warning(f"Item {getattr(tool_func, '__name__', repr(tool_func))} in STANDALONE_TOOL_FUNCTIONS is not a callable async function.")
            continue
            
        tool_name = tool_func.__name__
        
        # Apply tool filtering logic
        if filter_enabled:
            # Skip if not in included_tools when included_tools is specified
            if included_tools and tool_name not in included_tools:
                logger.debug(f"Skipping tool {tool_name} (not in included_tools)")
                continue
                
            # Skip if in excluded_tools
            if tool_name in excluded_tools:
                logger.debug(f"Skipping tool {tool_name} (in excluded_tools)")
                continue
        
        # Register the tool
        mcp_server.tool(name=tool_name)(tool_func)
        registered_tools[tool_name] = {
            "description": inspect.getdoc(tool_func) or "",
            "type": "standalone_function"
        }
        logger.info(f"Registered tool function: {tool_name}", emoji_key="⚙️")
        standalone_count += 1
    
    # Special handling for meta_api_tool which is a module rather than a function
    # Only register if it passes the filtering criteria
    if (not filter_enabled or 
        "register_api_meta_tools" in included_tools or 
        (not included_tools and "register_api_meta_tools" not in excluded_tools)):
        try:
            from ultimate_mcp_server.tools.meta_api_tool import register_api_meta_tools
            register_api_meta_tools(mcp_server)
            logger.info("Registered API Meta-Tool functions", emoji_key="⚙️")
            standalone_count += 1
        except ImportError:
            logger.warning("Meta API tools not found (ultimate_mcp_server.tools.meta_api_tool)")
        except Exception as e:
            logger.error(f"Failed to register Meta API tools: {e}", exc_info=True)
    
    # Special handling for excel_spreadsheet_automation which is a module rather than a function
    # Only register if it passes the filtering criteria AND Excel is available on Windows
    if (not filter_enabled or 
        "register_excel_spreadsheet_tools" in included_tools or 
        (not included_tools and "register_excel_spreadsheet_tools" not in excluded_tools)):
        try:
            from ultimate_mcp_server.tools.excel_spreadsheet_automation import (
                WINDOWS_EXCEL_AVAILABLE,
                register_excel_spreadsheet_tools,
            )
            if WINDOWS_EXCEL_AVAILABLE:
                register_excel_spreadsheet_tools(mcp_server)
                logger.info("Registered Excel spreadsheet tools", emoji_key="⚙️")
                standalone_count += 1
            else:
                # Automatically exclude Excel tools if not available
                logger.warning("Excel automation tools are only available on Windows with Excel installed. These tools will not be registered.")
                # If not already explicitly excluded, add to excluded_tools
                if "register_excel_spreadsheet_tools" not in excluded_tools:
                    if not cfg.tool_registration.filter_enabled:
                        cfg.tool_registration.filter_enabled = True
                    if not hasattr(cfg.tool_registration, "excluded_tools"):
                        cfg.tool_registration.excluded_tools = []
                    cfg.tool_registration.excluded_tools.append("register_excel_spreadsheet_tools")
        except ImportError:
            logger.warning("Excel spreadsheet tools not found (ultimate_mcp_server.tools.excel_spreadsheet_automation)")
        except Exception as e:
            logger.error(f"Failed to register Excel spreadsheet tools: {e}", exc_info=True)
    
    logger.info(
        f"Completed tool registration. Registered {standalone_count} tools.", 
        emoji_key="⚙️"
    )
    
    # Return info about registered tools
    return registered_tools
