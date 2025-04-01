"""MCP Tools for LLM Gateway."""

from llm_gateway.tools.base import (
    BaseTool,
    register_tool,
    with_retry,
    with_tool_metrics,
)
from llm_gateway.tools.completion import CompletionTools
from llm_gateway.tools.document import DocumentTools
from llm_gateway.tools.extraction import ExtractionTools, extract_code_from_response
from llm_gateway.tools.meta import MetaTools
from llm_gateway.tools.optimization import OptimizationTools
from llm_gateway.tools.rag import RAGTools

__all__ = [
    # Base tool classes and decorators
    "BaseTool",
    "with_tool_metrics",
    "with_retry",
    "register_tool",
    
    # Tool implementations
    "CompletionTools",
    "DocumentTools",
    "ExtractionTools",
    "MetaTools",
    "OptimizationTools",
    "RAGTools",
    "get_all_tools",
    
    # Utility functions
    "extract_code_from_response",
]

def get_all_tools():
    """Get all registered MCP tools.
    
    Returns:
        List of tool instances
    """
    return [
        CompletionTools(),
        DocumentTools(),
        ExtractionTools(),
        MetaTools(),
        OptimizationTools(),
        RAGTools(),
    ]