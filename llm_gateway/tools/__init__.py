"""Tools for LLM Gateway."""
from llm_gateway.tools.base import (
    BaseTool,
    with_tool_metrics,
    with_retry,
    register_tool,
)
from llm_gateway.tools.completion import CompletionTools
from llm_gateway.tools.document import DocumentTools
from llm_gateway.tools.extraction import ExtractionTools
from llm_gateway.tools.meta import MetaTools
from llm_gateway.tools.optimization import OptimizationTools

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
]