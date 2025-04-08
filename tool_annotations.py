"""
Tool annotations for MCP servers.

This module provides a standardized way to annotate tools with hints that help LLMs
understand when and how to use them effectively.
"""
from typing import List, Optional


class ToolAnnotations:
    """
    Tool annotations providing hints to LLMs about tool behavior and usage.
    
    These annotations help LLMs make better decisions about which tools to use
    in different situations, and understand the potential consequences of tool calls.
    
    All hints are advisory only, and don't restrict the tool's behavior - they
    simply provide guidance to the LLM.
    """
    
    def __init__(
        self,
        read_only_hint: bool = False,
        destructive_hint: bool = True,
        idempotent_hint: bool = False,
        open_world_hint: bool = True,
        audience: List[str] = None,
        priority: float = 0.5,
        title: Optional[str] = None,
        examples: List[dict] = None,
    ):
        """
        Initialize tool annotations.
        
        Args:
            read_only_hint: If True, indicates this tool does not modify its environment.
                Default: False
            destructive_hint: If True, the tool may perform destructive updates.
                Only meaningful when read_only_hint is False.
                Default: True
            idempotent_hint: If True, calling the tool repeatedly with the same arguments
                will have no additional effect. Only meaningful when read_only_hint is False.
                Default: False
            open_world_hint: If True, this tool may interact with external systems or entities.
                If False, the tool's domain is closed (e.g., memory tools).
                Default: True
            audience: Who is the intended user of this tool (e.g., ["assistant", "user"]).
                Default: ["assistant"]
            priority: How important this tool is (0.0-1.0, higher is more important).
                Default: 0.5
            title: Human-readable title for the tool.
            examples: List of usage examples, each containing 'input' and 'output'.
        """
        self.read_only_hint = read_only_hint
        self.destructive_hint = destructive_hint
        self.idempotent_hint = idempotent_hint
        self.open_world_hint = open_world_hint
        self.audience = audience or ["assistant"]
        self.priority = max(0.0, min(1.0, priority))  # Clamp between 0 and 1
        self.title = title
        self.examples = examples or []
        
    def to_dict(self) -> dict:
        """Convert annotations to dictionary for MCP protocol."""
        return {
            "readOnlyHint": self.read_only_hint,
            "destructiveHint": self.destructive_hint,
            "idempotentHint": self.idempotent_hint,
            "openWorldHint": self.open_world_hint,
            "title": self.title,
            "audience": self.audience,
            "priority": self.priority,
            "examples": self.examples
        }

# Pre-defined annotation templates for common tool types

READONLY_TOOL = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
    priority=0.8,
    title="Read-Only Tool"
)

QUERY_TOOL = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
    priority=0.7,
    title="Query Tool"
)

DESTRUCTIVE_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
    priority=0.3,
    title="Destructive Tool"
)

IDEMPOTENT_UPDATE_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
    priority=0.5,
    title="Idempotent Update Tool"
) 