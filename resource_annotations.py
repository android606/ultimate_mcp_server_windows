"""
Resource annotations for MCP servers.

This module provides standardized annotations for resources in MCP servers,
helping LLMs understand the importance, audience, and format of resources.
"""
from typing import List, Optional


class ResourceAnnotations:
    """
    Annotations for MCP resources.
    
    These annotations provide metadata about resources to help LLMs understand:
    - How important the resource is for a task
    - Who should see the resource (user, assistant, or both)
    - How the resource should be formatted or displayed
    """
    
    def __init__(
        self,
        priority: float = 0.5,
        audience: List[str] = None,
        structured_format: bool = False,
        chunking_recommended: bool = False,
        max_recommended_chunk_size: Optional[int] = None,
        description: Optional[str] = None
    ):
        """
        Initialize resource annotations.
        
        Args:
            priority: How important this resource is (0.0-1.0, higher is more important).
                0.0 = entirely optional, 1.0 = effectively required.
                Default: 0.5
            audience: Who should see this resource (e.g., ["user", "assistant"]).
                Default: ["assistant"]
            structured_format: Whether this resource has a structured format that
                should be preserved (e.g., code, JSON, tables).
                Default: False
            chunking_recommended: Whether this resource should be chunked if large.
                Default: False
            max_recommended_chunk_size: Maximum recommended chunk size in characters.
                Default: None (no specific recommendation)
            description: Optional description of the resource.
        """
        self.priority = max(0.0, min(1.0, priority))  # Clamp between 0 and 1
        self.audience = audience or ["assistant"]
        self.structured_format = structured_format
        self.chunking_recommended = chunking_recommended
        self.max_recommended_chunk_size = max_recommended_chunk_size
        self.description = description
        
    def to_dict(self) -> dict:
        """Convert annotations to dictionary for MCP protocol."""
        result = {
            "priority": self.priority,
            "audience": self.audience
        }
        
        # Add extended properties
        if self.description:
            result["description"] = self.description
        
        # Add chunking metadata if recommended
        if self.chunking_recommended:
            result["chunking"] = {
                "recommended": True
            }
            if self.max_recommended_chunk_size:
                result["chunking"]["maxSize"] = self.max_recommended_chunk_size
                
        # Add format information
        if self.structured_format:
            result["format"] = {
                "structured": True
            }
            
        return result


# Pre-defined annotation templates for common resource types

HIGH_PRIORITY_RESOURCE = ResourceAnnotations(
    priority=0.9,
    audience=["assistant", "user"],
    description="Critical resource that should be prioritized"
)

CODE_RESOURCE = ResourceAnnotations(
    priority=0.8,
    audience=["assistant"],
    structured_format=True,
    chunking_recommended=True,
    max_recommended_chunk_size=2000,
    description="Source code that should preserve formatting"
)

LARGE_TEXT_RESOURCE = ResourceAnnotations(
    priority=0.6,
    audience=["assistant"],
    chunking_recommended=True,
    max_recommended_chunk_size=4000,
    description="Large text that should be chunked for processing"
)

STRUCTURED_DATA_RESOURCE = ResourceAnnotations(
    priority=0.7,
    audience=["assistant"],
    structured_format=True,
    description="Structured data like JSON or tables"
)

OPTIONAL_RESOURCE = ResourceAnnotations(
    priority=0.2,
    audience=["assistant"],
    description="Supplementary information that isn't critical"
)

USER_FACING_RESOURCE = ResourceAnnotations(
    priority=0.7,
    audience=["user"],
    description="Resource meant for user consumption"
)


def format_chunked_content(content: str, chunk_size: int = 4000, overlap: int = 200) -> List[dict]:
    """
    Format content as chunked resources with appropriate annotations.
    
    Args:
        content: The content to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunked resources with annotations
    """
    chunks = []
    
    # Create chunks with overlap
    for i in range(0, len(content), chunk_size - overlap):
        chunk_text = content[i:i + chunk_size]
        if chunk_text:
            # Create chunk with annotations
            chunk = {
                "text": chunk_text,
                "annotations": {
                    "priority": 0.7,
                    "audience": ["assistant"],
                    "chunk_info": {
                        "index": len(chunks),
                        "total_chunks": (len(content) + chunk_size - 1) // (chunk_size - overlap),
                        "start_position": i,
                        "end_position": min(i + chunk_size, len(content)),
                        "has_more": i + chunk_size < len(content)
                    }
                }
            }
            chunks.append(chunk)
    
    return chunks 