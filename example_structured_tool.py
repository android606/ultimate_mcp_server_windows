"""
Example of a well-structured MCP tool with best practices.

This module demonstrates how to create a comprehensive MCP tool
that implements all the best practices for LLM usability:
- Tool annotations for better decision-making
- Standardized error handling
- Input validation
- Detailed documentation with examples
- Structured outputs with consistent formats
"""
import time
from typing import Any, Dict, Optional

from error_handling import non_empty_string, validate_inputs, with_error_handling
from tool_annotations import ToolAnnotations


class ExampleTool:
    """Example implementation of a well-structured MCP tool."""
    
    def __init__(self, mcp_server):
        """Initialize with an MCP server instance."""
        self.mcp = mcp_server
        self._register_tools()
        
    def _register_tools(self):
        """Register tools with the MCP server."""
        
        # Create tool annotations with appropriate hints
        search_annotations = ToolAnnotations(
            read_only_hint=True,             # This tool doesn't modify anything
            destructive_hint=False,          # No destructive operations
            idempotent_hint=True,            # Can be called repeatedly with same results
            open_world_hint=True,            # Interacts with external data sources
            audience=["assistant"],          # Intended for the LLM to use
            priority=0.8,                    # High priority tool
            title="Search Knowledge Base",   # Human-readable title
            examples=[
                {
                    "name": "Basic search",
                    "description": "Search for information about a topic",
                    "input": {"query": "climate change", "filters": {"type": "article"}},
                    "output": {
                        "results": [
                            {"title": "Climate Change Basics", "score": 0.92},
                            {"title": "Effects of Global Warming", "score": 0.87}
                        ],
                        "total_matches": 2,
                        "search_time_ms": 105
                    }
                },
                {
                    "name": "Advanced search",
                    "description": "Search with multiple filters and limits",
                    "input": {
                        "query": "machine learning",
                        "filters": {"type": "tutorial", "level": "beginner"},
                        "limit": 1
                    },
                    "output": {
                        "results": [
                            {"title": "Introduction to Machine Learning", "score": 0.95}
                        ],
                        "total_matches": 1,
                        "search_time_ms": 87
                    }
                }
            ]
        )
        
        @self.mcp.tool(
            name="search_knowledge_base",
            description=(
                "Search for information in the knowledge base using keywords and filters.\n\n"
                "This tool is ideal for finding relevant information on specific topics. "
                "It supports filtering by content type, date ranges, and other metadata. "
                "The tool returns a list of matching results sorted by relevance score.\n\n"
                "WHEN TO USE:\n"
                "- When you need to find specific information on a topic\n"
                "- When you want to discover relevant articles or documentation\n"
                "- Before generating content to ensure accuracy\n\n"
                "WHEN NOT TO USE:\n"
                "- When you need to modify or create content (use content_* tools instead)\n"
                "- When you need very recent information that might not be in the knowledge base\n"
                "- When you need exact answers to questions (use qa_* tools instead)"
            ),
            annotations=search_annotations.to_dict(),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (required)"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters to narrow results",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["article", "tutorial", "reference", "faq"],
                                "description": "Content type filter"
                            },
                            "level": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                                "description": "Difficulty level filter"
                            },
                            "date_after": {
                                "type": "string",
                                "format": "date",
                                "description": "Only include content after this date (YYYY-MM-DD)"
                            }
                        }
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Maximum number of results to return (1-20, default 5)"
                    }
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "summary": {"type": "string"},
                                "type": {"type": "string"},
                                "date": {"type": "string", "format": "date"},
                                "score": {"type": "number"}
                            }
                        }
                    },
                    "total_matches": {"type": "integer"},
                    "search_time_ms": {"type": "integer"}
                }
            }
        )
        @with_error_handling
        @validate_inputs(query=non_empty_string)
        async def search_knowledge_base(
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 5,
            ctx=None
        ) -> Dict[str, Any]:
            """
            Search for information in the knowledge base using keywords and filters.
            
            This tool is ideal for finding relevant information on specific topics.
            It supports filtering by content type, date ranges, and other metadata.
            
            Args:
                query: Search query string (required)
                filters: Optional filters to narrow results
                    - type: Content type filter (article, tutorial, reference, faq)
                    - level: Difficulty level filter (beginner, intermediate, advanced)
                    - date_after: Only include content after this date (YYYY-MM-DD)
                limit: Maximum number of results to return (1-20, default 5)
                ctx: Context object passed by the MCP server
                
            Returns:
                Dictionary containing:
                - results: List of matching items with metadata
                - total_matches: Total number of matches found
                - search_time_ms: Search execution time in milliseconds
                
            Examples:
                Basic search:
                  search_knowledge_base(query="climate change")
                  
                Filtered search:
                  search_knowledge_base(
                    query="machine learning", 
                    filters={"type": "tutorial", "level": "beginner"},
                    limit=3
                  )
            """
            # Start timing
            start_time = time.time()
            
            # Validate inputs
            if not query or not isinstance(query, str):
                return {
                    "error": "Invalid query. Query must be a non-empty string.",
                    "results": [],
                    "total_matches": 0,
                    "search_time_ms": 0
                }
                
            # Handle filters
            filters = filters or {}
            content_type = filters.get("type")
            level = filters.get("level")
            date_after = filters.get("date_after")
            
            # Validate limit
            limit = max(1, min(20, limit))  # Ensure limit is between 1-20
            
            try:
                # Simulate knowledge base search
                # In a real implementation, this would query a database, search engine, etc.
                sample_results = [
                    {
                        "id": "kb-001",
                        "title": "Introduction to Climate Change",
                        "summary": "An overview of climate change causes and effects.",
                        "type": "article",
                        "level": "beginner",
                        "date": "2023-01-15",
                        "score": 0.95
                    },
                    {
                        "id": "kb-002",
                        "title": "Machine Learning Fundamentals",
                        "summary": "Learn the basics of machine learning algorithms.",
                        "type": "tutorial",
                        "level": "beginner",
                        "date": "2023-02-20",
                        "score": 0.92
                    },
                    {
                        "id": "kb-003",
                        "title": "Advanced Neural Networks",
                        "summary": "Deep dive into neural network architectures.",
                        "type": "tutorial",
                        "level": "advanced",
                        "date": "2023-03-10",
                        "score": 0.88
                    },
                    {
                        "id": "kb-004",
                        "title": "Climate Policy FAQ",
                        "summary": "Frequently asked questions about climate policies.",
                        "type": "faq",
                        "level": "intermediate",
                        "date": "2023-04-05",
                        "score": 0.82
                    },
                    {
                        "id": "kb-005",
                        "title": "Python Reference for Data Science",
                        "summary": "Reference guide for Python in data science applications.",
                        "type": "reference",
                        "level": "intermediate",
                        "date": "2023-05-12",
                        "score": 0.78
                    }
                ]
                
                # Simulate relevance matching based on query
                matched_results = []
                for result in sample_results:
                    # Simple keyword matching (real implementation would use proper search)
                    if query.lower() in result["title"].lower() or query.lower() in result["summary"].lower():
                        
                        # Apply filters if specified
                        if content_type and result["type"] != content_type:
                            continue
                        if level and result["level"] != level:
                            continue
                        if date_after and result["date"] < date_after:
                            continue
                            
                        matched_results.append(result)
                
                # Sort by relevance score
                matched_results.sort(key=lambda x: x["score"], reverse=True)
                
                # Apply limit
                limited_results = matched_results[:limit]
                
                # Calculate search time
                search_time_ms = int((time.time() - start_time) * 1000)
                
                # Return formatted results
                return {
                    "results": limited_results,
                    "total_matches": len(matched_results),
                    "search_time_ms": search_time_ms,
                    "query": query  # Return the original query for reference
                }
                
            except Exception as e:
                # Log the error (in a real implementation)
                print(f"Search error: {str(e)}")
                
                # Return error response
                return {
                    "error": f"Search failed: {str(e)}",
                    "results": [],
                    "total_matches": 0,
                    "search_time_ms": int((time.time() - start_time) * 1000)
                }


def register_example_tools(mcp_server):
    """
    Register all example tools with the MCP server.
    
    Args:
        mcp_server: MCP server instance
    """
    ExampleTool(mcp_server) 