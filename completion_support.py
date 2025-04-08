"""
Completion support for MCP servers.

This module provides utilities for implementing argument completion suggestions
in MCP tools, making them more interactive and easier to use.
"""
from typing import Any, Callable, Dict, List, Optional


class CompletionProvider:
    """Base class for argument completion providers."""
    
    async def get_completions(self, argument_name: str, current_value: str, **context) -> List[str]:
        """
        Get completion suggestions for an argument.
        
        Args:
            argument_name: Name of the argument being completed
            current_value: Current value of the argument
            **context: Additional context for completion
            
        Returns:
            List of completion suggestions
        """
        raise NotImplementedError("Subclasses must implement get_completions")
        
    def supports_argument(self, argument_name: str) -> bool:
        """
        Check if this provider supports completion for a given argument.
        
        Args:
            argument_name: Name of the argument
            
        Returns:
            True if this provider supports completion for the argument
        """
        raise NotImplementedError("Subclasses must implement supports_argument")


class StaticCompletionProvider(CompletionProvider):
    """Completion provider with static suggestions."""
    
    def __init__(self, completions: Dict[str, List[str]]):
        """
        Initialize with static completion values.
        
        Args:
            completions: Dictionary mapping argument names to suggestion lists
        """
        self.completions = completions
        
    async def get_completions(self, argument_name: str, current_value: str, **context) -> List[str]:
        """Get completion suggestions from static values."""
        if not self.supports_argument(argument_name):
            return []
            
        # Filter suggestions based on current value
        suggestions = self.completions.get(argument_name, [])
        if current_value:
            return [s for s in suggestions if s.lower().startswith(current_value.lower())]
        return suggestions
        
    def supports_argument(self, argument_name: str) -> bool:
        """Check if static completions exist for this argument."""
        return argument_name in self.completions


class DynamicCompletionProvider(CompletionProvider):
    """Completion provider with dynamically generated suggestions."""
    
    def __init__(self, completion_functions: Dict[str, Callable]):
        """
        Initialize with dynamic completion functions.
        
        Args:
            completion_functions: Dictionary mapping argument names to completion functions
        """
        self.completion_functions = completion_functions
        
    async def get_completions(self, argument_name: str, current_value: str, **context) -> List[str]:
        """Get completion suggestions by calling appropriate function."""
        if not self.supports_argument(argument_name):
            return []
            
        # Call the function to get suggestions
        func = self.completion_functions.get(argument_name)
        if func:
            try:
                suggestions = await func(current_value, **context)
                return suggestions
            except Exception as e:
                # Log error and return empty list
                print(f"Error getting completions for {argument_name}: {str(e)}")
                return []
        return []
        
    def supports_argument(self, argument_name: str) -> bool:
        """Check if a completion function exists for this argument."""
        return argument_name in self.completion_functions


class CompletionRegistry:
    """
    Registry for completion providers.
    
    This class manages completion providers for different tools and arguments,
    providing a central interface for completion requests.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self.tool_providers = {}  # Map of tool_name -> provider
        self.default_provider = None
        
    def register_provider(self, tool_name: str, provider: CompletionProvider):
        """
        Register a completion provider for a tool.
        
        Args:
            tool_name: Name of the tool
            provider: Completion provider for the tool
        """
        self.tool_providers[tool_name] = provider
        
    def set_default_provider(self, provider: CompletionProvider):
        """
        Set a default provider for tools without a specific provider.
        
        Args:
            provider: Default completion provider
        """
        self.default_provider = provider
        
    def get_provider(self, tool_name: str) -> Optional[CompletionProvider]:
        """
        Get the completion provider for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Completion provider for the tool, or the default provider
        """
        return self.tool_providers.get(tool_name, self.default_provider)
        
    async def get_completions(
        self, 
        tool_name: str, 
        argument_name: str, 
        current_value: str, 
        **context
    ) -> Dict[str, Any]:
        """
        Get completion suggestions for a tool argument.
        
        Args:
            tool_name: Name of the tool
            argument_name: Name of the argument
            current_value: Current value of the argument
            **context: Additional context for completion
            
        Returns:
            Dictionary with completion results
        """
        provider = self.get_provider(tool_name)
        
        if not provider or not provider.supports_argument(argument_name):
            return {
                "values": [],
                "hasMore": False,
                "total": 0
            }
            
        try:
            # Get suggestions from provider
            suggestions = await provider.get_completions(
                argument_name=argument_name,
                current_value=current_value,
                tool_name=tool_name,
                **context
            )
            
            # Limit to 100 items as per MCP spec
            has_more = len(suggestions) > 100
            suggestions = suggestions[:100]
            
            return {
                "values": suggestions,
                "hasMore": has_more,
                "total": len(suggestions)
            }
        except Exception as e:
            # Log error and return empty result
            print(f"Error getting completions: {str(e)}")
            return {
                "values": [],
                "hasMore": False,
                "total": 0
            }


# Example usage for file path completion
async def complete_file_paths(current_value: str, **context) -> List[str]:
    """
    Complete file paths based on the current value.
    
    Args:
        current_value: Current path value
        **context: Additional context
        
    Returns:
        List of path suggestions
    """
    import glob
    import os
    
    # Handle empty value
    if not current_value:
        return ["./", "../", "/"]
        
    # Expand user directory if needed
    path = os.path.expanduser(current_value)
    
    # Get the directory to search in
    directory = os.path.dirname(path) if os.path.basename(path) else path
    if not directory:
        directory = "."
        
    # Get matching files/directories
    pattern = os.path.join(directory, f"{os.path.basename(path)}*")
    matches = glob.glob(pattern)
    
    # Format results
    results = []
    for match in matches:
        if os.path.isdir(match):
            results.append(f"{match}/")
        else:
            results.append(match)
            
    return results


# Example completions for common arguments
COMMON_COMPLETIONS = StaticCompletionProvider({
    "provider": ["openai", "anthropic", "gemini", "mistral", "custom"],
    "model": [
        "gpt-4-turbo", "gpt-4o", "claude-3-5-sonnet", "claude-3-opus", 
        "gemini-1.5-pro", "gemini-1.5-flash", "mistral-large"
    ],
    "format": ["json", "text", "markdown", "html", "csv"],
    "source_type": ["csv", "json", "excel", "database", "api"],
    "analysis_type": ["general", "sentiment", "entities", "summary"]
}) 