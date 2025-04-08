"""
Model preferences for MCP servers.

This module implements the ModelPreferences capability from the MCP protocol,
allowing servers to express preferences for model selection during sampling.
"""
from typing import List, Optional


class ModelHint:
    """
    Hint for model selection.
    
    Model hints allow the server to suggest specific models or model families
    that would be appropriate for a given task.
    """
    
    def __init__(self, name: str):
        """
        Initialize a model hint.
        
        Args:
            name: A hint for a model name (e.g., 'claude-3-5-sonnet', 'sonnet', 'claude').
                 This should be treated as a substring matching.
        """
        self.name = name
        
    def to_dict(self) -> dict:
        """Convert model hint to dictionary."""
        return {"name": self.name}


class ModelPreferences:
    """
    Preferences for model selection.
    
    Because LLMs can vary along multiple dimensions (capability, cost, speed),
    this interface allows servers to express their priorities to help
    clients make appropriate model selections.
    
    These preferences are always advisory. The client may ignore them.
    """
    
    def __init__(
        self,
        intelligence_priority: float = 0.5,
        speed_priority: float = 0.5,
        cost_priority: float = 0.5,
        hints: Optional[List[ModelHint]] = None
    ):
        """
        Initialize model preferences.
        
        Args:
            intelligence_priority: How much to prioritize intelligence/capabilities (0.0-1.0).
                Default: 0.5
            speed_priority: How much to prioritize sampling speed/latency (0.0-1.0).
                Default: 0.5
            cost_priority: How much to prioritize cost (0.0-1.0).
                Default: 0.5
            hints: Optional model hints in preference order.
        """
        # Clamp values between 0 and 1
        self.intelligence_priority = max(0.0, min(1.0, intelligence_priority))
        self.speed_priority = max(0.0, min(1.0, speed_priority))
        self.cost_priority = max(0.0, min(1.0, cost_priority))
        self.hints = hints or []
        
    def to_dict(self) -> dict:
        """Convert model preferences to dictionary."""
        return {
            "intelligencePriority": self.intelligence_priority,
            "speedPriority": self.speed_priority,
            "costPriority": self.cost_priority,
            "hints": [hint.to_dict() for hint in self.hints]
        }


# Pre-defined preference templates for common use cases

BALANCED_PREFERENCES = ModelPreferences(
    intelligence_priority=0.5,
    speed_priority=0.5,
    cost_priority=0.5
)

INTELLIGENCE_FOCUSED = ModelPreferences(
    intelligence_priority=0.9,
    speed_priority=0.3,
    cost_priority=0.3,
    hints=[ModelHint("claude-3-5-opus")]
)

SPEED_FOCUSED = ModelPreferences(
    intelligence_priority=0.3,
    speed_priority=0.9,
    cost_priority=0.5,
    hints=[ModelHint("claude-3-haiku"), ModelHint("gemini-flash")]
)

COST_FOCUSED = ModelPreferences(
    intelligence_priority=0.3,
    speed_priority=0.5,
    cost_priority=0.9,
    hints=[ModelHint("mistral"), ModelHint("gemini-flash")]
) 