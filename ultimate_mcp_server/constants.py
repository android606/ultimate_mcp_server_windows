"""
Global constants and enumerations for the Ultimate MCP Server.

This module defines system-wide constants, enumerations, and mappings used throughout
the Ultimate MCP Server codebase. Centralizing these values ensures consistency across
the application and simplifies maintenance when values need to be updated.

The module includes:

- Provider enum: Supported LLM providers (OpenAI, Anthropic, etc.)
- TaskType enum: Categories of tasks that can be performed with LLMs
- LogLevel enum: Standard logging levels
- COST_PER_MILLION_TOKENS: Cost estimates for different models
- DEFAULT_MODELS: Default model mappings for each provider
- EMOJI_MAP: Emoji icons for enhanced logging and visualization

These constants should be imported and used directly rather than duplicating their
values in other parts of the codebase. This approach ensures that when values need
to be updated (e.g., adding a new provider or updating pricing), changes only need
to be made in this central location.

Example usage:
    ```python
    from ultimate_mcp_server.constants import Provider, TaskType, EMOJI_MAP
    
    # Use provider enum
    default_provider = Provider.OPENAI
    
    # Get emoji for logging
    success_emoji = EMOJI_MAP["success"]  # ✅
    
    # Check task type
    if task_type == TaskType.COMPLETION:
        # Handle completion task
    ```
"""
from enum import Enum
from typing import Dict


class Provider(str, Enum):
    """
    Enumeration of supported LLM providers in the Ultimate MCP Server.
    
    This enum defines the canonical names for each supported large language model
    provider in the system. These identifiers are used consistently throughout the
    codebase for:
    
    - Configuration settings (provider-specific API keys, endpoints, etc.)
    - Tool parameters (selecting which provider to use for a task)
    - Logging and error reporting (identifying the source of requests/responses)
    - Cost calculation and billing (provider-specific pricing models)
    
    New providers should be added here as they are integrated into the system.
    The string values should be lowercase and match the provider's canonical name
    where possible, as these values appear in API requests/responses.
    
    Usage:
        ```python
        # Reference a provider by enum
        default_provider = Provider.OPENAI
        
        # Convert between string and enum
        provider_name = "anthropic"
        provider_enum = Provider(provider_name)  # Provider.ANTHROPIC
        
        # Check if a provider is supported
        if user_provider in Provider.__members__.values():
            use_provider(user_provider)
        ```
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    GROK = "grok"


class TaskType(str, Enum):
    """
    Enumeration of task types that can be performed by LLMs in the system.
    
    This enum categorizes the different types of operations that LLMs can perform
    within the MCP ecosystem. These task types are used for:
    
    - Logging and analytics (tracking usage patterns by task type)
    - Prompt selection (optimizing prompts for specific task types)
    - Resource allocation (prioritizing resources for different task types)
    - Performance monitoring (measuring success rates by task category)
    
    The categorization helps organize tools in a semantically meaningful way and
    provides metadata for optimizing the system's handling of different tasks.
    When tools register with the system, they typically specify which task type
    they represent.
    
    Task types are roughly organized into these categories:
    - Text generation (COMPLETION, GENERATION, etc.)
    - Analysis and understanding (ANALYSIS, CLASSIFICATION, etc.)
    - Data manipulation (EXTRACTION, TRANSLATION, etc.)
    - System interaction (DATABASE, BROWSER, etc.)
    - Document operations (DOCUMENT_PROCESSING, etc.)
    
    Usage:
        ```python
        # Log with task type
        logger.info("Generating text completion", task_type=TaskType.COMPLETION)
        
        # Register tool with its task type
        @register_tool(name="generate_text", task_type=TaskType.COMPLETION)
        async def generate_text(prompt: str):
            # Implementation
        ```
    """
    COMPLETION = "completion"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    QA = "qa"
    DATABASE = "database"
    QUERY = "query"
    BROWSER = "browser"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    DOCUMENT_PROCESSING = "document_processing"
    DOCUMENT = "document"


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Cost estimates for model pricing (in dollars per million tokens)
# This constant defines the estimated costs for different models, used for cost tracking and budgeting
# Values represent US dollars per million tokens, differentiated by input (prompt) and output (completion) costs
# These costs may change as providers update their pricing, and should be periodically reviewed
COST_PER_MILLION_TOKENS: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    
    # Claude models
    "claude-3-7-sonnet-20250219": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},

    # DeepSeek models
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    
    # Gemini models
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.35, "output": 1.05},
    "gemini-2.0-flash-thinking-exp-01-21": {"input": 0.0, "output": 0.0},
    "gemini-2.5-pro-exp-03-25": {"input": 1.25, "output": 10.0},

    # OpenRouter models
    "mistralai/mistral-nemo": {"input": 0.035, "output": 0.08},
    
    # Grok models (based on the provided documentation)
    "grok-3-latest": {"input": 3.0, "output": 15.0},
    "grok-3-fast-latest": {"input": 5.0, "output": 25.0},
    "grok-3-mini-latest": {"input": 0.30, "output": 0.50},
    "grok-3-mini-fast-latest": {"input": 0.60, "output": 4.0},
}


# Default models by provider
# This mapping defines the recommended default model for each supported provider
# Used when no specific model is requested in API calls or tool invocations
# These defaults aim to balance quality, speed, and cost for general-purpose usage
DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4.1-mini",
    Provider.ANTHROPIC: "claude-3-5-haiku-20241022",
    Provider.DEEPSEEK: "deepseek-chat",
    Provider.GEMINI: "gemini-2.5-pro-exp-03-25",
    Provider.OPENROUTER: "mistralai/mistral-nemo",
    Provider.GROK: "grok-3-latest"
}


# Emoji mapping by log type and action
# Provides visual indicators for different log types, components, and actions
# Used in rich logging output to improve readability and visual scanning
# Organized into sections: general status, components, tasks, and providers
EMOJI_MAP = {
    "start": "🚀",
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "debug": "🔍",
    "critical": "🔥",
    
    # Component-specific emojis
    "server": "🖥️",
    "cache": "💾",
    "provider": "🔌",
    "request": "📤",
    "response": "📥",
    "processing": "⚙️",
    "model": "🧠",
    "config": "🔧",
    "token": "🔢",
    "cost": "💰",
    "time": "⏱️",
    "tool": "🛠️",
    "tournament": "🏆",
    "cancel": "🛑",
    "database": "🗄️",
    "browser": "🌐",
    
    # Task-specific emojis
    "completion": "✍️",
    "summarization": "📝",
    "extraction": "🔍",
    "generation": "🎨",
    "analysis": "📊",
    "classification": "🏷️",
    "query": "🔍",
    "browser_automation": "🌐",
    "database_interactions": "🗄️",
    "download": "⬇️",
    "upload": "⬆️",
    "document_processing": "📄",
    "document": "📄",
    "translation": "🔄",
    "qa": "❓",
    
    # Provider-specific emojis
    Provider.OPENAI: "🟢",
    Provider.ANTHROPIC: "🟣",
    Provider.DEEPSEEK: "🟠", 
    Provider.GEMINI: "🔵",
    Provider.OPENROUTER: "🌐",
    Provider.GROK: "⚡"
}